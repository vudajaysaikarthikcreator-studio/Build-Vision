"""Guaranteed floor-plan engine.

The standard planner (layout_generator.py) is tuned for typical plots but can
give up ("rooms cannot fit") on narrow frontages, custom room sizes or unusual
setbacks.  This engine never gives up while a plan is physically possible:

1. Reserve the road-side strip for parking / garden / loading bay.
2. Choose a building footprint inside what is left (anchored to the south-west
   corner so open land stays north/east, as Vastu prefers).
3. Assign every room a Vastu zone with the capacity-aware assigner
   (vastu/vastu_engine.py) - or a public->private functional order when Vastu
   is off.
4. Tile the footprint with a zone-aware recursive guillotine packer: cuts follow
   the 3x3 Vastu grid (south|middle|north rows, west|middle|east columns), so a
   room assigned to the NE really ends up in the NE, while cut positions follow
   room areas and never violate the minimum dimensions.
5. If the plot is too tight for the standard minimum sizes, minimum dimensions
   are relaxed in small steps (down to 60%) and the compression is reported.

Rooms tile the footprint exactly (no gaps, no overlaps, nothing outside the
plot) by construction; run_all_checks() re-verifies it.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from planning.building_programs import ROOM_DIMS, MIN_DIMS
from vastu.vastu_engine import (
    ZONE_GRID, assign_zones_optimal, rule_for, zone_score, zone9_of_point, SITE_TYPES,
)
from vastu.vastu_rules import ACCEPTABLE_ZONES
from .collision_checker import run_all_checks
from . import sizing as _sizing
from . import quality as _quality
from .layout_generator import (
    DOOR_SIDE_FOR_ZONE, _setback_for_plot, _gap_report, _alignment_report,
)

_ZONE_RC_TOP = {z: (r, c) for r, row in enumerate(ZONE_GRID) for c, z in enumerate(row)}
_RHO_STEPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
_PUBLIC = {"living_room", "reception", "lobby", "waiting_area", "showroom", "restaurant_dining", "teller_area",
           "open_work_area", "multipurpose_hall", "auditorium", "corridor_lobby", "security_room", "cashier"}
_SEMI = {"dining_room", "dining_hall", "kitchen", "study_room", "pantry", "meeting_room", "conference_room",
         "library", "common_room", "staircase", "office_cabin", "classroom", "consultation_room", "pharmacy",
         "restaurant_kitchen", "stage", "guest_room", "hostel_room"}


class LayoutInfeasible(ValueError):
    """The requested spaces cannot physically fit, even at minimum sizes."""


class _OutOfBudget(Exception):
    pass


# --------------------------------------------------------------------- items
def _rc(zone: str) -> Tuple[int, int]:
    r_top, c = _ZONE_RC_TOP.get(zone, (1, 1))
    return 2 - r_top, c                          # r: 0 = south row, 2 = north row; c: 0 = west, 2 = east


def _make_items(allocations, rho: float) -> List[dict]:
    items = []
    for i, a in enumerate(allocations):
        dims = tuple(a.dims) if getattr(a, "dims", None) else _sizing.base_dims(a)
        mins = tuple(a.min_dims) if getattr(a, "min_dims", None) else MIN_DIMS.get(a.room_type, (6.0, 8.0))
        mins = (min(mins[0], dims[0]) * rho, min(mins[1], dims[1]) * rho)
        site = a.room_type in SITE_TYPES
        items.append({
            "i": i, "alloc": a, "type": a.room_type, "label": a.label, "site": site,
            "dims": dims, "mn": (round(mins[0], 2), round(mins[1], 2)),
            "area": float(dims[0] * dims[1]), "zone": a.zone, "rc": _rc(a.zone),
            # Optional fixed orientation: the rectangle must span >= (x-extent, y-extent) - used to
            # make an upper-floor staircase nest exactly over the staircase below.
            "orient_min": (tuple(round(v * rho, 2) for v in a.orient_min) if getattr(a, "orient_min", None) else None),
            "max_aspect": 6.0 if (site or a.room_type in {"corridor_lobby", "loading_bay", "balcony", "garden"}) else 3.6,
        })
    return items


def _functional_zones(items: List[dict], road: str) -> None:
    """Without Vastu: public rooms next to the road, private ones at the back."""
    road = road.lower()
    counters = [0, 0, 0]
    for it in items:
        rank = 0 if it["type"] in _PUBLIC else 1 if it["type"] in _SEMI else 2
        k = counters[rank]
        counters[rank] += 1
        spread = k % 3
        if road == "north":
            rc = (2 - rank, spread)
        elif road == "south":
            rc = (rank, spread)
        elif road == "east":
            rc = (spread, 2 - rank)
        else:
            rc = (spread, rank)
        it["rc"] = rc


# ------------------------------------------------------------------- packing
class _Ctx:
    def __init__(self, max_aspect_bonus: float = 0.0, limit: int = 12000):
        self.nodes = 0
        self.limit = limit
        self.aspect_bonus = max_aspect_bonus


def _leaf_ok(it: dict, w: float, h: float, ctx: _Ctx) -> bool:
    om = it.get("orient_min")
    if om and not (w >= om[0] - 1e-6 and h >= om[1] - 1e-6):
        return False
    a, b = it["mn"]
    if not ((w >= a - 1e-6 and h >= b - 1e-6) or (w >= b - 1e-6 and h >= a - 1e-6)):
        return False
    lo, hi = min(w, h), max(w, h)
    return lo > 0 and hi / lo <= it["max_aspect"] + ctx.aspect_bonus


def _min_extent(group: Sequence[dict], other: float) -> float:
    """Lower bound on the extent a group needs when its cross-dimension is ``other``."""
    if other <= 0:
        return math.inf
    per_room = 0.0
    area = 0.0
    for it in group:
        a, b = it["mn"]
        if other < min(a, b) - 1e-6:
            return math.inf
        per_room = max(per_room, min(a, b))
        area += a * b
    return max(per_room, area / other)


def _area(group: Sequence[dict]) -> float:
    return sum(g["area"] for g in group)


def _candidate_cuts(items: List[dict], w: float, h: float):
    """Yield (axis, groupA, groupB) cuts.  axis 'x' splits the N-S extent (A = south part), 'y' the E-W extent (A = west)."""
    cuts = []
    for axis, key in (("x", 0), ("y", 1)):
        values = sorted({it["rc"][key] for it in items})
        for t in values[:-1]:
            a = [it for it in items if it["rc"][key] <= t]
            b = [it for it in items if it["rc"][key] > t]
            if a and b:
                cuts.append((axis, a, b))
    if not cuts:                                            # all rooms share one zone cell: balanced split
        ordered = sorted(items, key=lambda it: (-it["area"], it["i"]))
        ga: List[dict] = []
        gb: List[dict] = []
        sa = sb = 0.0
        for it in ordered:
            if sa <= sb:
                ga.append(it); sa += it["area"]
            else:
                gb.append(it); sb += it["area"]
        ga.sort(key=lambda it: it["i"]); gb.sort(key=lambda it: it["i"])
        axis = "x" if w >= h else "y"
        cuts.append((axis, ga, gb))
        cuts.append(("y" if axis == "x" else "x", ga, gb))

    def quality(cut):
        axis, a, b = cut
        fa = _area(a) / max(_area(a) + _area(b), 1e-9)
        if axis == "x":
            ca, cb = (w * fa, h), (w * (1 - fa), h)
        else:
            ca, cb = (w, h * fa), (w, h * (1 - fa))
        worst = max(max(ca) / max(min(ca), 1e-6), max(cb) / max(min(cb), 1e-6))
        return worst + abs(0.5 - fa) * 0.5

    strict = sorted(cuts, key=quality)
    # Loose cuts: keep the zone ordering (south->north or west->east, ties by the other axis) but allow a
    # cut anywhere in that order.  Tried only after every strict zone cut failed, so tight plots still fit.
    seen = {(axis, tuple(sorted(it["i"] for it in a))) for axis, a, _ in strict}
    loose = []
    if len(items) > 2:
        for axis, key, other in (("x", 0, 1), ("y", 1, 0)):
            order = sorted(items, key=lambda it: (it["rc"][key], it["rc"][other], it["i"]))
            for k in range(1, len(order)):
                a, b = order[:k], order[k:]
                sig = (axis, tuple(sorted(it["i"] for it in a)))
                if sig in seen:
                    continue
                seen.add(sig)
                loose.append((axis, a, b))
    return strict + sorted(loose, key=quality)


def _pack(items: List[dict], rect: Tuple[float, float, float, float], ctx: _Ctx) -> Optional[List[Tuple[dict, tuple]]]:
    ctx.nodes += 1
    if ctx.nodes > ctx.limit:
        raise _OutOfBudget()
    x, y, w, h = rect
    if len(items) == 1:
        return [(items[0], rect)] if _leaf_ok(items[0], w, h, ctx) else None
    if sum(it["mn"][0] * it["mn"][1] for it in items) > w * h + 1e-6:
        return None
    for axis, ga, gb in _candidate_cuts(items, w, h):
        ext, other = (w, h) if axis == "x" else (h, w)
        need_a, need_b = _min_extent(ga, other), _min_extent(gb, other)
        if need_a + need_b > ext + 1e-6:
            continue
        fa = _area(ga) / max(_area(ga) + _area(gb), 1e-9)
        pos = round(min(max(ext * fa, need_a), ext - need_b), 2)
        pos = min(max(pos, need_a), ext - need_b)
        if axis == "x":
            ra, rb = (x, y, pos, h), (round(x + pos, 2), y, round(w - pos, 2), h)
        else:
            ra, rb = (x, y, w, pos), (x, round(y + pos, 2), w, round(h - pos, 2))
        left = _pack(ga, ra, ctx)
        if left is None:
            continue
        right = _pack(gb, rb, ctx)
        if right is None:
            continue
        return left + right
    return None


# ------------------------------------------------------------- site strip
def _place_site(items: List[dict], env: Tuple[float, float, float, float], road: str, depth: float,
                rho: float) -> Optional[Tuple[List[Tuple[dict, tuple]], Tuple[float, float, float, float]]]:
    """Place parking/garden/loading in a strip of ``depth`` ft along the road edge.

    ``env`` is (x0, y0, dx, dy).  Returns (placed rects, remaining indoor rect).
    """
    x0, y0, dx, dy = env
    road = road.lower()
    along = dy if road in ("north", "south") else dx            # length of the road edge
    order = sorted(items, key=lambda it: (it["rc"][1] if road in ("north", "south") else it["rc"][0], it["i"]))
    widths = []
    for it in order:
        a, b = it["dims"]
        widths.append(min(b, along))
    total = sum(widths)
    if total > along + 1e-6:
        scale = along / total
        widths = [w * scale for w in widths]
    placed = []
    cursor = 0.0
    for it, wd in zip(order, widths):
        wd = round(wd, 2)
        a, b = it["mn"]
        if depth + 1e-6 < min(a, b) or wd + 1e-6 < min(a, b):
            return None
        if not ((depth >= a - 1e-6 and wd >= b - 1e-6) or (depth >= b - 1e-6 and wd >= a - 1e-6)):
            return None
        if road == "north":
            rect = (round(x0 + dx - depth, 2), round(y0 + cursor, 2), round(depth, 2), wd)
        elif road == "south":
            rect = (round(x0, 2), round(y0 + cursor, 2), round(depth, 2), wd)
        elif road == "east":
            rect = (round(x0 + cursor, 2), round(y0 + dy - depth, 2), wd, round(depth, 2))
        else:
            rect = (round(x0 + cursor, 2), round(y0, 2), wd, round(depth, 2))
        placed.append((it, rect))
        cursor += wd
    if road == "north":
        rest = (x0, y0, round(dx - depth, 2), dy)
    elif road == "south":
        rest = (round(x0 + depth, 2), y0, round(dx - depth, 2), dy)
    elif road == "east":
        rest = (x0, y0, dx, round(dy - depth, 2))
    else:
        rest = (x0, round(y0 + depth, 2), dx, round(dy - depth, 2))
    return placed, rest


# -------------------------------------------------------------------- scoring
def _score_rooms(rooms: List[dict], env: dict, vastu: bool) -> float:
    if not vastu:
        return 0.0
    tw = ts = 0.0
    for r in rooms:
        if r["room_type"] in SITE_TYPES:
            continue
        rule = rule_for(r["room_type"])
        z = zone9_of_point(r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0, env)
        tw += rule.weight
        ts += rule.weight * zone_score(rule, z)
    return ts / tw if tw else 0.0


def _aspect_penalty(placed: List[Tuple[dict, tuple]]) -> float:
    worst = 1.0
    for it, (x, y, w, h) in placed:
        lo, hi = min(w, h), max(w, h)
        worst = max(worst, hi / max(lo, 1e-6))
    return worst


def _room_dict(it: dict, rect: tuple, env: dict, ext_env: dict) -> dict:
    a = it["alloc"]
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    actual = zone9_of_point(cx, cy, ext_env if it["site"] else env)
    rule = rule_for(it["type"])
    acceptable = (rule.grade(actual) in ("ideal", "acceptable")) or actual in ACCEPTABLE_ZONES.get(it["type"], [])
    return {
        "room_type": it["type"], "label": it["label"], "zone": actual,
        "preferred_zone": a.preferred_zone, "matched": actual == a.preferred_zone,
        "vastu_acceptable": bool(acceptable), "door_side": DOOR_SIDE_FOR_ZONE.get(actual, "S"),
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2),
        "min_dims": [round(it["mn"][0], 2), round(it["mn"][1], 2)],
        **({"custom": True} if getattr(a, "custom", False) else {}),
    }


# ------------------------------------------------------------------ main entry
def guaranteed_layout(
    plot_length: float,
    plot_width: float,
    allocations: Sequence,
    road_direction: str = "South",
    building_type: str = "custom",
    floors: int = 1,
    vastu: Optional[bool] = None,
    setback_ft: Optional[float] = None,
    setbacks: Optional[Dict[str, float]] = None,
    main_entrance_direction: Optional[str] = None,
    pin_stair: Optional[dict] = None,
) -> Dict:
    """Build a collision-free, boundary-safe layout or raise LayoutInfeasible.

    ``pin_stair`` = {"rect": (x0, y0, x1, y1), "env": (x, y, w, h)} forces the staircase to cover that
    rectangle (the staircase of the floor below) inside the given outer footprint.
    """
    if plot_length <= 0 or plot_width <= 0:
        raise ValueError("Plot dimensions must be positive.")
    allocations = list(allocations)
    if not allocations:
        raise ValueError("At least one room is required.")
    road = str(road_direction or "South").strip().lower()
    if road not in ("north", "south", "east", "west"):
        road = "south"
    entrance = str(main_entrance_direction or road_direction or "South").strip().lower()
    if entrance not in ("north", "south", "east", "west"):
        entrance = road
    if vastu is None:
        vastu = any((a.preferred_zone or "C") != "C" for a in allocations)
    base_sb = float(setback_ft) if setback_ft is not None else _setback_for_plot(plot_length, plot_width)
    user_sb = {k: float(v) for k, v in (setbacks or {}).items() if v is not None and k in ("north", "east", "south", "west")}
    sb_options = [base_sb] + [s for s in (1.5, 1.0, 0.5, 0.0) if s < base_sb]
    notes: List[str] = []
    best = None

    for sb_i, sb in enumerate(sb_options):
        sides = {k: (user_sb.get(k, sb) if sb_i == 0 else min(user_sb.get(k, sb), sb)) for k in ("north", "east", "south", "west")}
        ex0, ey0 = sides["south"], sides["west"]
        edx = plot_length - sides["south"] - sides["north"]
        edy = plot_width - sides["west"] - sides["east"]
        if edx < 6 or edy < 6:
            continue
        for rho in _RHO_STEPS:
            result = _try_footprint(plot_length, plot_width, allocations, road, vastu, (ex0, ey0, edx, edy), rho,
                                     building_type=str(building_type or "custom").strip().lower(),
                                     entrance_direction=entrance, pin_stair=pin_stair)
            if result is not None:
                best = (result, sb_i, sb, sides, rho)
                break
        if best:
            break
    if best is None:
        raise LayoutInfeasible(
            f"The requested {len(allocations)} spaces cannot fit on a {plot_length:g} x {plot_width:g} ft plot, "
            "even at 60% of their minimum sizes. Remove a few rooms, add a floor, or use a larger plot.")
    (rooms, env, score, ctx_note), sb_i, sb, sides, rho = best
    if sb_i > 0:
        notes.append(f"Setback was reduced from {base_sb:g} ft to {sb:g} ft to fit all rooms.")
    if rho < 1.0:
        notes.append(f"Rooms were compressed to {int(round(rho * 100))}% of their standard minimum sizes to fit this plot; "
                     "consider a larger plot, fewer rooms or an extra floor for comfortable sizes.")

    checks = run_all_checks(rooms, plot_length, plot_width)
    if not checks["passed"]:
        raise LayoutInfeasible("Internal layout validation failed: " + ", ".join(
            k for k in ("collisions", "boundary_violations", "dimension_violations") if checks[k]))

    indoor = [r for r in rooms if r["room_type"] not in SITE_TYPES]
    match_count = sum(1 for r in indoor if r.get("matched"))
    acceptable_count = sum(1 for r in indoor if r.get("vastu_acceptable"))
    indoor_rooms = [r for r in rooms if r["room_type"] not in SITE_TYPES]
    env_source = indoor_rooms or rooms
    all_x = [r["x"] for r in env_source]; all_y = [r["y"] for r in env_source]
    envelope = {"x": round(min(all_x), 2), "y": round(min(all_y), 2),
                "w": round(max(r["x"] + r["w"] for r in env_source) - min(all_x), 2),
                "h": round(max(r["y"] + r["h"] for r in env_source) - min(all_y), 2)}
    layout = {
        "plot": {"length": round(plot_length, 2), "width": round(plot_width, 2),
                 "area": round(plot_length * plot_width, 2)},
        "floors": floors, "road_direction": road_direction,
        "setback_ft": round(max(sides.values()), 2), "setbacks_ft": {k: round(v, 2) for k, v in sides.items()},
        "building_envelope": envelope, "occupied_envelope": dict(envelope),
        "rooms": rooms, "collisions": checks["collisions"], "boundary_violations": checks["boundary_violations"],
        "dimension_violations": checks["dimension_violations"], "passed_checks": True,
        "vastu_match_count": match_count, "vastu_acceptable_count": acceptable_count,
        "vastu_score": round(score, 4),
        "vastu_match_percent": round(100.0 * match_count / max(len(indoor), 1), 1),
        "vastu_enabled": bool(vastu),
        "vastu_layout_engine": "Capacity-aware Vastu zone assignment + zone-grid guillotine packing (guaranteed fit)",
        "building_type": str(building_type or "custom").strip().lower(),
        "compression_ratio": rho, "layout_notes": notes,
        "gap_report": _gap_report(rooms), "circulation": [], "alignment_fallback": None,
    }
    layout["planning_envelope"] = {"x": round(ex0, 2), "y": round(ey0, 2),
                                 "w": round(edx, 2), "h": round(edy, 2)}
    layout["site_spaces"] = [r for r in rooms if r.get("room_type") in SITE_TYPES]
    layout["no_unplanned_gaps"] = layout["gap_report"].get("max_internal_gap_ft", 0.0) <= 0.20
    try:
        rep = _alignment_report(rooms)
        rep["layout_grid"] = "ZONE_GRID_GUILLOTINE"
        layout["alignment_report"] = rep
    except Exception:  # noqa: BLE001 - reporting only
        layout["alignment_report"] = {"layout_grid": "ZONE_GRID_GUILLOTINE"}
    return layout


def _place_site_vastu(items: List[dict], env: Tuple[float, float, float, float], entrance: str, rho: float):
    """Place a single residential parking space in a Vastu-compatible corner.

    Road-side parking used to consume the entire entrance edge.  On a north/east
    facing Vastu plan that could leave the requested main-door room stranded away
    from the exterior wall.  Parking is therefore anchored to NW for north/west
    approaches and SE for south/east approaches, while staying on the road-facing
    side.
    """
    if len(items) != 1 or items[0]["type"] != "parking":
        return None
    it = items[0]
    x0, y0, dx, dy = env
    depth, frontage = [float(v) * max(0.75, rho) for v in it["dims"]]
    depth = min(depth, dx)
    frontage = min(frontage, dy)
    if depth < it["mn"][0] - 1e-6 or frontage < it["mn"][1] - 1e-6:
        return None
    entrance = str(entrance or "north").lower()
    use_nw = entrance in {"north", "south"}
    if use_nw:
        rect = (round(x0 + dx - depth, 2), round(y0, 2), round(depth, 2), round(frontage, 2))
        # Carve the west frontage from the planning envelope. The house keeps
        # the full south/north depth, including the requested entrance edge.
        rest = (round(x0, 2), round(y0 + frontage, 2), round(dx, 2), round(dy - frontage, 2))
    else:
        rect = (round(x0, 2), round(y0 + dy - frontage, 2), round(depth, 2), round(frontage, 2))
        # Carve the south depth from the planning envelope. The house keeps the
        # full west/east frontage, including an east/west requested entrance.
        rest = (round(x0 + depth, 2), round(y0, 2), round(dx - depth, 2), round(dy, 2))
    return [(it, rect)], rest


def _anchor_footprint(rest: Tuple[float, float, float, float], fx: float, fy: float,
                      entrance: str) -> Tuple[float, float, float, float]:
    """Anchor a compact house footprint to the requested entrance side."""
    rx, ry, rdx, rdy = rest
    direction = str(entrance or "south").lower()
    if direction == "north":
        return (round(rx + rdx - fx, 2), round(ry, 2), round(fx, 2), round(fy, 2))
    if direction == "east":
        return (round(rx, 2), round(ry + rdy - fy, 2), round(fx, 2), round(fy, 2))
    if direction == "west":
        return (round(rx, 2), round(ry, 2), round(fx, 2), round(fy, 2))
    return (round(rx, 2), round(ry, 2), round(fx, 2), round(fy, 2))


def _try_footprint(plot_length, plot_width, allocations, road, vastu, env_rect, rho,
                   building_type: str = "custom", entrance_direction: str = "south", pin_stair: Optional[dict] = None):
    ex0, ey0, edx, edy = env_rect
    items = _make_items(allocations, rho)
    site = [it for it in items if it["site"]]
    indoor = [it for it in items if not it["site"]]
    if not indoor:                                             # e.g. a "garden only" floor
        indoor, site = site, []

    site_rects, rest = [], env_rect
    # For residential Vastu plans, preserve the requested entrance edge and put
    # the usual parking corner on a different part of the frontage. Other site
    # spaces and non-Vastu plans retain the road-strip behaviour.
    residential = str(building_type or "custom").lower() in {"1bhk","2bhk","3bhk","4bhk","duplex","villa","studio","farmhouse","custom"}
    if site and vastu and residential and len(site) == 1 and site[0]["type"] == "parking":
        placed_site = _place_site_vastu(site, env_rect, entrance_direction, rho)
        if placed_site is not None:
            site_rects, rest = placed_site
    elif site:
        depth_targets: List[float] = [0.0]
        d0 = max(max(it["dims"][0], 0.0) for it in site)
        dmin = max(min(it["mn"]) for it in site)
        cands = sorted({round(max(dmin, d0 * f), 1) for f in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)}, reverse=True)
        depth_targets = [d for d in cands if d <= (edx if road in ("north", "south") else edy) * 0.6] or [dmin]
        for depth in depth_targets:
            placed_site = _place_site(site, env_rect, road, depth, rho)
            if placed_site is not None:
                site_rects, rest = placed_site
                break
        if not site_rects and site:
            return None

    rx, ry, rdx, rdy = rest
    if rdx < 5 or rdy < 5:
        return None
    need = sum(it["mn"][0] * it["mn"][1] for it in indoor)
    if need > rdx * rdy + 1e-6:
        return None
    total_target = sum(it["area"] for it in indoor)
    # Compact but not cramped: avoid the old long, narrow house block and leave
    # meaningful open space outside the building footprint. The room pack itself
    # is still gap-free within this compact footprint.
    target_factor = 1.22 if residential else 1.08
    min_factor = 1.38 if residential else 1.15
    fp_area = min(rdx * rdy, max(total_target * target_factor, need * min_factor))
    return _pack_indoor(indoor, (rx, ry, rdx, rdy), fp_area, road, vastu, rho, allocations, site_rects, env_rect,
                        entrance_direction=entrance_direction, pin_stair=pin_stair)


_MIN_STRIP = 4.5          # a strip thinner than this cannot hold a room; it is absorbed into the stair


def _pack_around(items: List[dict], stair: dict, fp: Tuple[float, float, float, float],
                 pin: Tuple[float, float, float, float], ctx: _Ctx):
    """Tile ``fp`` with the staircase at exactly ``pin`` and every other item around it.

    The footprint is cut (guillotine style) into the stair plus up to four rectangular regions
    (west/east strips and the pieces below/above the stair); the remaining rooms are shared out
    between those regions by Vastu zone and area, and each region is packed with the normal packer.
    """
    fx, fy, fw, fh = fp
    px0, py0 = max(pin[0], fx), max(pin[1], fy)
    px1, py1 = min(pin[2], fx + fw), min(pin[3], fy + fh)
    if px1 - px0 < 3.0 or py1 - py0 < 3.0:
        return None
    if px0 - fx < _MIN_STRIP:
        px0 = fx
    if fx + fw - px1 < _MIN_STRIP:
        px1 = fx + fw
    if py0 - fy < _MIN_STRIP:
        py0 = fy
    if fy + fh - py1 < _MIN_STRIP:
        py1 = fy + fh
    regions = []
    if px0 - fx > 1e-6:
        regions.append((fx, fy, px0 - fx, fh))
    if fx + fw - px1 > 1e-6:
        regions.append((px1, fy, fx + fw - px1, fh))
    if py0 - fy > 1e-6:
        regions.append((px0, fy, px1 - px0, py0 - fy))
    if fy + fh - py1 > 1e-6:
        regions.append((px0, py1, px1 - px0, fy + fh - py1))
    stair_rect = (round(px0, 2), round(py0, 2), round(px1 - px0, 2), round(py1 - py0, 2))
    if not regions:
        return [(stair, stair_rect)] if not items else None
    if not items:
        return None
    rcs = [(min(2, int(3 * (r[0] + r[2] / 2.0 - fx) / fw)), min(2, int(3 * (r[1] + r[3] / 2.0 - fy) / fh)))
           for r in regions]
    groups: List[List[dict]] = [[] for _ in regions]
    load_t = [0.0] * len(regions)
    load_m = [0.0] * len(regions)
    for it in items:
        best_j, best_s = None, None
        mn_area = it["mn"][0] * it["mn"][1]
        for j, r in enumerate(regions):
            area = r[2] * r[3]
            if load_m[j] + mn_area > area * 0.97:
                continue
            d = abs(it["rc"][0] - rcs[j][0]) + abs(it["rc"][1] - rcs[j][1])
            fill_after = (load_t[j] + it["area"]) / area
            score = -0.9 * d + 1.8 * max(0.0, 1.0 - load_t[j] / area) - 4.0 * max(0.0, fill_after - 1.25)
            if best_s is None or score > best_s:
                best_j, best_s = j, score
        if best_j is None:
            return None
        groups[best_j].append(it)
        load_t[best_j] += it["area"]
        load_m[best_j] += mn_area
    for j in range(len(regions)):                      # no region may be left without a room
        if not groups[j]:
            donor = max(range(len(regions)), key=lambda k: len(groups[k]))
            if len(groups[donor]) < 2:
                return None
            groups[j].append(groups[donor].pop())
    placed: List[Tuple[dict, tuple]] = [(stair, stair_rect)]
    for j, r in enumerate(regions):
        sub = _pack(groups[j], (round(r[0], 2), round(r[1], 2), round(r[2], 2), round(r[3], 2)), ctx)
        if sub is None:
            return None
        placed += sub
    return placed


def _slide_to_contain(fp: Tuple[float, float, float, float], rest: Tuple[float, float, float, float],
                      pin: Tuple[float, float, float, float]):
    """Move footprint ``fp`` (inside ``rest``) the shortest way so that it contains the stair ``pin``."""
    x0, y0, w, h = fp
    rx, ry, rdx, rdy = rest
    lo_x, hi_x = max(rx, pin[2] - w), min(pin[0], rx + rdx - w)
    lo_y, hi_y = max(ry, pin[3] - h), min(pin[1], ry + rdy - h)
    if lo_x > hi_x + 1e-6 or lo_y > hi_y + 1e-6:
        return None
    return (round(min(max(x0, lo_x), hi_x), 2), round(min(max(y0, lo_y), hi_y), 2), w, h)


def _pack_indoor(indoor, rest, fp_area, road, vastu, rho, allocations, site_rects, env_rect,
                 entrance_direction: str = "south", pin_stair: Optional[dict] = None):
    rx, ry, rdx, rdy = rest
    full = rdx * rdy
    stair_it = None
    pin = None
    if pin_stair:
        stair_it = next((it for it in indoor if it["type"] == "staircase"), None)
        pin = tuple(pin_stair["rect"]) if stair_it is not None else None
        if pin is not None and not (pin[0] >= rx - 1e-6 and pin[1] >= ry - 1e-6 and
                                    pin[2] <= rx + rdx + 1e-6 and pin[3] <= ry + rdy + 1e-6):
            return None                                 # the lower stair is outside this floor's buildable area
    shapes = []
    f = math.sqrt(min(1.0, fp_area / full))
    shapes.append((round(rdx * f, 2), round(rdy * f, 2)))                    # proportional
    if fp_area < full - 1e-6:
        shapes.append((round(rdx, 2), round(min(rdy, fp_area / rdx), 2)))      # full N-S, shrink E-W
        shapes.append((round(min(rdx, fp_area / rdy), 2), round(rdy, 2)))      # full E-W, shrink N-S
        shapes.append((round(rdx, 2), round(rdy, 2)))                          # everything
    footprints = []
    for fx, fy in shapes:
        fp = _anchor_footprint((rx, ry, rdx, rdy), fx, fy, entrance_direction)
        if pin is not None:
            fp = _slide_to_contain(fp, (rx, ry, rdx, rdy), pin)
            if fp is None:
                continue
        footprints.append(fp)
    if pin is not None and pin_stair.get("env"):                             # same outer walls as the floor below
        e = tuple(pin_stair["env"])
        if (e[0] >= rx - 1e-6 and e[1] >= ry - 1e-6 and e[0] + e[2] <= rx + rdx + 1e-6 and
                e[1] + e[3] <= ry + rdy + 1e-6 and e not in footprints):
            footprints.insert(0, (round(e[0], 2), round(e[1], 2), round(e[2], 2), round(e[3], 2)))
    best = None
    for fi_, fp in enumerate(footprints):
        fx, fy = fp[2], fp[3]
        if fx * fy < sum(it["mn"][0] * it["mn"][1] for it in indoor) - 1e-6:
            continue
        env = {"x": fp[0], "y": fp[1], "w": fp[2], "h": fp[3]}
        variants = _zone_variants(indoor, vastu, road, fx * fy)
        for variant_i, zones in enumerate(variants):
            for it, z in zip(indoor, zones):
                it["zone"] = z
                it["rc"] = _rc(z) if vastu else it["rc"]
            # Try several deterministic room orders. The old packer accepted
            # the first valid guillotine tree, which could satisfy the zone rules
            # logically while pushing a room across a neighbouring zone. Searching
            # a few orderings lets the final geometry keep more rooms inside their
            # preferred Vastu cells without sacrificing the hard collision checks.
            order_variants = [
                list(indoor),
                sorted(indoor, key=lambda it: (it["rc"][0], it["rc"][1], -it["area"], it["i"])),
                sorted(indoor, key=lambda it: (it["rc"][1], it["rc"][0], -it["area"], it["i"])),
                sorted(indoor, key=lambda it: (-it["area"], it["rc"][0], it["rc"][1], it["i"])),
            ]
            for order in order_variants:
                for bonus in (-1.0, 0.0, 1.5, 4.0):
                    ctx = _Ctx(max_aspect_bonus=bonus)
                    try:
                        if pin is not None:
                            placed = _pack_around([it for it in order if it is not stair_it], stair_it, fp, pin, ctx)
                        else:
                            placed = _pack(list(order), fp, ctx)
                    except _OutOfBudget:
                        placed = None
                    if placed is not None:
                        rooms = [_room_dict(it, rect, env, env_full(env_rect)) for it, rect in placed]
                        _q = _quality.evaluate({"rooms": rooms}, "custom")["penalty"]      # strips / oversized utilities
                        score = _score_rooms(rooms, env, vastu) - 0.03 * (_aspect_penalty(placed) - 1.0) - 0.05 * _q
                        if best is None or score > best[0]:
                            site_items = [(it, rect) for it, rect in site_rects]
                            site_rooms = [_room_dict(it, rect, env, env_full(env_rect)) for it, rect in site_items]
                            best = (score, rooms + site_rooms, env)
            if best is None:
                continue
        if best is not None and fi_ == 0 and best[0] >= 0.90:
            break
    if best is None:
        return None
    score, rooms, env = best
    return rooms, env, max(0.0, min(1.0, score)), None


def env_full(env_rect):
    return {"x": env_rect[0], "y": env_rect[1], "w": env_rect[2], "h": env_rect[3]}


def _zone_variants(indoor: List[dict], vastu: bool, road: str, footprint_area: float) -> List[List[str]]:
    if not vastu:
        _functional_zones(indoor, road)
        return [["C"] * len(indoor)]
    payload = [{"room_type": it["type"], "area": it["area"],
                "zone": it["alloc"].zone if getattr(it["alloc"], "fixed_zone", False) else None,
                "hint_zones": getattr(it["alloc"], "hint_zones", None)}
               for it in indoor]
    out: List[List[str]] = []
    for cap in (1.6, 1.3, 2.2):
        out.append(assign_zones_optimal(payload, footprint_area, capacity_factor=cap))
    out.append([it["alloc"].zone if it["alloc"].zone not in (None, "") else "C" for it in indoor])
    seen, uniq = set(), []
    for v in out:
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq
