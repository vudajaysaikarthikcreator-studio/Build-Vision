"""Never-fail layout entry point.

Order of attack for one floor:

1. Standard planner (layout_generator.generate_layout) - unless the request
   uses custom room sizes / setbacks, which only the guaranteed engine honours.
2. Guaranteed engine (guaranteed_layout.guaranteed_layout) - if the standard
   planner raised, returned nothing, or returned a plan with collisions or
   boundary violations.

Whatever comes back has passed run_all_checks(); a genuinely impossible request
raises ``LayoutInfeasible`` with a message the user can act on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .layout_generator import generate_layout, _apply_presentation_options
from .guaranteed_layout import guaranteed_layout, LayoutInfeasible  # noqa: F401  (re-exported)
from . import quality as _quality
from .collision_checker import run_all_checks
from vastu.vastu_engine import SITE_TYPES

# A plan whose quality penalty exceeds this is re-planned with the guaranteed engine
# and the better of the two candidates is kept (see layout/quality.py).
QUALITY_RETRY_PENALTY = 1.0
# Penalty for an upper-floor plan whose staircase cannot be put exactly over the one below.
STACK_PENALTY = 2.5
# Overlap (share of the smaller staircase) that counts as "stacked".
STACK_OK_RATIO = 0.85


def _needs_custom_engine(allocations: Sequence) -> bool:
    return any(getattr(a, "dims", None) or getattr(a, "min_dims", None) or getattr(a, "custom", False)
               for a in allocations)


def _primary_layout(
    plot_length: float,
    plot_width: float,
    allocations: Sequence,
    floors: int = 1,
    road_direction: str = "South",
    building_type: str = "custom",
    wall_width_ft: float = 0.5,
    room_size_details: bool = True,
    appearance: str = "technical",
    main_door_position: str = "center",
    parking_position: str = "road_side",
    staircase_position: str = "road_side",
    custom_mode: bool = False,
    main_entrance_direction: str | None = None,
    hall_shape: str = "auto",
    setback_ft: Optional[float] = None,
    setbacks: Optional[Dict[str, float]] = None,
    bay_frame: Optional[dict] = None,
    area_target: Optional[float] = None,
) -> Dict:
    allocations = list(allocations)
    presentation = dict(wall_width_ft=wall_width_ft, room_size_details=room_size_details, appearance=appearance,
                        main_door_position=main_door_position, parking_position=parking_position,
                        staircase_position=staircase_position)
    fallback_reason: Optional[str] = None
    use_custom = custom_mode or setback_ft is not None or bool(setbacks) or _needs_custom_engine(allocations)

    # Corridor-and-bay planner: regular, aligned plans for villas, duplexes, farmhouses and institutional
    # buildings (school, hostel, hospital ...).  Falls through to the older engines if it cannot fit.
    try:
        from .bay_planner import BAY_TYPES, plan_bays
        from planning.building_programs import canonical_type as _canon
        if _canon(building_type) in BAY_TYPES and not setbacks and not _needs_custom_engine(allocations):
            layout = plan_bays(plot_length, plot_width, allocations, main_entrance_direction or road_direction,
                               road_direction, True, building_type, setback_ft=setback_ft, bay_frame=bay_frame,
                               area_target=area_target)
            layout = _apply_presentation_options(layout, **presentation)
            layout["engine"] = "bay"
            return layout
    except Exception as exc:  # noqa: BLE001 - fall through to the older engines
        fallback_reason_bay = f"corridor planner skipped ({exc})"
    else:
        fallback_reason_bay = None

    # Hall-centred residential planner (per-room doors, attached / under-stair washrooms, L/U/rectangle hall).
    try:
        from .residential_planner import RESIDENTIAL_PLANNER_TYPES, plan_residential
        if str(building_type or "custom").lower() in RESIDENTIAL_PLANNER_TYPES and not setbacks and not _needs_custom_engine(allocations):
            layout = plan_residential(plot_length, plot_width, allocations, main_entrance_direction or road_direction,
                                      road_direction, True, hall_shape, building_type, setback_ft=setback_ft,
                                      parking_position=parking_position)
            layout = _apply_presentation_options(layout, **presentation)
            layout["engine"] = "residential"
            return layout
    except Exception as exc:  # noqa: BLE001 - fall through to the general planners
        fallback_reason_res = f"residential planner skipped ({exc})"
    else:
        fallback_reason_res = None

    if not use_custom:
        try:
            layout = generate_layout(plot_length, plot_width, allocations, floors=floors,
                                     road_direction=road_direction, building_type=building_type, **presentation)
            if (layout and not layout.get("collisions") and not layout.get("boundary_violations")
                    and len(layout.get("rooms", [])) == len(allocations)):
                layout["engine"] = "standard"
                return layout
            fallback_reason = "standard planner returned an incomplete or overlapping plan"
        except Exception as exc:  # noqa: BLE001 - any planner failure must fall through to the guaranteed engine
            fallback_reason = f"standard planner could not place every room ({exc})"
    else:
        fallback_reason = "custom sizes / setbacks use the guaranteed engine"

    layout = guaranteed_layout(plot_length, plot_width, allocations, road_direction=road_direction,
                               building_type=building_type, floors=floors, setback_ft=setback_ft, setbacks=setbacks,
                               main_entrance_direction=main_entrance_direction or road_direction)
    layout = _apply_presentation_options(layout, **presentation)
    layout["engine"] = "guaranteed"
    layout["fallback_reason"] = fallback_reason
    return layout


def _zone_fraction(zone: str):
    """(fx, fy) centre of a 3x3 Vastu grid cell as fractions of the footprint (x: south->north, y: west->east)."""
    from vastu.vastu_engine import ZONE_GRID
    for r_top, row in enumerate(ZONE_GRID):
        for c, z in enumerate(row):
            if z == zone:
                return (2.5 - r_top) / 3.0, (c + 0.5) / 3.0
    return 0.5, 0.5


def _best_stair_zone(g_xy, box_wh, buildable, default_zone):
    """Grid cell (zone) to force the upper-floor staircase into.

    After the plan is shifted so its stair sits exactly over the lower one, the footprint must still lie
    inside the buildable area.  That depends only on where the stair sits inside its own footprint, so
    choose the cell whose centre leaves the largest slack on all four sides.
    """
    from vastu.vastu_engine import ZONE_GRID
    gx, gy = g_xy
    bw, bh = box_wh
    lo_x, hi_x = buildable["x"], buildable["x"] + buildable["w"]
    lo_y, hi_y = buildable["y"], buildable["y"] + buildable["h"]
    best, best_key = default_zone, None
    for r_top, row in enumerate(ZONE_GRID):
        for c, z in enumerate(row):
            fx, fy = (2.5 - r_top) / 3.0, (c + 0.5) / 3.0
            slack = min(gx - lo_x - fx * bw, hi_x - gx - (1 - fx) * bw,
                        gy - lo_y - fy * bh, hi_y - gy - (1 - fy) * bh)
            key = (round(min(slack, bw / 6.0), 1), z == default_zone)      # enough slack beats exactness
            if best_key is None or key > best_key:
                best, best_key = z, key
    return best


def _compact_setbacks(plot_length, plot_width, allocations, factor, base_sb, centre=None, anchor=None):
    """Setbacks that shrink the buildable envelope to ~factor x the programme area.

    Placement of the compact box: centred in the buildable area; around ``centre`` (the middle of the
    floor below); or - for stair stacking - positioned so the lower-floor staircase point
    ``anchor = (x, y, fx, fy)`` falls at fraction (fx, fy) of the box, i.e. in the middle of the grid
    cell the upper staircase is forced into.  The stack then needs only a small shift.
    """
    need = sum(float(getattr(a, "target_area", 0) or 0) for a in allocations) * factor
    bx, by = plot_length - 2 * base_sb, plot_width - 2 * base_sb
    area = max(bx * by, 1.0)
    if need >= area * 0.92:
        return None                                   # plot is already tight; nothing to shrink
    k = (need / area) ** 0.5
    bw, bh = bx * k, by * k
    if anchor is not None:
        x0, y0 = anchor[0] - anchor[2] * bw, anchor[1] - anchor[3] * bh
    elif centre is not None:
        x0, y0 = centre[0] - bw / 2.0, centre[1] - bh / 2.0
    else:
        x0, y0 = base_sb + (bx - bw) / 2.0, base_sb + (by - bh) / 2.0
    x0 = min(max(x0, base_sb), plot_length - base_sb - bw)
    y0 = min(max(y0, base_sb), plot_width - base_sb - bh)
    return {"south": x0, "north": plot_length - x0 - bw, "west": y0, "east": plot_width - y0 - bh}


def _axis_shift(g0: float, g1: float, s0: float, s1: float, lo: float, hi: float, tol: float = 0.25):
    """Shift along one axis that lays the upper stair over the lower one.

    ``lo``/``hi`` bound the allowed shift (keeps the whole plan in the buildable area).  Returns
    (shift, overlap_length): the shift gives the largest overlap reachable inside [lo, hi], preferring
    the one that centres the two stairs.
    """
    if (g1 - g0) >= (s1 - s0):                       # upper stair can sit inside the lower stair
        a, b = g0 - s0, g1 - s1
    else:                                            # lower stair can sit inside the upper stair
        a, b = g1 - s1, g0 - s0
    a, b = a - tol, b + tol                          # shifts giving the full overlap
    lo2, hi2 = max(lo, a), min(hi, b)
    if lo2 <= hi2:
        d = min(max(0.0, lo2), hi2)                      # smallest move that still gives a full overlap
    else:                                            # full overlap unreachable: closest allowed shift
        d = hi if hi < a else lo
    overlap = max(0.0, min(g1, s1 + d) - max(g0, s0 + d))
    return d, overlap


def _stack_shift(layout: Dict, ref: Optional[Dict], box: Optional[Dict] = None, tol: float = 0.05):
    """(dx, dy, ratio) placing this floor's staircase over the lower-floor staircase.

    ``ratio`` is the overlap divided by the smaller stair's area (1.0 = one stair lies fully over
    the other).  Returns None when no shift keeps the plan inside the buildable area and the plot.
    """
    if not ref:
        return (0.0, 0.0, 1.0)
    stairs = [r for r in layout.get("rooms", []) if r.get("room_type") == "staircase"]
    if not stairs:
        return None
    s = max(stairs, key=lambda r: r["w"] * r["h"])
    g = ref["stair"]
    plot = layout.get("plot") or {}
    L, W = float(plot.get("length", 0) or 0), float(plot.get("width", 0) or 0)
    pe = box
    if not pe:
        sb = float(layout.get("setback_ft", 0) or 0)
        pe = {"x": sb, "y": sb, "w": L - 2 * sb, "h": W - 2 * sb}
    lo = {"x": -1e9, "y": -1e9}
    hi = {"x": 1e9, "y": 1e9}
    for r in layout["rooms"]:
        if r.get("room_type") in SITE_TYPES:
            bx0, by0, bx1, by1 = 0.0, 0.0, L, W
        else:
            bx0, by0, bx1, by1 = pe["x"], pe["y"], pe["x"] + pe["w"], pe["y"] + pe["h"]
        lo["x"] = max(lo["x"], bx0 - r["x"] - tol)
        hi["x"] = min(hi["x"], bx1 - (r["x"] + r["w"]) + tol)
        lo["y"] = max(lo["y"], by0 - r["y"] - tol)
        hi["y"] = min(hi["y"], by1 - (r["y"] + r["h"]) + tol)
    if lo["x"] > hi["x"] or lo["y"] > hi["y"]:
        return None
    dx, ox = _axis_shift(g["x"], g["x"] + g["w"], s["x"], s["x"] + s["w"], lo["x"], hi["x"])
    dy, oy = _axis_shift(g["y"], g["y"] + g["h"], s["y"], s["y"] + s["h"], lo["y"], hi["y"])
    ratio = (ox * oy) / max(min(g["w"] * g["h"], s["w"] * s["h"]), 1e-6)
    return (round(dx, 2), round(dy, 2), round(min(ratio, 1.0), 3))


def _stack_penalty(shift) -> float:
    if shift is None:
        return STACK_PENALTY + 0.5
    ratio = shift[2]
    return 0.0 if ratio >= STACK_OK_RATIO else 0.5 + STACK_PENALTY * (1.0 - ratio)


def _translate(layout: Dict, dx: float, dy: float) -> None:
    """Move every geometric element of a floor plan by (dx, dy) in place (shared dicts shifted once)."""
    seen = set()

    def shift(o):
        if isinstance(o, dict):
            if id(o) in seen:
                return
            seen.add(id(o))
            if isinstance(o.get("x"), (int, float)) and isinstance(o.get("y"), (int, float)):
                o["x"] = round(o["x"] + dx, 2)
                o["y"] = round(o["y"] + dy, 2)
            for k, v in o.items():
                if k in ("plot", "planning_envelope", "setbacks_ft"):
                    continue
                shift(v)
        elif isinstance(o, list):
            for v in o:
                shift(v)

    shift(layout)


def _finish(layout: Dict, shift, ref: Optional[Dict]) -> Dict:
    """Apply the stair-stacking translation (verifying the result) and record the outcome."""
    if not ref:
        return layout
    notes = layout.setdefault("layout_notes", [])
    if shift is not None and shift[2] > 0.0:
        dx, dy, ratio = shift
        _translate(layout, dx, dy)
        plot = layout.get("plot") or {}
        chk = run_all_checks(layout["rooms"], float(plot.get("length", 0)), float(plot.get("width", 0)))
        if not chk["collisions"] and not chk["boundary_violations"]:
            layout["stair_stacked"] = ratio >= STACK_OK_RATIO
            layout["stair_overlap"] = ratio
            if ratio >= STACK_OK_RATIO:
                return layout
            notes.append(f"Staircase overlaps the staircase below by {ratio * 100:.0f}% only; "
                         "verify the stair core alignment before construction.")
            return layout
        _translate(layout, -dx, -dy)                       # should never happen; undo and report
    layout["stair_stacked"] = False
    notes.append("Staircase could not be placed above the staircase on the floor below on this plot; "
                 "verify the stair core position before construction.")
    return layout


def safe_generate_layout(plot_length, plot_width, allocations, floors=1, road_direction="South",
                         building_type="custom", **kw) -> Dict:
    """Best-of planner: keep the candidate plan with the best architectural quality.

    Overlap-free is not enough - rooms must keep sensible proportions (no 85 ft
    principal office, no 52 ft restroom strip, no 540 sq.ft staircase) and the plan
    must not leave huge unplanned gaps.  Candidates, in order: the primary engine,
    the guaranteed engine, then the guaranteed engine on a compact envelope sized to
    the space programme (the rest of a large plot stays open site, as in a real school
    or factory compound).  The lowest quality penalty wins.

    For upper floors ``stair_ref`` (the ground-floor staircase rectangle) makes the winning plan
    be shifted so its staircase sits exactly above the one below; plans for which that is
    impossible are penalised so a stackable alternative is preferred.
    """
    stair_ref = kw.pop("stair_ref", None)
    if stair_ref and stair_ref.get("bay_frame"):
        kw["bay_frame"] = stair_ref["bay_frame"]
    allocations = list(allocations)
    n = len(allocations)
    # The real buildable area (plot minus legal setbacks); a candidate's own planning envelope may be
    # an artificially compact box, which must not limit where the stacked floor can sit.
    _sb0 = kw.get("setback_ft")
    if _sb0 is None:
        from .layout_generator import _setback_for_plot
        _sb0 = _setback_for_plot(plot_length, plot_width)
    _us = kw.get("setbacks") or {}
    _s = {k: float(_us.get(k, _sb0)) for k in ("north", "east", "south", "west")}
    buildable = {"x": _s["south"], "y": _s["west"],
                 "w": plot_length - _s["south"] - _s["north"], "h": plot_width - _s["west"] - _s["east"]}
    if stair_ref and stair_ref.get("box"):
        # The ground floor may have needed reduced setbacks to fit; the floors above may use the same area.
        _b = stair_ref["box"]
        buildable = {"x": min(buildable["x"], _b["x"]), "y": min(buildable["y"], _b["y"]),
                     "w": 0, "h": 0}
        buildable["w"] = max(plot_length - _s["south"] - _s["north"] + (_s["south"] - buildable["x"]),
                             _b["x"] + _b["w"] - buildable["x"])
        buildable["h"] = max(plot_width - _s["west"] - _s["east"] + (_s["west"] - buildable["y"]),
                             _b["y"] + _b["h"] - buildable["y"])

    def judge(layout):
        q = _quality.evaluate(layout, building_type, n)
        layout["quality"] = q
        shift = _stack_shift(layout, stair_ref, buildable) if stair_ref else (0.0, 0.0, 1.0)
        return q["penalty"] + (_stack_penalty(shift) if stair_ref else 0.0), shift

    primary = _primary_layout(plot_length, plot_width, allocations, floors=floors, road_direction=road_direction,
                              building_type=building_type, **kw)
    best = primary
    best_pen, best_shift = judge(primary)
    if best_pen <= QUALITY_RETRY_PENALTY:
        return _finish(best, best_shift, stair_ref)
    presentation = {k: kw[k] for k in ("wall_width_ft", "room_size_details", "appearance", "main_door_position",
                                       "parking_position", "staircase_position") if k in kw}
    entrance = kw.get("main_entrance_direction") or road_direction
    user_setbacks = kw.get("setbacks")
    base_sb = kw.get("setback_ft")
    if base_sb is None:
        from .layout_generator import _setback_for_plot
        base_sb = _setback_for_plot(plot_length, plot_width)

    centre = anchor = None
    if stair_ref and stair_ref.get("env"):
        e = stair_ref["env"]
        centre = (e["x"] + e["w"] / 2.0, e["y"] + e["h"] / 2.0)
        g = stair_ref["stair"]
        fx, fy = _zone_fraction(stair_ref.get("zone", "C"))
        anchor = (g["x"] + g["w"] / 2.0, g["y"] + g["h"] / 2.0, fx, fy)
    attempts = []                                      # (setback_ft, setbacks, tag, stair_zone, pin)
    pin = None
    if stair_ref:
        g = stair_ref["stair"]
        pin = {"rect": (g["x"], g["y"], g["x"] + g["w"], g["y"] + g["h"])}
        if stair_ref.get("env"):
            e = stair_ref["env"]
            pin["env"] = (e["x"], e["y"], e["w"], e["h"])
        attempts.append((kw.get("setback_ft"), user_setbacks, "staircase pinned over the floor below", None, pin))
    if primary.get("engine") != "guaranteed" or pin:
        attempts.append((kw.get("setback_ft"), user_setbacks, "guaranteed engine", None, None))
    if not user_setbacks:
        for factor in ((1.1, 1.25, 1.45, 1.9, 2.6) if anchor else (1.45, 1.9, 2.5, 1.2)):
            sb = _compact_setbacks(plot_length, plot_width, allocations, factor, base_sb, centre)
            if not sb:
                continue
            zone = None
            if anchor:
                zone = _best_stair_zone((anchor[0], anchor[1]),
                                        (plot_length - sb["south"] - sb["north"], plot_width - sb["west"] - sb["east"]),
                                        buildable, stair_ref.get("zone", "C"))
            attempts.append((kw.get("setback_ft"), sb, f"compact envelope x{factor}", zone, None))
    for setback_ft, setbacks, tag, stair_zone, pin_arg in attempts:
        if best_pen <= QUALITY_RETRY_PENALTY:
            break
        try:
            if stair_ref:
                g = stair_ref["stair"]
                for a in allocations:
                    if getattr(a, "room_type", "") == "staircase":
                        zone = stair_zone or stair_ref.get("zone")
                        if zone:
                            a.zone = a.preferred_zone = zone
                            a.fixed_zone = True
                        # Same footprint and orientation as the stair below, so the flights nest.
                        a.dims = (round(g["w"], 2), round(g["h"], 2))
                        a.min_dims = (round(g["w"] * 0.96, 2), round(g["h"] * 0.96, 2))
                        a.orient_min = (round(g["w"] * 0.96, 2), round(g["h"] * 0.96, 2))
            alt = guaranteed_layout(plot_length, plot_width, allocations, road_direction=road_direction,
                                    building_type=building_type, floors=floors, setback_ft=setback_ft,
                                    setbacks=setbacks, main_entrance_direction=entrance, pin_stair=pin_arg)
            alt = _apply_presentation_options(alt, **presentation)
            pen, shift = judge(alt)
        except Exception:  # noqa: BLE001 - an infeasible candidate is simply skipped
            continue
        if alt.get("collisions") or alt.get("boundary_violations"):
            continue
        if pen + 0.25 < best_pen:
            alt["engine"] = "guaranteed"
            alt["fallback_reason"] = f"quality gate ({tag}): primary plan had " + "; ".join(
                best["quality"]["issues"][:2] or ["no staircase stacking"])
            best, best_pen, best_shift = alt, pen, shift
    return _finish(best, best_shift, stair_ref)
