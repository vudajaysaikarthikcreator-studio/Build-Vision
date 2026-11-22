"""Hall-centred residential planner (1 BHK ... n BHK, duplex, villa, studio, farmhouse).

Why this exists
---------------
The generic planners tile rooms by Vastu zone only.  For houses that produced
three recurring problems:

* bathrooms landed *outside* the bedrooms (a separate room in a zone),
* doors were chosen by wall scoring, so a bedroom could open into another
  bedroom or a bathroom instead of the hall,
* only some BHK counts packed cleanly.

This planner builds the house around the hall instead:

* A **hall group** (living + dining [+ passage]) touches the entrance wall.
  Every other room is a *slot* that shares a wall with the hall group, so each
  room can have its own door straight onto the hall (no room is ever entered
  through another bedroom).
* Plan shapes (``hall_shape``): ``rectangle`` (hall band across the front,
  rooms behind), ``l_shape`` (rooms wrap the hall on two sides) and
  ``u_shape`` (rooms wrap the hall on three sides).  ``auto`` tries all of
  them and keeps the best Vastu-scoring fit.
* An **attached washroom** is carved *inside* its bedroom (door only from the
  bedroom).  The **common washroom** is carved inside the staircase (under the
  stairs) or, when there is no staircase, in a corner of the hall passage.
  Pooja sits in the north-east corner of the hall, a store inside the kitchen.
* Everything is laid out in compass coordinates (x = south->north,
  y = west->east), so Vastu zones are real zones whatever the entrance side.

Rooms are rectangles.  A contained room (bath, pooja, ...) is a rectangle lying
inside its parent and carries ``inside_of`` so the collision checker, renderer
and 3D view treat it as part of the parent instead of an overlap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from planning.building_programs import ROOM_DIMS, MIN_DIMS
from vastu.vastu_engine import rule_for, zone_score, zone9_of_point, ZONE_GRID, SITE_TYPES
from .collision_checker import run_all_checks
from .guaranteed_layout import LayoutInfeasible
from .layout_generator import _setback_for_plot

RESIDENTIAL_PLANNER_TYPES = {"1bhk", "2bhk", "3bhk", "4bhk", "duplex", "villa", "studio", "farmhouse", "custom"}

BEDROOM_TYPES = {"master_bedroom", "bedroom", "guest_room"}
BATH_TYPES = {"bathroom", "toilet"}
BATH_DIMS = (5.0, 7.0)
PASSAGE_DEPTH = 3.5
DOOR_CLEAR = 3.6           # shared wall needed for a door (door 2.75 + edge clearance)
_LABEL = {"living": "Living Room", "dining": "Dining Room", "passage": "Passage", "lounge": "Lounge"}

SHAPE_ALIASES = {
    "": "auto", "auto": "auto", "any": "auto",
    "rect": "rectangle", "rectangle": "rectangle", "rectangular": "rectangle", "straight": "rectangle",
    "l": "l_shape", "l_shape": "l_shape", "l-shape": "l_shape", "lshape": "l_shape", "l shape": "l_shape",
    "u": "u_shape", "u_shape": "u_shape", "u-shape": "u_shape", "ushape": "u_shape", "u shape": "u_shape",
}


def normalise_shape(value) -> str:
    return SHAPE_ALIASES.get(str(value or "auto").strip().lower().replace("__", "_"), "auto")


# ------------------------------------------------------------------ bath roles
def apply_bath_roles(allocations: Sequence, attached_count: Optional[int] = None) -> None:
    """Decide which bathrooms are attached to a bedroom and which are common.

    * 1 bathroom -> attached to the master bedroom.
    * 2+ bathrooms -> one common washroom, the rest attached (master first, then
      the other bedrooms in order).  ``attached_count`` overrides the split.
    The roles are stored on the allocation objects (``attached_to`` / ``common``)
    so the floor distributor can keep an attached bath on its bedroom's floor.
    """
    beds = [a for a in allocations if a.room_type in BEDROOM_TYPES]
    baths = [a for a in allocations if a.room_type in BATH_TYPES]
    for a in baths:
        a.attached_to = None
        a.common = False
    for a in beds:
        a.attached_bath = None
    if not baths:
        return
    n_bath, n_bed = len(baths), len(beds)
    if attached_count is None:
        attached = n_bath if n_bath == 1 else n_bath - 1
    else:
        attached = int(attached_count)
    attached = max(0, min(attached, n_bed, n_bath))
    order = sorted(beds, key=lambda a: 0 if a.room_type == "master_bedroom" else 1)
    for bed, bath in zip(order[:attached], baths[:attached]):
        bath.attached_to = bed.label
        bed.attached_bath = bath.label
    for bath in baths[attached:]:
        bath.common = True


# ---------------------------------------------------------------- small geometry
def _iv(a0: float, a1: float, b0: float, b1: float) -> Tuple[float, float]:
    return max(a0, b0), min(a1, b1)


def shared_edge(a: dict, b: dict, tol: float = 0.06):
    """(side_of_a, lo, hi) of the wall segment two rects share, else None.
    ``lo``/``hi`` are absolute coordinates along the wall (y for N/S walls, x for E/W walls)."""
    ax0, ax1, ay0, ay1 = a["x"], a["x"] + a["w"], a["y"], a["y"] + a["h"]
    bx0, bx1, by0, by1 = b["x"], b["x"] + b["w"], b["y"], b["y"] + b["h"]
    if abs(ax1 - bx0) <= tol:
        lo, hi = _iv(ay0, ay1, by0, by1)
        if hi - lo > 1e-6:
            return "N", lo, hi
    if abs(ax0 - bx1) <= tol:
        lo, hi = _iv(ay0, ay1, by0, by1)
        if hi - lo > 1e-6:
            return "S", lo, hi
    if abs(ay1 - by0) <= tol:
        lo, hi = _iv(ax0, ax1, bx0, bx1)
        if hi - lo > 1e-6:
            return "E", lo, hi
    if abs(ay0 - by1) <= tol:
        lo, hi = _iv(ax0, ax1, bx0, bx1)
        if hi - lo > 1e-6:
            return "W", lo, hi
    return None


def _side_axis_start(r: dict, side: str) -> float:
    return r["y"] if side in ("N", "S") else r["x"]


def _contains(p: dict, c: dict, tol: float = 0.05) -> bool:
    return (c["x"] >= p["x"] - tol and c["y"] >= p["y"] - tol and
            c["x"] + c["w"] <= p["x"] + p["w"] + tol and c["y"] + c["h"] <= p["y"] + p["h"] + tol)


# ---------------------------------------------------------------------- items
@dataclass
class Item:
    alloc: object
    rtype: str
    label: str
    area: float
    mins: Tuple[float, float]
    contained: List[dict] = field(default_factory=list)   # specs carved inside this item
    weight: float = 1.0

    @property
    def unit_area(self) -> float:
        return self.area + sum(c["area"] for c in self.contained)


def _dims_of(alloc, rtype: str) -> Tuple[float, float]:
    d = getattr(alloc, "dims", None)
    if d:
        return tuple(d)
    from .sizing import base_dims
    return base_dims(alloc, rtype)


def _mins_of(alloc, rtype: str) -> Tuple[float, float]:
    m = getattr(alloc, "min_dims", None)
    return tuple(m) if m else MIN_DIMS.get(rtype, (5.0, 5.0))


def _mk_item(alloc) -> Item:
    t = alloc.room_type
    a, b = _dims_of(alloc, t)
    return Item(alloc, t, alloc.label, a * b, _mins_of(alloc, t), weight=rule_for(t, getattr(alloc, "hint_zones", None)).weight)


def _contained_spec(alloc, kind: str, mode: str) -> dict:
    a, b = _dims_of(alloc, alloc.room_type)
    if alloc.room_type in BATH_TYPES:
        a, b = BATH_DIMS
    return {"alloc": alloc, "kind": kind, "mode": mode, "dims": (a, b), "area": a * b,
            "mins": _mins_of(alloc, alloc.room_type)}


# ------------------------------------------------------------- grid templates
@dataclass
class Region:
    kind: str                 # living | dining | passage | lounge | slot
    r0: int
    r1: int
    c0: int
    c1: int
    area: float = 0.0
    mins: Tuple[float, float] = (3.0, 3.0)
    item: Optional[Item] = None
    split: List[Tuple[str, float]] = field(default_factory=list)   # hall cell shared by several parts (kind, area)

    @property
    def weight(self) -> float:
        return max(self.area, 20.0)


@dataclass
class Template:
    name: str
    shape: str
    R: int
    C: int
    hall: List[Region]
    slots: List[Region]
    mirror: bool = False


def _hall_regions(kinds: List[str], c0: int, c1: int) -> List[Region]:
    """Merge consecutive rows of the same kind into one region."""
    out: List[Region] = []
    for r, k in enumerate(kinds):
        if out and out[-1].kind == k:
            out[-1].r1 = r + 1
        else:
            out.append(Region(k, r, r + 1, c0, c1))
    return out


def _mirror(t: Template) -> Template:
    def flip(rg: Region) -> Region:
        return Region(rg.kind, rg.r0, rg.r1, t.C - rg.c1, t.C - rg.c0, rg.area, rg.mins, rg.item, list(rg.split))
    return Template(t.name, t.shape, t.R, t.C, [flip(h) for h in t.hall], [flip(s) for s in t.slots], True)


def _tpl_rect(n: int, dining: bool) -> Template:
    C = n
    if dining and C >= 2:
        cl = max(1, int(round(C * 0.6)))
        hall = [Region("living", 0, 1, 0, cl), Region("dining", 0, 1, cl, C)]
    else:
        hall = [Region("living", 0, 1, 0, C)]
        if dining:
            hall[0].split = [("dining", 0.0)]
    slots = [Region("slot", 1, 2, c, c + 1) for c in range(C)]
    return Template(f"rect{C}", "rectangle", 2, C, hall, slots)


def _tpl_u(R: int, rear: bool, dining: bool) -> Template:
    H = R - 1 if rear else R
    kinds = ["living" if r == 0 else ("dining" if dining else "living") if r == 1 else "passage" for r in range(H)]
    hall = _hall_regions(kinds, 1, 2)
    if dining and H == 1:
        hall[0].split = [("dining", 0.0)]
    slots: List[Region] = []
    if rear:
        for i in range(R - 2):
            slots += [Region("slot", i, i + 1, 0, 1), Region("slot", i, i + 1, 2, 3)]
        slots += [Region("slot", R - 2, R, 0, 1), Region("slot", R - 2, R, 2, 3), Region("slot", R - 1, R, 1, 2)]
    else:
        for i in range(R):
            slots += [Region("slot", i, i + 1, 0, 1), Region("slot", i, i + 1, 2, 3)]
    return Template(f"u{R}{'r' if rear else ''}", "u_shape", R, 3, hall, slots)


def _tpl_l(R: int, dining: bool) -> Template:
    H = R - 1
    kinds = ["living" if r == 0 else ("dining" if dining else "living") if r == 1 else "passage" for r in range(H)]
    hall = _hall_regions(kinds, 1, 3)
    slots = [Region("slot", i, i + 1, 0, 1) for i in range(H - 1)]
    slots += [Region("slot", H - 1, R, 0, 1), Region("slot", H, R, 1, 2), Region("slot", H, R, 2, 3)]
    return Template(f"l{R}", "l_shape", R, 3, hall, slots)


def _tpl_band(m: int, left: bool, right: bool, dining: bool) -> Template:
    """Wide-plot plan: living on the entrance wall (flanked by up to two rooms), a dining/passage band
    across the middle and a row of ``m`` rooms at the back.  Every room touches the hall band."""
    C = m
    f_l, f_r = (1 if left else 0), (1 if right else 0)
    lc0, lc1 = f_l, C - f_r
    if lc1 - lc0 < 1:
        return None
    hall = [Region("living", 0, 1, lc0, lc1)]
    dc = max(1, (C + 1) // 2)
    if dining:
        hall.append(Region("dining", 1, 2, 0, dc))
        if dc < C:
            hall.append(Region("passage", 1, 2, dc, C))
    else:
        hall.append(Region("passage", 1, 2, 0, C))
    slots = []
    if left:
        slots.append(Region("slot", 0, 1, 0, 1))
    if right:
        slots.append(Region("slot", 0, 1, C - 1, C))
    slots += [Region("slot", 2, 3, c, c + 1) for c in range(C)]
    return Template(f"band{m}{'l' if left else ''}{'r' if right else ''}", "rectangle", 3, C, hall, slots)


def _candidates(n: int, dining: bool, shape: str) -> List[Template]:
    base: List[Template] = []
    if shape in ("auto", "rectangle") and 1 <= n <= 5:
        base.append(_tpl_rect(n, dining))
    if shape in ("auto", "u_shape"):
        if n >= 3 and n % 2 == 1:
            base.append(_tpl_u((n - 1) // 2 + 1, True, dining))
        if n >= 2 and n % 2 == 0:
            base.append(_tpl_u(n // 2, False, dining))
    if shape in ("auto", "l_shape") and 4 <= n <= 8:
        base.append(_tpl_l(n - 1, dining))
    if shape in ("auto", "rectangle") and 3 <= n <= 9:
        for m in range(3, 7):
            for lf, rt in ((False, False), (True, False), (True, True)):
                if m + int(lf) + int(rt) == n:
                    t = _tpl_band(m, lf, rt, dining)
                    if t is not None:
                        base.append(t)
    out: List[Template] = []
    for t in base:
        out.append(t)
        out.append(_mirror(t))
    return out


# ------------------------------------------------------------ compass mapping
def _norm_xy(entrance: str, cu: float, cv: float) -> Tuple[float, float]:
    """Normalised (x_north, y_east) of a canonical (depth, lateral) point in 0..1."""
    if entrance == "N":
        return 1.0 - cu, cv
    if entrance == "S":
        return cu, cv
    if entrance == "E":
        return cv, 1.0 - cu
    return cv, cu


def _zone_of_norm(xn: float, yn: float) -> str:
    row = 0 if xn >= 2 / 3 else (2 if xn <= 1 / 3 else 1)    # row 0 = north
    col = 2 if yn >= 2 / 3 else (0 if yn <= 1 / 3 else 1)
    return ZONE_GRID[row][col]


def _topo_zone(entrance: str, tpl: Template, rg: Region) -> str:
    cu = (rg.r0 + rg.r1) / 2.0 / tpl.R
    cv = (rg.c0 + rg.c1) / 2.0 / tpl.C
    return _zone_of_norm(*_norm_xy(entrance, cu, cv))


def _adjacent(a: Region, b: Region) -> bool:
    rows = min(a.r1, b.r1) - max(a.r0, b.r0)
    cols = min(a.c1, b.c1) - max(a.c0, b.c0)
    return (a.r1 == b.r0 or b.r1 == a.r0) and cols > 0 or (a.c1 == b.c0 or b.c1 == a.c0) and rows > 0


# --------------------------------------------------------------- size solver
def _orient_ok(w: float, d: float, mins: Tuple[float, float], slack: float = 0.0) -> bool:
    m1, m2 = mins
    return (w >= m1 - slack and d >= m2 - slack) or (w >= m2 - slack and d >= m1 - slack)


def _solve_sizes(tpl: Template, regions: List[Region], iters: int = 90) -> Tuple[List[float], List[float]]:
    R, C = tpl.R, tpl.C
    d = [11.0] * R
    w = [11.0] * C
    for h in tpl.hall:
        if h.kind == "passage":
            for r in range(h.r0, h.r1):
                d[r] = PASSAGE_DEPTH + 0.5
    if tpl.shape == "u_shape":
        w[1] = 13.0
    for _ in range(iters):
        nd, dd = [0.0] * R, [0.0] * R
        nw, dw = [0.0] * C, [0.0] * C
        for rg in regions:
            cd = sum(d[rg.r0:rg.r1])
            cw = sum(w[rg.c0:rg.c1])
            s = math.sqrt(max(rg.area, 1.0) / max(cd * cw, 1e-6))
            s = max(0.55, min(1.7, s))
            for r in range(rg.r0, rg.r1):
                nd[r] += rg.weight * d[r] * s
                dd[r] += rg.weight
            for c in range(rg.c0, rg.c1):
                nw[c] += rg.weight * w[c] * s
                dw[c] += rg.weight
        for r in range(R):
            if dd[r]:
                d[r] = 0.45 * d[r] + 0.55 * nd[r] / dd[r]
        for c in range(C):
            if dw[c]:
                w[c] = 0.45 * w[c] + 0.55 * nw[c] / dw[c]
        for h in tpl.hall:
            if h.kind == "passage":
                for r in range(h.r0, h.r1):
                    d[r] = max(d[r], PASSAGE_DEPTH)
    _enforce_mins(tpl, regions, d, w)
    return d, w


def _enforce_mins(tpl: Template, regions: List[Region], d: List[float], w: List[float]) -> None:
    passage_rows = {r for h in tpl.hall if h.kind == "passage" for r in range(h.r0, h.r1)}
    for _ in range(30):
        changed = False
        for rg in regions:
            cd = sum(d[rg.r0:rg.r1])
            cw = sum(w[rg.c0:rg.c1])
            m1, m2 = rg.mins
            if _orient_ok(cw, cd, rg.mins, 0.01):
                continue
            # pick the orientation that needs the smaller total growth
            need_a = max(0.0, m1 - cw) + max(0.0, m2 - cd)
            need_b = max(0.0, m2 - cw) + max(0.0, m1 - cd)
            tw, td = (m1, m2) if need_a <= need_b else (m2, m1)
            if cw < tw:
                cols = list(range(rg.c0, rg.c1))
                add = (tw - cw) / len(cols)
                for c in cols:
                    w[c] += add
                changed = True
            if cd < td:
                rows = [r for r in range(rg.r0, rg.r1) if r not in passage_rows] or list(range(rg.r0, rg.r1))
                add = (td - cd) / len(rows)
                for r in rows:
                    d[r] += add
                changed = True
        if not changed:
            break


# ------------------------------------------------------------------ the plan
class _Plan:
    """One concrete candidate: template + item->slot assignment + sizes."""

    def __init__(self, tpl: Template, assign: Dict[int, Item], d: List[float], w: List[float], vastu: float, notes: List[str]):
        self.tpl, self.assign, self.d, self.w, self.vastu, self.notes = tpl, assign, d, w, vastu, notes
        self.depth = sum(d)
        self.lat = sum(w)
        self.score = vastu


def _hungarian(cost: List[List[float]]) -> List[int]:
    n = len(cost)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        res = [0] * n
        for r, c in zip(rows, cols):
            res[r] = int(c)
        return res
    except Exception:  # noqa: BLE001 - scipy missing: greedy fallback
        used, res = set(), [0] * n
        for i in sorted(range(n), key=lambda k: min(cost[k])):
            j = min((j for j in range(n) if j not in used), key=lambda j: cost[i][j])
            used.add(j)
            res[i] = j
        return res


def _hall_target(kind: str, living: Optional[Item], dining: Optional[Item], extra: Dict[str, float], width_hint: float = 12.0) -> Tuple[float, Tuple[float, float]]:
    if kind == "living":
        base = living.area if living else ROOM_DIMS["living_room"][0] * ROOM_DIMS["living_room"][1] * 0.85
        mins = living.mins if living else MIN_DIMS["living_room"]
        return base + extra.get("living", 0.0), mins
    if kind == "dining":
        base = dining.area if dining else 90.0
        mins = dining.mins if dining else MIN_DIMS["dining_room"]
        return base + extra.get("dining", 0.0), mins
    return PASSAGE_DEPTH * 10.0 + extra.get("passage", 0.0), (PASSAGE_DEPTH - 0.4, 5.0)


def _build_plan(tpl: Template, entrance: str, slot_items: List[Item], living: Optional[Item], dining: Optional[Item],
                host_extra: Dict[str, float]) -> Optional[_Plan]:
    n = len(slot_items)
    if len(tpl.slots) != n:
        return None
    # --- Vastu assignment (Hungarian) over the topological zones
    hall_zone = {id(h): _topo_zone(entrance, tpl, h) for h in tpl.hall}
    dining_regions = [h for h in tpl.hall if h.kind == "dining"]
    cost: List[List[float]] = []
    for it in slot_items:
        rule = rule_for(it.rtype, getattr(it.alloc, "hint_zones", None))
        row = []
        for sl in tpl.slots:
            z = _topo_zone(entrance, tpl, sl)
            sc = rule.weight * zone_score(rule, z)
            if it.rtype == "kitchen" and any(_adjacent(sl, dr) for dr in dining_regions):
                sc += 0.6 * rule.weight
            # a big room in a tiny slot position is not an issue (sizes adapt), but keep stairs away from the hall front
            if it.rtype == "staircase" and sl.r0 == 0:
                sc -= 0.4
            row.append(-sc)
        cost.append(row)
    pick = _hungarian(cost)
    assign: Dict[int, Item] = {}
    vscore = vweight = 0.0
    for i, it in enumerate(slot_items):
        sl = tpl.slots[pick[i]]
        rule = rule_for(it.rtype, getattr(it.alloc, "hint_zones", None))
        assign[pick[i]] = it
        sl.item = it
        sl.area = it.unit_area
        sl.mins = it.mins
        vweight += rule.weight
        vscore += rule.weight * zone_score(rule, _topo_zone(entrance, tpl, sl))
        for c in it.contained:
            cr = rule_for(c["alloc"].room_type)
            vweight += cr.weight * 0.3
            vscore += cr.weight * 0.3 * 0.6        # contained rooms follow their parent; neutral credit
    for h in tpl.hall:
        area, mins = _hall_target(h.kind, living, dining, host_extra)
        if h.split:
            area += sum(((dining.area if dining else 90.0) if k == "dining" else 0.0) for k, _ in h.split)
        h.area, h.mins = area, mins
        if h.kind in ("living", "dining") and (living if h.kind == "living" else dining):
            ob = living if h.kind == "living" else dining
            rule = rule_for(ob.rtype)
            vweight += rule.weight
            vscore += rule.weight * zone_score(rule, hall_zone[id(h)])
    regions = list(tpl.hall) + list(tpl.slots)
    d, w = _solve_sizes(tpl, regions)
    return _Plan(tpl, assign, d, w, vscore / max(vweight, 1e-6), [])


# ------------------------------------------------------------- frame / mapping
class Frame:
    """Canonical (depth u, lateral v) -> compass rectangles for one entrance side."""

    def __init__(self, entrance: str, L: float, W: float, t0: float, s0: float):
        self.e, self.L, self.W, self.t0, self.s0 = entrance, L, W, t0, s0

    def rect(self, u0: float, u1: float, v0: float, v1: float) -> dict:
        t_lo, t_hi = self.t0 + u0, self.t0 + u1
        s_lo, s_hi = self.s0 + v0, self.s0 + v1
        if self.e == "N":
            x, w, y, h = self.L - t_hi, t_hi - t_lo, s_lo, s_hi - s_lo
        elif self.e == "S":
            x, w, y, h = t_lo, t_hi - t_lo, s_lo, s_hi - s_lo
        elif self.e == "E":
            x, w, y, h = s_lo, s_hi - s_lo, self.W - t_hi, t_hi - t_lo
        else:
            x, w, y, h = s_lo, s_hi - s_lo, t_lo, t_hi - t_lo
        return {"x": x, "y": y, "w": w, "h": h}


def _entrance_code(value: str) -> str:
    v = str(value or "North").strip().upper()
    return {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}.get(v, v[:1] if v[:1] in "NSEW" else "N")


# ------------------------------------------------------------ contained rooms
def _free_corner_rects(parent: dict, dims: Tuple[float, float]) -> List[Tuple[str, dict]]:
    """All 8 anchored placements (4 corners x 2 orientations) of a dims rect inside parent."""
    out = []
    for (p, q) in {(dims[0], dims[1]), (dims[1], dims[0])}:
        if p > parent["w"] - 0.01 or q > parent["h"] - 0.01:
            continue
        for cx, cy, tag in ((parent["x"], parent["y"], "SW"),
                            (parent["x"] + parent["w"] - p, parent["y"], "NW"),
                            (parent["x"], parent["y"] + parent["h"] - q, "SE"),
                            (parent["x"] + parent["w"] - p, parent["y"] + parent["h"] - q, "NE")):
            out.append((tag, {"x": cx, "y": cy, "w": p, "h": q}))
    return out


def _touch_sides(parent: dict, r: dict, tol: float = 0.05) -> List[str]:
    s = []
    if abs(r["x"] - parent["x"]) <= tol: s.append("S")
    if abs(r["x"] + r["w"] - (parent["x"] + parent["w"])) <= tol: s.append("N")
    if abs(r["y"] - parent["y"]) <= tol: s.append("W")
    if abs(r["y"] + r["h"] - (parent["y"] + parent["h"])) <= tol: s.append("E")
    return s


def _rect_overlap(a: dict, b: dict, pad: float = 0.0) -> bool:
    return (a["x"] < b["x"] + b["w"] + pad and b["x"] < a["x"] + a["w"] + pad and
            a["y"] < b["y"] + b["h"] + pad and b["y"] < a["y"] + a["h"] + pad)


def _place_contained(parent: dict, spec: dict, taken: List[dict], avoid_pts: List[Tuple[float, float]],
                     envelope: dict, want_wall: Optional[str], not_wall: Optional[str], toward: Tuple[int, int],
                     min_free: Tuple[float, float], rtype: Optional[str] = None) -> Optional[dict]:
    best, best_score = None, -1e9
    for tag, r in _free_corner_rects(parent, spec["dims"]):
        if any(_rect_overlap(r, t, 0.0) for t in taken):
            continue
        if any(r["x"] - 1.6 < px < r["x"] + r["w"] + 1.6 and r["y"] - 1.6 < py < r["y"] + r["h"] + 1.6 for px, py in avoid_pts):
            continue
        touches = _touch_sides(parent, r)
        if not_wall and not_wall in touches:
            continue
        # the parent must stay usable: a free rectangle >= min_free must remain beside/in front of the carve
        free_a = (parent["w"] - (r["w"] if "S" in touches or "N" in touches else 0.0), parent["h"])
        free_b = (parent["w"], parent["h"] - (r["h"] if "W" in touches or "E" in touches else 0.0))
        free_c = (parent["w"] - r["w"], parent["h"])
        free_d = (parent["w"], parent["h"] - r["h"])
        if not any(_orient_ok(f[0], f[1], min_free) for f in (free_c, free_d)):
            continue
        sc = 0.0
        if want_wall and want_wall in touches:
            sc += 10.0
        ext = sum(1 for sd in _touch_sides(envelope, r) if sd)
        sc += 2.5 * ext
        cx = r["x"] + r["w"] / 2 - (parent["x"] + parent["w"] / 2)
        cy = r["y"] + r["h"] / 2 - (parent["y"] + parent["h"] / 2)
        sc += 0.15 * (toward[0] * cx + toward[1] * cy)
        sc -= 0.01 * (r["w"] * r["h"])
        if rtype:
            # Vastu: the carved-in room has its own zone on the building (washroom -> W/NW/N, never SW/NE)
            zr = zone9_of_point(r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0, envelope)
            sc += 7.0 * zone_score(rule_for(rtype), zr)
        if sc > best_score:
            best, best_score = r, sc
    return best


def _inside_door(child: dict, parent: dict) -> None:
    """Door of a carved-in room: on a wall that faces the parent's interior (never on the parent's outer wall)."""
    touches = set(_touch_sides(parent, child))
    inner = [sd for sd in "SNWE" if sd not in touches]
    if not inner:
        return
    def wall_len(sd):
        return child["h"] if sd in ("S", "N") else child["w"]
    # prefer the wall that faces the larger free part of the parent
    def free_depth(sd):
        if sd == "N":  return parent["x"] + parent["w"] - (child["x"] + child["w"])
        if sd == "S":  return child["x"] - parent["x"]
        if sd == "E":  return parent["y"] + parent["h"] - (child["y"] + child["h"])
        return child["y"] - parent["y"]
    inner = [sd for sd in inner if wall_len(sd) >= 3.0] or inner
    side = max(inner, key=lambda sd: (free_depth(sd), wall_len(sd)))
    child["door_spec"] = {"wall": side, "position": round(wall_len(side) / 2.0, 2), "to": parent["label"]}
    child["door_side"] = side


def _toward(room_type: str) -> Tuple[int, int]:
    return {"pooja_room": (1, 1), "bathroom": (1, -1), "toilet": (1, -1), "store_room": (-1, -1),
            "balcony": (1, 1)}.get(room_type, (0, 0))


# ---------------------------------------------------------------- main driver
def _opposite(side: str) -> str:
    return {"N": "S", "S": "N", "E": "W", "W": "E"}[side]


def _door_center(lo: float, hi: float, width: float = 2.75, edge: float = 0.55) -> float:
    c = (lo + hi) / 2.0
    return min(max(c, lo + edge + width / 2), hi - edge - width / 2)


def plan_residential(
    plot_length: float,
    plot_width: float,
    allocations: Sequence,
    entrance_direction: str = "North",
    road_direction: str = "North",
    vastu: bool = True,
    hall_shape: str = "auto",
    building_type: str = "custom",
    floor: int = 0,
    setback_ft: Optional[float] = None,
    setbacks: Optional[Dict[str, float]] = None,
    parking_position: str = "road_side",
) -> Dict:
    """Return a layout dict (same shape as the other planners) or raise LayoutInfeasible."""
    allocs = list(allocations)
    if not allocs:
        raise LayoutInfeasible("No rooms to plan.")
    L, W = float(plot_length), float(plot_width)
    e = _entrance_code(entrance_direction)
    shape = normalise_shape(hall_shape)
    notes: List[str] = []

    # ------------------------------------------------------------ classify
    site_allocs = [a for a in allocs if a.room_type in SITE_TYPES]
    indoor = [a for a in allocs if a.room_type not in SITE_TYPES]
    living_a = next((a for a in indoor if a.room_type == "living_room"), None)
    dining_a = next((a for a in indoor if a.room_type == "dining_room"), None)
    stairs = [a for a in indoor if a.room_type == "staircase"]
    kitchens = [a for a in indoor if a.room_type == "kitchen"]
    beds = [a for a in indoor if a.room_type in BEDROOM_TYPES]
    baths = [a for a in indoor if a.room_type in BATH_TYPES]
    by_label_bed = {a.label: a for a in beds}
    pooja = [a for a in indoor if a.room_type == "pooja_room"]
    stores = [a for a in indoor if a.room_type == "store_room"]
    balconies = [a for a in indoor if a.room_type == "balcony"]
    consumed = {id(a) for a in [living_a, dining_a] if a}

    # bath roles (pipeline normally set them; derive when called directly)
    if any(getattr(b, "attached_to", None) or getattr(b, "common", False) for b in baths):
        pass
    else:
        apply_bath_roles(indoor)
    items_by_alloc: Dict[int, Item] = {}
    slot_items: List[Item] = []
    for a in indoor:
        if id(a) in consumed or a.room_type in BATH_TYPES or a.room_type in {"pooja_room", "balcony"}:
            continue
        if a.room_type == "store_room" and kitchens:
            continue
        it = _mk_item(a)
        items_by_alloc[id(a)] = it
        slot_items.append(it)
    living_item = _mk_item(living_a) if living_a else None
    dining_item = _mk_item(dining_a) if dining_a else None

    # attach contained rooms to their hosts
    host_extra: Dict[str, float] = {}
    common_baths = [b for b in baths if not getattr(b, "attached_to", None)]
    for b in baths:
        parent_label = getattr(b, "attached_to", None)
        if parent_label and parent_label in by_label_bed and id(by_label_bed[parent_label]) in items_by_alloc:
            items_by_alloc[id(by_label_bed[parent_label])].contained.append(_contained_spec(b, "attached_bath", "inside"))
        elif parent_label:       # bedroom not on this floor: it stays a stand-alone bath
            common_baths.append(b)
    stair_items = [items_by_alloc[id(s)] for s in stairs if id(s) in items_by_alloc]
    hall_baths: List[dict] = []
    for i, b in enumerate(common_baths):
        if stair_items and i < len(stair_items):
            stair_items[i].contained.append(_contained_spec(b, "under_stair_bath", "outside"))
        else:
            # no staircase to tuck it under: a small stand-alone washroom with its own door on the hall
            it = _mk_item(b)
            it.area = BATH_DIMS[0] * BATH_DIMS[1] * 1.25
            it.mins = (5.0, 6.5)
            items_by_alloc[id(b)] = it
            slot_items.append(it)
    for k, a in enumerate(stores):
        if kitchens:
            kit = items_by_alloc[id(kitchens[min(k, len(kitchens) - 1)])]
            kit.contained.append(_contained_spec(a, "store", "inside"))
    for a in pooja:
        living_spec = _contained_spec(a, "pooja", "inside")
        host_extra["living"] = host_extra.get("living", 0.0) + living_spec["area"]
        hall_baths.append({**living_spec, "host": "living"})
    for a in balconies:
        sp = _contained_spec(a, "balcony", "inside")
        host_extra["living"] = host_extra.get("living", 0.0) + sp["area"]
        hall_baths.append({**sp, "host": "living"})
    for sp in hall_baths:
        if sp.get("host") != "living":
            host_extra["passage"] = host_extra.get("passage", 0.0) + sp["area"]
            sp["host"] = "passage"

    n = len(slot_items)
    if n == 0 or n > 10:
        raise LayoutInfeasible(f"Residential planner supports 1-10 separate rooms around the hall (got {n}).")

    # ---------------------------------------------------------------- setbacks
    sb_base = float(setback_ft) if setback_ft is not None else _setback_for_plot(L, W)
    sides = {k: float((setbacks or {}).get(k, sb_base)) for k in ("north", "east", "south", "west")}
    front_key = {"N": "north", "S": "south", "E": "east", "W": "west"}[e]
    back_key = {"N": "south", "S": "north", "E": "west", "W": "east"}[e]
    # lateral axis: N/S entrance -> y (west..east); E/W entrance -> x (south..north)
    lat_lo_key, lat_hi_key = ("west", "east") if e in "NS" else ("south", "north")
    depth_len, lat_len = (L, W) if e in "NS" else (W, L)
    sb_f, sb_b = sides[front_key], sides[back_key]
    sb_lo, sb_hi = sides[lat_lo_key], sides[lat_hi_key]
    avail_depth = depth_len - sb_f - sb_b
    avail_lat = lat_len - sb_lo - sb_hi
    if avail_depth < 10 or avail_lat < 10:
        raise LayoutInfeasible("Plot is too small for a building after setbacks.")

    # ------------------------------------------------------- parking options
    park_a = site_allocs and next((a for a in site_allocs if a.room_type == "parking"), None)
    garden_a = next((a for a in site_allocs if a.room_type == "garden"), None)
    pdims = tuple(_dims_of(park_a, "parking")) if park_a else (16.0, 10.0)
    p_depth, p_wide = max(pdims), min(pdims)     # car length along the entrance axis, width across
    options: List[dict] = [{"name": "none"}] if not park_a else []
    if park_a:
        options = [{"name": "side_lo", "park": "side", "end": "lo"}, {"name": "side_hi", "park": "side", "end": "hi"},
                   {"name": "front_lo", "park": "front", "end": "lo"}, {"name": "front_hi", "park": "front", "end": "hi"}]

    dining_flag = dining_item is not None
    templates = _candidates(n, dining_flag, shape)
    if not templates and shape != "auto":
        notes.append(f"{shape.replace('_', ' ').title()} hall is not possible with {n} rooms around the hall; using the best available shape.")
        templates = _candidates(n, dining_flag, "auto")
    if not templates:
        raise LayoutInfeasible(f"No hall-centred template for {n} rooms.")

    best = None
    for tpl_proto in templates:
        tpl = _clone_template(tpl_proto)
        for sl in tpl.slots:
            sl.item = None
        plan = _build_plan(tpl, e, [_clone_item(i) for i in slot_items], living_item, dining_item, host_extra)
        if plan is None:
            continue
        for opt in options:
            res = _fit_plan(plan, opt, e, L, W, depth_len, lat_len, sb_f, sb_b, sb_lo, sb_hi,
                            p_depth, p_wide, park_a)
            if res is None:
                continue
            cand = (res["score"], plan, res)
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        raise LayoutInfeasible(
            f"The {n + 2} main spaces do not fit this {L:g} x {W:g} ft plot around a central hall. "
            "Try a larger plot, fewer rooms or more floors.")
    _, plan, fit = best
    return _materialise(plan, fit, e, L, W, allocs, living_item, dining_item, hall_baths, vastu, shape,
                        building_type, floor, road_direction, sides, sb_base, avail_depth, avail_lat,
                        park_a, garden_a, (sb_f, sb_b, sb_lo, sb_hi), notes)


def _clone_template(t: Template) -> Template:
    def c(rg: Region) -> Region:
        return Region(rg.kind, rg.r0, rg.r1, rg.c0, rg.c1, rg.area, rg.mins, None, list(rg.split))
    return Template(t.name, t.shape, t.R, t.C, [c(h) for h in t.hall], [c(s) for s in t.slots], t.mirror)


def _clone_item(i: Item) -> Item:
    return Item(i.alloc, i.rtype, i.label, i.area, i.mins, [dict(c) for c in i.contained], i.weight)


# ------------------------------------------------------------------ fitting

def _hall_pair(tpl: Template, sl: Region):
    """Static choice of the hall region a slot room opens onto, and how its shared wall is measured."""
    best = None
    for h in tpl.hall:
        rows = min(sl.r1, h.r1) - max(sl.r0, h.r0)
        cols = min(sl.c1, h.c1) - max(sl.c0, h.c0)
        if (sl.c1 == h.c0 or h.c1 == sl.c0) and rows > 0:
            cand = ("rows", range(max(sl.r0, h.r0), min(sl.r1, h.r1)), rows)
        elif (sl.r1 == h.r0 or h.r1 == sl.r0) and cols > 0:
            cand = ("cols", range(max(sl.c0, h.c0), min(sl.c1, h.c1)), cols)
        else:
            continue
        if best is None or cand[2] > best[2][2]:
            best = (h, cand[0], cand)
    return best


def _doors_ok(tpl: Template, d: List[float], w: List[float]) -> bool:
    for sl in tpl.slots:
        pair = _hall_pair(tpl, sl)
        if pair is None:
            return False
        _, kind, cand = pair
        length = sum(d[r] for r in cand[1]) if kind == "rows" else sum(w[c] for c in cand[1])
        if length < DOOR_CLEAR + 0.05:
            return False
    return True


def _optimise_sizes(plan: "_Plan", d0: List[float], w0: List[float], av_d: float, av_l: float):
    """Pick row depths / column widths that match each room's target area inside the available box.

    Fully vectorised (region membership matrices) so one solve takes a few milliseconds.
    """
    import numpy as np
    from scipy.optimize import minimize
    tpl = plan.tpl
    R, C = tpl.R, tpl.C
    regions = list(tpl.hall) + list(tpl.slots)
    pairs = [_hall_pair(tpl, sl) for sl in tpl.slots]
    if any(p is None for p in pairs):
        return None
    n = len(regions)
    Mr = np.zeros((n, R))
    Mc = np.zeros((n, C))
    for i, rg in enumerate(regions):
        Mr[i, rg.r0:rg.r1] = 1.0
        Mc[i, rg.c0:rg.c1] = 1.0
    tgt = np.array([max(rg.area, 20.0) for rg in regions])
    wgt = np.array([60.0 if (rg.item is not None and rg.item.rtype in BATH_TYPES) else max(rg.weight, 20.0) for rg in regions])
    asp = np.array([0.25 if rg.kind == "passage" else 0.06 for rg in regions]) * 40.0
    lo = np.array([min(rg.mins) for rg in regions])
    hi = np.array([max(rg.mins) for rg in regions])
    prod = np.array([rg.mins[0] * rg.mins[1] * 1.01 for rg in regions])
    # door-wall length rows: sum of selected depths (rows) or widths (cols) >= DOOR_CLEAR + margin
    Dm = np.zeros((len(pairs), R + C))
    for i, (h, kind, cand) in enumerate(pairs):
        for k in cand[1]:
            Dm[i, k if kind == "rows" else R + k] = 1.0
    passage_rows = {r for h in tpl.hall if h.kind == "passage" for r in range(h.r0, h.r1)}

    def obj(x):
        cd = Mr @ x[:R]
        cw = Mc @ x[R:]
        return float(np.sum(wgt * np.log(np.maximum(cd * cw, 1.0) / tgt) ** 2 +
                            asp * np.log(np.maximum(cw, 1.0) / np.maximum(cd, 1.0)) ** 2))

    # The master bedroom must be the largest bedroom (tried first; dropped only if the plot cannot allow it).
    master_idx = [i for i, rg in enumerate(regions) if rg.item is not None and rg.item.rtype == "master_bedroom"]
    other_idx = [i for i, rg in enumerate(regions) if rg.item is not None and rg.item.rtype == "bedroom"]
    master_pairs = [(m, o) for m in master_idx for o in other_idx]

    def cons_fn(x, with_master=False):
        cd = Mr @ x[:R]
        cw = Mc @ x[R:]
        parts = [[av_d - x[:R].sum(), av_l - x[R:].sum()], cw - lo, cd - lo, cw * cd - prod,
                 cw + cd - (lo + hi) - 0.02, Dm @ x - (DOOR_CLEAR + 0.2)]
        if with_master and master_pairs:
            area = cw * cd
            parts.append(np.array([area[m] - area[o] for m, o in master_pairs]))
        return np.concatenate(parts)

    bounds = [(PASSAGE_DEPTH + 0.3 if r in passage_rows else 4.5, av_d) for r in range(R)] + [(4.5, av_l)] * C
    scale = min(1.0, av_d / max(sum(d0), 1e-6), av_l / max(sum(w0), 1e-6))
    best = None
    for with_master in ((True, False) if master_pairs else (False,)):
        for k in (1.0, 0.85):
            x0 = np.array([max(b_[0], v * scale * k) for v, b_ in zip(list(d0) + list(w0), bounds)])
            try:
                r = minimize(obj, x0, method="SLSQP", bounds=bounds,
                             constraints=[{"type": "ineq", "fun": (lambda x, m=with_master: cons_fn(x, m))}],
                             options={"maxiter": 120, "ftol": 1e-6})
            except Exception:  # noqa: BLE001
                continue
            x = r.x
            if x[:R].sum() <= av_d + 0.05 and x[R:].sum() <= av_l + 0.05 and cons_fn(x, with_master).min() > -0.05:
                if best is None or r.fun < best[0]:
                    best = (r.fun, x)
                if r.success:
                    break
        if best is not None:
            break
    if best is None:
        return None
    x = best[1]
    return [float(v) for v in x[:R]], [float(v) for v in x[R:]]


REASONS: Dict[int,int] = {}


def _no(code: int):
    REASONS[code] = REASONS.get(code, 0) + 1
    return None


def _fit_plan(plan: _Plan, opt: dict, e: str, L: float, W: float, depth_len: float, lat_len: float,
              sb_f: float, sb_b: float, sb_lo: float, sb_hi: float, p_depth: float, p_wide: float, park_a) -> Optional[dict]:
    """Scale the solved plan to the available area for one parking option and place it on the plot."""
    av_d = depth_len - sb_f - sb_b
    av_l = lat_len - sb_lo - sb_hi
    d, w = list(plan.d), list(plan.w)
    t0 = sb_f
    s_lo, s_hi = sb_lo, lat_len - sb_hi
    park = None
    if opt["name"] != "none":
        if opt["park"] == "side":
            av_l -= p_wide
            if opt["end"] == "lo":
                park = ("side", s_lo, s_lo + p_wide)
                s_lo += p_wide
            else:
                park = ("side", s_hi - p_wide, s_hi)
                s_hi -= p_wide
        else:
            av_d -= p_depth
            park = ("front", None, None)
            t0 = sb_f + p_depth
    if av_d < 8 or av_l < 8:
        return _no(1)
    res = _optimise_sizes(plan, d, w, av_d - 0.04, av_l - 0.04)
    if res is None:
        return _no(2)
    d, w = res
    comp = 1.0
    # every region must satisfy its minimum dimensions, and every room needs a usable door wall on the hall
    for rg in list(plan.tpl.hall) + list(plan.tpl.slots):
        cd = sum(d[rg.r0:rg.r1])
        cw = sum(w[rg.c0:rg.c1])
        if not _orient_ok(cw, cd, rg.mins, 0.03):
            return _no(3)
        if rg.kind == "slot" and rg.item is not None:
            carved = sum(c["area"] for c in rg.item.contained if c["mode"] == "inside")
            if cw * cd - carved < max(rg.item.mins[0] * rg.item.mins[1] * 0.97, rg.item.area * 0.62):
                return _no(4)
            if rg.item.rtype in BATH_TYPES and cw * cd > 140.0:
                return _no(5)          # a washroom must stay washroom-sized
            if max(cw, cd) > (2.6 if rg.item.rtype in BATH_TYPES else 2.1) * min(cw, cd):
                return _no(6)
    if not _doors_ok(plan.tpl, d, w):
        return _no(7)
    fp_d, fp_l = sum(d), sum(w)
    # lateral placement: flush against the parking strip's opposite side so the strip stays free
    if park and park[0] == "side":
        s0 = s_lo if park[1] == sb_lo else s_hi - fp_l
    else:
        s0 = sb_lo + (lat_len - sb_lo - sb_hi - fp_l) / 2.0 if not park else sb_lo + (lat_len - sb_lo - sb_hi - fp_l) / 2.0
    frame = Frame(e, L, W, t0, s0)
    fit = {"opt": opt, "d": d, "w": w, "frame": frame, "park": park, "fp_d": fp_d, "fp_l": fp_l,
           "t0": t0, "s0": s0, "compression": comp}
    # parking quality (Vastu): zone of the parking rectangle on the plot
    pscore = 0.0
    if park_a is not None and park:
        prect = _parking_rect(fit, e, L, W, depth_len, lat_len, sb_f, sb_lo, sb_hi, p_depth, p_wide)
        if prect is None:
            return _no(8)
        fit["parking_rect"] = prect
        z = zone9_of_point(prect["x"] + prect["w"] / 2, prect["y"] + prect["h"] / 2, {"x": 0, "y": 0, "w": L, "h": W})
        pscore = zone_score(rule_for("parking"), z)
    waste = 1.0 - (fp_d * fp_l) / max(av_d * av_l, 1.0)
    fit["score"] = plan.vastu + 0.25 * pscore - 1.8 * max(0.0, 1.0 - comp) - 0.05 * max(0.0, waste - 0.5)
    return fit


def _parking_rect(fit: dict, e: str, L: float, W: float, depth_len: float, lat_len: float,
                  sb_f: float, sb_lo: float, sb_hi: float, p_depth: float, p_wide: float) -> Optional[dict]:
    park = fit["park"]
    if park[0] == "side":
        s_lo, s_hi = park[1], park[2]
        t_lo, t_hi = sb_f, sb_f + p_depth
    else:
        # front yard: park on the lateral side that leaves room beside the footprint if possible
        free_lo = fit["s0"] - sb_lo
        free_hi = (lat_len - sb_hi) - (fit["s0"] + fit["fp_l"])
        if fit["opt"]["end"] == "lo":
            s_lo = sb_lo
        else:
            s_lo = lat_len - sb_hi - p_wide
        s_hi = s_lo + p_wide
        t_lo, t_hi = sb_f, sb_f + p_depth
    fr = Frame(e, L, W, 0.0, 0.0)
    r = fr.rect(t_lo, t_hi, s_lo, s_hi)
    return r


# ----------------------------------------------------------- final assembly
def _round(v: float) -> float:
    return round(v + 0.0, 2)


def _materialise(plan: _Plan, fit: dict, e: str, L: float, W: float, allocs: Sequence, living_item: Optional[Item],
                 dining_item: Optional[Item], hall_specs: List[dict], vastu: bool, shape: str, building_type: str,
                 floor: int, road_direction: str, sides: Dict[str, float], sb_base: float, avail_depth: float,
                 avail_lat: float, park_a, garden_a, sb_tuple, notes: List[str]) -> Dict:
    tpl = plan.tpl
    d, w, frame = fit["d"], fit["w"], fit["frame"]
    # snap every grid line to 0.01 ft so neighbouring rooms share exactly the same coordinate
    row_pos = [0.0]
    for x in d:
        row_pos.append(round(row_pos[-1] + x, 2))
    col_pos = [0.0]
    for x in w:
        col_pos.append(round(col_pos[-1] + x, 2))
    frame.t0, frame.s0 = round(frame.t0, 2), round(frame.s0, 2)

    def cell_rect(rg: Region) -> dict:
        return frame.rect(row_pos[rg.r0], row_pos[rg.r1], col_pos[rg.c0], col_pos[rg.c1])

    rooms: List[dict] = []
    hall_rooms: List[dict] = []
    used_label = set()

    def uniq(label: str) -> str:
        base, k = label, 2
        while label in used_label:
            label = f"{base} {k}"
            k += 1
        used_label.add(label)
        return label

    # ---- hall group ---------------------------------------------------------
    hall_label_by_kind: Dict[str, str] = {}
    has_living = living_item is not None
    for h in tpl.hall:
        rect = cell_rect(h)
        parts: List[Tuple[str, Optional[Item], float]] = []
        if h.kind == "living":
            parts.append(("living", living_item, h.area - sum(a for _, a in [(k, 0) for k, _ in h.split])))
        elif h.kind == "dining":
            parts.append(("dining", dining_item, h.area))
        else:
            parts.append((h.kind, None, h.area))
        if h.split:
            # a dining area sharing the living cell: split along the depth axis (living stays on the entrance wall)
            living_area = (living_item.area if living_item else 120.0)
            dining_area = (dining_item.area if dining_item else 90.0)
            share = living_area / (living_area + dining_area)
            parts = [("living", living_item, share), ("dining", dining_item, 1 - share)]
            if e in "NS":
                u_len, base_rect = rect["w"], rect
            else:
                u_len, base_rect = rect["h"], rect
            cut = u_len * share
            if e == "N":
                r1 = {**rect, "x": rect["x"] + rect["w"] - cut, "w": cut}
                r2 = {**rect, "w": rect["w"] - cut}
            elif e == "S":
                r1 = {**rect, "w": cut}
                r2 = {**rect, "x": rect["x"] + cut, "w": rect["w"] - cut}
            elif e == "E":
                r1 = {**rect, "y": rect["y"] + rect["h"] - cut, "h": cut}
                r2 = {**rect, "h": rect["h"] - cut}
            else:
                r1 = {**rect, "h": cut}
                r2 = {**rect, "y": rect["y"] + cut, "h": rect["h"] - cut}
            rects = [r1, r2]
        else:
            rects = [rect]
        for (kind, it, _), rr in zip(parts, rects):
            if kind == "living":
                rtype = "living_room" if has_living else "lobby"
                label = it.label if it else "Lounge"
                alloc = it.alloc if it else None
            elif kind == "dining":
                rtype, label, alloc = "dining_room", (it.label if it else "Dining Room"), (it.alloc if it else None)
            else:
                rtype, label, alloc = "corridor_lobby", "Passage", None
            rd = {"room_type": rtype, "label": uniq(label), **{k: _round(v) for k, v in rr.items()},
                  "open_group": "hall", "hall_kind": kind, "no_door": kind != "living" or not has_living,
                  "_alloc": alloc, "min_dims": [3.0, 5.0] if kind == "passage" else list(it.mins if it else (6.0, 8.0))}
            hall_label_by_kind.setdefault(kind, rd["label"])
            hall_rooms.append(rd)
            rooms.append(rd)
    hall_labels = [r["label"] for r in hall_rooms]

    # envelope of the footprint
    fp = frame.rect(0.0, row_pos[-1], 0.0, col_pos[-1])
    envelope = {"x": _round(fp["x"]), "y": _round(fp["y"]), "w": _round(fp["w"]), "h": _round(fp["h"])}

    # ---- slot rooms ---------------------------------------------------------------
    slot_rooms: List[Tuple[dict, Region]] = []
    for idx, sl in enumerate(tpl.slots):
        it = sl.item
        rect = cell_rect(sl)
        rd = {"room_type": it.rtype, "label": uniq(it.label), **{k: _round(v) for k, v in rect.items()},
              "_alloc": it.alloc, "min_dims": list(it.mins), "_item": it}
        rooms.append(rd)
        slot_rooms.append((rd, sl))

    # ---- doors onto the hall + door-wall bookkeeping ------------------------------
    def best_hall_edge(r: dict, prefer_kind: Optional[str] = None):
        best = None
        for hr in hall_rooms:
            se = shared_edge(r, hr)
            if not se:
                continue
            side, lo, hi = se
            length = hi - lo
            bonus = 1.5 if prefer_kind and hr["hall_kind"] == prefer_kind else 0.0
            key = length + bonus
            if best is None or key > best[0]:
                best = (key, side, lo, hi, hr)
        return best

    avoid_by_room: Dict[str, List[Tuple[float, float]]] = {}
    for rd, sl in slot_rooms:
        it = rd["_item"]
        be = best_hall_edge(rd, "dining" if rd["room_type"] == "kitchen" else None)
        if be is None or (be[3] - be[2]) < DOOR_CLEAR:
            return_none = True
            raise LayoutInfeasible(f"{rd['label']} cannot open onto the hall (no shared wall); try another hall shape.")
        _, side, lo, hi, hr = be
        c = _door_center(lo, hi)
        rd["door_spec"] = {"wall": side, "position": round(c - _side_axis_start(rd, side), 2), "to": hr["label"]}
        rd["door_side"] = side
        rd["door_to"] = [hr["label"]]
        if side in ("N", "S"):
            wall_pt = (rd["x"] + (rd["w"] if side == "N" else 0.0), c)
        else:
            wall_pt = (c, rd["y"] + (rd["h"] if side == "E" else 0.0))
        avoid_by_room[rd["label"]] = [wall_pt]

    # ---- contained rooms -----------------------------------------------------------
    contained_rooms: List[dict] = []

    def add_contained(parent: dict, spec: dict, rect: dict, label_hint: Optional[str] = None) -> dict:
        alloc = spec["alloc"]
        label = uniq(alloc.label if alloc is not None else (label_hint or "Room"))
        rd = {"room_type": alloc.room_type, "label": label, **{k: _round(v) for k, v in rect.items()},
              "inside_of": parent["label"], "contained_kind": spec["kind"], "_alloc": alloc,
              "min_dims": [min(spec["dims"]) * 0.9, max(spec["dims"]) * 0.9]}
        return rd

    # ---- oversized stand-alone common washroom: keep a real washroom + a wash-area lobby --------
    # (a grid cell can be much bigger than a washroom; a 130 sq ft bathroom is not realistic)
    wash_children: List[dict] = []
    for rd, sl in slot_rooms:
        if rd["room_type"] in BATH_TYPES and rd["w"] * rd["h"] > 75.0:
            alloc = rd.get("_alloc")
            bath_label = rd["label"]
            used_label.discard(bath_label)
            old_label = rd["label"]
            rd["room_type"] = "corridor_lobby"
            rd["label"] = uniq("Wash Area")
            rd["min_dims"] = [3.0, 4.0]
            rd["_alloc"] = None
            avoid_by_room[rd["label"]] = avoid_by_room.pop(old_label, [])
            spec = {"alloc": alloc, "kind": "common_bath", "mode": "inside", "dims": BATH_DIMS,
                    "area": BATH_DIMS[0] * BATH_DIMS[1], "mins": (5.0, 6.5)}
            door_wall = rd["door_spec"]["wall"]
            rect = _place_contained(rd, spec, [], avoid_by_room[rd["label"]], envelope,
                                    want_wall=_opposite(door_wall), not_wall=door_wall, toward=_toward("bathroom"),
                                    min_free=(3.0, 4.0), rtype="bathroom")
            if rect is None:
                raise LayoutInfeasible("No space for the common washroom.")
            used_label.discard(bath_label)
            crd = add_contained(rd, spec, rect)
            crd["door_to"] = [rd["label"]]
            crd["door_mode"] = "inside"
            _inside_door(crd, rd)
            wash_children.append(crd)
            rd["_item"].contained = []
    contained_rooms_pre = wash_children

    placed_taken: Dict[str, List[dict]] = {}
    for rd, sl in slot_rooms:
        it = rd["_item"]
        taken = placed_taken.setdefault(rd["label"], [])
        for spec in it.contained:
            door_wall = rd["door_spec"]["wall"]
            if spec["kind"] == "attached_bath":
                rect = _place_contained(rd, spec, taken, avoid_by_room[rd["label"]], envelope,
                                        want_wall=_opposite(door_wall), not_wall=door_wall, toward=(1, -1),
                                        min_free=(6.0, 9.0), rtype=spec["alloc"].room_type)
            elif spec["kind"] == "under_stair_bath":
                # under the stairs: carved into the staircase block on the wall that faces the hall
                rect = _place_contained(rd, spec, taken, [], envelope,
                                        want_wall=door_wall, not_wall=None, toward=(1, -1), min_free=(3.0, 8.0),
                                        rtype="bathroom")
            else:       # store inside kitchen
                rect = _place_contained(rd, spec, taken, avoid_by_room[rd["label"]], envelope,
                                        want_wall=_opposite(door_wall), not_wall=door_wall, toward=_toward("store_room"),
                                        min_free=(5.5, 5.5), rtype="store_room")
            if rect is None:
                raise LayoutInfeasible(f"No space to fit {spec['alloc'].label} inside {rd['label']}.")
            crd = add_contained(rd, spec, rect)
            if spec["kind"] == "under_stair_bath":
                crd["door_spec"] = None
                be = None
                for hr in hall_rooms:
                    se = shared_edge(crd, hr)
                    if se and (be is None or se[2] - se[1] > be[2] - be[1]):
                        be = (hr, *se)
                if be is None:
                    # stair sits against the hall only on its own door wall: the bath must touch it
                    raise LayoutInfeasible("The under-stair washroom cannot open onto the hall.")
                hr, side, lo, hi = be
                c = _door_center(lo, hi, width=2.5)
                crd["door_spec"] = {"wall": side, "position": round(c - _side_axis_start(crd, side), 2), "to": hr["label"]}
                crd["door_side"] = side
                crd["door_to"] = [hr["label"]]
                # the staircase keeps its own opening on the part of the wall the washroom does not use
                sw = rd["door_spec"]["wall"]
                if side == sw:
                    w0 = _side_axis_start(rd, sw)
                    w1 = w0 + (rd["h"] if sw in ("S", "N") else rd["w"])
                    b0 = _side_axis_start(crd, side)
                    b1 = b0 + (crd["h"] if side in ("S", "N") else crd["w"])
                    left, right = (w0, b0 - 0.3), (b1 + 0.3, w1)
                    best_iv = max((left, right), key=lambda iv: iv[1] - iv[0])
                    if best_iv[1] - best_iv[0] < 3.4:
                        raise LayoutInfeasible("The staircase has no room for its own opening beside the under-stair washroom.")
                    cpos = (best_iv[0] + best_iv[1]) / 2.0
                    rd["door_spec"]["position"] = round(cpos - w0, 2)
                    avoid_by_room[rd["label"]] = []
            else:
                crd["door_to"] = [rd["label"]]
                crd["door_mode"] = "inside"
                _inside_door(crd, rd)
            taken.append(crd)
            contained_rooms.append(crd)
            # a second contained room must also keep clear of this one's door side: handled via `taken`
    # hall-hosted contained rooms (common bath w/o stairs, pooja, balcony)
    hall_taken: Dict[str, List[dict]] = {}
    main_wall_pt = None
    living_rd = next((r for r in hall_rooms if r["hall_kind"] == "living"), None)
    for spec in hall_specs:
        host_kind = spec["host"]
        host = None
        if host_kind == "living":
            host = living_rd
        else:
            host = next((r for r in hall_rooms if r["hall_kind"] == "passage"), None) or \
                   next((r for r in hall_rooms if r["hall_kind"] == "dining"), None) or living_rd
        if host is None:
            raise LayoutInfeasible("No hall space to host the washroom/pooja.")
        taken = hall_taken.setdefault(host["label"], [])
        # keep clear of every slot-room door that opens onto this host and of the main entrance
        pts = []
        for rd, _sl in slot_rooms:
            if rd["door_spec"]["to"] == host["label"]:
                side = rd["door_spec"]["wall"]
                c = rd["door_spec"]["position"] + _side_axis_start(rd, side)
                if side in ("N", "S"):
                    pts.append((rd["x"] + (rd["w"] if side == "N" else 0.0), c))
                else:
                    pts.append((c, rd["y"] + (rd["h"] if side == "E" else 0.0)))
        rt = spec["alloc"].room_type
        rect = _place_contained(host, spec, taken, pts, envelope, want_wall=None, not_wall=None,
                                toward=_toward(rt), min_free=(6.0, 6.0), rtype=rt)
        if rect is None and host_kind != "living":
            for alt in [r for r in hall_rooms if r is not host and r["hall_kind"] != "living"] + [living_rd]:
                if alt is None:
                    continue
                taken = hall_taken.setdefault(alt["label"], [])
                rect = _place_contained(alt, spec, taken, [], envelope, None, None, _toward(rt), (6.0, 6.0))
                if rect is not None:
                    host = alt
                    break
        if rect is None:
            raise LayoutInfeasible(f"No space for {spec['alloc'].label} inside the hall.")
        crd = add_contained(host, spec, rect)
        crd["door_to"] = [host["label"]]
        crd["door_mode"] = "inside"
        _inside_door(crd, host)
        hall_taken.setdefault(host["label"], []).append(crd)
        contained_rooms.append(crd)

    rooms.extend(contained_rooms)
    rooms.extend(contained_rooms_pre)

    # ---- parking / garden ------------------------------------------------------------
    site_spaces: List[dict] = []
    if park_a is not None and fit.get("parking_rect"):
        pr = fit["parking_rect"]
        site_spaces.append({"room_type": "parking", "label": uniq(park_a.label), **{k: _round(v) for k, v in pr.items()},
                            "_alloc": park_a, "no_door": True, "min_dims": [8.0, 10.0]})
    if garden_a is not None:
        gr = _garden_rect(L, W, envelope, [s for s in site_spaces], garden_a)
        if gr is None:
            raise LayoutInfeasible("No open space left for the garden.")
        site_spaces.append({"room_type": "garden", "label": uniq(garden_a.label), **{k: _round(v) for k, v in gr.items()},
                            "_alloc": garden_a, "no_door": True, "min_dims": [5.0, 5.0]})
    rooms.extend(site_spaces)

    # ---- zones / Vastu fields -------------------------------------------------------------
    plot_env = {"x": 0.0, "y": 0.0, "w": L, "h": W}
    out_rooms: List[dict] = []
    match = acceptable = 0
    sc_sum = sc_w = 0.0
    indoor_count = 0
    for r in rooms:
        alloc = r.pop("_alloc", None)
        r.pop("_item", None)
        env = plot_env if r["room_type"] in SITE_TYPES else envelope
        zone = zone9_of_point(r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0, env)
        rule = rule_for(r["room_type"], getattr(alloc, "hint_zones", None) if alloc else None)
        grade = rule.grade(zone)
        ok = grade in ("ideal", "acceptable")
        if r.get("contained_kind") == "attached_bath" and zone not in ("NE", "C"):
            ok = True          # an attached washroom belongs to its bedroom's wall; only NE / centre are refused
        pref = getattr(alloc, "preferred_zone", None) or (rule.ideal[0] if rule.ideal else "C")
        r.update({"zone": zone, "preferred_zone": pref, "matched": zone == pref, "vastu_acceptable": bool(ok)})
        r.setdefault("door_side", r.get("door_side", "S"))
        if getattr(alloc, "custom", False):
            r["custom"] = True
        if r["room_type"] not in SITE_TYPES:
            indoor_count += 1
            match += int(zone == pref)
            acceptable += int(ok)
            sc_sum += rule.weight * zone_score(rule, zone)
            sc_w += rule.weight
        out_rooms.append(r)

    # parents before children (drawing order)
    out_rooms.sort(key=lambda r: (1 if r.get("inside_of") else 0))
    checks = run_all_checks(out_rooms, L, W)
    if not checks["passed"]:
        raise LayoutInfeasible("Residential layout failed validation: " + ", ".join(
            k for k in ("collisions", "boundary_violations", "dimension_violations") if checks[k]))

    planning_env = {"x": _round(sb_tuple_x(e, L, W, sb_tuple)[0]), "y": _round(sb_tuple_x(e, L, W, sb_tuple)[1]),
                    "w": _round(sb_tuple_x(e, L, W, sb_tuple)[2]), "h": _round(sb_tuple_x(e, L, W, sb_tuple)[3])}
    if plan.tpl.mirror:
        notes.append("Plan mirrored so Vastu zones fall on the preferred sides.")
    shape_name = {"rectangle": "Rectangular hall", "l_shape": "L-shaped plan (rooms wrap the hall on two sides)",
                  "u_shape": "U-shaped plan (rooms wrap the hall on three sides)"}[tpl.shape]
    if shape != "auto" and tpl.shape != shape:
        notes.append(f"Requested {shape.replace('_', ' ')} was not possible here; {tpl.shape.replace('_', ' ')} used instead.")
    layout = {
        "plot": {"length": round(L, 2), "width": round(W, 2), "area": round(L * W, 2)},
        "floors": 1, "road_direction": road_direction,
        "setback_ft": round(max(sides.values()), 2), "setbacks_ft": {k: round(v, 2) for k, v in sides.items()},
        "building_envelope": dict(envelope), "occupied_envelope": dict(envelope), "planning_envelope": planning_env,
        "rooms": out_rooms, "collisions": checks["collisions"], "boundary_violations": checks["boundary_violations"],
        "dimension_violations": checks["dimension_violations"], "passed_checks": True,
        "vastu_match_count": match, "vastu_acceptable_count": acceptable,
        "vastu_score": round(sc_sum / max(sc_w, 1e-6), 4),
        "vastu_match_percent": round(100.0 * match / max(indoor_count, 1), 1),
        "vastu_enabled": bool(vastu),
        "vastu_layout_engine": "Hall-centred residential planner (hall group + per-room doors + attached/under-stair washrooms)",
        "building_type": str(building_type or "custom").lower(),
        "compression_ratio": round(fit["compression"], 3), "layout_notes": notes,
        "hall_shape": tpl.shape, "hall_shape_label": shape_name, "plan_template": tpl.name,
        "entrance_side": e, "gap_report": {"max_internal_gap_ft": 0.0, "gaps": []}, "circulation": [],
        "alignment_report": {"layout_grid": "HALL_GRID", "aligned_percent": 100.0},
        "alignment_fallback": None, "no_unplanned_gaps": True,
        "site_spaces": [r for r in out_rooms if r["room_type"] in SITE_TYPES],
        "floor": floor,
    }
    return layout


def sb_tuple_x(e: str, L: float, W: float, sbt) -> Tuple[float, float, float, float]:
    """Planning envelope (x, y, w, h) from front/back/lateral-low/lateral-high setbacks."""
    sb_f, sb_b, sb_lo, sb_hi = sbt
    if e == "N":
        return sb_b, sb_lo, L - sb_f - sb_b, W - sb_lo - sb_hi
    if e == "S":
        return sb_f, sb_lo, L - sb_f - sb_b, W - sb_lo - sb_hi
    if e == "E":
        return sb_lo, sb_b, L - sb_lo - sb_hi, W - sb_f - sb_b
    return sb_lo, sb_f, L - sb_lo - sb_hi, W - sb_f - sb_b


def _garden_rect(L: float, W: float, env: dict, taken: List[dict], alloc) -> Optional[dict]:
    """Largest free strip beside/behind the building (outside footprint and parking)."""
    gd = _dims_of(alloc, "garden")
    gmin = max(5.0, MIN_DIMS["garden"][0])
    cands = [
        {"x": 0.0, "y": 0.0, "w": env["x"], "h": W},                                   # south strip
        {"x": env["x"] + env["w"], "y": 0.0, "w": L - env["x"] - env["w"], "h": W},     # north strip
        {"x": 0.0, "y": 0.0, "w": L, "h": env["y"]},                                    # west strip
        {"x": 0.0, "y": env["y"] + env["h"], "w": L, "h": W - env["y"] - env["h"]},     # east strip
    ]
    best = None
    for c in cands:
        if c["w"] < gmin or c["h"] < gmin:
            continue
        w = min(c["w"] - 1.0, max(gd) * 1.0) if c["w"] > gmin + 1 else c["w"]
        h = min(c["h"] - 1.0, max(gd) * 1.0) if c["h"] > gmin + 1 else c["h"]
        r = {"x": c["x"] + 0.5 if c["w"] > gmin + 1 else c["x"], "y": c["y"] + 0.5 if c["h"] > gmin + 1 else c["y"], "w": w, "h": h}
        if any(_rect_overlap(r, t) for t in taken):
            continue
        if best is None or r["w"] * r["h"] > best["w"] * best["h"]:
            best = r
    return best
