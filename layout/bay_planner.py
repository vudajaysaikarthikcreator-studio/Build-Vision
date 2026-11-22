"""Corridor-and-bay planner: regular, aligned floor plans.

The general packers (guillotine / zone-grid) tile whatever rectangle is left over, which is why
a school came out with a sliver-shaped staff room, a villa as a 22 x 22 ft box on an 80 x 50 ft
plot, and a hostel with a 19 x 50 ft dining hall.  Real buildings of these types are laid out
the way architects do it:

    +------------------------------------------------------------+
    |  front bay  : entrance / public rooms (entry room centred) |
    +------------------------------------------------------------+
    |  CORRIDOR (or family hall in a house)                      |
    +------------------------------------------------------------+
    |  back bay   : repeated rooms, equal-width structural bays   |
    +------------------------------------------------------------+

* one rectangular footprint, sized from the room programme and fitted to the plot;
* two bays of rooms share ONE depth each, so every wall line runs straight through the plan;
* identical rooms (classrooms, hostel rooms, wards, bedrooms) have identical widths;
* very large halls (auditorium, dining hall ...) become end blocks spanning the full depth;
* the staircase always sits at the same corridor end of the same bay, so floors stack exactly,
  and an upper floor reuses the bay depths and outer walls of the floor below;
* every room has its own door onto the corridor; attached washrooms are carved inside their
  bedroom; the common washroom sits under the staircase.

``plan_bays`` returns the same layout dict as the other planners, or raises ``LayoutInfeasible``
so the caller can fall back to the older engines.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from planning.building_programs import MIN_DIMS, canonical_type
from vastu.vastu_engine import ZONE_GRID, rule_for, zone_score, zone9_of_point, SITE_TYPES
from vastu.vastu_rules import ACCEPTABLE_ZONES
from .collision_checker import run_all_checks
from .guaranteed_layout import LayoutInfeasible
from .layout_generator import DOOR_SIDE_FOR_ZONE, _setback_for_plot, _gap_report
from .quality import max_aspect, _CAP_AREA
from . import sizing as _sizing
from .residential_planner import (
    _inside_door, _place_contained, BATH_DIMS, BATH_TYPES, BEDROOM_TYPES, shared_edge,
)

RES_TYPES = {"villa", "duplex", "farmhouse"}
INST_TYPES = {"school", "college", "hostel", "hospital", "clinic", "hotel", "office", "bank", "apartment_block"}
BAY_TYPES = RES_TYPES | INST_TYPES

_ENTRY_PREF = ("living_room", "reception", "lobby", "waiting_area", "showroom", "teller_area", "open_work_area")
# Halls that become end blocks (full building depth) when they are large.
_HALL_END = {"multipurpose_hall", "auditorium", "dining_hall", "stage", "common_room", "open_work_area"}
_SMALL = {"restroom", "bathroom", "toilet", "store_room", "pantry", "security_room", "server_room",
          "utility_room", "service_room", "laundry", "housekeeping", "cashier", "vault", "nurse_station",
          "pooja_room", "balcony", "hr_room", "break_room"}
# Corridor / family-hall width and the bay depth range that suits each building type.
_CORRIDOR = {"school": 8.0, "college": 9.0, "hostel": 7.0, "hospital": 9.0, "clinic": 6.0, "hotel": 6.5,
             "office": 7.0, "bank": 7.0, "apartment_block": 7.0, "villa": 8.0, "duplex": 7.5, "farmhouse": 8.0}
_DEPTH_RANGE = {"school": (17, 27), "college": (17, 27), "hostel": (11, 18), "hospital": (13, 23),
                "clinic": (11, 18), "hotel": (12, 19), "office": (12, 22), "bank": (13, 23),
                "apartment_block": (19, 30), "villa": (12, 21), "duplex": (11.5, 19), "farmhouse": (12, 21)}
_GRID = 0.25
_FRONT_YARD_MIN = 6.0

REJ: Dict[str, int] = {}          # why candidate footprints were rejected (diagnostics only)


def _rej(reason: str):
    REJ[reason] = REJ.get(reason, 0) + 1
    return None


def _snap(v: float) -> float:
    return round(round(v / _GRID) * _GRID, 2)


# --------------------------------------------------------------------------- frame
class Frame:
    """Maps local (u along the corridor, v from the entrance side inwards) to plot rectangles (x, y, w, h).

    Plot axes: x = south -> north, y = west -> east.  ``e`` is the entrance side (N/S/E/W).
    """

    def __init__(self, e: str, x0: float, y0: float, fx: float, fy: float):
        self.e, self.x0, self.y0, self.fx, self.fy = e, x0, y0, fx, fy
        self.U = fy if e in "NS" else fx
        self.V = fx if e in "NS" else fy

    def rect(self, u0: float, v0: float, du: float, dv: float) -> Tuple[float, float, float, float]:
        e = self.e
        if e == "N":
            return (self.x0 + self.fx - (v0 + dv), self.y0 + u0, dv, du)
        if e == "S":
            return (self.x0 + v0, self.y0 + u0, dv, du)
        if e == "E":
            return (self.x0 + u0, self.y0 + self.fy - (v0 + dv), du, dv)
        return (self.x0 + u0, self.y0 + v0, du, dv)               # W

    def local_of_zone(self, zone: str) -> Tuple[float, float]:
        """(uf, vf) in 0..1 of a Vastu zone cell centre: uf along the corridor, vf from the entrance inwards."""
        fx, fy = 0.5, 0.5
        for r_top, row in enumerate(ZONE_GRID):
            for c, z in enumerate(row):
                if z == zone:
                    fx, fy = (2.5 - r_top) / 3.0, (c + 0.5) / 3.0
        e = self.e
        if e == "N":
            return fy, 1.0 - fx
        if e == "S":
            return fy, fx
        if e == "E":
            return fx, 1.0 - fy
        return fx, fy

    def wall_to_corridor(self, band: int) -> str:
        """Plot side of a room in bay ``band`` (0 = entrance side) that faces the corridor."""
        toward_back = {"N": "S", "S": "N", "E": "W", "W": "E"}[self.e]
        return toward_back if band == 0 else {"N": "S", "S": "N", "E": "W", "W": "E"}[toward_back]

    def wall_to_corridor_end(self, at_high_u: bool) -> str:
        """Wall of an end block that faces the corridor (the block sits at the low or high end of u)."""
        low_side = "W" if self.e in "NS" else "S"
        high_side = "E" if self.e in "NS" else "N"
        return low_side if at_high_u else high_side


# --------------------------------------------------------------------------- cells
class Cell:
    __slots__ = ("alloc", "rt", "label", "area", "mins", "uf", "vf", "role", "bath", "band", "end_block",
                 "fixed_w", "common_baths", "high_u")

    def __init__(self, alloc, area: float, mins: Tuple[float, float]):
        self.alloc, self.rt, self.label = alloc, alloc.room_type, alloc.label
        self.area, self.mins = area, mins
        self.uf = self.vf = 0.5
        self.role = "room"
        self.bath = None
        self.common_baths: List = []
        self.band = 0
        self.end_block = False
        self.fixed_w: Optional[float] = None
        self.high_u = False


def _mins_of(a) -> Tuple[float, float]:
    m = getattr(a, "min_dims", None)
    lo = tuple(m) if m else MIN_DIMS.get(a.room_type, (4.0, 5.0))
    return float(min(lo)), float(max(lo))


def _w_min(c: Cell, d: float) -> float:
    lo, hi = c.mins
    if d >= hi - 1e-6:
        base = lo
    elif d >= lo - 1e-6:
        base = hi
    else:
        return math.inf
    # a room in a deep bay must still be room-shaped: width >= depth / max aspect
    if c.rt not in _HALL_END:
        base = max(base, d / max_aspect(c.rt) + 0.05)
    return base


# ------------------------------------------------------------------ assignment
def _entry_of(cells: List[Cell]) -> Optional[Cell]:
    for t in _ENTRY_PREF:
        for c in cells:
            if c.rt == t and not c.end_block:
                return c
    return None


def _lpt(pool: List[Cell], start0: float, start1: float) -> None:
    """Longest-processing-time balancing of ``pool`` over the two bays."""
    loads = [start0, start1]
    for c in sorted(pool, key=lambda c: (-c.area, c.label)):
        b = 0 if loads[0] <= loads[1] else 1
        c.band = b
        loads[b] += c.area


def _assign_by_depth(cells: List[Cell], d_in: Tuple[float, float], ws: float, stair_band: Optional[int]) -> None:
    """Upper floor: give each bay a share of the rooms proportional to its (fixed) depth."""
    entry = next((c for c in cells if c.role == "entry"), None)
    loads = [0.0, 0.0]
    for s_ in (c for c in cells if c.role == "stair"):
        s_.band = stair_band if stair_band is not None else 0
        loads[s_.band] += (ws or 9.0) * d_in[s_.band]
    pool = [c for c in cells if not c.end_block and c.role != "stair"]
    if entry is not None:
        entry.band = 0
        loads[0] += entry.area
    for c in sorted((c for c in pool if c is not entry), key=lambda c: (-c.area, c.label)):
        b = 0 if (loads[0] + c.area) / d_in[0] <= (loads[1] + c.area) / d_in[1] else 1
        c.band = b
        loads[b] += c.area


def _assign_bands(cells: List[Cell], variant: int) -> None:
    """Choose a bay (0 = entrance side, 1 = far side) for every non-end-block cell.

    variant 0 / 1 : by Vastu zone (<0.5 / <=0.5 of the depth), then balance the two areas;
    variant 2 / 3 : by size class - small rooms form one shallow bay, big repeated rooms the other;
    variant 4 / 5 : area-balanced (all rooms / small rooms pinned to the entrance bay first).
    """
    movable = [c for c in cells if not c.end_block and c.role != "stair"]
    entry = next((c for c in cells if c.role == "entry"), None)
    for c in movable:
        if c is entry:
            c.band = 0
        elif variant == 0:
            c.band = 0 if c.vf < 0.5 - 1e-9 else 1
        elif variant == 1:
            c.band = 0 if c.vf <= 0.5 + 1e-9 else 1
        elif variant in (2, 3):
            small = c.rt in _SMALL or c.area < 150
            c.band = (0 if small else 1) if variant == 2 else (1 if small else 0)
    for s_ in (c for c in cells if c.role == "stair"):
        if variant in (2, 4, 5):
            s_.band = 0
        elif variant == 3:
            s_.band = 1
        else:
            s_.band = 0 if s_.vf < 0.5 else 1
    stair_area = sum(c.area for c in cells if c.role == "stair")
    if variant == 4:
        _lpt([c for c in movable if c is not entry], (entry.area if entry else 0.0) + stair_area, 0.0)
        return
    if variant == 5:
        small = [c for c in movable if c is not entry and (c.rt in _SMALL or c.area < 150)]
        for c in small:
            c.band = 0
        pool = [c for c in movable if c is not entry and c not in small]
        _lpt(pool, (entry.area if entry else 0.0) + stair_area + sum(c.area for c in small), 0.0)
        return
    if variant >= 2:
        return

    def load(b):
        return sum(c.area for c in cells if not c.end_block and c.band == b)

    for _ in range(60):
        l0, l1 = load(0), load(1)
        tot = l0 + l1
        if tot <= 0 or abs(l0 - l1) <= 0.07 * tot:
            break
        src, dst = (0, 1) if l0 > l1 else (1, 0)
        gap = abs(l0 - l1)
        cand = [c for c in movable if c.band == src and c is not entry and c.area <= gap * 0.9 + 1e-6]
        if not cand:
            break
        min(cand, key=lambda c: (abs(c.vf - 0.5), -c.area)).band = dst


def _order_band(cells: List[Cell], band: int) -> List[Cell]:
    row = [c for c in cells if not c.end_block and c.band == band and c.role != "stair"]
    row.sort(key=lambda c: (c.uf, c.vf, c.label))
    for bal in [c for c in row if c.rt == "balcony"]:          # a balcony closes the bay, it is never wedged between rooms
        row.remove(bal)
        if bal.uf >= 0.5:
            row.append(bal)
        else:
            row.insert(0, bal)
    entry = next((c for c in row if c.role == "entry"), None)
    if entry is not None:
        row.remove(entry)
        row.insert(len(row) // 2, entry)
    # kitchen next to the dining room whenever both share a bay
    k = next((c for c in row if c.rt in ("kitchen", "restaurant_kitchen")), None)
    dn = next((c for c in row if c.rt in ("dining_room", "dining_hall")), None)
    if k is not None and dn is not None and abs(row.index(k) - row.index(dn)) > 1:
        row.remove(dn)
        i = row.index(k)
        row.insert(i + 1 if dn.uf >= k.uf else i, dn)
    return row


# ------------------------------------------------------------------ widths
def _split_widths(cells: List[Cell], d: float, total: float) -> Optional[List[float]]:
    """Widths for one bay row of depth ``d`` summing exactly to ``total``.

    Area-proportional, clamped to each room's [minimum width, maximum width] (the maximum keeps the room
    room-shaped: width <= depth x max aspect); a clamped room's surplus / shortfall goes to the others.
    """
    n = len(cells)
    if n == 0:
        return []
    want = [c.fixed_w if c.fixed_w is not None else c.area / d for c in cells]
    lo = [c.fixed_w if c.fixed_w is not None else _w_min(c, d) for c in cells]
    for i, c in enumerate(cells):
        if c.role == "entry" and c.fixed_w is None:
            lo[i] = max(lo[i], min(0.2 * total, 16.0 if c.rt != "living_room" else 14.0))
    hi = []
    for c in cells:
        if c.fixed_w is not None:
            hi.append(c.fixed_w)
        elif c.rt in _HALL_END or c.rt == "corridor_lobby":
            hi.append(math.inf)
        else:
            hi.append(d * max_aspect(c.rt) - 0.05)
    if any(math.isinf(v) for v in lo):
        return _rej("split: room too deep for its minimum size")
    if sum(lo) > total + 1e-6:
        return _rej("split: rooms need more length than available")
    if sum(hi) < total - 1e-6:
        return _rej("split: too few rooms for the length")
    w = list(want)
    frozen = [c.fixed_w is not None for c in cells]
    for _ in range(2 * n + 4):
        free = [i for i in range(n) if not frozen[i]]
        if not free:
            break
        rest = total - sum(w[i] for i in range(n) if frozen[i])
        sw_ = sum(want[i] for i in free)
        if sw_ <= 0:
            return None
        f = rest / sw_
        for i in free:
            w[i] = want[i] * f
        changed = False
        for i in free:
            if w[i] < lo[i] - 1e-9:
                w[i], frozen[i], changed = lo[i], True, True
            elif w[i] > hi[i] + 1e-9:
                w[i], frozen[i], changed = hi[i], True, True
        if not changed:
            break
    if abs(sum(w) - total) > 1e-6:
        free = [i for i in range(n) if not frozen[i]]
        if not free:
            return None
        w[free[-1]] += total - sum(w)
    out, acc, prev = [], 0.0, 0.0
    for i in range(n):
        acc += w[i]
        edge = _snap(acc) if i < n - 1 else total
        out.append(round(edge - prev, 2))
        prev = edge
    if any(out[i] < lo[i] - 0.3 or out[i] > hi[i] + 0.3 or out[i] <= 0 for i in range(n)):
        return _rej("split: grid snapping")
    return out


def _cell_ok(c: Cell, w: float, d: float, tol: float = 0.0) -> bool:
    lo, hi = c.mins
    if not ((w >= lo - 0.05 and d >= hi - 0.05) or (w >= hi - 0.05 and d >= lo - 0.05)):
        return False
    ratio = max(w, d) / max(min(w, d), 1e-6)
    if ratio > max_aspect(c.rt) + tol + 1e-6:
        return False
    cap = _CAP_AREA.get(c.rt)
    if cap and w * d > cap + 1e-6:
        return False
    return True


# ------------------------------------------------------------------ site spaces
def _strip_width(sites: List) -> float:
    w = 0.0
    for a in sites:
        dd = _sizing.base_dims(a)
        w = max(w, min(dd) if a.room_type == "parking" else min(_mins_of(a)[1], 8.0))
    return round(w + 1.0, 2)


def _site_room(a, r, mn) -> dict:
    x, y, w, h = [round(v, 2) for v in r]
    return {"room_type": a.room_type, "label": a.label, "x": x, "y": y, "w": w, "h": h,
            "_alloc": a, "no_door": True, "min_dims": [min(mn[0], 5.0), min(mn[1], 5.0)]}


def _place_sites(sites: List, e: str, buildable: Tuple[float, float, float, float], frame: Frame,
                 mode: str, strip_w: float) -> List[dict]:
    """Parking / garden / loading bay: in the front yard, or in a side strip beside the building."""
    out: List[dict] = []
    if not sites:
        return out
    bx, by, bw, bh = buildable
    lateral_lo = {"N": True, "S": False, "E": True, "W": False}[e]       # Vastu corner: NW / SE for parking
    along_y = e in "NS"
    order = sorted(sites, key=lambda s: {"parking": 0, "loading_bay": 1, "garden": 2}.get(s.room_type, 3))

    def rect_for(depth_from_front: float, dep: float, lat_a: float, wid: float):
        if along_y:
            xx = (bx + bw - depth_from_front - dep) if e == "N" else (bx + depth_from_front)
            return (xx, lat_a, dep, wid)
        yy_ = (by + bh - depth_from_front - dep) if e == "E" else (by + depth_from_front)
        return (lat_a, yy_, wid, dep)

    lat_lo_edge = by if along_y else bx
    lat_len = bh if along_y else bw
    depth_len = bw if along_y else bh
    if mode == "side":
        lat_a0 = lat_lo_edge if lateral_lo else lat_lo_edge + lat_len - strip_w
        cursor = 0.0
        for a in order:
            dd = _sizing.base_dims(a)
            mn = _mins_of(a)
            remaining = depth_len - cursor
            if a.room_type == "garden":
                dep, wid = remaining, strip_w
            else:
                dep, wid = min(max(dd), remaining), min(min(dd), strip_w)
            if dep < min(mn) - 1e-6 or wid < min(mn) - 1e-6:
                raise LayoutInfeasible(f"Side strip is too small for the {a.label}.")
            lat_a = lat_a0 if lateral_lo else lat_a0 + (strip_w - wid)
            out.append(_site_room(a, rect_for(cursor, dep, lat_a, wid), mn))
            cursor += dep
        return out
    # front yard
    if e == "N":
        depth_avail = bx + bw - (frame.x0 + frame.fx)
    elif e == "S":
        depth_avail = frame.x0 - bx
    elif e == "E":
        depth_avail = by + bh - (frame.y0 + frame.fy)
    else:
        depth_avail = frame.y0 - by
    cur_lo, cur_hi = lat_lo_edge, lat_lo_edge + lat_len
    for a in order:
        dd = _sizing.base_dims(a)
        mn = _mins_of(a)
        dep = min(depth_avail if a.room_type == "garden" else max(dd), depth_avail)
        wid = min(30.0 if a.room_type == "garden" else min(dd), cur_hi - cur_lo)
        if dep < min(mn) - 1e-6 or wid < min(mn) - 1e-6:
            raise LayoutInfeasible(f"Front yard is too small for the {a.label}.")
        lat_a = cur_lo if lateral_lo else cur_hi - wid
        out.append(_site_room(a, rect_for(0.0, dep, lat_a, wid), mn))
        if lateral_lo:
            cur_lo += wid
        else:
            cur_hi -= wid
    return out


# ------------------------------------------------------------------ search
def _search(cells: List[Cell], bt: str, corridor_w: float, lateral_avail: float, depth_avail: float,
            frame_in: Optional[dict], stair_ws: Optional[float]) -> Optional[dict]:
    ends = [c for c in cells if c.end_block]
    stair = next((c for c in cells if c.role == "stair"), None)
    lo_d, hi_d = _DEPTH_RANGE.get(bt, (12, 22))

    def try_U(U_main: float, fixed_V: Optional[Tuple[float, float]] = None, ws_force: Optional[float] = None):
        rows = {0: _order_band(cells, 0), 1: _order_band(cells, 1)}
        ws = ws_force or (stair_ws if (stair is not None and stair_ws) else None)
        d = [0.0, 0.0]
        for b in (0, 1):
            fixed = 0.0
            if stair is not None and stair.band == b:
                if ws is None:
                    ws = min(max(stair.area / ((lo_d + hi_d) / 2.0), 8.0), 11.0)
                    if stair.common_baths:
                        ws = max(ws, 10.5)
                fixed = ws
            rest_area = sum(c.area for c in rows[b])
            avail = U_main - fixed
            if avail <= 4:
                return _rej("stair too wide")
            d[b] = fixed_V[b] if fixed_V else (rest_area / avail if rows[b] else lo_d)
        d = [_snap(max(v, 9.0)) for v in d]
        if not fixed_V and (min(d) < 9.0 or max(d) > 36.0):
            return _rej("depth")
        V = d[0] + corridor_w + d[1]
        end_w = [_snap(c.area / V) for c in ends]
        U_total = U_main + sum(end_w)
        if U_total > lateral_avail + 1e-6 or V > depth_avail + 1e-6:
            return _rej("plot")
        widths: List[List[float]] = []
        pen = 0.0
        if stair is not None:
            stair.fixed_w = ws
        for b in (0, 1):
            cs = list(rows[b])
            if stair is not None and stair.band == b:
                cs = ([stair] + cs) if not stair.high_u else (cs + [stair])
            if not cs:
                widths.append([])
                continue
            w = _split_widths(cs, d[b], _snap(U_main))
            if w is None:
                return _rej("split")
            for c, wi in zip(cs, w):
                if not _cell_ok(c, wi, d[b]):
                    return _rej("cell")
                pen += (max(wi, d[b]) / max(min(wi, d[b]), 1e-6) - 1.0) * 0.5
            widths.append(w)
            rows[b] = cs
        if stair is not None and ws:
            sd = d[stair.band]
            pen += 0.9 * max(0.0, max(ws, sd) / max(min(ws, sd), 1e-6) - 2.2)      # no 6 x 24 ft stair slot
        for c, wi in zip(ends, end_w):
            if not _cell_ok(c, wi, V, tol=0.8):
                return _rej("end block")
        aspect = U_total / max(V, 1e-6)             # length along the road / depth: wide frontage preferred
        pen += 4.0 * abs(math.log(aspect / 1.9)) + (6.0 * math.log(1.0 / aspect) if aspect < 1.0 else 0.0)
        pen += 6.0 * abs(d[0] - d[1]) / max(d[0] + d[1], 1.0)
        for dd_ in d:
            pen += 0.5 * max(0.0, dd_ - hi_d) + 0.25 * max(0.0, 11.0 - dd_)
        return {"score": pen, "U_main": _snap(U_main), "d": d, "V": V, "end_w": end_w, "U_total": U_total,
                "rows": rows, "widths": widths, "ws": ws}

    if frame_in:
        d_in = (frame_in["d"][0], frame_in["d"][1])
        V_in = d_in[0] + corridor_w + d_in[1]
        ew = sum(_snap(c.area / V_in) for c in ends)
        u_cap = _snap(frame_in["U_total"] - ew)
        need = 0.0
        for b in (0, 1):
            row = [c for c in cells if not c.end_block and c.band == b and c.role != "stair"]
            fixed = (stair_ws or 0.0) if (stair is not None and stair.band == b) else 0.0
            if row or fixed:
                need = max(need, sum(max(c.area / d_in[b], _w_min(c, d_in[b])) for c in row) + fixed)
        if need > u_cap * 1.12 or u_cap < 10:
            return _rej("frame too short")
        return try_U(_snap(min(u_cap, need)), fixed_V=d_in)
    def try_any(U, fixed_V=None):
        if stair is not None and stair.common_baths and not stair_ws:
            opts = [r for r in (try_U(U, fixed_V, 10.5), try_U(U, fixed_V, 12.0)) if r is not None]
            return min(opts, key=lambda r: r["score"]) if opts else None
        return try_U(U, fixed_V)

    best = None
    max_main = lateral_avail - sum(c.area / max(lo_d * 2 + corridor_w, 1.0) for c in ends)
    u = 14.0
    while u <= max_main + 1e-6:
        r = try_any(u)
        if r is not None and (best is None or r["score"] < best["score"]):
            best = r
        u += 0.5
    return best


# ------------------------------------------------------------------ main entry
def plan_bays(plot_length: float, plot_width: float, allocations: Sequence, entrance_direction: str = "North",
              road_direction: str = "North", vastu: bool = True, building_type: str = "custom", floor: int = 0,
              setback_ft: Optional[float] = None, setbacks: Optional[Dict[str, float]] = None,
              bay_frame: Optional[dict] = None, area_target: Optional[float] = None) -> Dict:
    bt = canonical_type(building_type)
    if bt not in BAY_TYPES:
        raise LayoutInfeasible("Corridor planner is not used for this building type.")
    REJ.clear()
    L, W = float(plot_length), float(plot_width)
    e = {"north": "N", "south": "S", "east": "E", "west": "W"}.get(
        str(entrance_direction or road_direction or "north").lower(), "N")
    allocs = list(allocations)
    sites = [a for a in allocs if a.room_type in SITE_TYPES]
    indoor = [a for a in allocs if a.room_type not in SITE_TYPES]
    if not indoor:
        raise LayoutInfeasible("No indoor rooms.")
    is_res = bt in RES_TYPES

    # ---- buildable rectangle
    sb = float(setback_ft) if setback_ft is not None else _setback_for_plot(L, W)
    sides = {k: float((setbacks or {}).get(k, sb)) for k in ("north", "east", "south", "west")}
    bx, by = sides["south"], sides["west"]
    bw, bh = L - sides["south"] - sides["north"], W - sides["west"] - sides["east"]
    if bw < 12 or bh < 12:
        raise LayoutInfeasible("Plot is too small for a building after setbacks.")
    along_front = bh if e in "NS" else bw          # lateral extent (corridor direction)
    depth_total = bw if e in "NS" else bh          # extent from the road inwards

    # ---- cells
    cells: List[Cell] = []
    by_label = {a.label: a for a in indoor}
    attached_baths, common_baths = {}, []
    for a in indoor:
        if a.room_type in BATH_TYPES and is_res:
            owner = getattr(a, "attached_to", None)
            if owner and owner in by_label and by_label[owner].room_type in BEDROOM_TYPES:
                attached_baths[owner] = a
            else:
                common_baths.append(a)
    skip = {id(a) for a in attached_baths.values()} | {id(a) for a in common_baths}
    for a in indoor:
        if id(a) in skip:
            continue
        c = Cell(a, _sizing.target_area_of(a), _mins_of(a))
        if a.room_type == "staircase":
            c.role = "stair"
        if a.label in attached_baths:
            c.bath = attached_baths[a.label]
            c.area += BATH_DIMS[0] * BATH_DIMS[1]
        cells.append(c)
    stair = next((c for c in cells if c.role == "stair"), None)
    if stair is not None:
        stair.common_baths = list(common_baths)
    elif common_baths:
        for a in common_baths:                        # no staircase to tuck it under: an ordinary washroom off the hall
            cells.append(Cell(a, max(_sizing.target_area_of(a), 45.0), _mins_of(a)))
        common_baths = []

    probe = Frame(e, 0.0, 0.0, 10.0, 10.0)
    for c in cells:
        c.uf, c.vf = probe.local_of_zone(c.alloc.zone or c.alloc.preferred_zone or "C")
    if not vastu:
        pub = {"reception", "lobby", "waiting_area", "living_room", "teller_area", "showroom", "open_work_area"}
        priv = BEDROOM_TYPES | {"hostel_room", "guest_room", "classroom", "office_cabin", "apartment_unit"}
        for i, c in enumerate(cells):
            c.vf = 0.1 if c.rt in pub else 0.9 if c.rt in priv else 0.5
            c.uf = (i + 0.5) / max(len(cells), 1)

    entry = _entry_of(cells)
    if entry is not None:
        entry.role, entry.vf = "entry", 0.0
    total_area = sum(c.area for c in cells)
    for c in cells:
        if (c.rt in _HALL_END and (c.area >= 0.14 * total_area or c.area >= 700)
                and c.role != "entry" and not is_res):
            c.end_block = True
            c.high_u = c.uf >= 0.5

    # The entry floor must be at least as big as the floors above it.
    own = sum(c.area for c in cells if c.role != "stair" and c.rt not in _HALL_END)
    if area_target and own > 0 and not bay_frame and not is_res and area_target > own * 1.10:
        lob_area = min((area_target - own) * 0.75, _sizing.MAX_AREA.get("lobby", 400.0))
        if lob_area >= 90.0 and not any(c.rt == "lobby" for c in cells):
            from .room_allocator import RoomAllocation
            la = RoomAllocation("lobby", "Entrance Lobby", "N", "N", True, lob_area)
            lc = Cell(la, lob_area, _mins_of(la))
            ent = next((c for c in cells if c.role == "entry"), None)
            lc.uf, lc.vf = (ent.uf if ent else 0.5), 0.05
            cells.append(lc)
            own += lob_area
    if area_target and own > 0 and not bay_frame and area_target > own * 1.03:
        k = min(area_target / own, 1.35)
        for c in cells:
            if c.role != "stair" and not c.end_block:
                c.area *= k
                cap = _CAP_AREA.get(c.rt)
                if cap:
                    c.area = min(c.area, cap * 0.95)
                c.area = min(c.area, _sizing.MAX_AREA.get(c.rt, c.area))

    # ---- search the footprint
    corridor_w = _CORRIDOR.get(bt, 7.0)
    frame_in = bay_frame
    stair_band = stair_end = stair_ws = None
    if frame_in:
        stair_band, stair_end, stair_ws = frame_in.get("stair_band"), frame_in.get("stair_end"), frame_in.get("stair_ws")
    yard_need = _FRONT_YARD_MIN
    for a in sites:
        dd = _sizing.base_dims(a)
        yard_need = max(yard_need, (max(dd) if a.room_type == "parking" else _mins_of(a)[1]) + 1.0)
    yard_need = max(yard_need, 8.0) if sites else 4.0
    strip_w = _strip_width(sites) if sites else 0.0
    cands: List[dict] = []
    for mode in (["none"] if not sites else ["front", "side"]):
        lat_avail = along_front - (strip_w if mode == "side" else 0.0)
        yard = yard_need if mode in ("front", "none") else 3.0
        if frame_in:
            lat_avail, yard = along_front, 0.0          # footprint already fixed by the floor below
        for variant in (0, 1, 2, 3, 4, 5) + ((6,) if frame_in else ()):
            if variant == 6:
                _assign_by_depth(cells, (frame_in["d"][0], frame_in["d"][1]), stair_ws or 0.0, stair_band)
            else:
                _assign_bands(cells, variant)
            if stair is not None:
                if stair_band is not None:
                    stair.band = stair_band
                stair.high_u = (stair_end == 1) if stair_end is not None else (stair.uf >= 0.5)
            for c in (x for x in cells if x.end_block):
                if stair is not None and c.high_u == stair.high_u:
                    c.high_u = not stair.high_u
            res = _search(cells, bt, corridor_w, lat_avail, depth_total - yard, frame_in, stair_ws)
            if res is not None:
                res["score"] += (0.4 if mode == "side" else 0.0)          # prefer the classic front-yard arrangement
                cands.append({**res, "mode": mode, "strip_w": strip_w, "yard": yard,
                              "cells_state": {c.label: (c.band, c.high_u, c.end_block) for c in cells}})
    if not cands:
        raise LayoutInfeasible("The corridor planner could not fit these rooms on this plot.")
    last_err: Optional[Exception] = None
    for best in sorted(cands, key=lambda r: r["score"]):          # first candidate that builds and validates wins
        for c in cells:
            c.band, c.high_u, c.end_block = best["cells_state"][c.label]
        try:
            return _build_layout(best, cells, stair, common_baths, sites, L, W, e, bt, vastu, floor, sides,
                                 (bx, by, bw, bh), corridor_w, road_direction, is_res, frame_in)
        except LayoutInfeasible as exc:
            last_err = exc
    raise last_err


# ------------------------------------------------------------------ layout build
def _build_layout(best: dict, cells: List[Cell], stair: Optional[Cell], common_baths: List, sites: List,
                  L: float, W: float, e: str, bt: str, vastu: bool, floor: int, sides: dict,
                  buildable: Tuple[float, float, float, float], corridor_w: float, road_direction: str,
                  is_res: bool, frame_in: Optional[dict]) -> Dict:
    bx, by, bw, bh = buildable
    U_main, d, V, end_w, U_total = best["U_main"], best["d"], best["V"], best["end_w"], best["U_total"]
    ends = [c for c in cells if c.end_block]
    lateral = bh if e in "NS" else bw
    depth = bw if e in "NS" else bh
    mode = best["mode"]
    sw_ = best["strip_w"] if mode == "side" else 0.0
    lat_lo_side = {"N": True, "S": False, "E": True, "W": False}[e]
    avail0 = (by if e in "NS" else bx) + (sw_ if lat_lo_side else 0.0)
    lat0 = avail0 + _snap((lateral - sw_ - U_total) / 2.0)
    free = depth - V - best["yard"]
    rear = _snap(max(0.0, free * 0.45))              # share the spare land between front and rear yards
    if e == "N":
        fx0, fy0, fx, fy = bx + rear, lat0, V, U_total
    elif e == "S":
        fx0, fy0, fx, fy = bx + depth - V - rear, lat0, V, U_total
    elif e == "E":
        fx0, fy0, fx, fy = lat0, by + rear, U_total, V
    else:
        fx0, fy0, fx, fy = lat0, by + depth - V - rear, U_total, V
    fr = Frame(e, round(fx0, 2), round(fy0, 2), round(fx, 2), round(fy, 2))
    if frame_in and frame_in.get("env"):
        fe = frame_in["env"]
        fr = Frame(e, fe["x"], fe["y"], fe["w"], fe["h"])          # same outer walls as the floor below
    u_lo = 0.0
    if frame_in and frame_in.get("stair_end") == 1:
        u_lo = round(frame_in["U_total"] - U_total, 2)             # upper floor hugs the staircase end
    low_end = sum(w for c, w in zip(ends, end_w) if not c.high_u)
    u_main0 = u_lo + low_end

    rooms: List[dict] = []
    corridor_label = "Family Hall" if is_res else "Corridor"
    cr = fr.rect(u_main0, d[0], U_main, corridor_w)
    corridor = {"room_type": "corridor_lobby", "label": corridor_label, "x": cr[0], "y": cr[1], "w": cr[2],
                "h": cr[3], "no_door": True, "min_dims": [5.0, 8.0], "_alloc": None}
    entry = next((c for c in cells if c.role == "entry"), None)
    rows, widths = best["rows"], best["widths"]
    placed: Dict[str, dict] = {}
    for b in (0, 1):
        v0 = 0.0 if b == 0 else d[0] + corridor_w
        u = u_main0
        for c, w in zip(rows[b], widths[b]):
            x, y, ww, hh = fr.rect(u, v0, w, d[b])
            rd = {"room_type": c.rt, "label": c.label, "x": x, "y": y, "w": ww, "h": hh, "_alloc": c.alloc,
                  "min_dims": [round(c.mins[0] * 0.97, 2), round(c.mins[1] * 0.97, 2)]}
            wall = fr.wall_to_corridor(b)
            seg = hh if wall in "NS" else ww
            rd["door_spec"] = {"wall": wall, "position": round(seg / 2.0, 2), "to": corridor_label}
            rd["door_to"] = [corridor_label]
            rd["door_side"] = wall
            placed[c.label] = rd
            rooms.append(rd)
            u += w
    lo_cursor, hi_cursor = u_lo, u_main0 + U_main
    for c, w in zip(ends, end_w):
        if not c.high_u:
            u0, lo_cursor = lo_cursor, lo_cursor + w
        else:
            u0, hi_cursor = hi_cursor, hi_cursor + w
        x, y, ww, hh = fr.rect(u0, 0.0, w, V)
        rd = {"room_type": c.rt, "label": c.label, "x": x, "y": y, "w": ww, "h": hh, "_alloc": c.alloc,
              "min_dims": [round(c.mins[0] * 0.97, 2), round(c.mins[1] * 0.97, 2)]}
        wall = fr.wall_to_corridor_end(c.high_u)
        seg = fr.rect(u0, d[0], w, corridor_w)
        seg_lo = seg[0] if wall in "EW" else seg[1]
        start = rd["x"] if wall in "EW" else rd["y"]
        rd["door_spec"] = {"wall": wall, "position": round(seg_lo + corridor_w / 2.0 - start, 2), "to": corridor_label}
        rd["door_to"] = [corridor_label]
        rd["door_side"] = wall
        placed[c.label] = rd
        rooms.append(rd)
    # the entry room opens straight into the corridor (one open hall) and carries the main door
    if entry is not None:
        er = placed[entry.label]
        er["door_spec"] = None
        er["open_group"] = "hall"
        er["hall_kind"] = "living" if is_res else "lobby"
        corridor["open_group"] = "hall"
        corridor["hall_kind"] = "passage"
    rooms.insert(0, corridor)

    # ---- contained rooms: attached baths (inside their bedroom), common bath (under the staircase)
    plot_box = {"x": fr.x0, "y": fr.y0, "w": fr.fx, "h": fr.fy}
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    contained: List[dict] = []
    for c in cells:
        if c.bath is not None and c.label in placed:
            parent = placed[c.label]
            spec = {"alloc": c.bath, "kind": "attached_bath", "mode": "inside", "dims": BATH_DIMS,
                    "area": BATH_DIMS[0] * BATH_DIMS[1], "mins": _mins_of(c.bath)}
            wall = parent["door_spec"]["wall"] if parent.get("door_spec") else None
            rect = _place_contained(parent, spec, [], [], plot_box, want_wall=(opposite[wall] if wall else None),
                                    not_wall=wall, toward=(1, -1), min_free=(6.0, 8.0), rtype=c.bath.room_type)
            if rect is None:
                raise LayoutInfeasible(f"No space to fit {c.bath.label} inside {c.label}.")
            ch = {"room_type": c.bath.room_type, "label": c.bath.label, **{k: round(v, 2) for k, v in rect.items()},
                  "inside_of": parent["label"], "contained_kind": "attached_bath", "_alloc": c.bath,
                  "min_dims": [round(min(BATH_DIMS) * 0.9, 2), round(max(BATH_DIMS) * 0.9, 2)],
                  "door_to": [parent["label"]]}
            _inside_door(ch, parent)
            contained.append(ch)
    if stair is not None and common_baths:
        sp = placed[stair.label]
        wall = sp["door_spec"]["wall"]
        for bath in common_baths:
            spec = {"alloc": bath, "kind": "under_stair_bath", "mode": "outside", "dims": BATH_DIMS,
                    "area": BATH_DIMS[0] * BATH_DIMS[1], "mins": _mins_of(bath)}
            rect = _place_contained(sp, spec, [x for x in contained if x["inside_of"] == sp["label"]], [], plot_box,
                                    want_wall=wall, not_wall=None, toward=(1, -1), min_free=(3.0, 8.0), rtype="bathroom")
            if rect is None:
                raise LayoutInfeasible("No space for the under-stair washroom.")
            ch = {"room_type": bath.room_type, "label": bath.label, **{k: round(v, 2) for k, v in rect.items()},
                  "inside_of": sp["label"], "contained_kind": "under_stair_bath", "_alloc": bath,
                  "min_dims": [round(min(BATH_DIMS) * 0.9, 2), round(max(BATH_DIMS) * 0.9, 2)]}
            se = shared_edge(ch, corridor)
            if se is None:
                raise LayoutInfeasible("The under-stair washroom cannot open onto the hall.")
            side, lo, hi = se
            start = ch["y"] if side in "NS" else ch["x"]
            ch["door_spec"] = {"wall": side, "position": round((lo + hi) / 2.0 - start, 2), "to": corridor_label}
            ch["door_side"] = side
            ch["door_to"] = [corridor_label]
            # the staircase keeps its own opening on the part of the shared wall the washroom does not use
            w0 = sp["y"] if wall in "NS" else sp["x"]
            w1 = w0 + (sp["h"] if wall in "NS" else sp["w"])
            b1 = start + (ch["h"] if side in "NS" else ch["w"])
            iv = max(((w0, start - 0.3), (b1 + 0.3, w1)), key=lambda t: t[1] - t[0])
            if iv[1] - iv[0] < 3.4:
                raise LayoutInfeasible("The staircase has no room for its own opening beside the washroom.")
            sp["door_spec"]["position"] = round((iv[0] + iv[1]) / 2.0 - w0, 2)
            contained.append(ch)
    rooms.extend(contained)
    rooms.extend(_place_sites(sites, e, buildable, fr, mode, best["strip_w"]))

    # ---- zones / Vastu fields (same convention as the other planners)
    env = {"x": fr.x0, "y": fr.y0, "w": fr.fx, "h": fr.fy}
    plot_env = {"x": 0.0, "y": 0.0, "w": L, "h": W}
    out_rooms: List[dict] = []
    match = acceptable = indoor_n = 0
    sc_sum = sc_w = 0.0
    for r in rooms:
        alloc = r.pop("_alloc", None)
        zone = zone9_of_point(r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0,
                              plot_env if r["room_type"] in SITE_TYPES else env)
        rule = rule_for(r["room_type"], getattr(alloc, "hint_zones", None) if alloc else None)
        ok = rule.grade(zone) in ("ideal", "acceptable") or zone in ACCEPTABLE_ZONES.get(r["room_type"], [])
        if r.get("contained_kind") == "attached_bath":
            ok = True
        pref = getattr(alloc, "preferred_zone", None) or (rule.ideal[0] if rule.ideal else "C")
        r.update({"zone": zone, "preferred_zone": pref, "matched": zone == pref, "vastu_acceptable": bool(ok)})
        r.setdefault("door_side", DOOR_SIDE_FOR_ZONE.get(zone, "S"))
        if getattr(alloc, "custom", False):
            r["custom"] = True
        if r["room_type"] not in SITE_TYPES and r["room_type"] != "corridor_lobby":
            indoor_n += 1
            match += int(zone == pref)
            acceptable += int(ok)
            sc_sum += rule.weight * zone_score(rule, zone)
            sc_w += rule.weight
        out_rooms.append(r)
    out_rooms.sort(key=lambda r: (1 if r.get("inside_of") else 0))

    checks = run_all_checks(out_rooms, L, W)
    if not checks["passed"]:
        raise LayoutInfeasible("Corridor layout failed validation: " + ", ".join(
            k for k in ("collisions", "boundary_violations", "dimension_violations") if checks[k]))
    stair_rect = None
    if stair is not None:
        sp = placed[stair.label]
        stair_rect = {k: sp[k] for k in ("x", "y", "w", "h")}
    planning_env = {"x": round(bx, 2), "y": round(by, 2), "w": round(bw, 2), "h": round(bh, 2)}
    indoor_rooms = [r for r in out_rooms if r["room_type"] not in SITE_TYPES and not r.get("inside_of")]
    xs0 = min(r["x"] for r in indoor_rooms); ys0 = min(r["y"] for r in indoor_rooms)
    xs1 = max(r["x"] + r["w"] for r in indoor_rooms); ys1 = max(r["y"] + r["h"] for r in indoor_rooms)
    occupied = {"x": round(xs0, 2), "y": round(ys0, 2), "w": round(xs1 - xs0, 2), "h": round(ys1 - ys0, 2)}
    return {
        "plot": {"length": round(L, 2), "width": round(W, 2), "area": round(L * W, 2)},
        "floors": 1, "road_direction": road_direction,
        "setback_ft": round(max(sides.values()), 2), "setbacks_ft": {k: round(v, 2) for k, v in sides.items()},
        "building_envelope": dict(occupied), "occupied_envelope": dict(occupied), "planning_envelope": planning_env,
        "rooms": out_rooms, "collisions": checks["collisions"], "boundary_violations": checks["boundary_violations"],
        "dimension_violations": checks["dimension_violations"], "passed_checks": True,
        "vastu_match_count": match, "vastu_acceptable_count": acceptable,
        "vastu_score": round(sc_sum / max(sc_w, 1e-6), 4),
        "vastu_match_percent": round(100.0 * match / max(indoor_n, 1), 1),
        "vastu_enabled": bool(vastu),
        "vastu_layout_engine": "Corridor-and-bay planner (aligned bays, equal-width repeated rooms, per-room doors)",
        "building_type": bt, "compression_ratio": 1.0, "layout_notes": [],
        "entrance_side": e, "gap_report": _gap_report(indoor_rooms), "circulation": [],
        "alignment_report": {"layout_grid": "CORRIDOR_BAYS", "aligned_percent": 100.0},
        "alignment_fallback": None, "no_unplanned_gaps": True,
        "site_spaces": [r for r in out_rooms if r["room_type"] in SITE_TYPES],
        "floor": floor, "engine": "bay",
        "bay_frame": {"U_main": U_main, "d": list(d), "V": V, "U_total": (frame_in["U_total"] if frame_in else U_total),
                      "corridor_w": corridor_w,
                      "stair_band": (stair.band if stair is not None else None),
                      "stair_end": ((1 if stair.high_u else 0) if stair is not None else None),
                      "stair_ws": best.get("ws"), "stair": stair_rect, "env": dict(env)},
    }
