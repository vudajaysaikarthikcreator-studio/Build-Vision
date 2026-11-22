"""3D model data for the web viewer: walls, doors and windows, all with names.

The browser used to receive only room rectangles, so the 3D view could draw
translucent boxes but no real walls, doors, windows or their names.  This
module turns one floor layout into:

* ``walls``   - merged wall runs (one per physical wall piece, shared walls only once)
                 with exterior/partition class, thickness and a readable name,
* ``doors``   - one record per door (same resolver as the 2D blueprint, so both agree),
* ``windows`` - one record per window (same resolver as the 2D blueprint).

Coordinates are the engineering axes used everywhere else: x = south->north,
y = west->east, feet.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .door_engine import build_doors

EXTERIOR_T = 0.75
PARTITION_T = 0.4
SITE_TYPES = {"parking", "garden", "loading_bay"}
_SIDE_NAME = {"N": "North", "S": "South", "E": "East", "W": "West"}


def _r(v: float) -> float:
    return round(float(v), 2)


def _edges(room: dict):
    x, y, w, h = float(room["x"]), float(room["y"]), float(room["w"]), float(room["h"])
    # (axis, fixed coordinate, start, end, side-of-room)
    return [
        ("y", x, y, y + h, "S"),          # south wall: runs along y at x
        ("y", x + w, y, y + h, "N"),
        ("x", y, x, x + w, "W"),          # west wall: runs along x at y
        ("x", y + h, x, x + w, "E"),
    ]


def build_walls(rooms: List[dict], envelope: Optional[dict]) -> List[dict]:
    indoor = [r for r in rooms if str(r.get("room_type", "")).lower() not in SITE_TYPES]
    if not indoor:
        return []
    if envelope:
        ex0, ey0 = float(envelope["x"]), float(envelope["y"])
        ex1, ey1 = ex0 + float(envelope["w"]), ey0 + float(envelope["h"])
    else:
        ex0 = min(float(r["x"]) for r in indoor)
        ey0 = min(float(r["y"]) for r in indoor)
        ex1 = max(float(r["x"]) + float(r["w"]) for r in indoor)
        ey1 = max(float(r["y"]) + float(r["h"]) for r in indoor)

    lines: Dict[tuple, List[tuple]] = {}
    for r in indoor:
        for axis, c, a, b, side in _edges(r):
            lines.setdefault((axis, round(c, 1)), []).append((a, b, r, side, c))

    walls: List[dict] = []
    for (axis, ckey), segs in lines.items():
        pts = sorted({round(p, 3) for a, b, *_ in segs for p in (a, b)})
        for p0, p1 in zip(pts, pts[1:]):
            if p1 - p0 < 0.2:
                continue
            mid = (p0 + p1) / 2.0
            owners = [(r, side, c) for a, b, r, side, c in segs if a - 1e-6 <= mid <= b + 1e-6]
            if not owners:
                continue
            labels = []
            for r, _, _ in owners:
                if r.get("label") not in labels:
                    labels.append(r.get("label"))
            groups = {r.get("open_group") for r, _, _ in owners}
            if len(owners) >= 2 and len(groups) == 1 and None not in groups:
                continue                       # open hall group: no partition between living/dining/passage
            c = sum(c for _, _, c in owners) / len(owners)
            if axis == "y":
                exterior = abs(c - ex0) < 0.8 or abs(c - ex1) < 0.8
                side = "S" if abs(c - ex0) < 0.8 else ("N" if abs(c - ex1) < 0.8 else owners[0][1])
            else:
                exterior = abs(c - ey0) < 0.8 or abs(c - ey1) < 0.8
                side = "W" if abs(c - ey0) < 0.8 else ("E" if abs(c - ey1) < 0.8 else owners[0][1])
            if exterior and len(labels) == 1:
                name = f"Exterior wall ({_SIDE_NAME[side]}) - {labels[0]}"
            elif exterior:
                name = f"Exterior wall ({_SIDE_NAME[side]})"
            else:
                name = "Partition wall - " + " | ".join(labels[:2])
            walls.append({
                "axis": axis, "c": _r(c), "a": _r(p0), "b": _r(p1), "exterior": bool(exterior),
                "thickness": EXTERIOR_T if exterior else PARTITION_T, "name": name, "rooms": labels,
            })
    return walls


def _opening(kind: str, rec: dict, label: str) -> dict:
    x1, y1, x2, y2 = rec["x1"], rec["y1"], rec["x2"], rec["y2"]
    axis = "x" if abs(y1 - y2) < 1e-6 else "y"       # direction the wall runs along
    pos = float(rec["position"])
    if axis == "x":
        cx, cy = x1 + pos, y1
    else:
        cx, cy = x1, y1 + pos
    return {"kind": kind, "room": rec.get("room"), "wall": rec.get("wall_code"), "axis": axis,
            "cx": _r(cx), "cy": _r(cy), "width": _r(rec["width"]), "label": label,
            "main": bool(rec.get("is_main_entrance"))}



_WIN_W = {"living_room": 5.0, "master_bedroom": 4.5, "bedroom": 4.0, "kitchen": 3.5, "dining_room": 4.0,
          "bathroom": 2.0, "toilet": 2.0, "pooja_room": 2.5, "study_room": 3.5, "store_room": 2.0, "default": 3.5}


def _windows(rooms: List[dict], doors: List[dict], env: Optional[dict]) -> List[dict]:
    """Windows on every wall that touches the building envelope (the plot edge is NOT the wall: setbacks)."""
    indoor = [r for r in rooms if str(r.get("room_type", "")).lower() not in SITE_TYPES]
    if not indoor:
        return []
    if env:
        ex0, ey0 = float(env["x"]), float(env["y"])
        ex1, ey1 = ex0 + float(env["w"]), ey0 + float(env["h"])
    else:
        ex0 = min(float(r["x"]) for r in indoor); ey0 = min(float(r["y"]) for r in indoor)
        ex1 = max(float(r["x"]) + float(r["w"]) for r in indoor); ey1 = max(float(r["y"]) + float(r["h"]) for r in indoor)
    door_at = {}
    for d in doors:
        door_at.setdefault(d["room"], []).append(d)
    out = []
    for r in indoor:
        if r.get("no_window") or r.get("room_type") in ("corridor_lobby", "staircase") and not r.get("open_group"):
            pass
        x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
        sides = []
        if abs(x - ex0) < 0.8: sides.append(("S", x, y, y + h, "y"))
        if abs(x + w - ex1) < 0.8: sides.append(("N", x + w, y, y + h, "y"))
        if abs(y - ey0) < 0.8: sides.append(("W", y, x, x + w, "x"))
        if abs(y + h - ey1) < 0.8: sides.append(("E", y + h, x, x + w, "x"))
        rt = str(r.get("room_type"))
        base = _WIN_W.get(rt, _WIN_W["default"])
        for side, c, a, b, axis in sides[:2 if rt not in ("bathroom", "toilet") else 1]:
            seg = b - a
            if seg < 3.0:
                continue
            # free stretches of this wall once door openings (+0.7 ft margin) are removed
            blocked = sorted((float(d["position"]) - d["width"] / 2 - 0.7, float(d["position"]) + d["width"] / 2 + 0.7)
                             for d in door_at.get(r.get("label"), []) if d["wall_code"] == side)
            free, cur = [], 0.5
            for lo_, hi_ in blocked:
                if lo_ > cur:
                    free.append((cur, min(lo_, seg - 0.5)))
                cur = max(cur, hi_)
            if cur < seg - 0.5:
                free.append((cur, seg - 0.5))
            free = [f for f in free if f[1] - f[0] >= 1.6]
            if not free:
                continue
            lo_, hi_ = max(free, key=lambda f: f[1] - f[0])
            width = min(base, hi_ - lo_)
            pos = (lo_ + hi_) / 2.0
            if axis == "y":
                rec = {"x1": c, "y1": a, "x2": c, "y2": b}
            else:
                rec = {"x1": a, "y1": c, "x2": b, "y2": c}
            out.append({"room": r.get("label"), "wall_code": side, "position": pos, "width": width, **rec})
    return out


def build_floor_model(layout: dict, entrance_direction: str = "North", is_ground_floor: bool = True) -> dict:
    rooms = layout.get("rooms", [])
    plot = layout.get("plot", {})
    L = float(plot.get("length", 0) or 0) or None
    W = float(plot.get("width", 0) or 0) or None
    env = layout.get("building_envelope") or layout.get("occupied_envelope")
    door_recs = build_doors(rooms, entrance_direction=entrance_direction, plot_length=L, plot_width=W,
                            building_envelope=env, main_door_position=layout.get("main_door_position", "center"),
                            allow_main_entrance=is_ground_floor)
    win_recs = _windows(rooms, door_recs, env)
    doors = []
    for d in door_recs:
        name = "Main door" if d.get("is_main_entrance") else f"Door - {d['room']}"
        doors.append(_opening("door", d, name))
    windows = [_opening("window", w, f"Window - {w['room']}") for w in win_recs]
    return {"walls": build_walls(rooms, env), "doors": doors, "windows": windows, "envelope": env}
