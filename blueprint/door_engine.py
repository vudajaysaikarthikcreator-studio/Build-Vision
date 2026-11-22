"""Authoritative architectural door placement for BuildVision.

Door geometry uses the same coordinate system as the layout engine:
  x = south -> north (depth), y = west -> east (frontage)

The previous implementation mixed the wall coordinate labels with the room
rectangle axes and the local renderer then redrew doors at a fixed midpoint.
That made East/West doors appear on the wrong wall and made the rendered main
entrance disagree with the logical door data. This module is now the single
source of truth for door wall, position, width and swing.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from blueprint.wall_engine import _room_sides

DOOR_WIDTH_FT = 2.75
MIN_DOOR_WIDTH_FT = 2.25
EDGE_CLEARANCE = 0.55
DOOR_GAP_FT = 0.20

_WALL_NAME = {"N": "north", "S": "south", "E": "east", "W": "west"}
_SIDE_FOR_DIRECTION = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}
_ENTRANCE_ROOM_TYPES = {
    "living_room", "lobby", "reception", "waiting_area", "showroom", "restaurant_dining",
    "teller_area", "open_work_area", "multipurpose_hall", "auditorium",
}
_PRIORITY = {
    "living_room": 0, "lobby": 1, "reception": 1, "waiting_area": 2,
    "showroom": 2, "restaurant_dining": 3, "teller_area": 3,
    "open_work_area": 3, "multipurpose_hall": 3, "auditorium": 3,
}


def _direction_code(direction: str, fallback: str = "N") -> str:
    return _SIDE_FOR_DIRECTION.get(str(direction or "").strip().upper(), fallback)


def _side_length(room: dict, side: str) -> float:
    return float(room["h"] if side in ("S", "N") else room["w"])


def _side_interval(room: dict, side: str) -> Tuple[float, float]:
    if side in ("S", "N"):
        return float(room["y"]), float(room["y"] + room["h"])
    return float(room["x"]), float(room["x"] + room["w"])


def _shared_length(room: dict, other: dict, side: str, tol: float = 0.18) -> float:
    """Length of a wall side shared with another room."""
    rx0, rx1 = room["x"], room["x"] + room["w"]
    ry0, ry1 = room["y"], room["y"] + room["h"]
    ox0, ox1 = other["x"], other["x"] + other["w"]
    oy0, oy1 = other["y"], other["y"] + other["h"]
    if side == "S" and abs(rx0 - ox1) <= tol:
        return max(0.0, min(ry1, oy1) - max(ry0, oy0))
    if side == "N" and abs(rx1 - ox0) <= tol:
        return max(0.0, min(ry1, oy1) - max(ry0, oy0))
    if side == "W" and abs(ry0 - oy1) <= tol:
        return max(0.0, min(rx1, ox1) - max(rx0, ox0))
    if side == "E" and abs(ry1 - oy0) <= tol:
        return max(0.0, min(rx1, ox1) - max(rx0, ox0))
    return 0.0


def _exterior_sides(room: dict, plot_length: Optional[float], plot_width: Optional[float],
                    building_envelope: Optional[dict] = None, tol: float = 0.35) -> List[str]:
    """Return room sides on the actual building envelope, including setbacks."""
    if building_envelope:
        x0 = float(building_envelope.get("x", 0.0))
        y0 = float(building_envelope.get("y", 0.0))
        x1 = x0 + float(building_envelope.get("w", 0.0))
        y1 = y0 + float(building_envelope.get("h", 0.0))
    elif plot_length is not None and plot_width is not None:
        x0, y0, x1, y1 = 0.0, 0.0, float(plot_length), float(plot_width)
    else:
        return []

    sides = []
    if abs(float(room["x"]) - x0) <= tol:
        sides.append("S")
    if abs(float(room["x"]) + float(room["w"]) - x1) <= tol:
        sides.append("N")
    if abs(float(room["y"]) - y0) <= tol:
        sides.append("W")
    if abs(float(room["y"]) + float(room["h"]) - y1) <= tol:
        sides.append("E")
    return sides


def _normal_side(side: str) -> str:
    side = str(side or "S").upper()
    return side if side in {"N", "S", "E", "W"} else "S"


def _normalize_position(position: str) -> str:
    p = str(position or "center").strip().lower()
    return p if p in {"left", "center", "right"} else "center"


def _position_fraction(side: str, position: str) -> float:
    """Street-facing left/center/right as seen by a person standing outside."""
    p = _normalize_position(position)
    if p == "center":
        return 0.50
    # Along S->N / W->E engineering coordinates.
    if side == "N":
        return {"left": 0.75, "right": 0.25}[p]
    if side == "S":
        return {"left": 0.25, "right": 0.75}[p]
    if side == "E":
        return {"left": 0.25, "right": 0.75}[p]
    return {"left": 0.75, "right": 0.25}[p]  # W


def _door_width(seg_len: float, room_min_dim: float) -> float:
    if seg_len <= 0:
        return MIN_DOOR_WIDTH_FT
    width = min(DOOR_WIDTH_FT, max(MIN_DOOR_WIDTH_FT, room_min_dim * 0.55))
    max_width = max(MIN_DOOR_WIDTH_FT, seg_len - 2 * EDGE_CLEARANCE)
    return round(min(width, max_width), 2)


def _clamp_position(position: float, seg_len: float, width: float) -> float:
    lo = EDGE_CLEARANCE + width / 2.0
    hi = max(lo, seg_len - EDGE_CLEARANCE - width / 2.0)
    return round(min(max(position, lo), hi), 2)


def _candidate_score(room: dict, side: str, preferred: str, entrance_side: str,
                     shared_len: float, exterior: bool) -> float:
    score = 0.0
    if side == preferred:
        score += 30.0
    if not exterior and shared_len > 0.5:
        score += 15.0 + min(shared_len, 20.0) * 0.5
    elif exterior:
        score += 2.0
    if side == entrance_side:
        score += 8.0
    # Keep toilets/wet/service rooms away from arbitrary exterior openings when
    # an equally good interior/shared wall exists. Bedrooms benefit from privacy.
    rt = str(room.get("room_type") or "").lower()
    if rt in {"bathroom", "toilet", "restroom", "wash_area", "laundry", "service_room"} and exterior:
        score -= 4.0
    return score


def _occupied_interval_conflict(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > DOOR_GAP_FT


def _safe_non_main_position(seg_len: float, width: float, preferred: float,
                            occupied_intervals: Sequence[Tuple[float, float]],
                            coordinate_offset: float = 0.0) -> float:
    """Move a non-main door away from an already-used physical wall opening."""
    candidates = [preferred, seg_len * 0.25, seg_len * 0.75, seg_len * 0.50]
    lo = EDGE_CLEARANCE + width / 2.0
    hi = max(lo, seg_len - EDGE_CLEARANCE - width / 2.0)
    for raw in candidates:
        pos = min(max(raw, lo), hi)
        a0, a1 = coordinate_offset + pos - width / 2.0, coordinate_offset + pos + width / 2.0
        if all(not _occupied_interval_conflict(a0, a1, b0, b1) for b0, b1 in occupied_intervals):
            return round(pos, 2)
    # On very short walls there may be no perfect separation. Keep the door
    # collision-free relative to the room corners, even if it shares a wall.
    return round(min(max(preferred, lo), hi), 2)


def _wall_line_key(x1: float, y1: float, x2: float, y2: float) -> Tuple:
    """Canonical identity for a physical wall line, independent of direction."""
    if abs(y2 - y1) >= abs(x2 - x1):
        return ("V", round(y1, 2))
    return ("H", round(x1, 2))


def _wall_coordinate_offset(x1: float, y1: float, x2: float, y2: float) -> float:
    """Coordinate of the segment start in the physical wall-line axis."""
    return float(y1 if abs(y2 - y1) >= abs(x2 - x1) else x1)


def _wall_has_overlap(x1: float, y1: float, x2: float, y2: float, position: float, width: float,
                      occupied: Dict[Tuple, List[Tuple[float, float]]]) -> bool:
    key = _wall_line_key(x1, y1, x2, y2)
    offset = _wall_coordinate_offset(x1, y1, x2, y2)
    a0, a1 = offset + position - width / 2.0, offset + position + width / 2.0
    return any(_occupied_interval_conflict(a0, a1, b0, b1) for b0, b1 in occupied.get(key, []))


def build_doors(rooms: List[dict], entrance_direction: str = "North",
                plot_length: float = None, plot_width: float = None,
                building_envelope: Optional[dict] = None,
                main_door_position: str = "center",
                allow_main_entrance: bool = True) -> List[dict]:
    """Resolve one authoritative door record per room.

    The main door is placed only on a real exterior building wall.  Internal
    doors prefer shared room walls and are shifted when another door already
    occupies the same physical opening.
    """
    rooms = list(rooms or [])
    entrance_side = _direction_code(entrance_direction, "N")
    main_pos_setting = _normalize_position(main_door_position)
    # When an explicit building envelope is unavailable (for example direct
    # callers/tests using only plot dimensions), infer the occupied building
    # envelope from indoor rooms instead of treating the entire plot as built.
    # This preserves the setback-aware behaviour used by the main pipeline.
    effective_envelope = building_envelope
    if effective_envelope is None:
        indoor = [r for r in rooms if str(r.get("room_type", "")).lower() not in {"parking", "garden", "loading_bay"}]
        if indoor:
            x0 = min(float(r["x"]) for r in indoor)
            y0 = min(float(r["y"]) for r in indoor)
            x1 = max(float(r["x"]) + float(r["w"]) for r in indoor)
            y1 = max(float(r["y"]) + float(r["h"]) for r in indoor)
            effective_envelope = {"x": x0, "y": y0, "w": max(1.0, x1 - x0), "h": max(1.0, y1 - y0)}
    exterior_by_label: Dict[str, List[str]] = {
        r.get("label", str(i)): _exterior_sides(r, plot_length, plot_width, effective_envelope)
        for i, r in enumerate(rooms)
    }

    public = [r for r in rooms if r.get("room_type") in _ENTRANCE_ROOM_TYPES] if allow_main_entrance else []
    # First require an exterior wall on the requested facing side. This prevents
    # the old bug where a living room in the middle of the footprint received an
    # East/North/South/West main-door flag without actually touching that edge.
    exact = [r for r in public if entrance_side in exterior_by_label.get(r.get("label"), [])]
    if exact:
        main_room = min(exact, key=lambda r: (_PRIORITY.get(r.get("room_type"), 9), r.get("label", "")))
        main_side = entrance_side
        main_fallback = False
    else:
        exterior_public = [r for r in public if exterior_by_label.get(r.get("label"))]
        if exterior_public:
            def fallback_key(r):
                sides = exterior_by_label.get(r.get("label"), [])
                return (
                    _PRIORITY.get(r.get("room_type"), 9),
                    0 if entrance_side in sides else 1,
                    min(abs("NSEW".index(s) - "NSEW".index(entrance_side)) for s in sides),
                    r.get("label", ""),
                )
            main_room = min(exterior_public, key=fallback_key)
            # Prefer a wall adjacent to the requested side if present; otherwise
            # use the room's closest actual exterior wall and report the fallback.
            sides = exterior_by_label.get(main_room.get("label"), [])
            main_side = entrance_side if entrance_side in sides else sides[0]
            main_fallback = main_side != entrance_side
        else:
            main_room = None
            main_side = entrance_side
            main_fallback = True

    main_label = main_room.get("label") if main_room else None
    doors: List[dict] = []
    # Physical wall-line occupancy is shared across adjacent rooms. The main
    # door is processed first so internal doors cannot accidentally overlap it.
    occupied: Dict[Tuple, List[Tuple[float, float]]] = {}
    ordered_rooms = sorted(list(enumerate(rooms)), key=lambda item: (0 if item[1].get("label") == main_label else 1, item[0]))

    for idx, room in ordered_rooms:
        label = room.get("label", f"ROOM {idx + 1}")
        exteriors = exterior_by_label.get(label, [])
        sides = _room_sides(room)
        # Open hall areas (dining, passage) and site spaces have no door of their own.
        if room.get("no_door") and label != main_label:
            continue
        spec = room.get("door_spec") if label != main_label else None
        if isinstance(spec, dict) and spec.get("wall") in sides:
            # Planner-assigned door: this exact wall, this exact position (one door per room).
            side = spec["wall"]
            x1, y1, x2, y2 = sides[side]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            width = _door_width(seg_len, min(float(room["w"]), float(room["h"])))
            if str(room.get("room_type")) in ("bathroom", "toilet"):
                width = min(width, 2.5)
            position = _clamp_position(float(spec.get("position", seg_len / 2.0)), seg_len, width)
            key = _wall_line_key(x1, y1, x2, y2)
            offset = _wall_coordinate_offset(x1, y1, x2, y2)
            occupied.setdefault(key, []).append((offset + position - width / 2.0, offset + position + width / 2.0))
            cx = x1 + (x2 - x1) * (position / max(seg_len, 1e-9))
            cy = y1 + (y2 - y1) * (position / max(seg_len, 1e-9))
            doors.append({
                "room": label, "room_type": room.get("room_type"), "wall": _WALL_NAME[side], "wall_code": side,
                "position": round(position, 2), "width": round(width, 2), "swing_direction": "inside",
                "is_main_entrance": False, "main_entrance_requested_side": None, "main_entrance_actual_side": None,
                "main_entrance_fallback": False, "is_exterior": side in exteriors,
                "opens_to": spec.get("to") or (room.get("door_to") or [None])[0],
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "center_x": round(cx, 2), "center_y": round(cy, 2),
            })
            continue
        preferred = _normal_side(room.get("door_side", "S"))
        candidates = []
        for side in ("S", "N", "W", "E"):
            if side not in sides:
                continue
            shared = sum(_shared_length(room, other, side) for other in rooms if other is not room)
            exterior = side in exteriors
            score = _candidate_score(room, side, preferred, entrance_side, shared, exterior)
            candidates.append((score, side, shared, exterior))

        if not candidates:
            continue

        is_main = label == main_label
        if is_main:
            # The selected main wall must be exterior. The opening cannot be on an
            # interior partition merely because the preferred Vastu side says so.
            chosen = next((c for c in candidates if c[1] == main_side and c[3]), None)
            if chosen is None:
                chosen = max((c for c in candidates if c[3]), key=lambda c: c[0], default=max(candidates))
            score, side, shared_len, exterior = chosen
        else:
            # Prefer the room's Vastu-selected wall, then a useful shared wall,
            # then any exterior side. Penalize an already-used physical wall
            # opening so two adjacent rooms do not receive the same doorway.
            def candidate_key(c):
                c_score, c_side, c_shared, c_exterior = c
                cx1, cy1, cx2, cy2 = sides[c_side]
                c_len = math.hypot(cx2 - cx1, cy2 - cy1)
                c_width = _door_width(c_len, min(float(room["w"]), float(room["h"])))
                overlap_penalty = 100.0 if _wall_has_overlap(cx1, cy1, cx2, cy2, c_len * 0.50, c_width, occupied) else 0.0
                return (c_score - overlap_penalty, c_shared, c_side)
            score, side, shared_len, exterior = max(candidates, key=candidate_key)

        x1, y1, x2, y2 = sides[side]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        width = _door_width(seg_len, min(float(room["w"]), float(room["h"])))

        if is_main:
            fraction = _position_fraction(side, main_pos_setting)
            position = _clamp_position(seg_len * fraction, seg_len, width)
        else:
            preferred_pos = seg_len * 0.50
            key = _wall_line_key(x1, y1, x2, y2)
            intervals = occupied.setdefault(key, [])
            offset = _wall_coordinate_offset(x1, y1, x2, y2)
            position = _safe_non_main_position(seg_len, width, preferred_pos, intervals, coordinate_offset=offset)

        key = _wall_line_key(x1, y1, x2, y2)
        offset = _wall_coordinate_offset(x1, y1, x2, y2)
        occupied.setdefault(key, []).append((offset + position - width / 2.0, offset + position + width / 2.0))

        cx = x1 + (x2 - x1) * (position / max(seg_len, 1e-9))
        cy = y1 + (y2 - y1) * (position / max(seg_len, 1e-9))
        doors.append({
            "room": label,
            "room_type": room.get("room_type"),
            "wall": _WALL_NAME[side],
            "wall_code": side,
            "position": round(position, 2),
            "width": round(width, 2),
            "swing_direction": "inside",
            "is_main_entrance": bool(is_main and exterior),
            "main_entrance_requested_side": entrance_side if is_main else None,
            "main_entrance_actual_side": side if is_main else None,
            "main_entrance_fallback": bool(main_fallback) if is_main else False,
            "is_exterior": bool(exterior),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "center_x": round(cx, 2), "center_y": round(cy, 2),
        })

    return doors


def validate_doors(rooms: List[dict], doors: List[dict], expect_main_entrance: bool = True) -> dict:
    """Geometry checks used by tests, report export and the renderer."""
    by_label = {r.get("label"): r for r in rooms}
    issues = []
    physical_openings: Dict[Tuple, List[Tuple[float, float, str]]] = {}
    for d in doors:
        room = by_label.get(d.get("room"))
        if room is None:
            issues.append(f"Unknown room: {d.get('room')}")
            continue
        side = d.get("wall_code")
        if side not in {"N", "S", "E", "W"}:
            issues.append(f"{d['room']}: invalid wall code {side}")
            continue
        expected = _room_sides(room).get(side)
        if expected is None:
            continue
        x1, y1, x2, y2 = expected
        if any(abs(float(d.get(k, 0)) - v) > 0.01 for k, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2))):
            issues.append(f"{d['room']}: door endpoints do not match its wall geometry")
        seg_len = _side_length(room, side)
        position = float(d.get("position", 0))
        width = float(d.get("width", 0))
        if position - width / 2 < EDGE_CLEARANCE - 1e-6:
            issues.append(f"{d['room']}: door too close to wall start")
        if position + width / 2 > seg_len - EDGE_CLEARANCE + 1e-6:
            issues.append(f"{d['room']}: door too close to wall end")
        if width <= 0:
            issues.append(f"{d['room']}: door width must be positive")
        key = _wall_line_key(x1, y1, x2, y2)
        offset = _wall_coordinate_offset(x1, y1, x2, y2)
        a0, a1 = offset + position - width / 2.0, offset + position + width / 2.0
        for b0, b1, other_room in physical_openings.get(key, []):
            if _occupied_interval_conflict(a0, a1, b0, b1):
                issues.append(f"{d['room']}: doorway overlaps {other_room} on the same physical wall")
        physical_openings.setdefault(key, []).append((a0, a1, d.get("room", "ROOM")))
        if d.get("is_main_entrance") and not d.get("is_exterior"):
            issues.append(f"{d['room']}: main entrance is not on an exterior wall")
    mains = [d for d in doors if d.get("is_main_entrance")]
    if expect_main_entrance and len(mains) != 1 and rooms:
        issues.append(f"Expected one main entrance, found {len(mains)}")
    if not expect_main_entrance and mains:
        issues.append(f"Upper floor must not contain a main entrance; found {len(mains)}")
    return {"passed": not issues, "issues": issues}
