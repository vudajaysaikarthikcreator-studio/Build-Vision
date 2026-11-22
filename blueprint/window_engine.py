"""
blueprint/window_engine.py

Professional window engine (brief section 4).

Places windows only on a room's exterior-facing walls (never on an
interior partition), skips the wall the door already occupies where a
non-door exterior wall is available, and applies a window count rule per
room type:

    living_room / dining_room : 2 windows where the room has 2 exterior
                                 walls to spread them across, else 1
    bedroom / master_bedroom  : 1, or 2 on a corner room with 2 exterior
                                 walls
    kitchen                   : 1
    bathroom                  : 1 small ventilation window
    everything else           : 1 if it has an exterior wall, else none
"""

from typing import Dict, List

from blueprint.wall_engine import _room_sides, room_exterior_sides, EXTERIOR_WALL_THICKNESS

WINDOW_WIDTH_FT = {
    "default": 3.5,
    "bathroom": 1.6,
    "kitchen": 3.0,
}
EDGE_CLEARANCE = 0.5

_WALL_NAME = {"N": "north", "S": "south", "E": "east", "W": "west"}

_WINDOW_COUNT_RULE = {
    "living_room": 2, "dining_room": 2,
    "master_bedroom": 2, "bedroom": 2,
    "kitchen": 1, "bathroom": 1,
}


def _rule_count(room_type: str) -> int:
    return _WINDOW_COUNT_RULE.get(room_type, 1)


def build_windows(rooms: List[dict], doors: List[dict], plot_length: float, plot_width: float) -> List[dict]:
    door_wall_by_room = {d["room"]: d["wall_code"] for d in doors}
    windows = []

    for room in rooms:
        ext_sides = room_exterior_sides(room, plot_length, plot_width)
        if not ext_sides:
            continue  # interior room — no exterior wall available for a window

        room_type = room.get("room_type", "")
        wanted = min(_rule_count(room_type), len(ext_sides) if len(ext_sides) > 1 else 1)
        # prefer walls that are NOT the door wall, so windows and doors
        # never land on the same opening
        door_wall = door_wall_by_room.get(room["label"])
        candidate_sides = [s for s in ext_sides if s != door_wall] or ext_sides

        chosen_sides = candidate_sides[:wanted]
        if len(chosen_sides) < wanted and door_wall in ext_sides and door_wall not in chosen_sides:
            # only exterior option left is the door wall itself (e.g. a
            # small room with a single outer wall) — allow it, the window
            # will be offset away from the door below
            chosen_sides.append(door_wall)

        win_w = WINDOW_WIDTH_FT.get(room_type, WINDOW_WIDTH_FT["default"])
        win_w = min(win_w, max(min(room["w"], room["h"]) * 0.6, 1.2))

        sides_geo = _room_sides(room)
        for i, side in enumerate(chosen_sides):
            x1, y1, x2, y2 = sides_geo[side]
            seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if seg_len < win_w + 2 * EDGE_CLEARANCE:
                win_w_local = max(seg_len - 2 * EDGE_CLEARANCE, 0.8)
            else:
                win_w_local = win_w

            if side == door_wall and len(chosen_sides) == 1:
                # same wall as the door: push the window to whichever half
                # of the wall the door is NOT on
                door = next(d for d in doors if d["room"] == room["label"])
                position = seg_len * 0.82 if door["position"] < seg_len / 2 else seg_len * 0.18
                position = min(max(position, EDGE_CLEARANCE + win_w_local / 2),
                                seg_len - EDGE_CLEARANCE - win_w_local / 2)
            else:
                # spread multiple windows evenly along the wall
                frac = (i + 1) / (len(chosen_sides) + 1) if len(chosen_sides) > 1 else 0.5
                position = seg_len * frac

            windows.append({
                "room": room["label"],
                "room_type": room_type,
                "wall": _WALL_NAME.get(side, "north"),
                "wall_code": side,
                "position": round(position, 2),
                "width": round(win_w_local, 2),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return windows


def draw_window(ax, window: dict, color: str, zorder: int = 5):
    """
    Draws the standard architectural window symbol in the wall opening:
    two thin parallel lines spanning the opening (representing the glass
    pane between the wall faces) with short end ticks closing the gap the
    wall-cutting step left behind.
    """
    length = ((window["x2"] - window["x1"]) ** 2 + (window["y2"] - window["y1"]) ** 2) ** 0.5 or 1e-6
    ux, uy = (window["x2"] - window["x1"]) / length, (window["y2"] - window["y1"]) / length
    nx, ny = -uy, ux
    t = EXTERIOR_WALL_THICKNESS / 2
    half = window["width"] / 2
    cx = window["x1"] + ux * window["position"]
    cy = window["y1"] + uy * window["position"]
    p1 = (cx - ux * half, cy - uy * half)
    p2 = (cx + ux * half, cy + uy * half)

    for sign in (-1, 1):
        ax.plot([p1[0] + nx * t * sign, p2[0] + nx * t * sign],
                 [p1[1] + ny * t * sign, p2[1] + ny * t * sign],
                 color=color, linewidth=1.1, zorder=zorder)
    # end ticks closing the wall faces at each jamb
    for p in (p1, p2):
        ax.plot([p[0] - nx * t, p[0] + nx * t], [p[1] - ny * t, p[1] + ny * t],
                 color=color, linewidth=1.1, zorder=zorder)
