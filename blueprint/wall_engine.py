"""
blueprint/wall_engine.py

Professional wall engine (brief section 2).

Turns the room rectangles already produced by layout/layout_generator.py
into wall geometry: one WallSegment per side of every room, classified as
EXTERIOR (thick, on the plot boundary) or INTERIOR (thin, between rooms).

Design note: the geometry engine aims for shared-wall room adjacency.
Rooms are not given decorative spacing; when two rooms are intended to be
adjacent their centerlines coincide at the partition, so the wall engine
draws one visually continuous interior partition. Deliberate open areas such
as corridors, courtyards or setbacks must be represented as explicit plan
space rather than as accidental gaps.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from matplotlib import patches


# ---- architectural constants (feet) --------------------------------------
EXTERIOR_WALL_THICKNESS = 0.5
INTERIOR_WALL_THICKNESS = 0.25
BOUNDARY_TOLERANCE = 1.0  # how close to the plot edge counts as "exterior"
# (covers layout_generator's WALL_MARGIN_FT + per-room padding, so the
# rooms actually touching the building envelope are correctly classified)


@dataclass
class WallSegment:
    side: str          # "N" | "S" | "E" | "W" — room-relative side
    room_label: str
    room_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    exterior: bool
    thickness: float

    @property
    def length(self) -> float:
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5

    @property
    def is_horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < 1e-6


def _room_sides(room: dict) -> Dict[str, Tuple[float, float, float, float]]:
    x, y, w, h = room["x"], room["y"], room["w"], room["h"]
    # Engineering axes: x = south->north, y = west->east.  Therefore
    # south/north walls keep x fixed and run along y; west/east walls keep
    # y fixed and run along x.  The old implementation swapped the N/S and
    # E/W geometry, which put East/West doors and windows on the wrong walls.
    return {
        "S": (x, y, x, y + h),
        "N": (x + w, y, x + w, y + h),
        "W": (x, y, x + w, y),
        "E": (x, y + h, x + w, y + h),
    }


def build_walls(rooms: List[dict], plot_length: float, plot_width: float) -> List[WallSegment]:
    """One WallSegment per side of every room, classified exterior/interior."""
    walls = []
    for idx, room in enumerate(rooms):
        for side, (x1, y1, x2, y2) in _room_sides(room).items():
            on_boundary = (
                (side == "S" and x1 <= BOUNDARY_TOLERANCE) or
                (side == "N" and x1 >= plot_length - BOUNDARY_TOLERANCE) or
                (side == "W" and y1 <= BOUNDARY_TOLERANCE) or
                (side == "E" and y1 >= plot_width - BOUNDARY_TOLERANCE)
            )
            walls.append(WallSegment(
                side=side, room_label=room["label"], room_index=idx,
                x1=x1, y1=y1, x2=x2, y2=y2,
                exterior=on_boundary,
                thickness=EXTERIOR_WALL_THICKNESS if on_boundary else INTERIOR_WALL_THICKNESS,
            ))
    return walls


def room_exterior_sides(room: dict, plot_length: float, plot_width: float) -> List[str]:
    """Which of a room's sides (if any) sit on the building envelope."""
    sides = []
    for side, (x1, y1, x2, y2) in _room_sides(room).items():
        if side == "S" and x1 <= BOUNDARY_TOLERANCE:
            sides.append(side)
        elif side == "N" and x1 >= plot_length - BOUNDARY_TOLERANCE:
            sides.append(side)
        elif side == "W" and y1 <= BOUNDARY_TOLERANCE:
            sides.append(side)
        elif side == "E" and y1 >= plot_width - BOUNDARY_TOLERANCE:
            sides.append(side)
    return sides


def wall_polygon_points(seg: WallSegment, extend: float = 0.0):
    """
    Returns the 4 corner points (data units) of the rectangle representing
    this wall segment at its true thickness, centered on the segment's
    centerline. `extend` lengthens both ends slightly (used for exterior
    walls so corners overlap cleanly instead of leaving hairline gaps).
    """
    x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
    length = seg.length or 1e-6
    ux, uy = (x2 - x1) / length, (y2 - y1) / length  # unit direction
    nx, ny = -uy, ux  # unit normal
    t = seg.thickness / 2.0
    ex1, ey1 = x1 - ux * extend, y1 - uy * extend
    ex2, ey2 = x2 + ux * extend, y2 + uy * extend
    return [
        (ex1 + nx * t, ey1 + ny * t),
        (ex2 + nx * t, ey2 + ny * t),
        (ex2 - nx * t, ey2 - ny * t),
        (ex1 - nx * t, ey1 - ny * t),
    ]


def draw_walls(ax, walls: List[WallSegment], exterior_color: str, interior_color: str,
                background_color: str = "white"):
    """
    Draws every wall as a filled polygon at true architectural thickness
    (exterior walls thicker/darker, interior walls thin), then hands back
    the wall list so the door/window engines can carve their openings out
    of the same polygons afterward (drawn on top, in the background
    color, before the door/window symbols go on).
    """
    for seg in walls:
        extend = seg.thickness / 2 if seg.exterior else 0.0
        pts = wall_polygon_points(seg, extend=extend)
        ax.add_patch(patches.Polygon(
            pts, closed=True,
            facecolor=exterior_color if seg.exterior else interior_color,
            edgecolor="none", zorder=3 if seg.exterior else 2,
        ))


def cut_opening(ax, seg: WallSegment, position: float, width: float, background_color: str = "white"):
    """
    Cuts a door/window opening out of a wall by painting a background-
    colored polygon over that stretch of the wall (drawn after the wall,
    before the door/window symbol). `position` is measured in feet from
    the segment's (x1, y1) start point along its length.
    """
    length = seg.length or 1e-6
    ux, uy = (seg.x2 - seg.x1) / length, (seg.y2 - seg.y1) / length
    half = width / 2
    cx, cy = seg.x1 + ux * position, seg.y1 + uy * position
    ox1, oy1 = cx - ux * half, cy - uy * half
    ox2, oy2 = cx + ux * half, cy + uy * half
    opening = WallSegment(side=seg.side, room_label=seg.room_label, room_index=seg.room_index,
                           x1=ox1, y1=oy1, x2=ox2, y2=oy2, exterior=seg.exterior,
                           thickness=seg.thickness * 1.02)
    pts = wall_polygon_points(opening)
    ax.add_patch(patches.Polygon(pts, closed=True, facecolor=background_color,
                                  edgecolor="none", zorder=4))
