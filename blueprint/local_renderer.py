"""Detailed Pillow renderer for a residential architectural floor plan.

The result is an original technical drawing inspired by common Indian house
plan presentation conventions: thick outer walls, thin partitions, door swings,
windows, staircase treads, parking/open space, room dimensions, north arrow and
road/entrance information. It is not a construction approval drawing.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from blueprint.dimension_engine import feet_to_arch_string
from blueprint.door_engine import DOOR_WIDTH_FT, build_doors, validate_doors
from planning.building_programs import property_title


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _room_fill(room_type: str, appearance: str = "technical") -> str:
    t = str(room_type or "").lower()
    appearance = str(appearance or "technical").lower()
    if appearance == "warm":
        if "parking" in t or "garden" in t or "loading" in t: return "#F2F0E8"
        if "bath" in t or "restroom" in t or "toilet" in t or "wash" in t: return "#EEF3EF"
        if "kitchen" in t or "pantry" in t or "utility" in t: return "#F6EBDD"
        if "bedroom" in t or "guest_room" in t or "hostel_room" in t or "apartment_unit" in t: return "#F4EEE5"
        if any(k in t for k in ("living", "dining", "lobby", "reception", "waiting", "hall")): return "#F7F0E7"
        return "#FAF8F3"
    if appearance == "minimal":
        return "#FFFFFF"
    if "parking" in t or "garden" in t or "loading" in t:
        return "#F1F1EF"
    if "bath" in t or "restroom" in t or "toilet" in t or "wash" in t:
        return "#F4F6F5"
    if "kitchen" in t or "pantry" in t or "utility" in t:
        return "#F7F4EE"
    if "bedroom" in t or "guest_room" in t or "hostel_room" in t or "apartment_unit" in t:
        return "#F6F8FA"
    if any(k in t for k in ("living", "dining", "lobby", "reception", "waiting", "hall", "classroom", "work_area", "showroom")):
        return "#F7F8F9"
    return "#FBFBFA"


def _map_point(x: float, y: float, L: float, scale: float, ox: float, oy: float) -> Tuple[float, float]:
    # Rotate the engineering coordinate system into a portrait drawing:
    # x=depth becomes vertical; y=frontage becomes horizontal.
    return ox + y * scale, oy + (L - x) * scale


def _box(room: dict, L: float, scale: float, ox: float, oy: float):
    p1 = _map_point(room["x"], room["y"], L, scale, ox, oy)
    p2 = _map_point(room["x"] + room["w"], room["y"] + room["h"], L, scale, ox, oy)
    return p1[0], p2[1], p2[0], p1[1]


def _draw_dimension(d: ImageDraw.ImageDraw, a, b, label, font, color="#505A66", arrow_size=5):
    """Draw a clean architectural dimension line with extension/tick ends."""
    d.line((a[0], a[1], b[0], b[1]), fill=color, width=1)
    horizontal = abs(a[1] - b[1]) < 2
    if horizontal:
        for x, direction in ((a[0], 1), (b[0], -1)):
            d.polygon([(x, a[1]), (x + direction * arrow_size, a[1] - 2),
                       (x + direction * arrow_size, a[1] + 2)], fill=color)
    else:
        for y, direction in ((a[1], 1), (b[1], -1)):
            d.polygon([(a[0], y), (a[0] - 2, y + direction * arrow_size),
                       (a[0] + 2, y + direction * arrow_size)], fill=color)
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    d.text((mx, my), label, font=font, fill=color, anchor="mm", stroke_width=2, stroke_fill="white")


def _draw_room_dimensions(d, box, depth_ft: float, frontage_ft: float, scale: float, color="#66717D"):
    """Draw verified room frontage/depth dimensions inside each room."""
    x1, y1, x2, y2 = box
    px_w, px_h = x2 - x1, y2 - y1
    fs = _font(max(7, min(10, int(scale * 0.55))), True)
    # Horizontal dimension represents project h = frontage.
    if px_w >= 95:
        y = y1 + min(24, max(14, px_h * 0.18))
        _draw_dimension(d, (x1 + 8, y), (x2 - 8, y),
                        feet_to_arch_string(frontage_ft), fs, color, arrow_size=4)
    # Vertical dimension represents project w = depth.
    if px_h >= 95:
        x = x2 - min(24, max(14, px_w * 0.18))
        _draw_dimension(d, (x, y1 + 8), (x, y2 - 8),
                        feet_to_arch_string(depth_ft), fs, color, arrow_size=4)


def _draw_door(d, door: dict, L: float, scale: float, ox: float, oy: float, wall_width_ft: float):
    """Render the exact authoritative door opening + inside swing."""
    p1 = _map_point(float(door["x1"]), float(door["y1"]), L, scale, ox, oy)
    p2 = _map_point(float(door["x2"]), float(door["y2"]), L, scale, ox, oy)
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) or 1e-6
    ux, uy = (p2[0] - p1[0]) / length, (p2[1] - p1[1]) / length
    width_px = max(10.0, float(door.get("width", DOOR_WIDTH_FT)) * scale)
    half = width_px / 2.0
    cx = p1[0] + ux * float(door["position"]) * scale
    cy = p1[1] + uy * float(door["position"]) * scale
    hinge = (cx - ux * half, cy - uy * half)
    jamb = (cx + ux * half, cy + uy * half)

    # Interior-facing normals in rendered pixels.  The drawing coordinate has
    # y-down orientation, while the engineering coordinates are south->north
    # / west->east.
    into = {
        "S": (0.0, -1.0),
        "N": (0.0, 1.0),
        "W": (1.0, 0.0),
        "E": (-1.0, 0.0),
    }.get(str(door.get("wall_code", "S")).upper(), (0.0, -1.0))
    leaf = (hinge[0] + into[0] * width_px, hinge[1] + into[1] * width_px)

    # Carve only the exact opening, then draw the standard open-door symbol.
    opening_px = max(6, int(scale * max(0.20, wall_width_ft * 0.55)))
    d.line((cx - ux * half, cy - uy * half, cx + ux * half, cy + uy * half),
           fill="white", width=opening_px)
    line_w = max(2, int(scale * 0.11))
    arc_w = max(1, int(scale * 0.07))
    d.line((hinge[0], hinge[1], leaf[0], leaf[1]), fill="#414D5A", width=line_w)

    def angle(a, b):
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0

    a1 = angle(hinge, leaf)
    a2 = angle(hinge, jamb)
    delta = (a2 - a1) % 360.0
    if delta > 180.0:
        a1, a2 = a2, a1
        delta = (a2 - a1) % 360.0
    if delta > 110.0:
        # Numerical/transform edge case: retain a conventional quarter arc.
        delta = 90.0
    r = width_px
    bbox = (hinge[0] - r, hinge[1] - r, hinge[0] + r, hinge[1] + r)
    d.arc(bbox, start=a1, end=a1 + delta, fill="#66717D", width=arc_w)



def _draw_window(d, box, side, scale):
    x1, y1, x2, y2 = box
    width = max(22, min(x2 - x1, y2 - y1) * 0.30)
    line_w = max(2, int(scale * 0.10))
    if side in {"S", "N"}:
        y = y2 if side == "S" else y1
        cx = (x1 + x2) / 2
        a, b = cx - width / 2, cx + width / 2
        d.line((a, y, b, y), fill="white", width=max(5, int(scale * 0.22)))
        d.line((a, y - 3, b, y - 3), fill="#6D8295", width=line_w)
        d.line((a, y + 3, b, y + 3), fill="#6D8295", width=line_w)
    else:
        x = x2 if side == "E" else x1
        cy = (y1 + y2) / 2
        a, b = cy - width / 2, cy + width / 2
        d.line((x, a, x, b), fill="white", width=max(5, int(scale * 0.22)))
        d.line((x - 3, a, x - 3, b), fill="#6D8295", width=line_w)
        d.line((x + 3, a, x + 3, b), fill="#6D8295", width=line_w)


def _measure_text(text: str, font) -> tuple[float, float]:
    scratch = ImageDraw.Draw(Image.new("RGB", (2, 2), "white"))
    b = scratch.textbbox((0, 0), text, font=font)
    return float(b[2] - b[0]), float(b[3] - b[1])


def _fit_room_label(label: str, box_width: float, base_size: int, fill=None) -> tuple[list[str], int, int]:
    """Fit room labels into narrow architectural rooms without clipping."""
    label = str(label or "ROOM").upper().replace("_", " ")
    size = int(max(7, base_size))
    max_width = max(22.0, box_width - 10.0)
    replacements = {
        "BATHROOM": "BATH", "BEDROOM": "BED", "LIVING ROOM": "LIVING",
        "DINING ROOM": "DINING", "MASTER BEDROOM": "MASTER", "STAIRCASE": "STAIR",
        "PARKING": "PARK", "POOJA ROOM": "POOJA", "UTILITY ROOM": "UTILITY",
    }
    for _ in range(12):
        font = _font(size, True)
        text_w, text_h = _measure_text(label, font)
        if text_w <= max_width:
            return [label], size, int(text_h)
        words = label.split()
        lines = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            trial_w, _ = _measure_text(trial, font)
            if trial_w <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if lines and len(lines) <= 2 and all(_measure_text(line, font)[0] <= max_width for line in lines):
            line_h = max(_measure_text(line, font)[1] for line in lines)
            return lines, size, int(line_h)
        label = replacements.get(label, label)
        if size == 7:
            return [label[:14]], 7, 8
        size -= 1
    return [label[:14]], max(7, size), 8


def _draw_furniture(d, room: dict, box, scale):
    t = str(room.get("room_type", "")).lower()
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ink = "#9AA4AF"
    thin = max(1, int(scale * 0.06))

    if "bedroom" in t and w > 55 and h > 55:
        bw, bh = w * 0.48, h * 0.42
        bx, by = x1 + (w - bw) / 2, y1 + (h - bh) / 2
        d.rectangle((bx, by, bx + bw, by + bh), outline=ink, width=thin)
        d.line((bx, by + bh * 0.22, bx + bw, by + bh * 0.22), fill=ink, width=thin)
        d.rectangle((bx + 6, by + 4, bx + bw * 0.30, by + bh * 0.18), outline=ink, width=thin)
        d.rectangle((bx + bw * 0.70, by + 4, bx + bw - 6, by + bh * 0.18), outline=ink, width=thin)
    elif "living" in t and w > 70 and h > 65:
        sw, sh = w * 0.52, h * 0.18
        sx, sy = x1 + (w - sw) / 2, y1 + h * 0.60
        d.rectangle((sx, sy, sx + sw, sy + sh), outline=ink, width=thin)
        d.rectangle((x1 + w * 0.27, y1 + h * 0.20, x1 + w * 0.73, y1 + h * 0.42), outline=ink, width=thin)
    elif "dining" in t and w > 65 and h > 60:
        tw, th = w * 0.38, h * 0.30
        tx, ty = x1 + (w - tw) / 2, y1 + (h - th) / 2
        d.ellipse((tx, ty, tx + tw, ty + th), outline=ink, width=thin)
    elif "kitchen" in t and w > 55 and h > 50:
        d.rectangle((x1 + 8, y1 + 10, x2 - 8, y1 + 25), outline=ink, width=thin)
        d.ellipse((x1 + w * 0.34, y1 + 13, x1 + w * 0.43, y1 + 22), outline=ink, width=thin)
    elif "bathroom" in t and w > 40 and h > 42:
        d.ellipse((x1 + w * 0.58, y1 + h * 0.28, x1 + w * 0.78, y1 + h * 0.52), outline=ink, width=thin)
        d.ellipse((x1 + w * 0.18, y1 + h * 0.65, x1 + w * 0.40, y1 + h * 0.82), outline=ink, width=thin)
    elif any(k in t for k in ("office", "cabin", "manager", "director", "principal", "staff", "faculty", "consultation", "reception", "cashier")) and w > 50 and h > 45:
        d.rectangle((x1 + w * 0.25, y1 + h * 0.25, x1 + w * 0.75, y1 + h * 0.48), outline=ink, width=thin)
        d.rectangle((x1 + w * 0.40, y1 + h * 0.53, x1 + w * 0.60, y1 + h * 0.80), outline=ink, width=thin)
    elif any(k in t for k in ("classroom", "library", "lab", "work_area", "conference", "meeting", "dining_hall", "restaurant_dining")) and w > 60 and h > 55:
        d.rectangle((x1 + w * 0.18, y1 + h * 0.30, x1 + w * 0.82, y1 + h * 0.55), outline=ink, width=thin)
        for xx in [x1 + w * 0.28, x1 + w * 0.50, x1 + w * 0.72]:
            d.rectangle((xx - 7, y1 + h * 0.62, xx + 7, y1 + h * 0.73), outline=ink, width=thin)
    elif "parking" in t and w > 70 and h > 50:
        car_w, car_h = w * 0.55, h * 0.55
        cx, cy = x1 + (w - car_w) / 2, y1 + (h - car_h) / 2
        d.rounded_rectangle((cx, cy, cx + car_w, cy + car_h), radius=max(5, int(scale)), outline="#A0A7AF", width=thin)
        d.ellipse((cx + 7, cy + car_h * 0.08, cx + car_w * 0.30, cy + car_h * 0.22), outline="#A0A7AF", width=thin)
        d.ellipse((cx + car_w * 0.70, cy + car_h * 0.78, cx + car_w - 7, cy + car_h * 0.92), outline="#A0A7AF", width=thin)
    elif "staircase" in t and w > 55 and h > 45:
        steps = 7
        for i in range(steps + 1):
            yy = y1 + 8 + i * (h - 16) / steps
            d.line((x1 + 8, yy, x2 - 8, yy), fill=ink, width=thin)
        d.polygon([(x2 - 15, y1 + 16), (x2 - 6, y1 + 16), (x2 - 10, y1 + 6)], outline=ink)
    elif "balcony" in t:
        d.line((x1 + 6, y1 + 8, x2 - 6, y1 + 8), fill=ink, width=thin)
        for xx in [x1 + 10 + (x2 - x1 - 20) * i / 5 for i in range(6)]:
            d.line((xx, y1 + 8, xx, y1 + 18), fill=ink, width=thin)



def _draw_main_entrance(d, px1, py1, px2, py2, road, position, scale, ink, soft):
    """Draw a clearly identified main door/gate on the road-facing plot edge."""
    road = str(road or "south").lower()
    position = str(position or "center").lower()
    ratio = {"left": 0.28, "center": 0.50, "right": 0.72}.get(position, 0.50)
    span = max(34, min(72, scale * 7))
    # Keep this label close to the wall/door itself, inside the inner half of
    # the road strip; the "ROAD" label (drawn separately, in the outer half
    # of the same strip) used to sit at the same offset, so a centered main
    # door collided with the word "ROAD" whenever both were centered.
    if road in {"north", "south"}:
        cx = px1 + (px2 - px1) * ratio
        y = py1 if road == "north" else py2
        d.line((cx - span/2, y, cx + span/2, y), fill="white", width=max(7, int(scale*.42)))
        d.line((cx - span/2, y, cx + span/2, y), fill=ink, width=max(2, int(scale*.12)))
        ty = y - 8 if road == "north" else y + 8
        d.text((cx, ty), f"MAIN DOOR • {position.upper()}", font=_font(8, True), fill=soft, anchor="mm")
    else:
        cy = py1 + (py2 - py1) * ratio
        x = px2 if road == "east" else px1
        d.line((x, cy - span/2, x, cy + span/2), fill="white", width=max(7, int(scale*.42)))
        d.line((x, cy - span/2, x, cy + span/2), fill=ink, width=max(2, int(scale*.12)))
        tx = x + 8 if road == "east" else x - 8
        d.text((tx, cy), f"MAIN DOOR • {position.upper()}", font=_font(8, True), fill=soft, anchor="mm")

def render_local_blueprint(layout: Dict, entrance_direction: str, out_path: str,
                           floor_title: str = "GROUND FLOOR", dpi: int = 180,
                           building_type: str = "custom",
                           road_direction: str | None = None) -> str:
    plot = layout["plot"]
    L, W = float(plot["length"]), float(plot["width"])
    rooms = layout.get("rooms", [])
    setback = float(layout.get("setback_ft", 2.0))
    scale = max(5.0, min(12.0, 1050.0 / max(L, W)))
    left, top, right, bottom = 150, 115, 330, (235 if layout.get("vastu_enabled") else 190)
    appearance = str(layout.get("appearance", "technical") or "technical").lower()
    wall_width_ft = float(layout.get("wall_width_ft", 0.5) or 0.5)
    room_size_details = bool(layout.get("room_size_details", True))
    canvas_w = int(W * scale + left + right)
    canvas_h = int(L * scale + top + bottom)
    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(img)
    ox, oy = float(left), float(top)
    ink = "#2F3945"
    soft = "#707B87"

    d.text((left, 18), floor_title, font=_font(25, True), fill=ink)
    subtitle = f"{feet_to_arch_string(W)} × {feet_to_arch_string(L)} {property_title(building_type).upper()} FLOOR PLAN"
    # Fit the subtitle into the left title block so long building names never
    # collide with road/Vastu metadata on compact plan sheets.
    max_title_width = max(240, canvas_w - left - right - 280)
    subtitle_size = 13
    while subtitle_size > 8 and d.textbbox((0, 0), subtitle, font=_font(subtitle_size, True))[2] > max_title_width:
        subtitle_size -= 1
    d.text((left, 51), subtitle, font=_font(subtitle_size, True), fill=soft)
    is_ground_floor = int(layout.get("floor", 0) or 0) == 0
    effective_road_direction = str(road_direction or layout.get("road_direction") or entrance_direction or "South").title()
    if is_ground_floor:
        d.text((canvas_w - right, 28),
               f"ROAD: {effective_road_direction.upper()}  |  MAIN ENTRANCE: {str(entrance_direction).upper()}",
               font=_font(9, True), fill=soft, anchor="ra")
    else:
        d.text((canvas_w - right, 28), "ACCESS: INTERNAL STAIRCASE",
               font=_font(9, True), fill=soft, anchor="ra")
    if layout.get("vastu_enabled"):
        _pool = [r for r in rooms if str(r.get("room_type", "")).lower() not in {"parking", "garden", "loading_bay"}]
        total_rooms = max(len(_pool), 1)
        acceptable = min(sum(1 for r in _pool if r.get("vastu_acceptable")), total_rooms)
        score = acceptable / total_rooms
        if score is not None:
            score_pct = round(float(score) * 100.0)
            vastu_text = f"VASTU ARRANGEMENT: ON  |  SCORE: {score_pct}%  |  ACCEPTABLE: {acceptable}/{total_rooms}"
        else:
            m = int(layout.get("vastu_match_count", sum(1 for r in rooms if r.get("matched"))))
            vastu_text = f"VASTU ARRANGEMENT: ON  |  PREFERRED: {m}/{total_rooms}"
        d.text((canvas_w - right, 45), vastu_text, font=_font(8, True), fill=soft, anchor="ra")
    if layout.get('alignment_report'):
        d.text((canvas_w - right, 61),
               "ORTHOGONAL WALL GRID: ON",
               font=_font(8, True), fill=soft, anchor="ra")

    # Plot boundary
    px1, py1 = ox, oy
    px2, py2 = ox + W * scale, oy + L * scale
    d.rectangle((px1, py1, px2, py2), outline=ink,
                 width=max(4, int(scale * max(0.28, wall_width_ft * 0.55))))

    # Show both the permissible planning envelope and the actual room bounding box.
    env = layout.get("building_envelope") or {"x": setback, "y": setback,
                                               "w": max(1.0, L - 2 * setback),
                                               "h": max(1.0, W - 2 * setback)}
    bx1, by1 = _map_point(float(env["x"]), float(env["y"]), L, scale, ox, oy)
    bx2, by2 = _map_point(float(env["x"]) + float(env["w"]),
                          float(env["y"]) + float(env["h"]), L, scale, ox, oy)
    d.rectangle((min(bx1, bx2), min(by1, by2), max(bx1, bx2), max(by1, by2)),
                outline="#A0A7AF", width=max(1, int(scale * 0.06)))

    occupied = layout.get("occupied_envelope")
    if occupied:
        ox1, oy1 = _map_point(float(occupied["x"]), float(occupied["y"]), L, scale, ox, oy)
        ox2, oy2 = _map_point(float(occupied["x"]) + float(occupied["w"]),
                              float(occupied["y"]) + float(occupied["h"]), L, scale, ox, oy)
        d.rectangle((min(ox1, ox2), min(oy1, oy2), max(ox1, ox2), max(oy1, oy2)),
                    outline="#5F6B78", width=max(2, int(scale * 0.10)))

    # Only the ground floor gets a road strip/gate. Upper floors use internal
    # staircase access and must not invent a second road or main entrance.
    if is_ground_floor:
        road = effective_road_direction.lower()
        road_h = 28
        # "ROAD" sits in the outer half of the strip; the main-door label sits
        # near the wall, so the two annotations remain separate.
        if road == "north":
            d.rectangle((px1, py1 - road_h, px2, py1), fill="#ECEDEA", outline="#A3A9AF")
            d.text(((px1 + px2) / 2, py1 - road_h + 8), "ROAD", font=_font(10, True), fill=soft, anchor="mm")
            d.line((px1 + W * scale * .40, py1, px1 + W * scale * .60, py1), fill="#46515D", width=max(5, int(scale*.3)))
        elif road == "east":
            d.rectangle((px2, py1, px2 + road_h, py2), fill="#ECEDEA", outline="#A3A9AF")
            d.text((px2 + road_h - 8, (py1 + py2) / 2), "ROAD", font=_font(10, True), fill=soft, anchor="mm")
            d.line((px2, py1 + L * scale * .42, px2, py1 + L * scale * .58), fill="#46515D", width=max(5, int(scale*.3)))
        elif road == "west":
            d.rectangle((px1 - road_h, py1, px1, py2), fill="#ECEDEA", outline="#A3A9AF")
            d.text((px1 - road_h + 8, (py1 + py2) / 2), "ROAD", font=_font(10, True), fill=soft, anchor="mm")
            d.line((px1, py1 + L * scale * .42, px1, py1 + L * scale * .58), fill="#46515D", width=max(5, int(scale*.3)))
        else:
            d.rectangle((px1, py2, px2, py2 + road_h), fill="#ECEDEA", outline="#A3A9AF")
            d.text(((px1 + px2) / 2, py2 + road_h - 8), "ROAD", font=_font(10, True), fill=soft, anchor="mm")
            d.line((px1 + W * scale * .40, py2, px1 + W * scale * .60, py2), fill="#46515D", width=max(5, int(scale*.3)))

    # Doors are resolved once from the same wall geometry used by the layout
    # engine. The renderer no longer invents a second midpoint door model.
    doors = build_doors(
        rooms, entrance_direction=entrance_direction, plot_length=L, plot_width=W,
        building_envelope=env, main_door_position=layout.get("main_door_position", "center"),
        allow_main_entrance=is_ground_floor,
    )
    door_check = validate_doors(rooms, doors, expect_main_entrance=is_ground_floor)
    if not door_check["passed"]:
        layout.setdefault("render_warnings", []).extend(door_check["issues"])
    main_door = next((dd for dd in doors if dd.get("is_main_entrance")), None)
    if is_ground_floor and main_door:
        d.text((canvas_w - right, 76),
               f"MAIN DOOR: {main_door.get('wall','').upper()} WALL / {main_door.get('position',0):.2f} FT",
               font=_font(8, True), fill=soft, anchor="ra")
    door_by_room = {d["room"]: d for d in doors}

    # Rooms: fills, walls, furniture and labels are drawn first so door openings
    # can safely be carved across either side of a shared partition afterward.
    def _free_box(parent):
        """Largest rectangle of ``parent`` not covered by rooms carved inside it (for labels/furniture)."""
        kids = [k for k in rooms if k.get("inside_of") == parent.get("label")]
        if not kids:
            return None
        px0, py0 = float(parent["x"]), float(parent["y"])
        px1, py1 = px0 + float(parent["w"]), py0 + float(parent["h"])
        best, best_a = None, 0.0
        for k in kids:
            kx0, ky0 = float(k["x"]), float(k["y"])
            kx1, ky1 = kx0 + float(k["w"]), ky0 + float(k["h"])
            for cand in ((px0, py0, kx0, py1), (kx1, py0, px1, py1), (px0, py0, px1, ky0), (px0, ky1, px1, py1)):
                cw, ch = cand[2] - cand[0], cand[3] - cand[1]
                if cw < 3.0 or ch < 3.0:
                    continue
                if any(not (cand[2] <= float(o["x"]) + 0.01 or cand[0] >= float(o["x"]) + float(o["w"]) - 0.01 or
                            cand[3] <= float(o["y"]) + 0.01 or cand[1] >= float(o["y"]) + float(o["h"]) - 0.01) for o in kids):
                    continue
                if cw * ch > best_a:
                    best, best_a = cand, cw * ch
        return best

    for r in rooms:
        box = _box(r, L, scale, ox, oy)
        fill = _room_fill(r.get("room_type"), appearance)
        d.rectangle(box, fill=fill, outline=ink,
                     width=max(2, int(scale * max(0.10, wall_width_ft * 0.20))))
        fb = _free_box(r)
        if fb is not None:
            sub = dict(r, x=fb[0], y=fb[1], w=fb[2] - fb[0], h=fb[3] - fb[1])
            lbox = _box(sub, L, scale, ox, oy)
        else:
            lbox = box
        _draw_furniture(d, r if fb is None else sub, lbox, scale)

        label = str(r.get("label", "ROOM")).upper()
        rw = float(r.get("w", 0)); rh = float(r.get("h", 0))
        dim = f'{feet_to_arch_string(rh)} × {feet_to_arch_string(rw)}'
        cx, cy = (lbox[0] + lbox[2]) / 2, (lbox[1] + lbox[3]) / 2
        fs = max(7, min(14, int(7 + min(lbox[2] - lbox[0], lbox[3] - lbox[1]) / 45)))
        label_lines, label_size, line_h = _fit_room_label(label, lbox[2] - lbox[0], fs, fill)
        total_label_h = line_h * len(label_lines) + max(0, len(label_lines) - 1) * 2
        label_top = cy - (total_label_h + label_size + 8) / 2
        d.multiline_text((cx, label_top), "\n".join(label_lines), font=_font(label_size, True),
                         fill=ink, anchor="ma", align="center", spacing=2,
                         stroke_width=1, stroke_fill=fill)
        d.text((cx, label_top + total_label_h + 5), dim, font=_font(max(7, label_size - 1)),
               fill=soft, anchor="ma")
        if room_size_details and min(box[2] - box[0], box[3] - box[1]) >= 92:
            _draw_room_dimensions(d, box, rw, rh, scale)

        rt = str(r.get("room_type", "")).lower()
        if "parking" in rt:
            d.text(((box[0]+box[2])/2, box[3]-12), "PARKING ENTRY", font=_font(7, True), fill=soft, anchor="mm")
        if "staircase" in rt:
            d.text(((box[0]+box[2])/2, (box[1]+box[3])/2), "UP", font=_font(10, True), fill=soft, anchor="mm")

    # Open-plan hall group (living + dining + passage): no partition between its areas.
    try:
        from layout.residential_planner import shared_edge as _shared_edge
    except Exception:  # noqa: BLE001
        _shared_edge = None
    if _shared_edge:
        grp = [r for r in rooms if r.get("open_group")]
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                if a.get("open_group") != b.get("open_group"):
                    continue
                se = _shared_edge(a, b)
                if not se:
                    continue
                side, lo, hi = se
                wpx = max(3, int(scale * max(0.10, wall_width_ft * 0.20)) + 3)
                gap = 0.18
                if side in ("N", "S"):
                    xx = a["x"] + (a["w"] if side == "N" else 0.0)
                    p1 = _map_point(xx, lo + gap, L, scale, ox, oy); p2 = _map_point(xx, hi - gap, L, scale, ox, oy)
                else:
                    yy = a["y"] + (a["h"] if side == "E" else 0.0)
                    p1 = _map_point(lo + gap, yy, L, scale, ox, oy); p2 = _map_point(hi - gap, yy, L, scale, ox, oy)
                d.line((p1[0], p1[1], p2[0], p2[1]), fill=_room_fill(a.get("room_type"), appearance), width=wpx)

    # Doors and windows are drawn after every room outline so openings are not
    # overwritten by a neighboring room's border.
    for door in doors:
        _draw_door(d, door, L, scale, ox, oy, wall_width_ft)
        if door.get("is_main_entrance"):
            p = _map_point(door["center_x"], door["center_y"], L, scale, ox, oy)
            side = door.get("wall_code")
            label_pos = {"S": (p[0], p[1] + 15), "N": (p[0], p[1] - 15),
                         "W": (p[0] - 18, p[1]), "E": (p[0] + 18, p[1])}.get(side, (p[0], p[1]))
            text = "MAIN DOOR" if not door.get("main_entrance_fallback") else "MAIN DOOR • ACTUAL EDGE"
            d.text(label_pos, text, font=_font(8, True), fill=soft, anchor="mm")

    for r in rooms:
        box = _box(r, L, scale, ox, oy)
        current_door = door_by_room.get(r.get("label"), {})
        door_side = current_door.get("wall_code")
        eps = 0.35
        sides = []
        env_x0 = float(env.get("x", setback))
        env_y0 = float(env.get("y", setback))
        env_x1 = env_x0 + float(env.get("w", max(1.0, L - 2 * setback)))
        env_y1 = env_y0 + float(env.get("h", max(1.0, W - 2 * setback)))
        if float(r["x"]) <= env_x0 + eps: sides.append("S")
        if float(r["x"]) + float(r["w"]) >= env_x1 - eps: sides.append("N")
        if float(r["y"]) <= env_y0 + eps: sides.append("W")
        if float(r["y"]) + float(r["h"]) >= env_y1 - eps: sides.append("E")
        for candidate in sides:
            if candidate != door_side:
                _draw_window(d, box, candidate, scale)
                break

    # Overall dimensions: frontage at bottom, depth at left.
    dim_font = _font(12, True)
    # Overall plot dimensions are taken directly from the input plot geometry.
    dim_font = _font(12, True)
    _draw_dimension(d, (px1, py2 + 55), (px2, py2 + 55), feet_to_arch_string(W), dim_font)
    _draw_dimension(d, (px1 - 55, py2), (px1 - 55, py1), feet_to_arch_string(L), dim_font)

    # Setback note and compass
    d.text((px1, py2 + 82), f"Indicative planning setback: {setback:g} ft (verify local bylaws before construction)",
           font=_font(10), fill=soft)
    cx, cy = canvas_w - 45, 68
    d.line((cx, cy + 18, cx, cy - 18), fill=ink, width=2)
    d.polygon([(cx, cy - 28), (cx - 5, cy - 16), (cx + 5, cy - 16)], fill=ink)
    d.text((cx, cy - 34), "N", font=_font(12, True), fill=ink, anchor="mb")

    env_w, env_h = float(env.get("w", 0)), float(env.get("h", 0))
    d.text((px1, py2 + 98),
           f"PLANNING ENVELOPE: {feet_to_arch_string(env_h)} × {feet_to_arch_string(env_w)}",
           font=_font(9, True), fill=soft)
    if occupied:
        occ_w, occ_h = float(occupied.get("w", 0)), float(occupied.get("h", 0))
        d.text((px1, py2 + 114),
               f"ROOM EXTENT: {feet_to_arch_string(occ_h)} × {feet_to_arch_string(occ_w)}",
               font=_font(9, True), fill=soft)
    d.text((px1, py2 + 130), "ROOM DIMENSIONS: FRONTAGE × DEPTH  |  ALL DIMENSIONS IN FEET-INCHES",
           font=_font(8, True), fill=soft)
    if layout.get("vastu_enabled"):
        key_rooms = {
            "master_bedroom": "MASTER=SW", "kitchen": "KITCHEN=SE",
            "pooja_room": "POOJA=NE", "living_room": "LIVING=N/NE",
            "bathroom": "BATH=NW/SE", "restroom": "RESTROOM=NW/SE",
            "staircase": "STAIR=S/W", "parking": "PARKING=NW/N/W",
        }
        present = []
        seen = set()
        for r in rooms:
            rt = r.get("room_type")
            if rt in key_rooms and rt not in seen:
                present.append(key_rooms[rt]); seen.add(rt)
        if present:
            d.text((px1, py2 + 145),
                   "VASTU GUIDANCE: " + "  •  ".join(present[:5]),
                   font=_font(8, True), fill=soft)

    d.text((px2 + 24, py1 + 12),
           "ROOM SIZE DETAILS", font=_font(11, True), fill=ink)
    schedule_y = py1 + 34
    for idx, rr in enumerate(rooms[:24], 1):
        rrw, rrh = float(rr.get("w", 0)), float(rr.get("h", 0))
        text = f"{idx}. {str(rr.get('label', 'ROOM'))[:22]}  {feet_to_arch_string(rrh)} × {feet_to_arch_string(rrw)}"
        d.text((px2 + 24, schedule_y), text, font=_font(8), fill=soft)
        schedule_y += 16
    if len(rooms) > 24:
        d.text((px2 + 24, schedule_y), f"+ {len(rooms)-24} more rooms", font=_font(8), fill=soft)
        schedule_y += 16
    d.text((px2 + 24, schedule_y + 8),
           f"WALL WIDTH: {feet_to_arch_string(wall_width_ft)}  |  APPEARANCE: {appearance.upper()}",
           font=_font(8, True), fill=ink)
    entry_display = str(entrance_direction if is_ground_floor else "Internal staircase access").upper()
    entry_position = str(layout.get('main_door_position','center')).upper() if is_ground_floor else ""
    entry_text = f"ENTRY: {entry_display}" + (f" / {entry_position}" if entry_position else "")
    d.text((px2 + 24, schedule_y + 26), entry_text, font=_font(8, True), fill=ink)
    has_parking = any("parking" in str(rr.get("room_type", "")).lower() for rr in rooms)
    parking_text = str(layout.get('parking_position','road_side')).replace('_',' ').upper() if has_parking else "NOT INCLUDED"
    d.text((px2 + 24, schedule_y + 42), f"PARKING: {parking_text}", font=_font(8), fill=soft)
    has_staircase = any("staircase" in str(rr.get("room_type", "")).lower() for rr in rooms)
    staircase_text = str(layout.get('staircase_position','road_side')).replace('_',' ').upper() if has_staircase else "NOT INCLUDED"
    d.text((px2 + 24, schedule_y + 58), f"STAIRCASE: {staircase_text}", font=_font(8), fill=soft)

    # Legend
    legend_y = py2 + (190 if layout.get("vastu_enabled") else 173)
    d.text((px1, legend_y), "PLAN ELEMENTS", font=_font(10, True), fill=ink)
    items = ["THICK LINE = EXTERNAL WALL", "THIN LINE = PARTITION", "D = DOOR / SWING", "W = WINDOW", "OPEN SPACE = SETBACK / EXTERIOR"]
    for i, text in enumerate(items):
        d.text((px1 + 150 * (i % 2), legend_y + 20 + 16 * (i // 2)), text, font=_font(9), fill=soft)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG", optimize=True)
    return out_path
