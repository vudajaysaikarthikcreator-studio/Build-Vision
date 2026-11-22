"""Practical architectural floor-plan generator.

This version keeps the existing BuildVision API contract but replaces the old
3x3 Vastu-zone rectangles with a room-packing planner that behaves more like
an Indian residential floor-plan drawing: setbacks, road-side open space,
parking/stair placement, realistic room proportions, and Vastu-aware room
position scoring are used before collision validation.

Coordinates:
  x = plot depth (south -> north)
  y = plot frontage (west -> east)
"""
from __future__ import annotations

from typing import List, Dict, Tuple

from .room_allocator import RoomAllocation
from .collision_checker import run_all_checks
from vastu.vastu_rules import ZONE_GRID, ACCEPTABLE_ZONES
from planning.building_programs import ROOM_DIMS, MIN_DIMS

DOOR_SIDE_FOR_ZONE = {
    "N": "S", "NE": "W", "E": "W", "SE": "N", "S": "N", "SW": "E",
    "W": "E", "NW": "S", "C": "S",
}

# Room geometry is centralized in planning/building_programs.py so every building type
# uses the same target/minimum dimensions in allocation, validation and rendering.


def _setback_for_plot(plot_length: float, plot_width: float) -> float:
    min_side = min(plot_length, plot_width)
    if min_side < 24:
        return 1.5
    if min_side < 35:
        return 2.0
    return 3.0


def _rect_intersects(a: dict, b: dict) -> bool:
    return not (
        a["x"] + a["w"] <= b["x"] + 1e-6
        or b["x"] + b["w"] <= a["x"] + 1e-6
        or a["y"] + a["h"] <= b["y"] + 1e-6
        or b["y"] + b["h"] <= a["y"] + 1e-6
    )


def _zone_bounds(plot_length: float, plot_width: float, zone: str,
                 envelope: dict | None = None) -> Tuple[float, float, float, float]:
    """Return a Vastu zone rectangle using the buildable planning envelope.

    Setbacks are excluded from the directional grid so a room touching the
    setback does not get pushed across a zone boundary merely because the plot
    has a mandatory open strip.
    """
    index = {z: (r, c) for r, row in enumerate(ZONE_GRID) for c, z in enumerate(row)}
    r, c = index.get(zone, (1, 1))
    ex = float(envelope['x']) if envelope else 0.0
    ey = float(envelope['y']) if envelope else 0.0
    ew = float(envelope['w']) if envelope else plot_length
    eh = float(envelope['h']) if envelope else plot_width
    row_h = ew / 3.0
    col_w = eh / 3.0
    x = ex + (2 - r) * row_h
    y = ey + c * col_w
    return x, y, row_h, col_w


def _zone_center(plot_length: float, plot_width: float, zone: str,
                 envelope: dict | None = None) -> Tuple[float, float]:
    x, y, w, h = _zone_bounds(plot_length, plot_width, zone, envelope)
    return x + w / 2.0, y + h / 2.0


def _room_priority(room_type: str) -> int:
    # Larger / harder-to-place spaces first; service rooms last.
    return {
        "parking": 1, "loading_bay": 1, "reception": 2, "lobby": 2, "living_room": 2,
        "master_bedroom": 3, "open_work_area": 3, "classroom": 3, "showroom": 3,
        "production_hall": 3, "storage_hall": 3, "multipurpose_hall": 3, "auditorium": 3,
        "bedroom": 4, "guest_room": 4, "hostel_room": 4, "apartment_unit": 4,
        "kitchen": 5, "restaurant_kitchen": 5, "dining_room": 6, "dining_hall": 6,
        "restaurant_dining": 6, "staircase": 7, "conference_room": 7, "meeting_room": 8,
        "office_cabin": 8, "consultation_room": 8, "treatment_room": 8, "procedure_room": 8,
        "library": 8, "computer_lab": 8, "science_lab": 8, "diagnostic_lab": 8,
        "bathroom": 9, "restroom": 9, "toilet": 9, "pooja_room": 10, "study_room": 10,
        "pantry": 11, "store_room": 11, "stock_room": 11, "server_room": 11, "utility_room": 11,
    }.get(room_type, 12)


def _candidate_positions(free: dict, room_type: str, desired: Tuple[float, float]):
    """Yield a few high-quality positions inside a free rectangle."""
    fw, fh = free["w"], free["h"]
    dw, dh = desired
    options = [(dw, dh), (dh, dw)] if abs(dw - dh) > 0.05 else [(dw, dh)]
    for w, h in options:
        if w <= fw + 1e-6 and h <= fh + 1e-6:
            yield free["x"], free["y"], w, h
            yield free["x"] + (fw - w) / 2.0, free["y"] + (fh - h) / 2.0, w, h
            yield free["x"] + fw - w, free["y"] + fh - h, w, h

        # On tight plots, shrink toward the minimum viable size while keeping
        # a sensible aspect ratio.
        min_w, min_h = MIN_DIMS.get(room_type, (3.0, 4.0))
        if fw >= min_w and fh >= min_h:
            s = min(fw / w, fh / h, 1.0)
            sw, sh = w * s, h * s
            if sw >= min_w and sh >= min_h:
                yield free["x"] + (fw - sw) / 2.0, free["y"] + (fh - sh) / 2.0, sw, sh


def _subtract_free_rects(free_rects: List[dict], used: dict) -> List[dict]:
    out: List[dict] = []
    for f in free_rects:
        if not _rect_intersects(f, used):
            out.append(f)
            continue

        fx, fy, fw, fh = f["x"], f["y"], f["w"], f["h"]
        ux, uy, uw, uh = used["x"], used["y"], used["w"], used["h"]
        # left, right, bottom, top free pieces
        pieces = [
            {"x": fx, "y": fy, "w": max(0.0, ux - fx), "h": fh},
            {"x": min(fx + fw, ux + uw), "y": fy,
             "w": max(0.0, fx + fw - (ux + uw)), "h": fh},
            {"x": fx, "y": fy, "w": fw, "h": max(0.0, uy - fy)},
            {"x": fx, "y": min(fy + fh, uy + uh), "w": fw,
             "h": max(0.0, fy + fh - (uy + uh))},
        ]
        out.extend(p for p in pieces if p["w"] > 0.35 and p["h"] > 0.35)

    # Remove rectangles completely contained inside another free rectangle.
    pruned: List[dict] = []
    for i, a in enumerate(out):
        contained = False
        for j, b in enumerate(out):
            if i == j:
                continue
            if (
                a["x"] >= b["x"] - 1e-6 and a["y"] >= b["y"] - 1e-6
                and a["x"] + a["w"] <= b["x"] + b["w"] + 1e-6
                and a["y"] + a["h"] <= b["y"] + b["h"] + 1e-6
            ):
                contained = True
                break
        if not contained:
            pruned.append(a)
    return pruned


def _front_reservation(
    room_type: str, plot_length: float, plot_width: float, road_direction: str,
    occupied: List[dict], setback: float,
) -> dict | None:
    """Place parking/garden/staircase along the road-facing edge.

    Spaces are scanned along the frontage, so parking + staircase + garden can
    coexist on a compact plot instead of all competing for one corner.
    """
    target_w, target_h = ROOM_DIMS[room_type]
    min_w, min_h = MIN_DIMS[room_type]
    rd = str(road_direction or "South").strip().lower()
    x_min, y_min = setback, setback
    x_max = plot_length - setback
    y_max = plot_width - setback
    w = min(target_w, max(1.0, x_max - x_min))
    h = min(target_h, max(1.0, y_max - y_min))
    if w < min_w or h < min_h:
        return None

    candidates = []
    if rd in {"north", "south"}:
        front_x = x_max - w if rd == "north" else x_min
        # Scan across the 25-ft-ish frontage in both directions.
        ys = [y_min + i for i in range(int(max(0, (y_max - h) // 1)) + 1)]
        candidates.extend({"x": front_x, "y": y, "w": w, "h": h} for y in ys)
        candidates.extend({"x": front_x, "y": y, "w": w, "h": h} for y in reversed(ys))
    else:
        front_y = y_max - h if rd == "east" else y_min
        xs = [x_min + i for i in range(int(max(0, (x_max - w) // 1)) + 1)]
        candidates.extend({"x": x, "y": front_y, "w": w, "h": h} for x in xs)
        candidates.extend({"x": x, "y": front_y, "w": w, "h": h} for x in reversed(xs))

    for c in candidates:
        if c["x"] < x_min or c["y"] < y_min:
            continue
        if c["x"] + c["w"] > x_max + 1e-6 or c["y"] + c["h"] > y_max + 1e-6:
            continue
        if not any(_rect_intersects(c, r) for r in occupied):
            return c
    return None


def _pack_remaining(
    rooms: List[RoomAllocation], plot_length: float, plot_width: float,
    setback: float, road_direction: str, occupied: List[dict],
    allowed_envelope: dict | None = None,
) -> List[dict]:
    envelope = allowed_envelope or {
        'x': setback, 'y': setback,
        'w': max(1.0, plot_length - 2 * setback),
        'h': max(1.0, plot_width - 2 * setback),
    }

    free_area = envelope["w"] * envelope["h"] - sum(r["w"] * r["h"] for r in occupied)
    desired_area = sum(ROOM_DIMS.get(a.room_type, (8.0, 8.0))[0] * ROOM_DIMS.get(a.room_type, (8.0, 8.0))[1] for a in rooms)
    min_area = sum(MIN_DIMS.get(a.room_type, (3.0, 4.0))[0] * MIN_DIMS.get(a.room_type, (3.0, 4.0))[1] for a in rooms)

    scale_area = 1.0
    if desired_area > free_area and desired_area > 0:
        scale_area = min(1.0, max(0.0, (free_area * 0.88) / desired_area))
    if free_area >= min_area and desired_area > 0:
        scale_area = max(scale_area, min(1.0, (min_area * 1.03) / desired_area))
    dim_scale = max(0.55, scale_area ** 0.5)

    effective_dims = {}
    for a in rooms:
        dw, dh = ROOM_DIMS.get(a.room_type, (8.0, 8.0))
        mw, mh = MIN_DIMS.get(a.room_type, (3.0, 4.0))
        effective_dims[a.room_type, a.label] = (max(mw, dw * dim_scale), max(mh, dh * dim_scale))

    placed = list(occupied)
    unplaced = []
    ordered = sorted(
        rooms,
        key=lambda a: (
            -_room_vastu_weight(a.room_type),
            _room_priority(a.room_type),
            -a.target_area,
        )
    )
    zone_map = {a.label: (a.zone or a.preferred_zone or "C") for a in ordered}

    # Scan the entire building envelope rather than fragmented free rectangles.
    # This is slower than pure guillotine packing but much more reliable for
    # small 25-ft/30-ft frontage plots with several compact service rooms.
    for item in ordered:
        desired = effective_dims[item.room_type, item.label]
        zx, zy = _zone_center(plot_length, plot_width, zone_map[item.label], envelope)
        best = None
        for w, h in [(desired[0], desired[1]), (desired[1], desired[0])]:
            if w > envelope["w"] or h > envelope["h"]:
                continue
            min_env_x = envelope['x']
            min_env_y = envelope['y']
            max_x = envelope['x'] + envelope['w'] - w
            max_y = envelope['y'] + envelope['h'] - h
            # Adaptive anchor search: small residential plots retain 0.5-ft
            # resolution, while large institutional/industrial plots use
            # 1-2-ft candidates around Vastu anchors and existing room edges.
            # This preserves directional placement without millions of scans.
            step = 0.5 if max(plot_length, plot_width) <= 60 else (1.0 if max(plot_length, plot_width) <= 120 else 2.0)
            zone_dx = envelope['w'] / 3.0
            zone_dy = envelope['h'] / 3.0
            def _clamp(v, lo, hi):
                return max(lo, min(hi, v))
            x_values = {envelope['x'], envelope['x'] + envelope['w'] - w}
            y_values = {envelope['y'], envelope['y'] + envelope['h'] - h}
            # Preferred-zone anchors.
            for anchor in (zx - w/2.0, zx - w, zx, zx + w/2.0):
                x_values.add(round(_clamp(anchor, envelope['x'], max_x) / step) * step)
            for anchor in (zy - h/2.0, zy - h, zy, zy + h/2.0):
                y_values.add(round(_clamp(anchor, envelope['y'], max_y) / step) * step)
            # Pack against already placed room edges to keep the footprint tight.
            for r in placed:
                for anchor in (r.get('x', 0), r.get('x', 0) + r.get('w', 0) - w, r.get('x', 0) + r.get('w', 0), r.get('x', 0) - w):
                    x_values.add(round(_clamp(anchor, envelope['x'], max_x) / step) * step)
                for anchor in (r.get('y', 0), r.get('y', 0) + r.get('h', 0) - h, r.get('y', 0) + r.get('h', 0), r.get('y', 0) - h):
                    y_values.add(round(_clamp(anchor, envelope['y'], max_y) / step) * step)
            # Ensure a small local grid exists around the target Vastu zone.
            x0 = _clamp(zx - zone_dx/2.0 - w, envelope['x'], max_x)
            x1 = _clamp(zx + zone_dx/2.0, envelope['x'], max_x)
            y0 = _clamp(zy - zone_dy/2.0 - h, envelope['y'], max_y)
            y1 = _clamp(zy + zone_dy/2.0, envelope['y'], max_y)
            xv = x0
            local_guard = 0
            while xv <= x1 + 1e-6 and local_guard < 250:
                x_values.add(round(xv / step) * step)
                xv += step; local_guard += 1
            yv = y0
            local_guard = 0
            while yv <= y1 + 1e-6 and local_guard < 250:
                y_values.add(round(yv / step) * step)
                yv += step; local_guard += 1

            for x in sorted(x_values):
                if x < min_env_x - 1e-6 or x + w > envelope['x'] + envelope['w'] + 1e-6:
                    continue
                for y in sorted(y_values):
                    if y < min_env_y - 1e-6 or y + h > envelope['y'] + envelope['h'] + 1e-6:
                        continue
                    candidate = {"x": x, "y": y, "w": w, "h": h}
                    if any(_rect_intersects(candidate, r) for r in placed):
                        continue
                    cx, cy = x + w / 2.0, y + h / 2.0
                    zone_distance = ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5
                    # Prefer locations that leave the candidate close to one
                    # envelope edge: this creates clean architectural bands.
                    edge_dist = min(x - setback, y - setback,
                                    plot_length - setback - (x + w),
                                    plot_width - setback - (y + h))
                    score = _compact_candidate_score(
                        candidate, placed, plot_length, plot_width, item, envelope
                    ) + max(0.0, edge_dist) * 0.01
                    if best is None or score < best[0]:
                        best = (score, candidate)

        if best is None:
            # Robust fallback for compact plots: use a coarse complete grid so
            # greedy anchor placement never rejects a layout merely because
            # an anchor candidate set was too sparse.
            fallback_step = 0.5 if max(plot_length, plot_width) <= 50 else 1.0
            grid_best = None
            xi = envelope['x']
            guard_x = 0
            while xi <= plot_length - envelope.get('x', setback) - w + envelope['x'] + 1e-6 and guard_x < 500:
                yj = envelope['y']
                guard_y = 0
                while yj <= envelope['y'] + envelope['h'] - h + 1e-6 and guard_y < 500:
                    candidate = {'x': round(xi / fallback_step) * fallback_step,
                                 'y': round(yj / fallback_step) * fallback_step,
                                 'w': w, 'h': h}
                    if (candidate['x'] >= envelope['x'] - 1e-6 and
                        candidate['y'] >= envelope['y'] - 1e-6 and
                        candidate['x'] + w <= envelope['x'] + envelope['w'] + 1e-6 and
                        candidate['y'] + h <= envelope['y'] + envelope['h'] + 1e-6 and
                        not any(_rect_intersects(candidate, r) for r in placed)):
                        score = _compact_candidate_score(
                            candidate, placed, plot_length, plot_width, item, envelope
                        )
                        if grid_best is None or score < grid_best[0]:
                            grid_best = (score, candidate)
                    yj += fallback_step
                    guard_y += 1
                xi += fallback_step
                guard_x += 1
            if grid_best is None:
                unplaced.append(item.label)
                continue
            best = grid_best

        chosen = best[1]
        room = {
            "room_type": item.room_type,
            "label": item.label,
            "zone": item.zone,
            "preferred_zone": item.preferred_zone,
            "matched": item.matched,
            **{k: round(v, 2) for k, v in chosen.items()},
            "door_side": DOOR_SIDE_FOR_ZONE.get(item.zone, "S"),
        }
        placed.append(room)

    if unplaced:
        raise ValueError(
            "The requested rooms cannot fit inside the available planning envelope. "
            f"Unplaced rooms: {', '.join(unplaced)}. Try adding a floor, reducing rooms, or increasing plot size."
        )
    return placed


def _compact_grid_layout(
    plot_length: float, plot_width: float, allocations: List[RoomAllocation],
    setback: float, road_direction: str, floors: int, building_type: str = 'custom',
) -> Dict | None:
    """Clean two-column grid used for compact residential plots.

    It deliberately produces the clear room bands seen in professional 2D
    house-plan sheets instead of filling the whole plot with irregular zones.
    """
    inner_w = plot_width - 2 * setback
    inner_l = plot_length - 2 * setback
    if inner_w < 14 or inner_l < 22 or len(allocations) > 12:
        return None
    residential_types = {"master_bedroom", "bedroom", "bathroom", "kitchen", "living_room", "dining_room",
                         "pooja_room", "study_room", "store_room", "balcony", "parking", "garden", "staircase"}
    if any(a.room_type not in residential_types for a in allocations):
        return None

    cols = 2 if inner_w >= 18 else 1
    col_w = inner_w / cols
    gap = 0.0  # rooms share walls; circulation is represented explicitly, not as accidental gaps
    cell_w = max(1.0, col_w - gap)

    order = {
        "parking": 1, "staircase": 2, "garden": 3, "living_room": 4,
        "master_bedroom": 5, "bedroom": 6, "kitchen": 7, "dining_room": 8,
        "bathroom": 9, "pooja_room": 10, "study_room": 11, "store_room": 12,
        "balcony": 13,
    }
    ordered = sorted(allocations, key=lambda a: (order.get(a.room_type, 20), -a.target_area))
    groups = [ordered[i:i + cols] for i in range(0, len(ordered), cols)]

    row_heights = []
    for group in groups:
        rh = 0.0
        for item in group:
            target = ROOM_DIMS.get(item.room_type, (8.0, 8.0))
            minimum = MIN_DIMS.get(item.room_type, (3.0, 4.0))
            # Compact-plan target: keep rooms practical but avoid oversized
            # target areas that would crowd a 25x40/30x40 plot.
            target_area = min(target[0] * target[1], minimum[0] * minimum[1] * 1.05)
            rh = max(rh, target_area / cell_w, minimum[0])
        row_heights.append(rh)

    required_depth = sum(row_heights)
    if required_depth > inner_l:
        # Scale room heights as a group. Keep the room proportions practical.
        factor = inner_l / required_depth
        row_heights = [h * factor for h in row_heights]
        for idx, group in enumerate(groups):
            if any(row_heights[idx] + 1e-6 < MIN_DIMS.get(item.room_type, (3, 4))[0] for item in group):
                return None

    rooms = []
    x = setback
    for row_idx, group in enumerate(groups):
        rh = row_heights[row_idx]
        # Front row is placed at the road-facing side. For north-facing roads,
        # start from the opposite x end so the first band is still front-most.
        if str(road_direction or "South").lower() == "north":
            row_x = plot_length - setback - sum(row_heights[:row_idx + 1])
        else:
            row_x = x

        # East/West road-facing plots read more naturally when the entry-side
        # room is drawn first in the frontage direction.
        cols_order = list(range(cols))
        if str(road_direction or "South").lower() == "east":
            cols_order = list(reversed(cols_order))

        for pos, item in enumerate(group):
            col = cols_order[pos]
            y = setback + col * col_w
            room = {
                "room_type": item.room_type, "label": item.label,
                "zone": item.zone, "preferred_zone": item.preferred_zone,
                "matched": item.matched,
                "x": round(row_x, 2), "y": round(y, 2),
                # x/w = depth, y/h = frontage. Row depth is `rh`; column
                # frontage is `cell_w`. Keeping this convention consistent is
                # important for collision checks, 3D rendering and dimensions.
                "w": round(max(1.0, rh), 2),
                "h": round(max(1.0, cell_w), 2),
                "door_side": DOOR_SIDE_FOR_ZONE.get(item.zone, "S"),
            }
            rooms.append(room)
        if str(road_direction or "South").lower() != "north":
            x += rh

    checks = run_all_checks(rooms, plot_length, plot_width)
    vastu_active = any((a.preferred_zone or 'C') != 'C' for a in allocations)
    if not checks['collisions'] and not checks['boundary_violations']:
        min_x = min(r['x'] for r in rooms) if rooms else setback
        min_y = min(r['y'] for r in rooms) if rooms else setback
        max_x = max((r['x'] + r['w']) for r in rooms) if rooms else setback
        max_y = max((r['y'] + r['h']) for r in rooms) if rooms else setback
        footprint = {'x': min_x, 'y': min_y, 'w': max_x - min_x, 'h': max_y - min_y}
        if vastu_active:
            for r in rooms:
                actual = _preferred_zone_from_point(plot_length, plot_width,
                                                     r['x'] + r['w'] / 2.0,
                                                     r['y'] + r['h'] / 2.0, None)
                r['zone'] = actual
                r['matched'] = actual == r.get('preferred_zone')
                try:
                    from vastu.vastu_rules import ACCEPTABLE_ZONES
                    r['vastu_acceptable'] = actual in set(ACCEPTABLE_ZONES.get(r.get('room_type'), [r.get('preferred_zone')]))
                except Exception:
                    r['vastu_acceptable'] = r['matched']
                r['door_side'] = DOOR_SIDE_FOR_ZONE.get(actual, r.get('door_side', 'S'))
    return {
        "plot": {"length": round(plot_length, 2), "width": round(plot_width, 2),
                  "area": round(plot_length * plot_width, 2)},
        "floors": floors, "road_direction": road_direction,
        "setback_ft": round(setback, 2),
        "building_envelope": {"x": round(setback, 2), "y": round(setback, 2),
                               "w": round(plot_length - 2 * setback, 2),
                               "h": round(plot_width - 2 * setback, 2)},
        "rooms": rooms, "collisions": checks["collisions"],
        "boundary_violations": checks["boundary_violations"], "passed_checks": checks["passed"],
        "vastu_match_count": sum(1 for r in rooms if r.get('matched')),
        "vastu_acceptable_count": sum(1 for r in rooms if r.get('vastu_acceptable')),
        "vastu_score": round(_actual_zone_score(plot_length, plot_width, rooms), 4),
        "vastu_match_percent": round(100.0 * sum(1 for r in rooms if r.get('matched')) / max(len(rooms), 1), 1),
    }



def _preferred_zone_from_point(plot_length: float, plot_width: float, x: float, y: float,
                              envelope: dict | None = None) -> str:
    """Resolve the actual 3x3 Vastu zone occupied by a room centre.

    When a compact building sits inside a larger plot, Vastu zones are resolved
    within the building footprint rather than the entire land parcel. This keeps
    a 100x150 plot from making every room appear stuck in one plot-level zone.
    """
    if envelope:
        ex, ey = float(envelope['x']), float(envelope['y'])
        ew, eh = max(float(envelope['w']), 1e-6), max(float(envelope['h']), 1e-6)
        rx = min(0.999999, max(0.0, (x - ex) / ew))
        ry = min(0.999999, max(0.0, (y - ey) / eh))
    else:
        rx = min(0.999999, max(0.0, x / max(plot_length, 1e-6)))
        ry = min(0.999999, max(0.0, y / max(plot_width, 1e-6)))
    col = min(2, max(0, int(ry * 3)))
    row_from_south = min(2, max(0, int(rx * 3)))
    row = 2 - row_from_south
    return ZONE_GRID[row][col]


def _local_dimensions(room_type: str, scale: float) -> Tuple[float, float]:
    """Return (depth, frontage) after uniform scaling, never below minima."""
    depth, frontage = ROOM_DIMS.get(room_type, (8.0, 8.0))
    min_depth, min_frontage = MIN_DIMS.get(room_type, (3.0, 4.0))
    return max(min_depth, depth * scale), max(min_frontage, frontage * scale)


def _orient_for_frontage(room_type: str, scale: float, max_frontage: float) -> Tuple[float, float]:
    """Choose the orientation that gives the smallest depth while fitting frontage."""
    d1, f1 = _local_dimensions(room_type, scale)
    d2, f2 = f1, d1
    options = [(d, f) for d, f in ((d1, f1), (d2, f2)) if f <= max_frontage + 1e-6]
    if not options:
        return (d1, f1) if f1 <= f2 else (d2, f2)
    return min(options, key=lambda item: (item[0], -item[1]))


def _actual_zone_score(plot_length: float, plot_width: float, rooms: List[dict]) -> float:
    """Lower is better: normalized distance from each room centre to its preferred Vastu zone."""
    diagonal = max((plot_length ** 2 + plot_width ** 2) ** 0.5, 1.0)
    total = 0.0
    for r in rooms:
        px, py = r['x'] + r['w'] / 2.0, r['y'] + r['h'] / 2.0
        setback = _setback_for_plot(plot_length, plot_width)
        planning_env = {
            'x': setback, 'y': setback,
            'w': max(1.0, plot_length - 2 * setback),
            'h': max(1.0, plot_width - 2 * setback),
        }
        zx, zy = _zone_center(plot_length, plot_width, r.get('preferred_zone') or r.get('zone') or 'C', planning_env)
        total += ((px - zx) ** 2 + (py - zy) ** 2) ** 0.5 / diagonal
    return total


ZONE_WEIGHTS = {
    # Important residential Vastu spaces get stronger placement preference.
    'pooja_room': 2.8, 'master_bedroom': 2.6, 'kitchen': 2.4, 'living_room': 2.0,
    'bedroom': 1.8, 'study_room': 1.7, 'dining_room': 1.5, 'bathroom': 1.4,
    'toilet': 1.4, 'restroom': 1.2, 'staircase': 1.1, 'parking': 0.8, 'garden': 0.7,
}


def _room_vastu_weight(room_type: str) -> float:
    return ZONE_WEIGHTS.get(room_type, 1.0)


def _placement_score(plot_length: float, plot_width: float, item: RoomAllocation,
                     rect: dict, building_type: str = 'custom',
                     reference_envelope: dict | None = None) -> float:
    """Score a candidate room rectangle using Vastu first, geometry second.

    Vastu is a placement objective, not a post-generation label: the candidate
    receives a lower score when its centre is nearer the room's preferred zone.
    A small edge/shape penalty keeps plans compact and legible.
    """
    cx = rect['x'] + rect['w'] / 2.0
    cy = rect['y'] + rect['h'] / 2.0
    setback = _setback_for_plot(plot_length, plot_width)
    planning_env = reference_envelope or {
        'x': setback, 'y': setback,
        'w': max(1.0, plot_length - 2 * setback),
        'h': max(1.0, plot_width - 2 * setback),
    }
    zx, zy = _zone_center(plot_length, plot_width, item.preferred_zone or item.zone or 'C', planning_env)
    diagonal = max((plot_length ** 2 + plot_width ** 2) ** 0.5, 1.0)
    zone_distance = ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5 / diagonal
    aspect = max(rect['w'] / max(rect['h'], 1e-6), rect['h'] / max(rect['w'], 1e-6))
    aspect_penalty = max(0.0, aspect - 2.2) * 0.03
    # Institutional buildings still honour the same Vastu zones, but at a
    # softer weight so circulation and functional adjacency remain possible.
    vastu_weight = _room_vastu_weight(item.room_type) if building_type in {
        '1bhk','2bhk','3bhk','4bhk','duplex','villa','studio','farmhouse','custom'
    } else max(0.7, _room_vastu_weight(item.room_type) * 0.55)
    return zone_distance * vastu_weight + aspect_penalty



def _contact_metrics(candidate: dict, placed: List[dict], tol: float = 0.12) -> tuple[float, float]:
    """Return (shared_edge_length, nearest_edge_gap) for a candidate rectangle.

    A good architectural packer should prefer rooms that share a wall with an
    already placed room.  Tiny numerical differences are treated as touching.
    """
    if not placed:
        return 0.0, 999.0
    shared = 0.0
    nearest = 999.0
    cx1, cy1 = candidate['x'], candidate['y']
    cx2, cy2 = cx1 + candidate['w'], cy1 + candidate['h']
    for r in placed:
        rx1, ry1 = r['x'], r['y']
        rx2, ry2 = rx1 + r['w'], ry1 + r['h']
        if abs(cx2 - rx1) <= tol or abs(rx2 - cx1) <= tol:
            overlap = max(0.0, min(cy2, ry2) - max(cy1, ry1))
            shared = max(shared, overlap)
            nearest = min(nearest, abs(cx2 - rx1), abs(rx2 - cx1))
        if abs(cy2 - ry1) <= tol or abs(ry2 - cy1) <= tol:
            overlap = max(0.0, min(cx2, rx2) - max(cx1, rx1))
            shared = max(shared, overlap)
            nearest = min(nearest, abs(cy2 - ry1), abs(ry2 - cy1))
        # Distance between non-overlapping projections.  This is only a tie
        # breaker after shared-wall contact is considered.
        dx = max(rx1 - cx2, cx1 - rx2, 0.0)
        dy = max(ry1 - cy2, cy1 - ry2, 0.0)
        nearest = min(nearest, (dx * dx + dy * dy) ** 0.5)
    return shared, nearest


def _compact_candidate_score(candidate: dict, placed: List[dict],
                             plot_length: float, plot_width: float,
                             item: RoomAllocation, reference_envelope: dict) -> float:
    """Combine Vastu preference with strong shared-wall/compactness preference."""
    vastu = _placement_score(plot_length, plot_width, item, candidate,
                              reference_envelope=reference_envelope)
    if not placed:
        return vastu
    contact, nearest_gap = _contact_metrics(candidate, placed)
    min_x = min([r['x'] for r in placed] + [candidate['x']])
    min_y = min([r['y'] for r in placed] + [candidate['y']])
    max_x = max([r['x'] + r['w'] for r in placed] + [candidate['x'] + candidate['w']])
    max_y = max([r['y'] + r['h'] for r in placed] + [candidate['y'] + candidate['h']])
    footprint_area = max(1.0, (max_x - min_x) * (max_y - min_y))
    footprint_perimeter = (max_x - min_x) + (max_y - min_y)
    contact_bonus = min(contact / max(min(candidate['w'], candidate['h']), 1.0), 1.0)
    # Shared wall is strongly preferred. A 12-ft+ contact should beat a
    # directionally perfect but isolated room in nearly all cases.
    return (
        vastu * 90.0
        - contact_bonus * 8.0
        + min(nearest_gap, 6.0) * 0.45
        + footprint_area * 0.00025
        + footprint_perimeter * 0.002
    )


def _candidate_join_positions(room: dict, others: List[dict], env: dict) -> list[dict]:
    """Generate edge-to-edge positions that make rooms share walls."""
    candidates = []
    eps = 1e-6
    for ref in others:
        x_aligned = [ref['x'] - room['w'], ref['x'] + ref['w']]
        y_aligned = [ref['y'] - room['h'], ref['y'] + ref['h']]
        y_vals = [ref['y'], ref['y'] + ref['h'] - room['h'], ref['y'] + (ref['h'] - room['h']) / 2.0]
        x_vals = [ref['x'], ref['x'] + ref['w'] - room['w'], ref['x'] + (ref['w'] - room['w']) / 2.0]
        for x in x_aligned:
            for y in y_vals:
                candidates.append({'x': x, 'y': y, 'w': room['w'], 'h': room['h']})
        for y in y_aligned:
            for x in x_vals:
                candidates.append({'x': x, 'y': y, 'w': room['w'], 'h': room['h']})
    # Also pull rooms toward the current compact envelope edges.
    for x in (env['x'], env['x'] + env['w'] - room['w']):
        for y in (env['y'], env['y'] + env['h'] - room['h']):
            candidates.append({'x': x, 'y': y, 'w': room['w'], 'h': room['h']})
    return candidates


def _close_internal_gaps(rooms: List[dict], plot_length: float, plot_width: float,
                         reference_envelope: dict, iterations: int = 3) -> None:
    """Post-process rooms so they form a tight, connected plan.

    Rooms keep their dimensions.  We only translate rectangles to valid
    edge-to-edge positions, so all annotated dimensions remain authoritative.
    Vastu score is retained as a weighted objective, while adjacency prevents
    accidental empty strips/holes from appearing between rooms.
    """
    if len(rooms) < 2:
        return
    for _ in range(max(1, iterations)):
        ordered = sorted(rooms, key=lambda r: (-_room_vastu_weight(r.get('room_type', '')), r['label']))
        moved = False
        for room in ordered:
            others = [r for r in rooms if r is not room]
            current = {'x': room['x'], 'y': room['y'], 'w': room['w'], 'h': room['h']}
            current_contact, current_gap = _contact_metrics(current, others)
            current_score = _compact_candidate_score(current, others, plot_length, plot_width,
                                                     type('A', (), {'room_type': room.get('room_type'),
                                                                    'preferred_zone': room.get('preferred_zone'),
                                                                    'zone': room.get('zone')})(),
                                                     reference_envelope)
            best = None
            for candidate in _candidate_join_positions(room, others, reference_envelope):
                if candidate['x'] < reference_envelope['x'] - 1e-6 or candidate['y'] < reference_envelope['y'] - 1e-6:
                    continue
                if candidate['x'] + candidate['w'] > reference_envelope['x'] + reference_envelope['w'] + 1e-6:
                    continue
                if candidate['y'] + candidate['h'] > reference_envelope['y'] + reference_envelope['h'] + 1e-6:
                    continue
                if any(_rect_intersects(candidate, r) for r in others):
                    continue
                contact, nearest_gap = _contact_metrics(candidate, others)
                # Never trade away an existing wall contact for an isolated room.
                if current_contact > 0.25 and contact + 1e-6 < current_contact:
                    continue
                score = _compact_candidate_score(candidate, others, plot_length, plot_width,
                                                  type('A', (), {'room_type': room.get('room_type'),
                                                                 'preferred_zone': room.get('preferred_zone'),
                                                                 'zone': room.get('zone')})(),
                                                  reference_envelope)
                # Strongly reward any newly-created contact.
                if contact > current_contact + 0.25:
                    score -= 6.0
                if best is None or score < best[0]:
                    best = (score, candidate, contact, nearest_gap)
            if best and best[0] + 1e-6 < current_score:
                cand = best[1]
                room['x'] = round(cand['x'], 2)
                room['y'] = round(cand['y'], 2)
                moved = True
        if not moved:
            break




def _edge_alignment_metrics(candidate: dict, others: List[dict], tol: float = 0.20) -> tuple[int, float, float]:
    """Return (aligned_edge_count, shared_edge_length, nearest_parallel_gap)."""
    aligned = 0
    shared = 0.0
    nearest_gap = 999.0
    x_edges = (candidate['x'], candidate['x'] + candidate['w'])
    y_edges = (candidate['y'], candidate['y'] + candidate['h'])
    for r in others:
        rx = (r['x'], r['x'] + r['w'])
        ry = (r['y'], r['y'] + r['h'])
        for a in x_edges:
            for b in rx:
                gap = abs(a - b)
                nearest_gap = min(nearest_gap, gap)
                if gap <= tol:
                    aligned += 1
        for a in y_edges:
            for b in ry:
                gap = abs(a - b)
                nearest_gap = min(nearest_gap, gap)
                if gap <= tol:
                    aligned += 1
        if abs(candidate['x'] + candidate['w'] - r['x']) <= tol or abs(r['x'] + r['w'] - candidate['x']) <= tol:
            shared = max(shared, max(0.0, min(candidate['y'] + candidate['h'], r['y'] + r['h']) - max(candidate['y'], r['y'])))
        if abs(candidate['y'] + candidate['h'] - r['y']) <= tol or abs(r['y'] + r['h'] - candidate['y']) <= tol:
            shared = max(shared, max(0.0, min(candidate['x'] + candidate['w'], r['x'] + r['w']) - max(candidate['x'], r['x'])))
    return aligned, shared, nearest_gap


def _align_room_edges(rooms: List[dict], plot_length: float, plot_width: float,
                      reference_envelope: dict, iterations: int = 6,
                      edge_snap_tolerance: float = 1.0, max_shift: float = 4.0) -> None:
    """Straighten the plan onto common orthogonal wall lines.

    Rooms keep their dimensions.  Their x/y origins are translated to existing
    room edges (or envelope edges), which produces continuous horizontal and
    vertical partition lines and removes small zig-zag offsets between adjacent
    rooms. Vastu remains part of the score, but alignment is a first-class
    architectural objective.
    """
    if len(rooms) < 2:
        return

    env = reference_envelope
    env_x = float(env['x']); env_y = float(env['y'])
    env_x2 = env_x + float(env['w']); env_y2 = env_y + float(env['h'])

    def score(room, candidate, others):
        pseudo = type('A', (), {
            'room_type': room.get('room_type'),
            'preferred_zone': room.get('preferred_zone') or 'C',
            'zone': room.get('zone') or 'C',
        })
        vastu = _placement_score(plot_length, plot_width, pseudo, candidate,
                                  reference_envelope=env)
        aligned, shared, nearest = _edge_alignment_metrics(candidate, others)
        contact, nearest_contact = _contact_metrics(candidate, others)
        displacement = ((candidate['x'] - room['x']) ** 2 + (candidate['y'] - room['y']) ** 2) ** 0.5
        # Alignment dominates tiny Vastu differences; large Vastu moves are
        # still penalized, so the algorithm does not throw rooms across zones.
        return (
            vastu * 32.0
            - aligned * 5.0
            - min(shared / max(min(candidate['w'], candidate['h']), 1.0), 1.0) * 2.0
            - min(contact / max(min(candidate['w'], candidate['h']), 1.0), 1.0) * 1.5
            + min(displacement, max_shift * 2.0) * 0.12
            + min(nearest, edge_snap_tolerance) * 0.15
            + min(nearest_contact, 4.0) * 0.06
        )

    for _ in range(max(1, iterations)):
        moved_any = False
        ordered = sorted(rooms, key=lambda r: (-_room_vastu_weight(r.get('room_type', '')), r['label']))
        for room in ordered:
            others = [r for r in rooms if r is not room]
            old = {k: float(room[k]) for k in ('x','y','w','h')}
            best = old.copy()
            best_score = score(room, best, others)

            x_lines = {env_x, env_x2 - old['w']}
            y_lines = {env_y, env_y2 - old['h']}
            for r in others:
                for line in (r['x'], r['x'] + r['w']):
                    x_lines.add(line)
                    x_lines.add(line - old['w'])
                for line in (r['y'], r['y'] + r['h']):
                    y_lines.add(line)
                    y_lines.add(line - old['h'])

            # Only consider nearby wall lines. This prevents a room from being
            # dragged across a large plot just to improve a single alignment.
            x_candidates = [x for x in x_lines if env_x - 1e-6 <= x <= env_x2 - old['w'] + 1e-6
                             and abs(x - old['x']) <= max_shift + edge_snap_tolerance]
            y_candidates = [y for y in y_lines if env_y - 1e-6 <= y <= env_y2 - old['h'] + 1e-6
                             and abs(y - old['y']) <= max_shift + edge_snap_tolerance]

            candidates = []
            for x in x_candidates:
                candidates.append({'x': x, 'y': old['y'], 'w': old['w'], 'h': old['h']})
            for y in y_candidates:
                candidates.append({'x': old['x'], 'y': y, 'w': old['w'], 'h': old['h']})
            for x in x_candidates:
                for y in y_candidates:
                    # Only test combined moves when both axes are near an
                    # existing line; this avoids diagonal drifting.
                    if abs(x-old['x']) <= max_shift and abs(y-old['y']) <= max_shift:
                        candidates.append({'x': x, 'y': y, 'w': old['w'], 'h': old['h']})

            for cand in candidates:
                if any(_rect_intersects(cand, r) for r in others):
                    continue
                sc = score(room, cand, others)
                # Require a meaningful improvement; do not jitter rooms between
                # equally scored grid lines.
                if sc + 0.04 < best_score:
                    best_score = sc
                    best = cand

            if best != old:
                room['x'] = round(best['x'], 2)
                room['y'] = round(best['y'], 2)
                moved_any = True
        if not moved_any:
            break



def _cluster_snap_room_edges(rooms: List[dict], plot_length: float, plot_width: float,
                            reference_envelope: dict, iterations: int = 3,
                            cluster_tolerance: float = 2.0, max_shift: float = 3.0) -> None:
    """Snap nearby room edges to common orthogonal wall/grid lines.

    This is deliberately conservative: a room moves only along one axis at a
    time, stays inside the envelope, keeps its dimensions, and must remain
    collision-free. The candidate is scored against Vastu plus the number of
    neighboring edges it makes coincident with.
    """
    if len(rooms) < 2:
        return
    env = reference_envelope

    def clusters(values):
        vals = sorted(values)
        groups=[]
        for v in vals:
            if not groups or v - groups[-1][-1] > cluster_tolerance:
                groups.append([v])
            else:
                groups[-1].append(v)
        return [sum(g)/len(g) for g in groups if len(g) >= 2]

    for _ in range(max(1, iterations)):
        x_lines=[v for r in rooms for v in (r['x'], r['x']+r['w'])]
        y_lines=[v for r in rooms for v in (r['y'], r['y']+r['h'])]
        x_centers=clusters(x_lines)
        y_centers=clusters(y_lines)
        moved=False
        for room in sorted(rooms, key=lambda r: (-_room_vastu_weight(r.get('room_type','')), r['label'])):
            others=[r for r in rooms if r is not room]
            pseudo=type('A', (), {'room_type':room.get('room_type'),'preferred_zone':room.get('preferred_zone') or 'C','zone':room.get('zone') or 'C'})()
            old={'x':float(room['x']),'y':float(room['y']),'w':float(room['w']),'h':float(room['h'])}

            def candidate_score(c):
                vastu=_placement_score(plot_length,plot_width,pseudo,c,reference_envelope=env)
                aligned,shared,_=_edge_alignment_metrics(c,others,tol=0.25)
                displacement=((c['x']-old['x'])**2+(c['y']-old['y'])**2)**0.5
                return vastu*28.0 - aligned*5.5 - min(shared/max(min(c['w'],c['h']),1.0),1.0)*2.0 + displacement*0.10

            best=old; best_score=candidate_score(old)
            xs=sorted(x_centers,key=lambda v:abs(v-old['x']))
            ys=sorted(y_centers,key=lambda v:abs(v-old['y']))
            candidates=[]
            for line in xs[:12]:
                for x in (line, line-old['w']):
                    if abs(x-old['x'])<=max_shift:
                        candidates.append({'x':x,'y':old['y'],'w':old['w'],'h':old['h']})
            for line in ys[:12]:
                for y in (line, line-old['h']):
                    if abs(y-old['y'])<=max_shift:
                        candidates.append({'x':old['x'],'y':y,'w':old['w'],'h':old['h']})
            for c in candidates:
                if c['x'] < env['x']-1e-6 or c['y'] < env['y']-1e-6 or c['x']+c['w'] > env['x']+env['w']+1e-6 or c['y']+c['h'] > env['y']+env['h']+1e-6:
                    continue
                if any(_rect_intersects(c,r) for r in others):
                    continue
                sc=candidate_score(c)
                if sc+0.05 < best_score:
                    best,best_score=c,sc
            if best != old:
                room['x']=round(best['x'],2); room['y']=round(best['y'],2); moved=True
        if not moved:
            break

def _alignment_report(rooms: List[dict], tolerance: float = 0.20) -> dict:
    """Measure meaningful room-to-room wall alignment at T-junctions.

    An edge is evaluated only when another room has a parallel edge whose
    perpendicular span overlaps it by at least 0.5 ft. This avoids penalizing
    legitimate outer corners or isolated perimeter edges.
    """
    if len(rooms) < 2:
        return {'aligned_edges': 0, 'total_edges': 0, 'alignment_percent': 100.0, 'max_edge_misalignment_ft': 0.0}
    aligned = 0
    total = 0
    max_misalignment = 0.0
    span_tol = 0.5
    for i, room in enumerate(rooms):
        others = rooms[:i] + rooms[i+1:]
        # Vertical edges: compare x coordinates when y ranges overlap.
        for e in (room['x'], room['x'] + room['w']):
            refs = []
            for r in others:
                overlap = min(room['y'] + room['h'], r['y'] + r['h']) - max(room['y'], r['y'])
                if overlap >= span_tol:
                    refs.extend((r['x'], r['x'] + r['w']))
            if refs:
                nearest = min(abs(e - ref) for ref in refs)
                total += 1
                max_misalignment = max(max_misalignment, nearest)
                if nearest <= tolerance:
                    aligned += 1
        # Horizontal edges: compare y coordinates when x ranges overlap.
        for e in (room['y'], room['y'] + room['h']):
            refs = []
            for r in others:
                overlap = min(room['x'] + room['w'], r['x'] + r['w']) - max(room['x'], r['x'])
                if overlap >= span_tol:
                    refs.extend((r['y'], r['y'] + r['h']))
            if refs:
                nearest = min(abs(e - ref) for ref in refs)
                total += 1
                max_misalignment = max(max_misalignment, nearest)
                if nearest <= tolerance:
                    aligned += 1
    return {
        'aligned_edges': aligned,
        'total_edges': total,
        'alignment_percent': round(100.0 * aligned / max(total, 1), 1),
        'max_edge_misalignment_ft': round(max_misalignment, 2),
    }



def _zone_band(zone: str) -> tuple[int, int]:
    """Return (south_to_north band, west_to_east band) for a Vastu zone."""
    index = {z: (r, c) for r, row in enumerate(ZONE_GRID) for c, z in enumerate(row)}
    r, c = index.get(str(zone or 'C').upper(), (1, 1))
    return 2 - r, c


_ZONE_LATERAL = {
    # Centre of each Vastu column: west=1/6, centre=1/2, east=5/6.
    # Using actual column centres avoids rooms drifting across a zone boundary
    # merely because the previous width allocator favoured the largest spare room.
    'W': 1/6, 'SW': 1/6, 'NW': 1/6,
    'S': 0.50, 'C': 0.50, 'N': 0.50,
    'E': 5/6, 'SE': 5/6, 'NE': 5/6,
}

_RESIDENTIAL_VASTU_TYPES = {'1bhk','2bhk','3bhk','4bhk','duplex','villa','studio','farmhouse','custom'}


def _choose_join_groups(rooms: List[dict], inner_depth: float, inner_frontage: float, building_type: str = "custom") -> list[list[dict]]:
    """Group rooms into compact, full-width Vastu-aware horizontal bands.

    Rows are assigned by the preferred Vastu north/south band plus a small load
    balancing term. This keeps SW/SE spaces together toward the south, N/NE/NW
    spaces toward the north, while still avoiding an impractical one-room row.
    """
    if not rooms:
        return []

    # Residential Vastu is directional rather than a generic score. Build the
    # plan around a stable three-band hierarchy: north/public, centre/transition,
    # and south/service. This avoids the previous failure where the final tiler
    # moved a bedroom or bathroom into the centre merely to fill a row.
    if str(building_type or 'custom').lower() in _RESIDENTIAL_VASTU_TYPES:
        all_rooms=list(rooms)
        north=[]; south=[]; middle=[]

        def take_first(predicate):
            for r in list(all_rooms):
                if predicate(r):
                    all_rooms.remove(r); return r
            return None

        # Public/quiet north band. One NW bedroom plus the first NW bathroom
        # can sit beside the north/NE living room without forcing service rooms
        # into the centre of the house. Pooja/study stay in NE/N.
        living=take_first(lambda r:r.get('room_type')=='living_room')
        if living: north.append(living)
        first_bed=take_first(lambda r:r.get('room_type')=='bedroom' and str(r.get('preferred_zone','')).upper() in {'NW','N'})
        if first_bed: north.insert(0,first_bed)
        if not first_bed:
            first_bed=take_first(lambda r:r.get('room_type')=='bedroom')
            if first_bed: north.insert(0,first_bed)
        pooja=take_first(lambda r:r.get('room_type')=='pooja_room')
        if pooja: north.append(pooja)
        study=take_first(lambda r:r.get('room_type')=='study_room')
        if study: north.append(study)
        # Only a bathroom/toilet whose preferred zone is NW/N belongs in the
        # north band. West/east/south service rooms stay in their own directional
        # bands so the final retile cannot move them across Vastu sectors.
        def north_bath_fits(r):
            if r.get('room_type') not in {'bathroom','toilet','restroom'}:
                return False
            if str(r.get('preferred_zone','')).upper() not in {'NW','N'}:
                return False
            # If a NW bedroom already occupies the western edge, the bathroom
            # remains NW only when the two room centres fit inside the west third.
            # Otherwise route the bathroom to the middle/west band instead of
            # allowing it to spill into the north/centre zone.
            if first_bed is not None:
                bed_min=MIN_DIMS.get(first_bed.get('room_type'),(3.0,4.0))[1]
                bath_min=MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1]
                return bed_min + bath_min/2.0 <= inner_frontage/3.0 + 1e-6
            return True
        first_bath=take_first(north_bath_fits)
        if first_bath: north.insert(1 if first_bed else 0, first_bath)

        # South band contains the strongest traditional service/heat/stability
        # zones: master bedroom, kitchen, remaining bathrooms and staircase.
        master=take_first(lambda r:r.get('room_type')=='master_bedroom')
        if master: south.append(master)
        staircase=take_first(lambda r:r.get('room_type')=='staircase')
        # Take one S/W secondary bedroom into the south band where it fits.
        south_bed=take_first(lambda r:r.get('room_type')=='bedroom' and str(r.get('preferred_zone','')).upper() in {'S','SW','W'})
        if south_bed: south.append(south_bed)
        # All kitchens should remain in the south/east service band.
        while True:
            kitchen=take_first(lambda r:r.get('room_type') in {'kitchen','restaurant_kitchen'})
            if not kitchen: break
            south.append(kitchen)
        if staircase: south.append(staircase)
        # Remaining bathrooms/toilets are service spaces and are deliberately kept
        # out of the NE/NW public band.
        # Remaining bathrooms follow their preferred lateral/vertical Vastu zone:
        # W/NW bathrooms stay north or middle; S/SE bathrooms stay south.
        for r in list(all_rooms):
            if r.get('room_type') in {'bathroom','toilet','restroom'}:
                all_rooms.remove(r)
                z=str(r.get('preferred_zone','C')).upper()
                if z in {'NW','W','C','E'}:
                    # NW/W service rooms are both traditional west-side choices;
                    # keeping them in the middle band preserves the west sector
                    # instead of forcing multiple bathrooms to share the north row.
                    middle.append(r)
                elif z in {'N','NE'}:
                    north.append(r)
                else:
                    south.append(r)
        store=take_first(lambda r:r.get('room_type')=='store_room')
        if store: south.append(store)

        # Dining belongs beside the kitchen, and any remaining bedrooms remain in
        # the centre/west band rather than displacing the main north rooms.
        dining=take_first(lambda r:r.get('room_type')=='dining_room')
        if dining: middle.append(dining)
        while True:
            bed=take_first(lambda r:r.get('room_type')=='bedroom')
            if not bed: break
            # A bedroom explicitly assigned to the north band stays there. This is
            # primarily used by duplex/villa upper floors where a singleton W row
            # would geometrically resolve to the centre column.
            if str(bed.get('preferred_zone','')).upper() in {'N','NE'}:
                north.append(bed)
            else:
                middle.append(bed)
        # Balconies follow their N/E preference; if there is no room in the north
        # band, keep them in the middle rather than putting them into SW.
        while True:
            bal=take_first(lambda r:r.get('room_type')=='balcony')
            if not bal: break
            (north if len(north) < 4 else middle).append(bal)

        # Any optional remaining small rooms join the centre band.
        middle.extend(all_rooms)

        groups=[g for g in (south,middle,north) if g]
        for g in groups:
            g.sort(key=lambda r:(
                _ZONE_LATERAL.get(str(r.get('preferred_zone') or r.get('zone') or 'C').upper(),0.50),
                -_room_vastu_weight(r.get('room_type','')),
                r.get('label',''),
            ))

        # Only accept the explicit template when each band can physically fit at
        # minimum dimensions. Otherwise use the generic optimizer below.
        min_heights=[max(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[0] for r in g) for g in groups]
        min_frontages=[sum(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1] for r in g) for g in groups]
        if groups and sum(min_heights) <= inner_depth + 1e-6 and max(min_frontages) <= inner_frontage + 1e-6:
            return groups

    n=len(rooms)
    min_rows=max(1,(n+3)//4)
    max_rows=max(min_rows, min(n,(n+1)//2))
    best=None
    for rows_count in range(min_rows,max_rows+1):
        base_cap=(n+rows_count-1)//rows_count
        row_centers=[(i+0.5)/rows_count for i in range(rows_count)]
        row_groups=[[] for _ in range(rows_count)]
        # Strong Vastu rooms are assigned first so their desired vertical band
        # is not displaced by low-priority service spaces.
        items=sorted(rooms,key=lambda r:(-_room_vastu_weight(r.get('room_type','')), _zone_band(r.get('preferred_zone') or r.get('zone') or 'C')[0]))
        for r in items:
            band=_zone_band(r.get('preferred_zone') or r.get('zone') or 'C')[0]
            desired_y=band/2.0
            candidates=[i for i in range(rows_count) if len(row_groups[i]) < base_cap]
            if not candidates:
                candidates=list(range(rows_count))
            # Slight penalty for a heavily loaded row keeps bands balanced.
            i=min(candidates,key=lambda idx:abs(row_centers[idx]-desired_y)+0.08*len(row_groups[idx]))
            row_groups[i].append(r)
        # Empty rows are not permitted because that would create a true void.
        empties=[i for i,g in enumerate(row_groups) if not g]
        while empties:
            empty=empties.pop()
            donor=max(range(rows_count),key=lambda i:len(row_groups[i]))
            if len(row_groups[donor])<=1:
                break
            move=min(row_groups[donor], key=lambda r:abs(row_centers[empty]-_zone_band(r.get('preferred_zone') or 'C')[0]/2.0))
            row_groups[donor].remove(move); row_groups[empty].append(move)
        if any(not g for g in row_groups):
            continue
        for g in row_groups:
            g.sort(key=lambda r:(_ZONE_LATERAL.get(str(r.get('preferred_zone') or r.get('zone') or 'C').upper(),0.50), -_room_vastu_weight(r.get('room_type',''))))
        min_heights=[max(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[0] for r in g) for g in row_groups]
        target_heights=[max(ROOM_DIMS.get(r.get('room_type'),(8.0,8.0))[0] for r in g) for g in row_groups]
        min_frontages=[sum(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1] for r in g) for g in row_groups]
        if sum(min_heights)>inner_depth+1e-6 or max(min_frontages)>inner_frontage+1e-6:
            continue
        score=sum(abs((i+0.5)/rows_count - _zone_band(r.get('preferred_zone') or 'C')[0]/2.0) for i,g in enumerate(row_groups) for r in g)*1.4 + rows_count*0.25 + max(len(g) for g in row_groups)*0.03
        if best is None or score<best[0]:
            best=(score,row_groups)
    if best:
        return best[1]
    # Last-resort balanced chunks that still tile without gaps.
    rows_count=min(max_rows,n)
    ordered=sorted(rooms,key=lambda r:(_zone_band(r.get('preferred_zone') or 'C')[0],_ZONE_LATERAL.get(str(r.get('preferred_zone') or 'C').upper(),0.50)))
    groups=[]; pos=0
    for i in range(rows_count):
        remaining=n-pos; left=rows_count-i
        size=(remaining+left-1)//left
        groups.append(ordered[pos:pos+size]); pos+=size
    return groups

def _retile_rooms_without_internal_gaps(rooms: List[dict], plot_length: float, plot_width: float,
                                        reference_envelope: dict, setback: float, road_direction: str = 'South',
                                        building_type: str = 'custom') -> dict:
    """Retile interior rooms into touching full-width bands.

    This replaces the previous strategy of drawing large blank rectangles as
    PASSAGE / CIRCULATION.  Rooms become the actual floorplan partitions: every
    horizontal band spans the occupied frontage, rooms within a band share wall
    edges, and consecutive bands share full horizontal boundaries.
    """
    external_types={'parking','garden','loading_bay'}
    interior=[r for r in rooms if r.get('room_type') not in external_types]
    external=[r for r in rooms if r.get('room_type') in external_types]
    if len(interior) < 2:
        return {'rooms': rooms, 'envelope': dict(reference_envelope), 'external': external}

    ref_env=dict(reference_envelope or {})
    inner_depth=max(1.0, float(ref_env.get('w', plot_length-2*setback)))
    inner_frontage=max(1.0, float(ref_env.get('h', plot_width-2*setback)))
    groups=_choose_join_groups(interior, inner_depth, inner_frontage, building_type=building_type)
    if not groups:
        return {'rooms': rooms, 'envelope': dict(reference_envelope), 'external': external}

    target_area=sum(
        max(
            ROOM_DIMS.get(r.get('room_type'), (8.0,8.0))[0] * ROOM_DIMS.get(r.get('room_type'), (8.0,8.0))[1],
            MIN_DIMS.get(r.get('room_type'), (3.0,4.0))[0] * MIN_DIMS.get(r.get('room_type'), (3.0,4.0))[1] * 1.08,
        ) for r in interior
    )
    min_height_sum=sum(max(MIN_DIMS.get(r.get('room_type'), (3.0,4.0))[0] for r in g) for g in groups)
    max_min_frontage=max(sum(MIN_DIMS.get(r.get('room_type'), (3.0,4.0))[1] for r in g) for g in groups)
    plot_aspect=inner_depth/max(inner_frontage,1e-6)
    desired_area=max(target_area*1.10, min_height_sum*max_min_frontage*1.08)
    depth=(desired_area*plot_aspect)**0.5
    frontage=desired_area/max(depth,1e-6)
    depth=min(inner_depth, max(min_height_sum*1.03, depth))
    frontage=min(inner_frontage, max(max_min_frontage*1.03, frontage))
    # If the target area still does not fit, scale toward the plot while never
    # dropping below the hard minimum dimensions encoded in MIN_DIMS.
    if depth*frontage < min_height_sum*max_min_frontage:
        depth=min(inner_depth, min_height_sum)
        frontage=min(inner_frontage, max_min_frontage)
    if depth < min_height_sum-1e-6 or frontage < max_min_frontage-1e-6:
        return {'rooms': rooms, 'envelope': dict(reference_envelope), 'external': external}

    rd=str(road_direction or 'South').strip().lower()
    # Keep the final joined block inside the Vastu/site envelope already chosen
    # by the packer. The envelope has road-side parking/open-space carved out.
    if ref_env:
        x0=float(ref_env.get('x',setback)); y0=float(ref_env.get('y',setback))
        # When a large plot permits a smaller building, keep the block toward
        # the requested road-facing edge instead of centering it arbitrarily.
        if rd == 'north':
            x0=float(ref_env.get('x',setback))+max(0.0, float(ref_env.get('w',inner_depth))-depth)
        elif rd == 'east':
            y0=float(ref_env.get('y',setback))+max(0.0, float(ref_env.get('h',inner_frontage))-frontage)
        elif rd == 'south':
            x0=float(ref_env.get('x',setback))
        elif rd == 'west':
            y0=float(ref_env.get('y',setback))
    elif rd == 'north':
        x0=max(setback,plot_length-setback-depth); y0=max(setback,(plot_width-frontage)/2.0)
    elif rd == 'east':
        x0=max(setback,(plot_length-depth)/2.0); y0=max(setback,plot_width-setback-frontage)
    elif rd == 'west':
        x0=max(setback,(plot_length-depth)/2.0); y0=setback
    else:
        x0=setback; y0=max(setback,(plot_width-frontage)/2.0)
    # Road-facing side gets the first architectural band, but Vastu still
    # determines which rooms share that band.
    # The common local coordinate convention is south->north along x and
    # west->east along y. Keep this physical orientation fixed for Vastu.
    groups_y=[]
    room_rows=[]
    for group in groups:
        room_rows.append(list(group))

    # Allocate row heights from target depth, then fill the complete footprint.
    row_heights=[max(ROOM_DIMS.get(r.get('room_type'), (8.0,8.0))[0] for r in g) for g in room_rows]
    factor=depth/max(sum(row_heights),1e-6)
    row_heights=[max(max(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[0] for r in g), h*factor) for g,h in zip(room_rows,row_heights)]
    total_h=sum(row_heights)
    if total_h>depth+1e-6:
        spare=[max(0.0,h-max(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[0] for r in g)) for g,h in zip(room_rows,row_heights)]
        need=total_h-depth
        for i in range(len(row_heights)):
            take=min(spare[i],need); row_heights[i]-=take; need-=take
        total_h=sum(row_heights)
    if total_h>depth+1e-6:
        depth=total_h
        if depth>inner_depth+1e-6:
            return {'rooms':rooms,'envelope':dict(reference_envelope),'external':external}

    # Round row heights once, then make the final row absorb any rounding delta.
    rounded_rows=[round(h,2) for h in row_heights]
    if rounded_rows:
        rounded_rows[-1]=round(depth-sum(rounded_rows[:-1]),2)
    row_heights=rounded_rows

    new_rooms=[]
    current_x=x0
    for row_idx,(group,row_h) in enumerate(zip(room_rows,row_heights)):
        baselines=[]
        residential_mode = str(building_type or 'custom').lower() in _RESIDENTIAL_VASTU_TYPES
        for r in group:
            min_front = MIN_DIMS.get(r.get('room_type'), (3.0,4.0))[1]
            target_front = ROOM_DIMS.get(r.get('room_type'), (8.0,8.0))[1]
            zone = str(r.get('preferred_zone') or r.get('zone') or 'C').upper()
            # West/NW/SW rooms must remain narrow enough that their centre stays
            # inside the intended western Vastu sector after the row is packed.
            # Spare frontage is allocated to North/East/public rooms instead.
            if residential_mode and zone in {'W','NW','SW'}:
                baselines.append(min_front)
            else:
                baselines.append(max(min_front, target_front))
        # For small rows, choose the actual left-to-right permutation that puts
        # each room closest to its Vastu lateral target. This prevents a wide
        # N-zone living room from being pushed into NE simply because two NW
        # rooms happened to sort before it.
        if len(group) <= 7 and str(building_type or 'custom').lower() not in _RESIDENTIAL_VASTU_TYPES:
            import itertools
            base_total=sum(baselines) or len(group)
            target_widths=[(b+max(0.0,frontage-base_total)*(b/base_total)) for b in baselines] if base_total < frontage else [b*frontage/base_total for b in baselines]
            def perm_score(order):
                pos=0.0; score=0.0
                approx_env={'x':x0,'y':y0,'w':depth,'h':frontage}
                for idx in order:
                    center_rel=(pos+target_widths[idx]/2.0)/max(frontage,1e-6)
                    target=_ZONE_LATERAL.get(str(group[idx].get('preferred_zone') or group[idx].get('zone') or 'C').upper(),0.50)
                    score += (center_rel-target)**2
                    cx=x0+sum(row_heights[:row_idx])+row_h/2.0
                    cy=y0+pos+target_widths[idx]/2.0
                    actual_zone=_preferred_zone_from_point(plot_length,plot_width,cx,cy,approx_env)
                    rtype=group[idx].get('room_type')
                    acceptable=set(ACCEPTABLE_ZONES.get(rtype,[group[idx].get('preferred_zone') or 'C']))
                    if actual_zone not in acceptable:
                        score += 25.0
                    # When multiple rooms share the same preferred lateral band,
                    # place larger living/bedroom spaces before small service rooms
                    # so the service rooms do not push the main room into SE/SW.
                    score += idx * 0.0001
                    pos += target_widths[idx]
                return score
            best_order=min(itertools.permutations(range(len(group))), key=perm_score)
            group=[group[i] for i in best_order]
            baselines=[baselines[i] for i in best_order]
        base_sum=sum(baselines)
        if base_sum>=frontage:
            scale=frontage/base_sum
            widths=[max(MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1],b*scale) for r,b in zip(group,baselines)]
        elif str(building_type or 'custom').lower() in _RESIDENTIAL_VASTU_TYPES:
            # Vastu-first sequential allocation. Each room is sized so its centre
            # moves toward the centre of its assigned Vastu column (W=1/6, C=1/2,
            # E=5/6), while always reserving the minimum frontage for rooms still
            # to come. This prevents kitchens/bathrooms from crossing zone lines.
            min_widths=[MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1] for r in group]
            widths=[]; pos=0.0
            for i,r in enumerate(group[:-1]):
                remaining_min=sum(min_widths[i+1:])
                lo=min_widths[i]
                hi=max(lo, frontage-pos-remaining_min)
                z=str(r.get('preferred_zone') or r.get('zone') or 'C').upper()
                target_center=_ZONE_LATERAL.get(z,0.50)*frontage
                desired=2.0*(target_center-pos)
                # Keep close to the normal room frontage when that still reaches
                # the requested Vastu column; otherwise use the zone constraint.
                normal=max(lo, ROOM_DIMS.get(r.get('room_type'),(8.0,8.0))[1])
                if residential_mode and z in {'W','NW','SW'}:
                    normal=lo
                # Prefer the width that places the room centre on its Vastu
                # column. Fall back to the normal architectural frontage only when
                # the exact centre is impossible because of minimum-size constraints.
                if z in {'W','NW','SW'}:
                    # Keep western rooms at their minimum practical frontage so
                    # multiple W/NW/SW rooms can share the same third without one
                    # wide room pushing the next service room across the centre line.
                    w=lo
                elif lo <= desired <= hi:
                    w=desired
                else:
                    w=max(lo,min(hi,normal))
                widths.append(w); pos+=w
            widths.append(max(min_widths[-1], frontage-pos))
            if len(widths)==1:
                widths=[frontage]
        else:
            extra=frontage-base_sum
            widths=list(baselines)
            raw=[_ZONE_LATERAL.get(str(r.get('preferred_zone') or r.get('zone') or 'C').upper(),0.50) for r in group]
            order=sorted(range(len(group)),key=lambda i:(raw[i],i),reverse=True)
            caps=[max(baselines[i],baselines[i]*2.25) for i in range(len(group))]
            remaining_extra=extra
            for i in order:
                if remaining_extra<=1e-9: break
                add=min(remaining_extra,max(0.0,caps[i]-widths[i]))
                widths[i]+=add; remaining_extra-=add
            if remaining_extra>1e-9:
                widths[-1]+=remaining_extra

        # Final lateral correction for strict east-side rooms. We first shrink an
        # over-wide kitchen, then borrow spare frontage from rooms after it, and
        # only then expand the compact footprint. Every adjustment preserves room
        # minimums and keeps the rooms joined on the same shared wall line.
        if residential_mode and len(group) > 1:
            east_safe=0.668
            for i,r in enumerate(group):
                z=str(r.get('preferred_zone') or r.get('zone') or 'C').upper()
                if r.get('room_type') not in {'kitchen','restaurant_kitchen'} or z not in {'SE','E','NE'} or i==0:
                    continue
                pos=sum(widths[:i])
                center=pos+widths[i]/2.0
                shift_needed=max(0.0,east_safe*frontage-center)
                if shift_needed<=1e-6:
                    continue

                kitchen_min=MIN_DIMS.get(r.get('room_type'),(3.0,4.0))[1]
                # Shrinking the kitchen by d and expanding its predecessor by d
                # moves the kitchen centre by d/2.
                kitchen_available=max(0.0,widths[i]-kitchen_min)
                delta_k=min(2.0*shift_needed,kitchen_available)
                if delta_k>1e-9:
                    widths[i]-=delta_k
                    widths[i-1]+=delta_k
                    shift_needed-=delta_k/2.0

                # Now move the shared boundary by borrowing spare frontage from
                # east-side rooms after the kitchen. This shifts the kitchen centre
                # by the full transferred amount without changing its width.
                if shift_needed>1e-6:
                    remaining=shift_needed
                    moved=0.0
                    for j in range(len(group)-1,i,-1):
                        min_j=MIN_DIMS.get(group[j].get('room_type'),(3.0,4.0))[1]
                        take=min(remaining,max(0.0,widths[j]-min_j))
                        if take>1e-9:
                            widths[j]-=take; remaining-=take; moved+=take
                        if remaining<=1e-9:
                            break
                    if moved>1e-9:
                        widths[i-1]+=moved
                        shift_needed-=moved

                # If the existing row cannot provide enough movable width, enlarge
                # the compact frontage only when the plot envelope permits it.
                if shift_needed>1e-6:
                    expand=shift_needed/max(1e-6,1.0-east_safe)
                    if frontage+expand <= inner_frontage+1e-6:
                        frontage=round(frontage+expand,2)
                        widths[i-1]+=expand

        rounded_widths=[round(w,2) for w in widths]
        if rounded_widths:
            rounded_widths[-1]=round(frontage-sum(rounded_widths[:-1]),2)
        y=y0
        row_x=round(x0+sum(row_heights[:row_idx]),2)
        for idx,(room,w) in enumerate(zip(group,rounded_widths)):
            room_width=round(row_h,2)
            # Force the final room in each row to terminate exactly at the
            # common east/west edge after decimal rounding.
            if idx==len(group)-1:
                w=round(y0+frontage-y,2)
            new=dict(room)
            new.update({'x':row_x,'y':round(y,2),'w':room_width,'h':round(w,2)})
            new_rooms.append(new)
            y=round(y+w,2)
        current_x=round(row_x+row_h,2)

    # Recompute actual Vastu zones from the final physical footprint.
    envelope={'x':round(x0,2),'y':round(y0,2),'w':round(depth,2),'h':round(frontage,2)}
    for r in new_rooms:
        actual=_preferred_zone_from_point(plot_length,plot_width,r['x']+r['w']/2,r['y']+r['h']/2,envelope)
        r['zone']=actual
        r['matched']=actual==r.get('preferred_zone')
        try:
            r['vastu_acceptable']=actual in set(ACCEPTABLE_ZONES.get(r.get('room_type'),[r.get('preferred_zone')]))
        except Exception:
            r['vastu_acceptable']=r['matched']
        r['door_side']=DOOR_SIDE_FOR_ZONE.get(actual,r.get('door_side','S'))

    # Re-seat external/site spaces against the road-facing plot edge after the
    # indoor rooms have been compacted, so parking/loading/garden never overlap
    # the joined building block.
    reseated=[]
    for original in external:
        reservation=_front_reservation(
            original.get('room_type','parking'), plot_length, plot_width,
            road_direction,
            new_rooms + reseated, setback
        )
        if reservation is None:
            reservation={k: original[k] for k in ('x','y','w','h')}
        ext=dict(original)
        ext.update({k:round(v,2) for k,v in reservation.items()})
        reseated.append(ext)
    all_rooms=new_rooms+reseated
    return {'rooms': all_rooms, 'envelope': envelope, 'external': reseated}

def _derive_circulation_rects(rooms: List[dict], footprint: dict, min_dim: float = 0.60) -> list[dict]:
    """Fill every internal rectangular void with explicit circulation space.

    The void is not counted as a requested room. It becomes an intentional
    PASSAGE/CIRCULATION element so generated plans never show unexplained blank
    gaps between rooms. Small wall-tolerance slivers are omitted because the
    wall thickness visually closes them.
    """
    if not rooms:
        return []
    xs = sorted({round(float(v), 4) for r in rooms for v in (r['x'], r['x'] + r['w'])})
    ys = sorted({round(float(v), 4) for r in rooms for v in (r['y'], r['y'] + r['h'])})
    if len(xs) < 2 or len(ys) < 2:
        return []
    bx, by = float(footprint['x']), float(footprint['y'])
    bw, bh = float(footprint['w']), float(footprint['h'])
    bx2, by2 = bx + bw, by + bh
    # Keep only grid lines inside the actual occupied footprint.
    xs = [x for x in xs if bx - 1e-5 <= x <= bx2 + 1e-5]
    ys = [y for y in ys if by - 1e-5 <= y <= by2 + 1e-5]
    cells=[]
    for yi in range(len(ys)-1):
        y1,y2=ys[yi],ys[yi+1]
        if y2-y1 < min_dim:
            continue
        row=[]
        for xi in range(len(xs)-1):
            x1,x2=xs[xi],xs[xi+1]
            if x2-x1 < min_dim:
                continue
            cx,cy=(x1+x2)/2,(y1+y2)/2
            covered=any(r['x'] <= cx < r['x']+r['w']-1e-7 and r['y'] <= cy < r['y']+r['h']-1e-7 for r in rooms)
            if not covered:
                row.append((x1,x2))
        # collapse consecutive void cells across x
        runs=[]
        if row:
            sx,ex=row[0]
            for x1,x2 in row[1:]:
                if abs(x1-ex) <= 1e-5:
                    ex=x2
                else:
                    runs.append((sx,ex)); sx,ex=x1,x2
            runs.append((sx,ex))
        for x1,x2 in runs:
            if x2-x1 >= min_dim:
                cells.append({'x':x1,'y':y1,'w':x2-x1,'h':y2-y1})
    # Merge equal-width vertical runs and equal-height horizontal runs.
    merged=[dict(c) for c in cells]
    changed=True
    while changed:
        changed=False
        out=[]
        for c in sorted(merged, key=lambda r:(round(r['y'],4),round(r['x'],4),round(r['h'],4),round(r['w'],4))):
            did_merge=False
            for m in out:
                horizontal=(abs(m['y']-c['y']) <= 1e-5 and abs(m['h']-c['h']) <= 1e-5
                            and abs(m['x']+m['w']-c['x']) <= 1e-5)
                vertical=(abs(m['x']-c['x']) <= 1e-5 and abs(m['w']-c['w']) <= 1e-5
                          and abs(m['y']+m['h']-c['y']) <= 1e-5)
                if horizontal:
                    m['w'] += c['w']; did_merge=True; changed=True; break
                if vertical:
                    m['h'] += c['h']; did_merge=True; changed=True; break
            if not did_merge:
                out.append(dict(c))
        merged=out
    # Filter out negligible wall-tolerance fragments.
    return [
        {k: round(v,2) for k,v in r.items()}
        for r in merged if r['w'] >= min_dim and r['h'] >= min_dim and r['w']*r['h'] >= 2.0
    ]


def _finalize_layout_geometry(layout: Dict) -> Dict:
    """Apply the no-gap finalization to any generated floor layout."""
    rooms = layout.get('rooms') or []
    if not rooms:
        layout['circulation'] = []
        layout['gap_report'] = {'max_internal_gap_ft': 0.0, 'gaps': []}
        return layout
    env = layout.get('building_envelope') or layout.get('occupied_envelope')
    if not env:
        min_x=min(r['x'] for r in rooms); min_y=min(r['y'] for r in rooms)
        max_x=max(r['x']+r['w'] for r in rooms); max_y=max(r['y']+r['h'] for r in rooms)
        env={'x':min_x,'y':min_y,'w':max_x-min_x,'h':max_y-min_y}
    else:
        env=dict(env)
    # Replace unexplained blank internal voids with a true joined room tile.
    # Rooms themselves occupy the footprint; no PASSAGE / CIRCULATION rectangles
    # are synthesized inside the building.
    original_rooms=[dict(r) for r in rooms]
    original_env=dict(env)
    joined=_retile_rooms_without_internal_gaps(
        rooms, float(layout['plot']['length']), float(layout['plot']['width']), env,
        _setback_for_plot(float(layout['plot']['length']), float(layout['plot']['width'])),
        str(layout.get('road_direction') or 'South'),
        str(layout.get('building_type') or 'custom')
    )
    rooms[:] = joined['rooms']
    env = joined['envelope']

    # The joined tiler is preferred, but site reservations such as parking may
    # make a retile geometrically impossible on a tight plot. In that case keep
    # the already-valid authoritative room geometry instead of rejecting the
    # entire plan. This fallback never changes room dimensions/counts.
    checks = run_all_checks(rooms, float(layout['plot']['length']), float(layout['plot']['width']))
    if not checks['passed']:
        rooms[:] = original_rooms
        env = original_env
        checks = run_all_checks(rooms, float(layout['plot']['length']), float(layout['plot']['width']))
        if not checks['passed']:
            raise ValueError('Generated geometry failed hard collision/boundary validation.')
        layout['alignment_fallback']='AUTHORITATIVE_PRE_RETile_GEOMETRY'
    else:
        layout['alignment_fallback']=None
    # Tighten actual footprint around final rooms.
    min_x=min(r['x'] for r in rooms); min_y=min(r['y'] for r in rooms)
    max_x=max(r['x']+r['w'] for r in rooms); max_y=max(r['y']+r['h'] for r in rooms)
    footprint={'x':round(min_x,2),'y':round(min_y,2),'w':round(max_x-min_x,2),'h':round(max_y-min_y,2)}
    layout['occupied_envelope']=footprint
    layout['building_envelope']=footprint
    # No synthetic internal corridors are generated. Any open space outside the
    # occupied footprint remains true exterior/setback/open space.
    layout['circulation']=[]
    actual_gap=_gap_report(rooms)
    layout['gap_report']=actual_gap
    report = _alignment_report(rooms)
    report['layout_grid'] = 'JOINED_ORTHOGONAL_BANDS'
    report['joined_room_count'] = len([r for r in rooms if r.get('room_type') not in {'parking','garden','loading_bay'}])
    layout['alignment_report'] = report
    layout['no_unplanned_gaps']=actual_gap.get('max_internal_gap_ft',0.0) <= 0.20
    return layout

def _gap_report(rooms: List[dict], tolerance: float = 0.20) -> dict:
    """Report only true empty separations between adjacent room edges.

    A room hidden behind another room in the same projection is not a gap; the
    previous pairwise implementation incorrectly reported those cases. We only
    examine the nearest room on each side along overlapping projections.
    """
    if len(rooms) < 2:
        return {'max_internal_gap_ft': 0.0, 'gaps': []}
    gaps=[]
    for a in rooms:
        for axis in ('x','y'):
            a_start=a[axis]; a_end=a_start+a['w' if axis=='x' else 'h']
            cross_start=a['y'] if axis=='x' else a['x']
            cross_end=cross_start+(a['h'] if axis=='x' else a['w'])
            nearest=None
            nearest_pair=None
            for b in rooms:
                if b is a or b.get('room_type') in {'parking','garden','loading_bay'}:
                    continue
                b_start=b[axis]; b_end=b_start+b['w' if axis=='x' else 'h']
                b_cross_start=b['y'] if axis=='x' else b['x']
                b_cross_end=b_cross_start+(b['h'] if axis=='x' else b['w'])
                overlap=min(cross_end,b_cross_end)-max(cross_start,b_cross_start)
                if overlap <= tolerance:
                    continue
                if b_start >= a_end-tolerance:
                    gap=b_start-a_end
                    if nearest is None or gap<nearest:
                        nearest=gap; nearest_pair=b
                elif a_start >= b_end-tolerance:
                    gap=a_start-b_end
                    if nearest is None or gap<nearest:
                        nearest=gap; nearest_pair=b
            if nearest is not None and nearest > tolerance:
                gaps.append({'between':(a['label'],nearest_pair['label']),'gap_ft':round(nearest,2),'axis':axis})
    # Deduplicate symmetric pairs.
    unique=[]; seen=set()
    for g in gaps:
        key=(frozenset(g['between']),g['axis'])
        if key not in seen:
            seen.add(key); unique.append(g)
    return {'max_internal_gap_ft':round(max([g['gap_ft'] for g in unique] or [0.0]),2),'gaps':unique}

def _convert_local_rect(road_direction: str, setback: float, plot_length: float, plot_width: float,
                        depth: float, frontage: float, d: float, f: float) -> dict:
    """Convert local (depth/frontage) planning coordinates to project x/y/w/h."""
    rd = str(road_direction or 'South').strip().lower()
    if rd == 'north':
        return {'x': plot_length - setback - d - depth, 'y': setback + f, 'w': depth, 'h': frontage}
    if rd == 'east':
        return {'x': setback + f, 'y': plot_width - setback - d - depth, 'w': frontage, 'h': depth}
    if rd == 'west':
        return {'x': setback + f, 'y': setback + d, 'w': frontage, 'h': depth}
    return {'x': setback + d, 'y': setback + f, 'w': depth, 'h': frontage}


def _shelf_layout(plot_length: float, plot_width: float, allocations: List[RoomAllocation],
                  setback: float, road_direction: str, floors: int, building_type: str = 'custom') -> Dict | None:
    """Build a compact, dimension-first room plan using frontage shelves.

    Unlike the previous equal-column grid, rooms keep practical target/minimum
    dimensions and are grouped into rows that fit the available frontage.
    This prevents large plots from producing 70-ft-wide bedrooms and makes
    small plots fail clearly when their requested spaces genuinely cannot fit.
    """
    rd = str(road_direction or 'South').strip().lower()
    # For north/south roads, plot x is depth and plot y is frontage. For
    # east/west roads those axes are swapped because the road-facing edge is
    # the east/west side of the plot.
    if rd in {'east', 'west'}:
        inner_depth = plot_width - 2 * setback
        inner_frontage = plot_length - 2 * setback
    else:
        inner_depth = plot_length - 2 * setback
        inner_frontage = plot_width - 2 * setback
    if inner_depth < 12 or inner_frontage < 12 or not allocations:
        return None

    # Front-to-back order chosen to form useful residential bands. Service
    # rooms are grouped later so kitchens/baths can share a compact row.
    type_order = {
        'parking': 0, 'garden': 0, 'pooja_room': 0,
        'living_room': 1, 'dining_room': 2,
        'master_bedroom': 3, 'bedroom': 4,
        'staircase': 5,
        'kitchen': 6, 'bathroom': 7,
        'study_room': 8, 'store_room': 9, 'balcony': 10,
    }
    def _preferred_local_position(item):
        planning_env = {
            'x': _setback_for_plot(plot_length, plot_width),
            'y': _setback_for_plot(plot_length, plot_width),
            'w': max(1.0, plot_length - 2 * _setback_for_plot(plot_length, plot_width)),
            'h': max(1.0, plot_width - 2 * _setback_for_plot(plot_length, plot_width)),
        }
        zx, zy = _zone_center(plot_length, plot_width, item.preferred_zone or item.zone or 'C', planning_env)
        if rd == 'east':
            return plot_width - zy, zx
        if rd == 'west':
            return zy, zx
        if rd == 'north':
            return plot_length - zx, zy
        return zx, zy

    # Vastu-aware ordering: rooms with stronger Vastu requirements are placed
    # first and then arranged in the order of their preferred local zone. This
    # prevents the packer from treating the Vastu assignment as decorative data.
    vastu_first = {'pooja_room', 'master_bedroom', 'kitchen', 'living_room', 'bedroom', 'bathroom', 'toilet', 'restroom'}
    ordered = sorted(
        allocations,
        key=lambda a: (
            0 if a.room_type in vastu_first else 1,
            _preferred_local_position(a)[0],
            _preferred_local_position(a)[1],
            -_room_vastu_weight(a.room_type),
            -a.target_area,
            type_order.get(a.room_type, 20),
            a.label,
        ),
    )

    # Do not allow a large plot to turn the building into one extremely long
    # hallway of rooms. The room set determines a preferred building frontage;
    # the actual plot frontage is still respected as the hard maximum.
    room_area_target = sum(
        ROOM_DIMS.get(a.room_type, (8.0, 8.0))[0] * ROOM_DIMS.get(a.room_type, (8.0, 8.0))[1]
        for a in allocations
    )
    preferred_frontage = max(20.0, (room_area_target * 1.30 * 1.45) ** 0.5)
    frontage_limit = min(inner_frontage, preferred_frontage)

    # Try progressively smaller target dimensions. The absolute minimums remain
    # fixed so the algorithm never silently creates unusably tiny rooms.
    for scale in [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
        rows = []
        remaining = list(ordered)
        depth_used = 0.0
        possible = True

        while remaining:
            seed = remaining.pop(0)
            sd, sf = _orient_for_frontage(seed.room_type, scale, frontage_limit)
            if sf > inner_frontage + 1e-6 or sd > inner_depth - depth_used + 1e-6:
                possible = False
                break
            row = [(seed, sd, sf)]
            frontage_used = sf
            row_depth = sd

            # Add the next item that best fills the unused frontage without
            # increasing row depth unnecessarily.
            while remaining:
                candidates = []
                for idx, item in enumerate(remaining):
                    d, f = _orient_for_frontage(item.room_type, scale, frontage_limit - frontage_used)
                    if f <= frontage_limit - frontage_used + 1e-6 and d <= inner_depth - depth_used + 1e-6:
                        candidates.append((max(row_depth, d) - row_depth, -(f), idx, item, d, f))
                if not candidates:
                    break
                _, _, idx, item, d, f = min(candidates, key=lambda v: (v[0], v[1], v[2]))
                remaining.pop(idx)
                row.append((item, d, f))
                frontage_used += f
                row_depth = max(row_depth, d)

            rows.append(row)
            depth_used += row_depth

        if not possible or depth_used > inner_depth + 1e-6:
            continue

        # Center the compact building footprint along the frontage. Keep the
        # depth immediately behind the road-facing setback so the entrance side
        # remains visually obvious.
        building_depth = depth_used
        building_frontage = max(sum(f for _, _, f in row) for row in rows)
        front_offset = max(0.0, (inner_frontage - building_frontage) / 2.0)

        rooms = []
        d_cursor = 0.0
        for row in rows:
            row_depth = max(d for _, d, _ in row)
            row_frontage = sum(f for _, _, f in row)
            f_cursor = front_offset

            # Place each room near its Vastu target within the available shelf.
            # The shelf remains clean/aligned, but room sequence is adjusted by
            # candidate score rather than blindly following input order.
            row_items = []
            for item, depth, frontage in row:
                row_items.append((item, depth, frontage))
            if len(row_items) > 1:
                candidates = []
                for perm in (row_items, list(reversed(row_items))):
                    score = 0.0
                    cursor = f_cursor
                    for item, depth, frontage in perm:
                        rect = _convert_local_rect(rd, setback, plot_length, plot_width,
                                                   depth, frontage, d_cursor, cursor)
                        score += _placement_score(plot_length, plot_width, item, rect)
                        cursor += frontage
                    candidates.append((score, perm))
                row_items = min(candidates, key=lambda x: x[0])[1]

            for item, depth, frontage in row_items:
                rect = _convert_local_rect(rd, setback, plot_length, plot_width,
                                           depth, frontage, d_cursor, f_cursor)
                room = {
                    'room_type': item.room_type,
                    'label': item.label,
                    'preferred_zone': item.preferred_zone,
                    'zone': item.zone,
                    'matched': item.matched,
                    **{k: round(v, 2) for k, v in rect.items()},
                    'door_side': DOOR_SIDE_FOR_ZONE.get(item.zone, 'S'),
                }
                rooms.append(room)
                f_cursor += frontage
            d_cursor += row_depth

        checks = run_all_checks(rooms, plot_length, plot_width)
        if checks['collisions'] or checks['boundary_violations']:
            continue

        min_x = min(r['x'] for r in rooms)
        min_y = min(r['y'] for r in rooms)
        max_x = max(r['x'] + r['w'] for r in rooms)
        max_y = max(r['y'] + r['h'] for r in rooms)
        footprint = {'x': min_x, 'y': min_y, 'w': max_x - min_x, 'h': max_y - min_y}
        vastu_active = any((a.preferred_zone or 'C') != 'C' for a in allocations)

        # Re-resolve Vastu zone from actual geometry so the API does not claim a
        # preferred-zone match that the final coordinates do not support. Zones
        # are measured against the plot orientation (south/west to north/east),
        # which is the same reference used by the placement objective.
        if vastu_active:
            for r in rooms:
                actual = _preferred_zone_from_point(plot_length, plot_width,
                                                     r['x'] + r['w'] / 2.0,
                                                     r['y'] + r['h'] / 2.0,
                                                     None)
                r['zone'] = actual
                preferred = r.get('preferred_zone')
                acceptable = {preferred}
                try:
                    from vastu.vastu_rules import ACCEPTABLE_ZONES
                    acceptable = set(ACCEPTABLE_ZONES.get(r.get('room_type'), [preferred]))
                except Exception:
                    pass
                # `matched` remains strict: True means the final room centre is in
                # the preferred zone, not merely an acceptable fallback zone.
                r['matched'] = actual == preferred
                r['vastu_acceptable'] = actual in acceptable
                r['door_side'] = DOOR_SIDE_FOR_ZONE.get(actual, r.get('door_side', 'S'))

        checks = run_all_checks(rooms, plot_length, plot_width)
        if checks['collisions'] or checks['boundary_violations']:
            continue

        envelope = {
            'x': round(footprint['x'], 2), 'y': round(footprint['y'], 2),
            'w': round(footprint['w'], 2), 'h': round(footprint['h'], 2),
        }
        return _finalize_layout_geometry({
            'building_type': str(building_type or 'custom').strip().lower(),
            'plot': {'length': round(plot_length, 2), 'width': round(plot_width, 2),
                     'area': round(plot_length * plot_width, 2)},
            'floors': floors, 'road_direction': road_direction,
            'setback_ft': round(setback, 2),
            'building_envelope': envelope,
            'occupied_envelope': envelope,
            'rooms': rooms,
            'collisions': checks['collisions'],
            'boundary_violations': checks['boundary_violations'],
            'passed_checks': checks['passed'],
            'dimension_scale': scale,
            'building_footprint_sqft': round(envelope['w'] * envelope['h'], 2),
            'vastu_match_count': sum(1 for r in rooms if r.get('matched')),
            'vastu_acceptable_count': sum(1 for r in rooms if r.get('vastu_acceptable')),
            'vastu_score': round(_actual_zone_score(plot_length, plot_width, rooms), 4),
            'vastu_match_percent': round(100.0 * sum(1 for r in rooms if r.get('matched')) / max(len(rooms), 1), 1),
        })
    return None

def _optimize_vastu_positions(rooms: List[dict], plot_length: float, plot_width: float,
                               reference_envelope: dict, iterations: int = 2) -> None:
    """Greedy local-improvement pass that moves rooms toward open preferred zones."""
    from vastu.vastu_rules import ACCEPTABLE_ZONES
    step = 0.5 if max(plot_length, plot_width) <= 60 else (1.0 if max(plot_length, plot_width) <= 120 else 2.0)
    diagonal = max((reference_envelope['w'] ** 2 + reference_envelope['h'] ** 2) ** 0.5, 1.0)

    def candidate_positions(room, zone):
        zx, zy, zw, zh = _zone_bounds(plot_length, plot_width, zone, reference_envelope)
        max_x = min(reference_envelope['x'] + reference_envelope['w'] - room['w'], zx + zw - room['w'])
        max_y = min(reference_envelope['y'] + reference_envelope['h'] - room['h'], zy + zh - room['h'])
        min_x = max(reference_envelope['x'], zx)
        min_y = max(reference_envelope['y'], zy)
        if max_x < min_x or max_y < min_y:
            return []
        xs=[]; x=min_x
        while x <= max_x + 1e-6 and len(xs) < 300:
            xs.append(round(x / step) * step); x += step
        ys=[]; y=min_y
        while y <= max_y + 1e-6 and len(ys) < 300:
            ys.append(round(y / step) * step); y += step
        # Include exact corners/centre even when step rounding skips them.
        xs.extend([min_x, max_x, (min_x + max_x) / 2.0])
        ys.extend([min_y, max_y, (min_y + max_y) / 2.0])
        return [(max(min_x, min(max_x, xx)), max(min_y, min(max_y, yy))) for xx in dict.fromkeys(xs) for yy in dict.fromkeys(ys)]

    for _ in range(max(1, iterations)):
        ordered = sorted(rooms, key=lambda r: (-_room_vastu_weight(r.get('room_type', '')), 0 if not r.get('matched') else 1))
        for room in ordered:
            preferred = room.get('preferred_zone') or 'C'
            current = {'x': room['x'], 'y': room['y'], 'w': room['w'], 'h': room['h']}
            current_score = _placement_score(plot_length, plot_width, type('A', (), {
                'room_type': room.get('room_type'), 'preferred_zone': preferred, 'zone': room.get('zone')
            })(), current, reference_envelope=reference_envelope)
            if room.get('matched'):
                continue
            best = None
            for x, y in candidate_positions(room, preferred):
                candidate = {'x': x, 'y': y, 'w': room['w'], 'h': room['h']}
                if any(other is not room and _rect_intersects(candidate, other) for other in rooms):
                    continue
                displacement = ((x - room['x']) ** 2 + (y - room['y']) ** 2) ** 0.5 / diagonal
                score = _placement_score(plot_length, plot_width, type('A', (), {
                    'room_type': room.get('room_type'), 'preferred_zone': preferred, 'zone': room.get('zone')
                })(), candidate, reference_envelope=reference_envelope) + displacement * 0.04
                if best is None or score < best[0]:
                    best = (score, candidate)
            if best and best[0] + 1e-6 < current_score:
                room['x'] = round(best[1]['x'], 2)
                room['y'] = round(best[1]['y'], 2)

        for room in rooms:
            actual = _preferred_zone_from_point(
                plot_length, plot_width,
                room['x'] + room['w'] / 2.0, room['y'] + room['h'] / 2.0,
                reference_envelope,
            )
            room['zone'] = actual
            room['matched'] = actual == room.get('preferred_zone')
            room['vastu_acceptable'] = actual in set(ACCEPTABLE_ZONES.get(room.get('room_type'), [room.get('preferred_zone')]))
            room['door_side'] = DOOR_SIDE_FOR_ZONE.get(actual, room.get('door_side', 'S'))


def _compact_vastu_envelope(plot_length: float, plot_width: float,
                             allocations: List[RoomAllocation], setback: float,
                             road_direction: str) -> dict:
    """Choose a compact building footprint before Vastu room placement."""
    inner_l = max(1.0, plot_length - 2 * setback)
    inner_w = max(1.0, plot_width - 2 * setback)
    indoor = [a for a in allocations if a.room_type not in {'parking', 'garden'}]
    target_area = sum(
        ROOM_DIMS.get(a.room_type, (8.0, 8.0))[0] * ROOM_DIMS.get(a.room_type, (8.0, 8.0))[1]
        for a in indoor
    )
    minimum_area = sum(
        MIN_DIMS.get(a.room_type, (3.0, 4.0))[0] * MIN_DIMS.get(a.room_type, (3.0, 4.0))[1]
        for a in indoor
    )
    area_factor = 1.60 if plot_length * plot_width >= 10000 else 1.35
    required = max(target_area * area_factor, minimum_area * 1.55, 18.0 * 18.0)
    # Preserve the plot's approximate depth/frontage proportion but keep the
    # room block substantially smaller than a very large plot.
    aspect = inner_l / max(inner_w, 1e-6)
    depth = (required * aspect) ** 0.5
    width = required / max(depth, 1e-6)
    depth = min(inner_l, max(16.0, depth))
    width = min(inner_w, max(16.0, width))
    area_now = depth * width
    if area_now < required:
        missing = required / max(area_now, 1e-6)
        width = min(inner_w, width * missing ** 0.5)
        depth = min(inner_l, depth * missing ** 0.5)

    rd = str(road_direction or 'South').strip().lower()
    if rd == 'south':
        x = setback
        y = setback + max(0.0, (inner_w - width) / 2.0)
    elif rd == 'north':
        x = plot_length - setback - depth
        y = setback + max(0.0, (inner_w - width) / 2.0)
    elif rd == 'east':
        x = setback + max(0.0, (inner_l - depth) / 2.0)
        y = plot_width - setback - width
    else:
        x = setback + max(0.0, (inner_l - depth) / 2.0)
        y = setback
    return {
        'x': round(x, 2), 'y': round(y, 2),
        'w': round(min(depth, inner_l), 2), 'h': round(min(width, inner_w), 2),
    }


def _vastu_pack_layout(plot_length: float, plot_width: float,
                        allocations: List[RoomAllocation], setback: float,
                        road_direction: str, floors: int, building_type: str = 'custom') -> Dict | None:
    """Generate a Vastu-directed layout using the authoritative geometry packer.

    This path is selected before the legacy shelf/grid planners when Vastu is
    enabled. Rooms are placed close to their preferred directional zones first,
    while the collision/boundary/minimum-dimension rules remain hard constraints.
    """
    plot_area = plot_length * plot_width

    # Reserve road-side external areas first. Parking/loading/garden are site
    # elements, not indoor Vastu rooms. Removing their strip from the indoor
    # envelope prevents the Vastu packer from failing and falling back to the
    # generic shelf layout.
    site_allocations = [a for a in allocations if a.room_type in {"parking", "garden", "loading_bay"}]
    remaining = [a for a in allocations if a.room_type not in {"parking", "garden", "loading_bay"}]
    site_probe=[]
    for item in site_allocations:
        reservation=_front_reservation(item.room_type, plot_length, plot_width, road_direction, site_probe, setback)
        if reservation is None:
            return None
        site_probe.append(reservation)

    if plot_area >= 5000:
        compact_env = _compact_vastu_envelope(plot_length, plot_width, remaining + [], setback, road_direction)
    else:
        compact_env = {
            'x': setback, 'y': setback,
            'w': max(1.0, plot_length - 2 * setback),
            'h': max(1.0, plot_width - 2 * setback),
        }

    # Carve a one-foot planning buffer between the indoor footprint and all
    # road-side site spaces, while leaving the opposite side available.
    buffer=1.0
    rd=str(road_direction or 'South').strip().lower()
    if site_probe:
        if rd == 'east':
            limit=min(r['y'] for r in site_probe) - buffer
            compact_env['h']=min(compact_env['h'], max(1.0, limit-compact_env['y']))
        elif rd == 'west':
            limit=max(r['y']+r['h'] for r in site_probe) + buffer
            new_y=max(compact_env['y'], limit)
            compact_env['h']=max(1.0, compact_env['y']+compact_env['h']-new_y)
            compact_env['y']=new_y
        elif rd == 'south':
            # _front_reservation anchors 'south' road-side spaces at front_x =
            # x_min (the same edge compact_env starts from), so the indoor
            # block must be pushed forward past them, not shrunk from x_min.
            limit=max(r['x']+r['w'] for r in site_probe) + buffer
            new_x=max(compact_env['x'], limit)
            compact_env['w']=max(1.0, compact_env['x']+compact_env['w']-new_x)
            compact_env['x']=new_x
        elif rd == 'north':
            # 'north' road-side spaces are anchored at front_x = x_max - w
            # (the far edge compact_env extends to), so only the width needs
            # to shrink; compact_env['x'] (the near edge) is unaffected.
            limit=min(r['x'] for r in site_probe) - buffer
            compact_env['w']=min(compact_env['w'], max(1.0, limit-compact_env['x']))

    if compact_env['w'] <= 8 or compact_env['h'] <= 8:
        return None

    # Residential plans use the joined-band Vastu tiler directly. This keeps
    # master/kitchen/pooja/living/service rooms in their directional bands and
    # prevents the generic greedy shelf fallback from undoing the Vastu plan.
    if str(building_type or 'custom').lower() in _RESIDENTIAL_VASTU_TYPES:
        placeholders=[]
        for item in remaining:
            placeholders.append({
                'room_type': item.room_type, 'label': item.label,
                'zone': item.zone, 'preferred_zone': item.preferred_zone,
                'matched': item.matched, 'door_side': DOOR_SIDE_FOR_ZONE.get(item.zone,'S'),
            })
        joined=_retile_rooms_without_internal_gaps(
            placeholders + [
                {
                    'room_type': item.room_type, 'label': item.label, 'zone': item.zone,
                    'preferred_zone': item.preferred_zone, 'matched': item.matched,
                    **{k:round(v,2) for k,v in reservation.items()}
                }
                for item,reservation in zip(site_allocations, site_probe)
            ],
            plot_length, plot_width, compact_env, setback, road_direction, building_type=str(building_type or 'custom').lower()
        )
        placed_all=joined.get('rooms',[])
        # When the joined-band tiler cannot fit the requested rooms inside the
        # target envelope, it bails out by returning the untouched input room
        # placeholders (see the early `return` statements above), which have no
        # x/y/w/h yet. Treat that as a retile failure and fall through to the
        # general Vastu packer below instead of crashing on the missing keys.
        retile_ok = bool(placed_all) and all(
            all(k in r for k in ('x', 'y', 'w', 'h')) for r in placed_all
        )
        hard = run_all_checks(placed_all, plot_length, plot_width) if retile_ok else {'passed': False}
        if hard['passed']:
            # Recalculate Vastu zones against the final indoor footprint.
            indoor=[r for r in placed_all if r.get('room_type') not in {'parking','garden','loading_bay'}]
            final_env=joined.get('envelope') or compact_env
            for r in indoor:
                actual=_preferred_zone_from_point(plot_length,plot_width,r['x']+r['w']/2,r['y']+r['h']/2,final_env)
                r['zone']=actual; r['matched']=actual==r.get('preferred_zone')
                r['vastu_acceptable']=actual in set(ACCEPTABLE_ZONES.get(r.get('room_type'),[r.get('preferred_zone')]))
                r['door_side']=DOOR_SIDE_FOR_ZONE.get(actual,r.get('door_side','S'))
            if all(r.get('vastu_acceptable',False) for r in indoor):
                env=joined.get('envelope') or compact_env
                return {
                    'plot': {'length':round(plot_length,2),'width':round(plot_width,2),'area':round(plot_length*plot_width,2)},
                    'floors':floors,'road_direction':road_direction,'setback_ft':round(setback,2),
                    'building_envelope':env,'occupied_envelope':env,'rooms':placed_all,
                    'collisions':hard['collisions'],'boundary_violations':hard['boundary_violations'],
                    'dimension_violations':hard.get('dimension_violations',[]),'passed_checks':True,
                    'vastu_match_count':sum(1 for r in indoor if r.get('matched')),
                    'vastu_acceptable_count':sum(1 for r in indoor if r.get('vastu_acceptable')),
                    'vastu_score':round(_actual_zone_score(plot_length,plot_width,indoor),4),
                    'vastu_match_percent':round(100*sum(1 for r in indoor if r.get('matched'))/max(len(indoor),1),1),
                    'vastu_enabled':True,
                    'vastu_layout_engine':'Traditional Vastu Zone Assignment + Vastu Band Beam Search + Joined Orthogonal Room Tiling',
                    'building_type':str(building_type or 'custom').strip().lower(),
                    'gap_report':_gap_report(placed_all),
                    'no_unplanned_gaps':_gap_report(placed_all).get('max_internal_gap_ft',0.0) <= 0.20,
                    'alignment_report':{**_alignment_report(placed_all), 'layout_grid':'JOINED_ORTHOGONAL_BANDS', 'joined_room_count':len(indoor)},
                }
        # Fall through to the general Vastu packer for genuinely tight/irregular
        # cases; it may find a valid compact arrangement when the band tiler cannot.

    occupied: List[dict] = []
    try:
        placed = _pack_remaining(
            remaining, plot_length, plot_width, setback, road_direction, occupied,
            allowed_envelope=compact_env
        )
    except ValueError:
        return None

    checks = run_all_checks(placed, plot_length, plot_width)
    if not checks['passed']:
        return None

    _optimize_vastu_positions(placed, plot_length, plot_width, compact_env, iterations=2)
    checks = run_all_checks(placed, plot_length, plot_width)
    if not checks['passed']:
        return None

    for r in placed:
        actual = _preferred_zone_from_point(
            plot_length, plot_width,
            r['x'] + r['w'] / 2.0, r['y'] + r['h'] / 2.0,
            compact_env,
        )
        r['zone'] = actual
        r['matched'] = actual == r.get('preferred_zone')
        try:
            r['vastu_acceptable'] = actual in set(
                ACCEPTABLE_ZONES.get(r.get('room_type'), [r.get('preferred_zone')])
            )
        except Exception:
            r['vastu_acceptable'] = r['matched']
        r['door_side'] = DOOR_SIDE_FOR_ZONE.get(actual, r.get('door_side', 'S'))

    # Re-seat parking/garden/loading outside the indoor building block.
    site_rooms=[]
    for item in site_allocations:
        reservation=_front_reservation(item.room_type, plot_length, plot_width, road_direction, placed+site_rooms, setback)
        if reservation is None:
            # Site space is requested; return None rather than allowing overlap.
            return None
        site_rooms.append({
            'room_type': item.room_type, 'label': item.label, 'zone': item.zone,
            'preferred_zone': item.preferred_zone, 'matched': item.matched,
            **{k:round(v,2) for k,v in reservation.items()},
            'door_side': DOOR_SIDE_FOR_ZONE.get(item.zone,'S'),
        })
    placed_all=placed+site_rooms

    min_x = min(r['x'] for r in placed_all) if placed_all else setback
    min_y = min(r['y'] for r in placed) if placed else setback
    max_x = max(r['x'] + r['w'] for r in placed) if placed_all else setback
    max_y = max(r['y'] + r['h'] for r in placed_all) if placed_all else setback
    envelope = {
        'x': round(min_x, 2), 'y': round(min_y, 2),
        'w': round(max_x - min_x, 2), 'h': round(max_y - min_y, 2),
    }
    match_count = sum(1 for r in placed if r.get('matched'))
    acceptable_count = sum(1 for r in placed if r.get('vastu_acceptable'))
    return _finalize_layout_geometry({
        'plot': {'length': round(plot_length, 2), 'width': round(plot_width, 2),
                 'area': round(plot_length * plot_width, 2)},
        'floors': floors,
        'road_direction': road_direction,
        'setback_ft': round(setback, 2),
        'building_envelope': envelope,
        'occupied_envelope': envelope,
        'rooms': placed_all,
        'collisions': checks['collisions'],
        'boundary_violations': checks['boundary_violations'],
        'dimension_violations': checks.get('dimension_violations', []),
        'passed_checks': checks['passed'],
        'vastu_match_count': match_count,
        'vastu_acceptable_count': acceptable_count,
        'vastu_score': round(_actual_zone_score(plot_length, plot_width, placed), 4),
        'vastu_match_percent': round(100.0 * match_count / max(len(placed), 1), 1),
        'vastu_enabled': True,
        'vastu_layout_engine': 'Traditional Vastu Zone Assignment + Anchor Packing + Directional Band Tiling + Orthogonal Room Join',
        'building_type': str(building_type or 'custom').strip().lower(),
        'gap_report': _gap_report(placed),
    })


def _apply_presentation_options(layout: Dict, wall_width_ft: float = 0.5,
                                 room_size_details: bool = True,
                                 appearance: str = "technical",
                                 main_door_position: str = "center",
                                 parking_position: str = "road_side",
                                 staircase_position: str = "road_side") -> Dict:
    """Attach non-destructive architectural presentation controls to a validated layout.

    Geometry remains authoritative; these settings control drawing detail and
    entry/space annotations so the generated blueprint stays collision-safe.
    """
    layout["wall_width_ft"] = round(max(0.25, min(float(wall_width_ft or 0.5), 1.5)), 2)
    layout["room_size_details"] = bool(room_size_details)
    layout["appearance"] = str(appearance or "technical").lower()
    layout["main_door_position"] = str(main_door_position or "center").lower()
    layout["parking_position"] = str(parking_position or "road_side").lower()
    layout["staircase_position"] = str(staircase_position or "road_side").lower()

    # Explicitly identify the most natural room for the main entry label.
    candidates = {"living_room", "lobby", "reception", "waiting_area"}
    target = next((r for r in layout.get("rooms", []) if r.get("room_type") in candidates), None)
    if target is None and layout.get("rooms"):
        target = layout["rooms"][0]
    if target:
        layout["main_door_room"] = target.get("label", target.get("room_type", "Entry"))
    return layout


def generate_layout(
    plot_length: float,
    plot_width: float,
    allocations: List[RoomAllocation],
    floors: int = 1,
    road_direction: str = "South",
    building_type: str = "custom",
    wall_width_ft: float = 0.5,
    room_size_details: bool = True,
    appearance: str = "technical",
    main_door_position: str = "center",
    parking_position: str = "road_side",
    staircase_position: str = "road_side",
) -> Dict:
    if plot_length <= 0 or plot_width <= 0:
        raise ValueError("Plot dimensions must be positive.")

    setback = _setback_for_plot(plot_length, plot_width)
    building_type = str(building_type or 'custom').strip().lower()
    vastu_active = any((a.preferred_zone or 'C') != 'C' for a in allocations)

    # When Vastu is enabled, use the weighted Vastu anchor + geometry packer
    # first. Legacy shelf/grid planners remain safe fallbacks for combinations
    # that genuinely cannot fit through the Vastu-first arrangement.
    if vastu_active:
        vastu_layout = _vastu_pack_layout(
            plot_length, plot_width, allocations, setback, road_direction, floors
        , building_type=building_type)
        if vastu_layout is not None:
            return _apply_presentation_options(vastu_layout, wall_width_ft, room_size_details,
                                               appearance, main_door_position,
                                               parking_position, staircase_position)

    # Dimension-first shelf planner for non-Vastu plans or as a fallback.
    def _vastu_ok(candidate: Dict | None) -> bool:
        if candidate is None or not vastu_active:
            return candidate is not None
        for rr in candidate.get('rooms', []):
            if rr.get('room_type') in {'parking','garden','loading_bay'}:
                continue
            if not rr.get('vastu_acceptable', False):
                return False
        return True

    shelf = _shelf_layout(plot_length, plot_width, allocations, setback, road_direction, floors, building_type=building_type)
    if shelf is not None and _vastu_ok(shelf):
        return _apply_presentation_options(shelf, wall_width_ft, room_size_details,
                                           appearance, main_door_position,
                                           parking_position, staircase_position)

    compact = _compact_grid_layout(plot_length, plot_width, allocations, setback, road_direction, floors, building_type=building_type)
    if compact is not None and compact.get('passed_checks', False) and _vastu_ok(compact):
        return _apply_presentation_options(compact, wall_width_ft, room_size_details,
                                           appearance, main_door_position,
                                           parking_position, staircase_position)

    # Larger/irregular plots use the more flexible Vastu-scored packer.
    occupied: List[dict] = []
    remaining: List[RoomAllocation] = []
    for item in allocations:
        if item.room_type in {"parking", "garden", "staircase", "loading_bay", "reception", "security_room"}:
            reservation = _front_reservation(
                item.room_type, plot_length, plot_width, road_direction, occupied, setback
            )
            if reservation is None:
                remaining.append(item)
                continue
            occupied.append({
                "room_type": item.room_type, "label": item.label, "zone": item.zone,
                "preferred_zone": item.preferred_zone, "matched": item.matched,
                **{k: round(v, 2) for k, v in reservation.items()},
                "door_side": DOOR_SIDE_FOR_ZONE.get(item.zone, "S"),
            })
        else:
            remaining.append(item)

    placed = _pack_remaining(remaining, plot_length, plot_width, setback, road_direction, occupied)
    checks = run_all_checks(placed, plot_length, plot_width)
    if not checks['passed']:
        problems = []
        if checks['collisions']:
            problems.append(f"{len(checks['collisions'])} room collision(s)")
        if checks['boundary_violations']:
            problems.append(f"{len(checks['boundary_violations'])} boundary violation(s)")
        if checks.get('dimension_violations'):
            problems.append(f"{len(checks['dimension_violations'])} room dimension violation(s)")
        raise ValueError('Layout validation failed: ' + ', '.join(problems) + '.')
    finalized = _finalize_layout_geometry({
        "building_type": str(building_type or 'custom').strip().lower(),
        "plot": {"length": round(plot_length, 2), "width": round(plot_width, 2),
                  "area": round(plot_length * plot_width, 2)},
        "floors": floors, "road_direction": road_direction,
        "setback_ft": round(setback, 2),
        "building_envelope": {"x": round(setback, 2), "y": round(setback, 2),
                               "w": round(plot_length - 2 * setback, 2),
                               "h": round(plot_width - 2 * setback, 2)},
        "occupied_envelope": {
            "x": round(min(r["x"] for r in placed), 2),
            "y": round(min(r["y"] for r in placed), 2),
            "w": round(max(r["x"] + r["w"] for r in placed) - min(r["x"] for r in placed), 2),
            "h": round(max(r["y"] + r["h"] for r in placed) - min(r["y"] for r in placed), 2),
        },
        "rooms": placed, "collisions": checks["collisions"],
        "boundary_violations": checks["boundary_violations"],
        "dimension_violations": checks.get('dimension_violations', []),
        "passed_checks": checks["passed"],
        "vastu_match_count": sum(1 for r in placed if r.get('matched')),
        "vastu_acceptable_count": sum(1 for r in placed if r.get('zone') in {r.get('preferred_zone')}),
        "vastu_score": round(_actual_zone_score(plot_length, plot_width, placed), 4),
        "vastu_match_percent": round(100.0 * sum(1 for r in placed if r.get('matched')) / max(len(placed), 1), 1),
        "gap_report": _gap_report(placed),
    })
    return _apply_presentation_options(finalized, wall_width_ft, room_size_details,
                                       appearance, main_door_position,
                                       parking_position, staircase_position)

