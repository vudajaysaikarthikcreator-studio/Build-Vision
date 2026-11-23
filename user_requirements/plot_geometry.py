"""Plot geometry from user-supplied lengths and degrees.

Three ways to describe a plot (first one present wins):

1. ``plot_traverse`` - surveyor style: a list of legs, each with a length in
   feet and a compass bearing in degrees (clockwise from true north).  Any
   number of legs >= 3.  Small closing errors are distributed with the
   Bowditch (compass) rule; large ones are rejected with a clear message.

2. ``plot_sides`` (+ optional ``plot_angles``) - the four side lengths
   (north / east / south / west) and, optionally, the interior corner angles
   (NE / SE / SW / NW) in degrees.  Missing angles are derived.  If no angles
   are given and opposite sides are equal the plot is a rectangle, otherwise
   it is solved as a quadrilateral with a right angle at the SW corner.

3. ``plot_length`` x ``plot_width`` - a plain rectangle.

``plot_bearing_deg`` says how far the plot's drawn "north" edge is rotated from
true north (clockwise, degrees).  The layout engine works on a rectangle whose
edges point to the true compass directions (that is what Vastu zones need), so
the resolver returns the largest such rectangle that fits inside the plot
polygon (or, with ``alignment='plot'``, the largest rectangle aligned to the
plot's own edges when the rotation is small).

Coordinate convention shared with the layout engine:
    x = north-south extent (south -> north)  == ``plot_length``
    y = east-west extent  (west  -> east)    == ``plot_width``
Polygons here are lists of (east, north) tuples in feet.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]  # (east, north)

COMPASS_8 = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
_SIDE_BEARING = {"north": 0.0, "north-east": 45.0, "east": 90.0, "south-east": 135.0,
                 "south": 180.0, "south-west": 225.0, "west": 270.0, "north-west": 315.0}
_CORNER_KEYS = ("SW", "SE", "NE", "NW")


class PlotError(ValueError):
    """The supplied plot description cannot describe a real plot."""


# ------------------------------------------------------------------ helpers
def _norm_name(value) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def compass_to_bearing(name: str) -> float:
    key = _norm_name(name)
    aliases = {"n": "north", "e": "east", "s": "south", "w": "west", "ne": "north-east",
               "nw": "north-west", "se": "south-east", "sw": "south-west"}
    key = aliases.get(key, key)
    if key not in _SIDE_BEARING:
        raise PlotError(f"Unknown direction '{name}'. Use North, East, South, West or a diagonal such as North-East.")
    return _SIDE_BEARING[key]


def bearing_to_compass(bearing: float) -> str:
    return COMPASS_8[int(((bearing % 360.0) + 22.5) // 45.0) % 8]


def snap_cardinal(bearing: float) -> Tuple[str, float]:
    """Nearest of North/East/South/West and the residual error in degrees."""
    b = bearing % 360.0
    idx = int((b + 45.0) // 90.0) % 4
    target = idx * 90.0
    err = ((b - target + 180.0) % 360.0) - 180.0
    return ["North", "East", "South", "West"][idx], err


def _area(poly: Sequence[Point]) -> float:
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _perimeter(poly: Sequence[Point]) -> float:
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _rotate(poly: Sequence[Point], bearing_deg: float, about: Point = (0.0, 0.0)) -> List[Point]:
    """Rotate clockwise (as seen on a map) by ``bearing_deg`` degrees."""
    a = math.radians(-bearing_deg)
    ca, sa = math.cos(a), math.sin(a)
    ox, oy = about
    return [((x - ox) * ca - (y - oy) * sa + ox, (x - ox) * sa + (y - oy) * ca + oy) for x, y in poly]


def _is_convex(poly: Sequence[Point]) -> bool:
    sign = 0
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        cx, cy = poly[(i + 2) % n]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cr) < 1e-9:
            continue
        s = 1 if cr > 0 else -1
        if sign and s != sign:
            return False
        sign = s
    return True


def _self_intersects(poly: Sequence[Point]) -> bool:
    n = len(poly)

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) in (1, n - 1):
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if ccw(a, b, c) * ccw(a, b, d) < 0 and ccw(c, d, a) * ccw(c, d, b) < 0:
                return True
    return False


def _point_in_poly(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def _interior_angles(poly: Sequence[Point]) -> List[float]:
    """Interior angle (degrees) at every vertex, for a counter-clockwise polygon."""
    n = len(poly)
    out = []
    for i in range(n):
        p0, p1, p2 = poly[i - 1], poly[i], poly[(i + 1) % n]
        a1 = math.atan2(p0[1] - p1[1], p0[0] - p1[0])
        a2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        ang = math.degrees((a1 - a2) % (2 * math.pi))
        out.append(ang)
    return out


# ---------------------------------------------------------- polygon builders
def _polygon_from_traverse(legs: Sequence[dict]) -> Tuple[List[Point], List[str]]:
    warnings: List[str] = []
    if not isinstance(legs, (list, tuple)) or len(legs) < 3:
        raise PlotError("plot_traverse needs at least 3 legs, each with a length (ft) and a bearing (degrees).")
    pts: List[Point] = [(0.0, 0.0)]
    lengths = []
    for i, leg in enumerate(legs, 1):
        try:
            length = float(leg.get("length", leg.get("length_ft")))
            bearing = float(leg.get("bearing", leg.get("bearing_deg")))
        except (TypeError, ValueError, AttributeError):
            raise PlotError(f"Leg {i}: give a numeric length (ft) and bearing (degrees).")
        if not (0 < length <= 5000):
            raise PlotError(f"Leg {i}: length must be between 0 and 5000 ft.")
        b = math.radians(bearing)
        x, y = pts[-1]
        pts.append((x + length * math.sin(b), y + length * math.cos(b)))
        lengths.append(length)
    perimeter = sum(lengths)
    mis_e, mis_n = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
    misclosure = math.hypot(mis_e, mis_n)
    if misclosure / perimeter > 0.03:
        raise PlotError(
            f"The traverse does not close: it ends {misclosure:.1f} ft away from where it started "
            f"({100 * misclosure / perimeter:.1f}% of the perimeter). Re-check the lengths and bearings.")
    if misclosure > 0.05:
        warnings.append(f"Traverse misclosure of {misclosure:.2f} ft was distributed along the boundary (Bowditch rule).")
    poly: List[Point] = []
    run = 0.0
    for i in range(len(legs)):
        frac = run / perimeter
        poly.append((pts[i][0] - mis_e * frac, pts[i][1] - mis_n * frac))
        run += lengths[i]
    return poly, warnings


def _solve_quad_from_sides(s_len: float, e_len: float, n_len: float, w_len: float,
                           theta_sw: float = 90.0) -> List[Point]:
    """Quadrilateral from four sides with a given interior angle at SW."""
    th = math.radians(theta_sw)
    sw = (0.0, 0.0)
    se = (s_len, 0.0)
    nw = (w_len * math.cos(th), w_len * math.sin(th))
    dx, dy = nw[0] - se[0], nw[1] - se[1]
    d = math.hypot(dx, dy)
    if d > e_len + n_len + 1e-9 or d < abs(e_len - n_len) - 1e-9 or d == 0:
        raise PlotError(
            "These four side lengths cannot close into a plot with that corner angle. "
            "Check that no side is longer than the other three combined.")
    a = (e_len ** 2 - n_len ** 2 + d ** 2) / (2 * d)
    h2 = e_len ** 2 - a ** 2
    h = math.sqrt(max(0.0, h2))
    mx, my = se[0] + a * dx / d, se[1] + a * dy / d
    cand = [(mx + h * dy / d, my - h * dx / d), (mx - h * dy / d, my + h * dx / d)]
    ne = max(cand, key=lambda p: p[1])
    return [sw, se, ne, nw]


def _polygon_from_sides(sides: Dict[str, float], angles: Dict[str, float]) -> Tuple[List[Point], List[str]]:
    warnings: List[str] = []
    s, e, n, w = (sides["south"], sides["east"], sides["north"], sides["west"])
    ang = {k: v for k, v in angles.items() if v is not None}
    for k, v in ang.items():
        if not (20.0 <= v <= 160.0):
            raise PlotError(f"Corner angle {k} = {v:g}° is outside the supported 20°-160° range "
                            "(plots must be convex; a corner of 180° or more is not a corner).")
    missing = [k for k in _CORNER_KEYS if k not in ang]
    if len(ang) == 4:
        total = sum(ang.values())
        if abs(total - 360.0) > 2.0:
            raise PlotError(f"The four interior angles add up to {total:g}°, but a four-sided plot must add up to 360°.")
        if abs(total - 360.0) > 0.01:
            warnings.append(f"Corner angles summed to {total:g}°; they were scaled to 360°.")
            ang = {k: v * 360.0 / total for k, v in ang.items()}
    elif len(ang) == 3:
        ang[missing[0]] = 360.0 - sum(ang.values())
        if not (20.0 <= ang[missing[0]] <= 160.0):
            raise PlotError(f"The three angles given force {missing[0]} to {ang[missing[0]]:g}°, which is not a valid corner.")
    elif len(ang) == 2:
        if all(k in ang for k in ("SW", "SE")) or all(k in ang for k in ("NW", "NE")):
            pass  # enough to place the polygon; remaining angles are computed from the geometry
        else:
            rest = (360.0 - sum(ang.values())) / 2.0
            for k in missing:
                ang[k] = rest
    elif len(ang) == 1:
        k0, v0 = next(iter(ang.items()))
        if abs(s - n) < 0.01 and abs(e - w) < 0.01:      # parallelogram
            opp = {"SW": "NE", "NE": "SW", "SE": "NW", "NW": "SE"}
            adj = 180.0 - v0
            ang = {k0: v0, opp[k0]: v0}
            for k in _CORNER_KEYS:
                ang.setdefault(k, adj)
        # otherwise handled below: use as the SW angle if it is SW, else solved with 90
    # ---- build from the south edge (or mirror from the north edge)
    def from_south(th_sw: float, th_se: float) -> List[Point]:
        sw = (0.0, 0.0)
        se = (s, 0.0)
        ne = (se[0] + e * math.cos(math.radians(180.0 - th_se)), e * math.sin(math.radians(180.0 - th_se)))
        nw = (w * math.cos(math.radians(th_sw)), w * math.sin(math.radians(th_sw)))
        return [sw, se, ne, nw]

    if "SW" in ang and "SE" in ang:
        poly = from_south(ang["SW"], ang["SE"])
    elif "NW" in ang and "NE" in ang:
        # Build downward from the north edge, then shift so the south-most point is at y = 0.
        nw, ne = (0.0, 0.0), (n, 0.0)
        sw_p = (w * math.cos(math.radians(ang["NW"])), -w * math.sin(math.radians(ang["NW"])))
        se_p = (ne[0] + e * math.cos(math.radians(180.0 - ang["NE"])), -e * math.sin(math.radians(180.0 - ang["NE"])))
        poly = [sw_p, se_p, ne, nw]
        dy = -min(p[1] for p in poly)
        poly = [(x, y + dy) for x, y in poly]
    else:
        theta = ang.get("SW", 90.0)
        if len(ang) == 0 and abs(s - n) < 0.01 and abs(e - w) < 0.01:
            poly = from_south(90.0, 90.0)
        else:
            poly = _solve_quad_from_sides(s, e, n, w, theta)
    # report when the given north side disagrees with the geometry
    ne, nw = poly[2], poly[3]
    got_n = math.dist(ne, nw)
    if abs(got_n - n) > 0.5:
        warnings.append(f"With the given angles the north side works out to {got_n:.1f} ft "
                        f"(you entered {n:.1f} ft); the angles were treated as authoritative.")
    return poly, warnings


# ------------------------------------------------- largest inscribed rectangle
def largest_axis_rectangle(poly: Sequence[Point], resolution: float = 0.5) -> Tuple[float, float, float, float]:
    """Largest axis-aligned rectangle inside ``poly``.

    Returns (min_east, min_north, width_east_west, depth_north_south).  Works
    for any simple polygon (convex or not) by rasterising and using the
    "largest rectangle in a histogram" scan; the result is then verified
    against the exact polygon and trimmed if a corner pokes out.
    """
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    span = max(maxx - minx, maxy - miny)
    res = resolution if span <= 150 else (1.0 if span <= 400 else 2.0)
    cols = max(1, int(math.ceil((maxx - minx) / res)))
    rows = max(1, int(math.ceil((maxy - miny) / res)))
    grid = [[_point_in_poly((minx + (c + 0.5) * res, miny + (r + 0.5) * res), poly) for c in range(cols)]
            for r in range(rows)]
    heights = [0] * cols
    best = (0, 0, 0, 0, 0)  # area, r_end, c_start, w, h
    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if grid[r][c] else 0
        stack: List[int] = []
        for c in range(cols + 1):
            cur = heights[c] if c < cols else 0
            start = c
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = heights[top] * (c - left)
                if area > best[0]:
                    best = (area, r, left, c - left, heights[top])
            stack.append(c) if c < cols else None
    _, r_end, c0, wc, hr = best
    if wc == 0 or hr == 0:
        raise PlotError("No usable rectangular building area fits inside this plot.")
    x0 = minx + c0 * res
    y0 = miny + (r_end - hr + 1) * res
    w, h = wc * res, hr * res

    def inside(px, py, ww, hh):
        pts = [(px, py), (px + ww, py), (px + ww, py + hh), (px, py + hh)]
        steps = 8
        for i in range(steps + 1):
            t = i / steps
            pts += [(px + ww * t, py), (px + ww * t, py + hh), (px, py + hh * t), (px + ww, py + hh * t)]
        return all(_point_in_poly(p, poly) or _on_edge(p, poly) for p in pts)

    tries = 0
    while not inside(x0, y0, w, h) and tries < 40 and w > 1 and h > 1:
        x0 += 0.125; y0 += 0.125; w -= 0.25; h -= 0.25; tries += 1
    return x0, y0, w, h


def _on_edge(p: Point, poly: Sequence[Point], tol: float = 1e-6) -> bool:
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
        if abs(cross) > tol * max(1.0, math.dist(a, b)):
            continue
        if min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol:
            return True
    return False


# ----------------------------------------------------------------- resolver
def _num(v, name):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise PlotError(f"{name} must be a number.")
    if f != f or f in (float("inf"), float("-inf")):
        raise PlotError(f"{name} must be a finite number.")
    return f


def _clean_sides(sides: Optional[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (sides or {}).items():
        key = _norm_name(k)
        key = {"n": "north", "e": "east", "s": "south", "w": "west"}.get(key, key)
        if key not in ("north", "east", "south", "west"):
            raise PlotError(f"Unknown plot side '{k}'. Use north, east, south, west.")
        f = _num(v, f"{k} side length")
        if f is not None:
            if not (5.0 <= f <= 5000.0):
                raise PlotError(f"The {key} side must be between 5 ft and 5000 ft (got {f:g}).")
            out[key] = f
    return out


def _clean_angles(angles: Optional[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (angles or {}).items():
        key = str(k).strip().upper().replace("-", "").replace("_", "")
        key = {"NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW"}.get(key, key)
        if key not in _CORNER_KEYS:
            raise PlotError(f"Unknown corner '{k}'. Use NE, SE, SW, NW.")
        f = _num(v, f"{k} corner angle")
        if f is not None:
            out[key] = f
    return out


def resolve_plot(
    plot_length: Optional[float] = None,
    plot_width: Optional[float] = None,
    plot_sides: Optional[dict] = None,
    plot_angles: Optional[dict] = None,
    plot_traverse: Optional[Sequence[dict]] = None,
    plot_bearing_deg: Optional[float] = 0.0,
    road_direction: Optional[str] = None,
    road_side: Optional[str] = None,
    road_bearing_deg: Optional[float] = None,
    alignment: str = "auto",
) -> dict:
    """Turn any supported plot description into engine-ready numbers."""
    warnings: List[str] = []
    bearing = _num(plot_bearing_deg, "plot_bearing_deg") or 0.0
    bearing = ((bearing + 180.0) % 360.0) - 180.0
    sides = _clean_sides(plot_sides)
    angles = _clean_angles(plot_angles)
    mode = "rectangle"
    true_frame_poly: List[Point]

    if plot_traverse:
        poly, w = _polygon_from_traverse(plot_traverse)
        warnings += w
        mode = "traverse"
        plot_frame = poly
        true_frame_poly = poly            # bearings are already true bearings
        bearing_used = 0.0
    else:
        if sides:
            pl, pw = _num(plot_length, "plot_length"), _num(plot_width, "plot_width")
            # fill missing sides: N/S from width, E/W from length, mirroring opposite sides
            for a, b in (("north", "south"), ("east", "west")):
                if a not in sides and b in sides:
                    sides[a] = sides[b]
            if "north" not in sides:
                sides["north"] = sides["south"] = pw if pw else None
            if "east" not in sides:
                sides["east"] = sides["west"] = pl if pl else None
            if any(sides.get(k) is None for k in ("north", "east", "south", "west")):
                raise PlotError("Give at least one north/south side and one east/west side length (ft), "
                                "or use plot_length and plot_width.")
            poly, w = _polygon_from_sides(sides, angles)
            warnings += w
            is_rect = all(abs(a - 90.0) < 0.01 for a in _interior_angles(poly)) if _is_ccw_fix(poly) else False
            mode = "rectangle" if is_rect else "quadrilateral"
        else:
            pl, pw = _num(plot_length, "plot_length"), _num(plot_width, "plot_width")
            if not pl or not pw:
                raise PlotError("Give the plot size: plot_length and plot_width, or the four side lengths.")
            if angles:
                sides = {"north": pw, "south": pw, "east": pl, "west": pl}
                poly, w = _polygon_from_sides(sides, angles)
                warnings += w
                mode = "quadrilateral"
            else:
                poly = [(0.0, 0.0), (pw, 0.0), (pw, pl), (0.0, pl)]   # SW, SE, NE, NW as (east, north)
        plot_frame = poly
        bearing_used = bearing
        true_frame_poly = _rotate(poly, bearing, about=(0.0, 0.0)) if abs(bearing) > 1e-9 else list(poly)

    if _area(plot_frame) < 0:                                   # force counter-clockwise
        plot_frame = list(reversed(plot_frame))
        true_frame_poly = list(reversed(true_frame_poly))
    if _self_intersects(plot_frame):
        raise PlotError("The plot boundary crosses itself. Check the order of the sides / legs.")
    area = _area(plot_frame)
    if area < 100:
        raise PlotError(f"The plot area works out to only {area:.0f} sq.ft - too small to plan a building on.")

    # ---- choose which frame the building rectangle is aligned to
    residual = ((bearing_used + 45.0) % 90.0) - 45.0            # rotation away from the nearest compass axis
    align = str(alignment or "auto").lower()
    if align not in ("auto", "compass", "plot"):
        raise PlotError("alignment must be 'auto', 'compass' or 'plot'.")
    exact_rect = (mode == "rectangle" and abs(bearing_used - round(bearing_used / 90.0) * 90.0) < 1e-6)
    use_plot_frame = (align == "plot") or (align == "auto" and abs(residual) <= 15.0 and mode != "traverse")
    if mode == "traverse":
        use_plot_frame = False
    if use_plot_frame:
        if exact_rect:
            x0, y0, wq, hq = 0.0, 0.0, poly[1][0] - poly[0][0], poly[3][1] - poly[0][1]
        else:
            x0, y0, wq, hq = largest_axis_rectangle(plot_frame)  # rectangle in the plot's own axes
        frame_rot = bearing_used if abs(bearing_used) > 1e-9 else 0.0
        n90 = round(frame_rot / 90.0) * 90.0
        vastu_north_offset = frame_rot - n90                      # plot axes vs. true axes, degrees
        build_poly_frame = plot_frame
        rect_local = [(x0, y0), (x0 + wq, y0), (x0 + wq, y0 + hq), (x0, y0 + hq)]
        quarter = int(round(n90 / 90.0)) % 4
        if quarter in (1, 3):                                      # plot rotated by 90/270: swap extents
            length_ns, width_ew = wq, hq
        else:
            length_ns, width_ew = hq, wq
        alignment_used = "plot"
        rect_true = _rotate(rect_local, frame_rot) if abs(frame_rot) > 1e-9 else rect_local
    else:
        if exact_rect:
            bb = true_frame_poly
            x0, y0 = min(p[0] for p in bb), min(p[1] for p in bb)
            wq, hq = max(p[0] for p in bb) - x0, max(p[1] for p in bb) - y0
        else:
            x0, y0, wq, hq = largest_axis_rectangle(true_frame_poly)
        length_ns, width_ew = hq, wq
        vastu_north_offset = 0.0
        alignment_used = "compass"
        rect_true = [(x0, y0), (x0 + wq, y0), (x0 + wq, y0 + hq), (x0, y0 + hq)]
        if abs(residual) > 1e-6 or mode == "traverse":
            pass

    # exact rectangle -> keep the user's numbers verbatim
    if exact_rect:
        quarter = int(round(bearing_used / 90.0)) % 4
        ax = [poly[1][0] - poly[0][0], poly[3][1] - poly[0][1]] if plot_traverse is None else [wq, hq]
        ew, ns = (ax[0], ax[1])
        if quarter in (1, 3):
            ew, ns = ns, ew
        length_ns, width_ew = ns, ew

    length_ns = round(max(length_ns, 0.0), 2)
    width_ew = round(max(width_ew, 0.0), 2)
    if length_ns < 8 or width_ew < 8:
        raise PlotError(
            f"Only a {width_ew:g} x {length_ns:g} ft rectangle fits inside this plot - too small for a building. "
            "Check the side lengths and angles.")
    usable_pct = round(100.0 * length_ns * width_ew / area, 1)
    if usable_pct < 60.0:
        warnings.append(f"Only {usable_pct:.0f}% of the plot area fits a rectangular building block; "
                        "the rest of the land stays open (garden / driveway / setback).")

    # ---- road / entrance direction (true compass)
    road_info = _resolve_road(road_direction, road_side, road_bearing_deg, bearing_used, warnings)

    # every polygon is finally expressed relative to the building rectangle's SW corner
    ox = min(p[0] for p in rect_true); oy = min(p[1] for p in rect_true)
    poly_rel = [(round(px - ox, 2), round(py - oy, 2)) for px, py in true_frame_poly]

    angles_out = _interior_angles(plot_frame)
    edges = [round(math.dist(plot_frame[i], plot_frame[(i + 1) % len(plot_frame)]), 2) for i in range(len(plot_frame))]
    return {
        "mode": mode,
        "vertex_count": len(plot_frame),
        "area_sqft": round(area, 1),
        "perimeter_ft": round(_perimeter(plot_frame), 1),
        "edge_lengths_ft": edges,
        "corner_angles_deg": [round(a, 2) for a in angles_out],
        "polygon_true_rel": poly_rel,
        "buildable": {"length_ns_ft": length_ns, "width_ew_ft": width_ew,
                      "area_sqft": round(length_ns * width_ew, 1), "usable_percent": usable_pct},
        "alignment": alignment_used,
        "plot_bearing_deg": round(bearing_used, 2),
        "vastu_north_offset_deg": round(vastu_north_offset, 2),
        "road": road_info,
        "warnings": warnings,
        # engine-ready values
        "plot_length": length_ns,
        "plot_width": width_ew,
        "road_direction": road_info["direction"],
    }


def _is_ccw_fix(poly) -> bool:
    return _area(poly) > 0


def _resolve_road(road_direction, road_side, road_bearing_deg, plot_bearing, warnings) -> dict:
    """Road side -> true compass direction snapped to N/E/S/W for the layout engine."""
    if road_bearing_deg not in (None, ""):
        b = _num(road_bearing_deg, "road_bearing_deg") % 360.0
        src = "road_bearing_deg"
    elif road_side:
        b = (compass_to_bearing(road_side) + plot_bearing) % 360.0
        src = "road_side"
    else:
        b = compass_to_bearing(road_direction or "North")
        src = "road_direction"
    cardinal, err = snap_cardinal(b)
    if abs(err) > 1.0:
        warnings.append(
            f"The road faces {bearing_to_compass(b)} ({b:.0f}°). The building block is aligned to the compass, so its "
            f"entrance side was set to {cardinal} ({abs(err):.0f}° off) - the usual practice for diagonal plots. "
            "Give road_bearing_deg or pick a main direction to choose the other side.")
    return {"source": src, "bearing_deg": round(b, 1), "compass": bearing_to_compass(b),
            "direction": cardinal, "snapped_error_deg": round(err, 1)}
