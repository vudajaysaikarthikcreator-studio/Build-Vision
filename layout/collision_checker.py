def _overlap(a, b):
    return (a["x"] < b["x"] + b["w"] - 1e-6 and b["x"] < a["x"] + a["w"] - 1e-6 and
            a["y"] < b["y"] + b["h"] - 1e-6 and b["y"] < a["y"] + a["h"] - 1e-6)

def _dimension_ok(room):
    """Accept the room's minimum dimensions in either orientation."""
    from planning.building_programs import MIN_DIMS
    custom = room.get("min_dims")
    mw, mh = tuple(custom) if custom else MIN_DIMS.get(room.get("room_type"), (1.0, 1.0))
    w, h = float(room.get("w", 0)), float(room.get("h", 0))
    return (w >= mw - 1e-6 and h >= mh - 1e-6) or (w >= mh - 1e-6 and h >= mw - 1e-6)

def run_all_checks(rooms, plot_length, plot_width):
    collisions = []
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms[i+1:], start=i+1):
            # A room carved inside its parent (attached bath, under-stair bath, pooja,
            # store) is meant to overlap the parent; it must not overlap anything else.
            if a.get("inside_of") == b.get("label") or b.get("inside_of") == a.get("label"):
                continue
            if a.get("inside_of") and a.get("inside_of") == b.get("inside_of") and False:
                continue
            if _overlap(a, b):
                collisions.append({"a": a["label"], "b": b["label"], "indices": [i, j]})
    boundary = []
    dimension_violations = []
    for r in rooms:
        if r["x"] < -1e-6 or r["y"] < -1e-6 or r["x"] + r["w"] > plot_length + 1e-6 or r["y"] + r["h"] > plot_width + 1e-6:
            boundary.append(r["label"])
        if r.get("w", 0) <= 0 or r.get("h", 0) <= 0 or not _dimension_ok(r):
            dimension_violations.append(r["label"])
    return {
        "collisions": collisions,
        "boundary_violations": boundary,
        "dimension_violations": dimension_violations,
        "passed": not collisions and not boundary and not dimension_violations,
    }
