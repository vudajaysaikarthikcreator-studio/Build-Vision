"""Vastu Shastra rule engine: zone geometry, capacity-aware zone assignment,
entrance guidance and a per-room audit of a finished layout.

Coordinates follow the layout engine: x = south->north, y = west->east.  A
compass zone is decided two ways and both are reported:

* 9-zone grid (Nava-vastu): the building footprint is cut in thirds each way.
* 16-zone bearing (Shodasha-vastu): the bearing from the footprint centre to the
  room centre, plus ``north_offset_deg`` when the building is not aligned to
  true north, falls into one of sixteen 22.5° sectors.

Rules are the widely published traditional ones; where sources disagree (dining,
study, second bedrooms, west-facing entrances) the rule table lists the common
choice as "ideal" and the alternatives as "acceptable" rather than pretending
there is a single canonical answer.  This is planning guidance, not a
religious or statutory requirement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from planning.building_programs import PREFERRED_ZONES, ACCEPTABLE_ZONES, RESIDENTIAL

ZONE_GRID = (("NW", "N", "NE"), ("W", "C", "E"), ("SW", "S", "SE"))   # top row = north
ZONES_9 = [z for row in ZONE_GRID for z in row]
ZONES_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
_ZONE16_TO_8 = {"NNE": "N", "ENE": "E", "ESE": "E", "SSE": "S", "SSW": "S", "WSW": "W", "WNW": "W", "NNW": "N"}
_ZONE_RC = {z: (r, c) for r, row in enumerate(ZONE_GRID) for c, z in enumerate(row)}

GRADE_SCORE = {"ideal": 1.0, "acceptable": 0.75, "neutral": 0.4, "avoid": 0.15, "forbidden": 0.0}
SITE_TYPES = {"parking", "garden", "loading_bay"}


# ------------------------------------------------------------------ geometry
def zone16_from_bearing(bearing_deg: float) -> str:
    return ZONES_16[int(((bearing_deg % 360.0) + 11.25) // 22.5) % 16]


def zone8_from_bearing(bearing_deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((bearing_deg % 360.0) + 22.5) // 45.0) % 8]


def to_zone8(zone16: str) -> str:
    return _ZONE16_TO_8.get(zone16, zone16)


def zone9_of_point(x: float, y: float, env: dict) -> str:
    """Grid zone of a point inside footprint ``env`` ({x,y,w,h}; x south->north)."""
    ex, ey, ew, eh = float(env["x"]), float(env["y"]), float(env["w"]), float(env["h"])
    r = 2 - min(2, max(0, int((x - ex) / (ew / 3.0 + 1e-9))))
    c = min(2, max(0, int((y - ey) / (eh / 3.0 + 1e-9))))
    return ZONE_GRID[r][c]


def bearing_of_point(x: float, y: float, env: dict, north_offset_deg: float = 0.0) -> float:
    cx = float(env["x"]) + float(env["w"]) / 2.0
    cy = float(env["y"]) + float(env["h"]) / 2.0
    return (math.degrees(math.atan2(y - cy, x - cx)) + north_offset_deg) % 360.0


# --------------------------------------------------------------------- rules
@dataclass
class Rule:
    ideal: List[str]
    ok: List[str] = field(default_factory=list)
    avoid: List[str] = field(default_factory=list)
    forbidden: List[str] = field(default_factory=list)
    weight: float = 1.0
    hard: bool = False           # a forbidden placement counts as a violation, not just a low score
    note: str = ""

    def grade(self, zone: str) -> str:
        if zone in self.forbidden:
            return "forbidden"
        if zone in self.ideal:
            return "ideal"
        if zone in self.ok:
            return "acceptable"
        if zone in self.avoid:
            return "avoid"
        return "neutral"


_TOILET_LIKE = {"bathroom", "toilet", "restroom", "wash_area", "laundry", "housekeeping", "service_room"}
_FIRE_LIKE = {"kitchen", "restaurant_kitchen", "pantry", "server_room", "computer_lab", "science_lab",
              "diagnostic_lab", "pharmacy", "cashier", "break_room", "utility_room"}
_HEAVY_LIKE = {"vault", "stock_room", "storage_hall", "warehouse_office", "manager_office", "director_office",
               "principal_office", "warden_room", "store_room", "production_hall"}

_CORE = {
    "pooja_room": Rule(["NE"], ["N", "E"], ["W", "NW"], ["S", "SW", "SE", "C"], 3.0, True,
                       "Prayer space belongs in the north-east (Ishanya)."),
    "kitchen": Rule(["SE"], ["NW"], ["N", "S"], ["NE", "SW", "C"], 3.0, True,
                    "Fire element: south-east (Agneya), north-west as second choice."),
    "master_bedroom": Rule(["SW"], ["S", "W"], ["SE", "N"], ["NE", "C"], 3.0, True,
                           "Head of the family sleeps in the heavy south-west."),
    "bedroom": Rule(["NW", "W"], ["S", "SW", "N"], ["SE"], ["NE", "C"], 2.0, True,
                    "Secondary bedrooms suit north-west/west; avoid the north-east and the centre."),
    "guest_room": Rule(["NW"], ["W", "N"], ["SE"], ["NE", "C"], 1.5, False, "Guest rooms: north-west."),
    "living_room": Rule(["N", "NE", "E"], ["NW"], ["SW", "S"], [], 2.0, False,
                        "Living areas are best in the light, open north/east."),
    "dining_room": Rule(["W", "E"], ["N", "S", "NW", "C"], ["SW"], [], 1.5, False,
                        "Dining: west/east is preferred; north/south or centre can be a compact-plan fallback."),
    "study_room": Rule(["NE", "N", "E", "W"], ["NW"], ["SE"], ["C"], 1.5, False,
                       "Study/work: facing east or north; west is the common alternative."),
    "staircase": Rule(["S", "SW", "W"], ["SE", "NW"], ["N", "E"], ["NE", "C"], 2.0, True,
                      "Stairs are heavy: south/west/south-west, never the north-east or the centre."),
    "store_room": Rule(["SW", "W"], ["S", "NW"], ["N", "E"], ["NE", "C"], 1.5, False, "Heavy storage: south-west/west."),
    "balcony": Rule(["N", "E", "NE"], ["NW"], ["SW"], [], 1.0, False, "Balconies open to the north/east."),
    "parking": Rule(["NW", "SE"], ["N", "E", "W", "S"], [], ["SW", "C", "NE"], 1.0, False,
                    "Vehicles: north-west or south-east."),
    "garden": Rule(["N", "E", "NE"], ["NW"], ["SW"], ["C"], 1.0, False, "Open/green space: north and east."),
}


def rule_for(room_type: str, hint_zones: Optional[Sequence[str]] = None) -> Rule:
    if hint_zones:
        return Rule(list(hint_zones[:1]), list(hint_zones[1:]) + ["W"], [], ["NE", "C"] if room_type != "custom_room" else ["C"], 1.0, False,
                    "User-named space placed by its usual Vastu function.")
    if room_type in _CORE:
        return _CORE[room_type]
    if room_type in _TOILET_LIKE:
        return Rule(["NW", "W"], ["S", "SE"], ["E", "N"], ["NE", "SW", "C"], 3.0, True,
                    "Toilets/wash areas: north-west or west; never north-east, south-west or the centre.")
    if room_type in _FIRE_LIKE:
        return Rule(["SE"], ["E", "NW"], ["N"], ["NE", "SW", "C"], 2.0, False, "Heat/fire-related: south-east.")
    if room_type in _HEAVY_LIKE:
        return Rule(["SW"], ["W", "S"], ["N", "NE"], ["NE", "C"], 2.0, False, "Heavy/valuable: south-west.")
    ideal = [PREFERRED_ZONES.get(room_type, "W")]
    ok = [z for z in ACCEPTABLE_ZONES.get(room_type, []) if z not in ideal]
    return Rule(ideal, ok, [], ["C"] if room_type not in {"lobby", "reception", "open_work_area", "multipurpose_hall",
                                                           "auditorium", "showroom", "restaurant_dining"} else [], 1.0, False,
                "Traditional preference for this space type.")


def zone_score(rule: Rule, zone: str) -> float:
    return GRADE_SCORE[rule.grade(zone)]


# ------------------------------------------------------------ zone assignment
def assign_zones_optimal(
    rooms: Sequence[dict],
    footprint_area: float,
    capacity_factor: float = 1.6,
    centre_capacity_factor: float = 0.35,
) -> List[str]:
    """Capacity-aware Vastu zone assignment.

    ``rooms``: dicts with ``room_type``, ``area`` (sq.ft), optional ``zone``
    (user-fixed) and ``hint_zones``.  Returns one zone per room.

    A plain "first choice" rule table piles every bathroom into the north-west
    and cannot be built; here each of the nine zones has an area budget
    (about 1/9 of the footprint x ``capacity_factor``; the Brahmasthan centre is
    kept nearly empty) and rooms are placed most-constrained-first, then
    improved by pairwise swaps that raise the total weighted score.
    """
    n = len(rooms)
    cap = {z: footprint_area / 9.0 * capacity_factor for z in ZONES_9}
    cap["C"] = footprint_area / 9.0 * centre_capacity_factor
    rules = [rule_for(r["room_type"], r.get("hint_zones")) for r in rooms]
    areas = [max(1.0, float(r.get("area", 60.0))) for r in rooms]
    result: List[Optional[str]] = [None] * n
    used = {z: 0.0 for z in ZONES_9}

    for i, r in enumerate(rooms):                                   # user-fixed zones first
        if r.get("zone"):
            result[i] = r["zone"]
            used[r["zone"]] += areas[i]

    def options(i):
        return sorted(ZONES_9, key=lambda z: -zone_score(rules[i], z))

    order = sorted((i for i in range(n) if result[i] is None),
                   key=lambda i: (-(rules[i].weight * (1.0 + (1.0 if rules[i].hard else 0.0))),
                                  -areas[i]))
    for i in order:
        placed = False
        for z in options(i):
            if zone_score(rules[i], z) <= 0.0:
                continue
            if used[z] + areas[i] <= cap[z] + 1e-6 or (used[z] == 0.0 and areas[i] > cap[z]):
                result[i] = z
                used[z] += areas[i]
                placed = True
                break
        if not placed:                                              # every good zone is full: least-loaded acceptable one
            cands = [z for z in ZONES_9 if zone_score(rules[i], z) > 0.0] or ZONES_9
            z = min(cands, key=lambda zz: (used[zz] / max(cap[zz], 1.0)) - 0.5 * zone_score(rules[i], zz))
            result[i] = z
            used[z] += areas[i]

    fixed = {i for i, r in enumerate(rooms) if r.get("zone")}
    improved = True
    guard = 0
    while improved and guard < 20:
        improved = False
        guard += 1
        for i in range(n):
            if i in fixed:
                continue
            for j in range(i + 1, n):
                if j in fixed or result[i] == result[j]:
                    continue
                zi, zj = result[i], result[j]
                before = rules[i].weight * zone_score(rules[i], zi) + rules[j].weight * zone_score(rules[j], zj)
                after = rules[i].weight * zone_score(rules[i], zj) + rules[j].weight * zone_score(rules[j], zi)
                di = used[zj] - areas[j] + areas[i]
                dj = used[zi] - areas[i] + areas[j]
                if after > before + 1e-9 and di <= cap[zj] * 1.15 and dj <= cap[zi] * 1.15:
                    used[zi] += areas[j] - areas[i]
                    used[zj] += areas[i] - areas[j]
                    result[i], result[j] = zj, zi
                    improved = True
    return [z or "C" for z in result]


# -------------------------------------------------------------------- entrance
_ENTRANCE = {
    # facing: (best 16-zone, acceptable 16-zones, avoid 16-zones, door position seen from the street)
    "North": (["NNE", "N"], ["NE"], ["NNW", "NW"], "left"),
    "East": (["ENE", "E"], ["NE"], ["ESE", "SE"], "right"),
    "South": (["SSE", "S"], ["SE"], ["SSW", "SW"], "right"),
    "West": (["WNW", "W"], ["NW"], ["WSW", "SW"], "left"),
}


def entrance_advice(road_direction: str) -> dict:
    d = str(road_direction or "North").strip().title()
    d = d if d in _ENTRANCE else "North"
    best, ok, avoid, pos = _ENTRANCE[d]
    return {"facing": d, "best_zones": best, "acceptable_zones": ok, "avoid_zones": avoid,
            "recommended_door_position": pos,
            "note": (f"For {'an' if d == 'East' else 'a'} {d}-facing plot the main door is best in the {best[0]} part of the {d.lower()} wall "
                     f"(the {pos} side when you stand outside facing the house).")}


def entrance_zone16(road_direction: str, door_position: str) -> str:
    """16-zone of the main door for a given facing and left/centre/right position (seen from the street)."""
    d = str(road_direction or "North").strip().title()
    pos = str(door_position or "center").lower()
    table = {
        "North": {"left": "NNE", "center": "N", "right": "NNW"},
        "East": {"right": "ENE", "center": "E", "left": "ESE"},
        "South": {"right": "SSE", "center": "S", "left": "SSW"},
        "West": {"left": "WNW", "center": "W", "right": "WSW"},
    }
    return table.get(d, table["North"]).get(pos, table.get(d, table["North"])["center"])


# ----------------------------------------------------------------------- audit
def _shared_wall(a: dict, b: dict, tol: float = 0.35) -> float:
    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if abs((a["x"] + a["w"]) - b["x"]) <= tol or abs((b["x"] + b["w"]) - a["x"]) <= tol:
        return max(0.0, oy)
    if abs((a["y"] + a["h"]) - b["y"]) <= tol or abs((b["y"] + b["h"]) - a["y"]) <= tol:
        return max(0.0, ox)
    return 0.0


def _overlap_area(a: dict, b: dict) -> float:
    ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return max(0.0, ox) * max(0.0, oy)


def audit_layout(floor_layouts: Sequence[dict], road_direction: str = "North", door_position: str = "center",
                 north_offset_deg: float = 0.0, vastu_enabled: bool = True, building_type: str = "custom",
                 entrance_direction: Optional[str] = None) -> dict:
    """Score a finished layout against the rule table (independent of how it was generated)."""
    items: List[dict] = []
    issues: List[dict] = []
    total_w = 0.0
    total_s = 0.0
    residential = str(building_type or "custom").lower() in RESIDENTIAL
    ground_by_type: Dict[str, List[dict]] = {}
    for fl in floor_layouts:
        rooms = fl.get("rooms") or []
        env = fl.get("occupied_envelope") or fl.get("building_envelope")
        indoor = [r for r in rooms if r.get("room_type") not in SITE_TYPES]
        if not env and indoor:
            env = {"x": min(r["x"] for r in indoor), "y": min(r["y"] for r in indoor),
                   "w": max(r["x"] + r["w"] for r in indoor) - min(r["x"] for r in indoor),
                   "h": max(r["y"] + r["h"] for r in indoor) - min(r["y"] for r in indoor)}
        if not env:
            continue
        fi = int(fl.get("floor", 0))
        # Brahmasthan: the central ninth of the footprint should stay open / light
        cx, cy = env["x"] + env["w"] / 2.0, env["y"] + env["h"] / 2.0
        centre_box = {"x": cx - env["w"] / 6.0, "y": cy - env["h"] / 6.0, "w": env["w"] / 3.0, "h": env["h"] / 3.0}
        for r in indoor:
            rule = rule_for(r["room_type"])
            mx, my = r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0
            z9 = zone9_of_point(mx, my, env)
            z16 = zone16_from_bearing(bearing_of_point(mx, my, env, north_offset_deg))
            grade = rule.grade(z9) if vastu_enabled else "neutral"
            score = GRADE_SCORE[grade]
            items.append({"floor": fi, "room": r.get("label", r["room_type"]), "room_type": r["room_type"],
                          "zone": z9, "zone16": z16, "grade": grade, "score": round(score, 2),
                          "ideal_zones": rule.ideal, "acceptable_zones": rule.ok})
            total_w += rule.weight
            total_s += rule.weight * score
            if vastu_enabled and grade in ("forbidden", "avoid") and (rule.hard or grade == "forbidden"):
                issues.append({"severity": "violation" if grade == "forbidden" else "warning", "floor": fi,
                               "room": r.get("label"), "zone": z9,
                               "message": f"{r.get('label')} sits in the {z9} zone. {rule.note}",
                               "better_zones": rule.ideal + rule.ok[:2]})
            if r["room_type"] in {"bathroom", "toilet", "restroom", "kitchen", "staircase", "store_room"} and vastu_enabled:
                if _overlap_area(r, centre_box) > 0.55 * r["w"] * r["h"] and env["w"] * env["h"] > 400:
                    issues.append({"severity": "warning", "floor": fi, "room": r.get("label"), "zone": "C",
                                   "message": f"{r.get('label')} occupies the Brahmasthan (centre); keep the centre open and light."})
            if fi == 0:
                ground_by_type.setdefault(r["room_type"], []).append(r)
        if vastu_enabled and residential:
            pool = [r for r in indoor if r["room_type"] == "pooja_room"]
            wet = [r for r in indoor if r["room_type"] in _TOILET_LIKE]
            kit = [r for r in indoor if r["room_type"] == "kitchen"]
            for p in pool:
                for t in wet:
                    if _shared_wall(p, t) > 1.0:
                        issues.append({"severity": "warning", "floor": fi, "room": p.get("label"),
                                       "message": f"{p.get('label')} shares a wall with {t.get('label')}; keep the prayer room away from toilets."})
            for k in kit:
                for t in wet:
                    if _shared_wall(k, t) > 1.0:
                        issues.append({"severity": "advisory", "floor": fi, "room": k.get("label"),
                                       "message": f"{k.get('label')} shares a wall with {t.get('label')}; a buffer (passage/utility) is preferred."})
            if fi > 0:
                for t in wet:
                    for lower in ground_by_type.get("pooja_room", []) + ground_by_type.get("kitchen", []):
                        if _overlap_area(t, lower) > 6.0:
                            issues.append({"severity": "warning", "floor": fi, "room": t.get("label"),
                                           "message": f"{t.get('label')} is directly above the ground-floor {lower.get('label')}."})
    # The road side and the requested main-door facing are separate inputs.
    # Entrance Vastu must be audited against the actual requested entrance side,
    # not whichever side the road happens to occupy.
    entrance_facing = entrance_direction or road_direction
    entrance = entrance_advice(entrance_facing)
    ez = entrance_zone16(entrance_facing, door_position)
    if ez in entrance["best_zones"]:
        e_grade = "ideal"
    elif ez in entrance["acceptable_zones"]:
        e_grade = "acceptable"
    elif ez in entrance["avoid_zones"]:
        e_grade = "avoid"
    else:
        e_grade = "neutral"
    if vastu_enabled and e_grade == "avoid":
        issues.append({"severity": "warning", "message": f"Main door in {ez} is not favoured. {entrance['note']}"})
    pct = round(100.0 * total_s / total_w, 1) if total_w else 0.0
    grade = "Excellent" if pct >= 85 else "Good" if pct >= 70 else "Fair" if pct >= 55 else "Needs work"
    counts = {g: sum(1 for i in items if i["grade"] == g) for g in GRADE_SCORE}
    return {
        "enabled": bool(vastu_enabled), "score_percent": pct, "grade": grade if vastu_enabled else "Not applied",
        "counts": counts, "rooms": items, "issues": issues,
        "entrance": {**entrance, "door_position": door_position, "door_zone16": ez, "grade": e_grade},
        "north_offset_deg": round(float(north_offset_deg or 0.0), 1),
        "method": "9-zone grid + 16-zone bearing audit with capacity-aware assignment",
    }
