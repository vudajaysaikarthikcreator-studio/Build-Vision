"""Plan-quality gate shared by every building type.

Collision/boundary checks only prove that rooms do not overlap. They do not
catch a "Principal Office" stretched into an 85 ft x 8 ft corridor, a 52 ft
restroom strip, or a warehouse that uses a quarter of its plot. This module
scores a finished floor layout on those architectural sanity rules so that
`safe_generate_layout` can pick the better of several candidate plans.
"""
from __future__ import annotations

from typing import Dict, List

# Largest sensible long-side : short-side ratio per room type.
_HALL = {"multipurpose_hall", "auditorium", "storage_hall", "production_hall", "showroom",
         "restaurant_dining", "dining_hall", "ward", "open_work_area", "stage", "common_room"}
_STRIP_OK = {"corridor_lobby", "lobby", "loading_bay", "parking", "staircase", "balcony", "garden",
             "waiting_area", "reception"}
_WET = {"bathroom", "toilet", "restroom", "wash_area", "laundry", "housekeeping"}
_SMALL = {"pooja_room", "store_room", "security_room", "pantry", "server_room", "vault",
          "cashier", "nurse_station", "utility_room", "service_room"}

# Utility spaces that must not balloon to fill leftover land (a 700 sq.ft staircase is wrong).
_CAP_AREA = {"staircase": 180.0, "restroom": 260.0, "bathroom": 90.0, "toilet": 90.0, "pooja_room": 40.0,
             "store_room": 100.0, "security_room": 90.0, "pantry": 130.0, "server_room": 140.0,
             "cashier": 110.0, "vault": 200.0, "utility_room": 200.0, "service_room": 140.0,
             "nurse_station": 150.0, "wash_area": 150.0, "laundry": 200.0, "housekeeping": 160.0}

# Rooms that are not part of the enclosed building mass.
_SITE = {"parking", "garden", "loading_bay"}


def max_aspect(room_type: str) -> float:
    if room_type == "corridor_lobby":
        return 30.0          # a corridor / family hall is meant to be long
    if room_type in _HALL:
        return 3.0
    if room_type in _STRIP_OK:
        return 4.5
    if room_type in _WET:
        return 2.6
    if room_type in _SMALL:
        return 2.6
    return 2.3          # bedrooms, offices, classrooms, labs, kitchens, living rooms ...


def evaluate(layout: Dict, building_type: str = "custom", expected_rooms: int | None = None) -> Dict:
    rooms: List[dict] = [r for r in layout.get("rooms", []) if r.get("w") and r.get("h")]
    issues: List[str] = []
    penalty = 0.0
    for r in rooms:
        w, h = float(r["w"]), float(r["h"])
        ratio = max(w, h) / max(min(w, h), 0.01)
        lim = max_aspect(r.get("room_type", ""))
        if ratio > lim:
            over = ratio / lim
            penalty += min(over - 1.0, 3.0) * 4.0
            issues.append(f"{r.get('label') or r.get('room_type')} is {w:.0f}x{h:.0f} ft (ratio {ratio:.1f}:1, max {lim:g}:1)")
    for r in rooms:
        cap = _CAP_AREA.get(r.get("room_type", ""))
        area = float(r["w"]) * float(r["h"])
        if cap and area > cap:
            penalty += min(area / cap - 1.0, 4.0) * 2.0
            issues.append(f"{r.get('label') or r.get('room_type')} is {area:.0f} sq.ft (expected <= {cap:.0f})")
    masters = [float(r["w"]) * float(r["h"]) for r in rooms if r.get("room_type") == "master_bedroom"]
    beds = [float(r["w"]) * float(r["h"]) for r in rooms if r.get("room_type") == "bedroom"]
    if masters and beds and max(masters) < 0.97 * max(beds):
        penalty += 1.5 + 4.0 * (max(beds) / max(masters) - 1.0)
        issues.append(f"master bedroom ({max(masters):.0f} sq.ft) is smaller than another bedroom ({max(beds):.0f} sq.ft)")
    if expected_rooms is not None and len(rooms) < expected_rooms:
        penalty += 50.0 * (expected_rooms - len(rooms))
        issues.append("not every requested space was placed")
    plot = layout.get("plot") or {}
    plot_area = float(plot.get("length", 0) or 0) * float(plot.get("width", 0) or 0)
    built = [r for r in rooms if r.get("room_type") not in _SITE]
    built_area = sum(float(r["w"]) * float(r["h"]) for r in built)
    coverage = built_area / plot_area if plot_area else 0.0
    gap = float((layout.get("gap_report") or {}).get("max_internal_gap_ft", 0.0) or 0.0)
    if gap > 1.0:
        penalty += gap * 1.5
        issues.append(f"{gap:.1f} ft internal gap")
    return {"penalty": round(penalty, 2), "issues": issues, "coverage": round(coverage, 3), "max_gap_ft": gap}
