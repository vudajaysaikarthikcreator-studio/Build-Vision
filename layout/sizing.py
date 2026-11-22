"""Plot-aware room sizing.

Every room type has a fixed reference size in ``planning/building_programs.ROOM_DIMS``
(a classroom is 20 x 25 ft, a living room 12 x 15 ft ...).  Those numbers do not know
anything about the plot, so on an 80 x 50 ft villa plot the planners tiled ~700 sq.ft
of rooms in the middle of 4,000 sq.ft of land, and on a school plot they squeezed the
rooms into whatever odd shapes were left.

This module computes, per floor, one *plan scale* for every room so that the rooms
together occupy a realistic share of the plot (``COVERAGE``).  Habitable rooms grow
fully, service rooms (stairs, washrooms, store ...) grow only a little, large halls grow
gently, and every room type has a maximum sensible area, so a bathroom never balloons to
fill leftover land.

The scale is stored on the allocation as ``plan_scale`` (linear factor applied to both
reference dimensions).  It is deliberately NOT written to ``dims``: an explicit ``dims``
means "the user asked for exactly this size" and routes the plan to the custom-size engine.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from planning.building_programs import ROOM_DIMS, canonical_type

SITE_TYPES = {"parking", "garden", "loading_bay"}

# Share of the *plot* the rooms of one floor should cover.  Types that are not listed
# (the 1-4 BHK homes, studio) keep their reference sizes: they were already well proportioned.
COVERAGE: Dict[str, float] = {
    "duplex": 0.40, "villa": 0.38, "farmhouse": 0.30,
    "office": 0.48, "school": 0.38, "college": 0.36, "hostel": 0.46,
    "hospital": 0.42, "clinic": 0.52, "hotel": 0.44, "restaurant": 0.50,
    "retail": 0.58, "mall": 0.46, "bank": 0.52, "apartment_block": 0.48,
    # warehouse / factory / community_hall: intentionally absent - single-hall buildings keep their reference sizes
}

# Growth exponent: 1.0 = grows fully with the habitable scale; smaller = grows less.
_SERVICE = {
    "staircase": 0.25, "bathroom": 0.30, "toilet": 0.30, "restroom": 0.45, "pooja_room": 0.25,
    "store_room": 0.40, "security_room": 0.35, "pantry": 0.45, "server_room": 0.40,
    "cashier": 0.40, "vault": 0.40, "utility_room": 0.45, "service_room": 0.45,
    "nurse_station": 0.45, "wash_area": 0.45, "laundry": 0.50, "housekeeping": 0.50,
    "corridor_lobby": 0.60, "lobby": 0.70, "balcony": 0.60,
    # large halls grow gently: a 60-bed hostel needs a bigger dining hall, not a 1,300 sq.ft one
    "dining_hall": 0.45, "multipurpose_hall": 0.45, "auditorium": 0.40, "stage": 0.45, "common_room": 0.65,
    "restaurant_dining": 0.6, "showroom": 0.8,
}

# Largest sensible area (sq.ft) of one room of the type.
MAX_AREA: Dict[str, float] = {
    "master_bedroom": 300, "bedroom": 220, "guest_room": 300, "living_room": 420, "dining_room": 230,
    "kitchen": 200, "study_room": 170, "store_room": 70, "pooja_room": 36, "staircase": 160,
    "bathroom": 55, "toilet": 55, "restroom": 130, "balcony": 120,
    "reception": 260, "lobby": 400, "waiting_area": 320, "security_room": 70, "pantry": 110,
    "office_cabin": 190, "open_work_area": 1500, "meeting_room": 330, "conference_room": 560,
    "manager_office": 230, "director_office": 300, "server_room": 120, "hr_room": 190, "break_room": 220,
    "classroom": 640, "principal_office": 260, "staff_room": 380, "library": 760, "computer_lab": 720,
    "science_lab": 720, "multipurpose_hall": 1600, "auditorium": 2400, "admin_office": 260, "faculty_room": 380,
    "hostel_room": 190, "warden_room": 230, "common_room": 520, "dining_hall": 800,
    "laundry": 180, "housekeeping": 140,
    "consultation_room": 220, "treatment_room": 280, "procedure_room": 280, "pharmacy": 200, "nurse_station": 150,
    "ward": 600, "diagnostic_lab": 560,
    "showroom": 1300, "cashier": 90, "stock_room": 260, "restaurant_dining": 1500, "restaurant_kitchen": 520,
    "storage_hall": 6000, "production_hall": 7000, "utility_room": 180, "warehouse_office": 240,
    "teller_area": 560, "vault": 180, "stage": 800, "apartment_unit": 1100,
}


def _ref_area(room_type: str) -> float:
    a, b = ROOM_DIMS.get(room_type, (10.0, 12.0))
    return float(a) * float(b)


def base_dims(alloc, room_type: Optional[str] = None) -> Tuple[float, float]:
    """Reference (depth, frontage) of an allocation, including its plot scale.

    Explicit user sizes (``alloc.dims``) always win and are never scaled.
    """
    rt = room_type or getattr(alloc, "room_type", "custom_room")
    d = getattr(alloc, "dims", None)
    if d:
        return float(d[0]), float(d[1])
    a, b = ROOM_DIMS.get(rt, (10.0, 12.0))
    s = float(getattr(alloc, "plan_scale", 1.0) or 1.0)
    return float(a) * s, float(b) * s


def target_area_of(alloc) -> float:
    a, b = base_dims(alloc)
    return a * b


def _solve_factor(entries: List[dict], target: float) -> float:
    """Bisection for the habitable growth factor f so that sum(area_i(f)) == target."""
    def total(f: float) -> float:
        return sum(min(e["cap"], e["base"] * (f ** e["power"])) for e in entries)

    lo, hi = 0.78, 12.0            # rooms may shrink to ~88% of their reference size when the floor must be smaller
    if total(lo) >= target:
        return lo
    if total(hi) <= target:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if total(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def apply_plot_scaling(allocations: Iterable, plot_length: float, plot_width: float, building_type: str,
                       coverage: Optional[float] = None) -> float:
    """Set ``plan_scale`` on every allocation of ONE floor.  Returns the habitable growth factor."""
    allocs = list(allocations)
    bt = canonical_type(building_type)
    cov = coverage if coverage is not None else COVERAGE.get(bt)
    for a in allocs:
        a.plan_scale = 1.0
        a.target_area = target_area_of(a)
    if not cov:
        return 1.0
    indoor = [a for a in allocs if a.room_type not in SITE_TYPES and not getattr(a, "dims", None)]
    fixed_area = sum(float(a.dims[0]) * float(a.dims[1]) for a in allocs
                     if a.room_type not in SITE_TYPES and getattr(a, "dims", None))
    if not indoor:
        return 1.0
    entries = []
    for a in indoor:
        base = _ref_area(a.room_type)
        cap = max(base, MAX_AREA.get(a.room_type, base * 4.0))
        power = _SERVICE.get(a.room_type, 1.0)
        entries.append({"alloc": a, "base": base, "cap": cap, "power": power})
    target = max(cov * plot_length * plot_width - fixed_area, 0.0)
    f = _solve_factor(entries, target)
    for e in entries:
        area = min(e["cap"], e["base"] * (f ** e["power"]))
        e["alloc"].plan_scale = round((area / e["base"]) ** 0.5, 4)
        e["alloc"].target_area = target_area_of(e["alloc"])
    return f
