"""Building-type space programming and common planning algorithms.

The program layer converts a high-level building type into explicit spaces.
It is intentionally deterministic and auditable: room counts are derived from
user inputs and no space is silently invented after this stage.
"""
from __future__ import annotations

from collections import Counter

# Target dimensions are (depth_ft, frontage_ft). Minima follow the same order.
ROOM_DIMS = {
    # Residential
    "master_bedroom": (13.0, 13.0), "bedroom": (11.5, 12.0), "bathroom": (5.0, 7.0),
    "kitchen": (9.0, 10.0), "living_room": (12.0, 15.0), "dining_room": (9.0, 10.0),
    "pooja_room": (4.0, 5.0), "study_room": (8.0, 9.0), "store_room": (5.0, 6.0),
    "balcony": (5.0, 10.0), "parking": (16.0, 10.0), "garden": (10.0, 7.0),
    "staircase": (12.0, 6.0),
    # Shared/common
    "reception": (10.0, 12.0), "lobby": (12.0, 20.0), "waiting_area": (12.0, 15.0),
    "corridor_lobby": (8.0, 18.0), "security_room": (6.0, 8.0),
    "pantry": (6.0, 8.0), "toilet": (5.0, 7.0), "restroom": (8.0, 10.0),
    "service_room": (8.0, 10.0),
    # Office / company
    "office_cabin": (8.0, 10.0), "open_work_area": (18.0, 24.0),
    "meeting_room": (10.0, 14.0), "conference_room": (14.0, 20.0),
    "manager_office": (10.0, 12.0), "director_office": (12.0, 14.0),
    "server_room": (8.0, 10.0), "hr_room": (8.0, 10.0), "break_room": (8.0, 10.0),
    # Education
    "classroom": (20.0, 25.0), "principal_office": (10.0, 12.0), "staff_room": (12.0, 15.0),
    "library": (15.0, 20.0), "computer_lab": (18.0, 24.0), "science_lab": (18.0, 24.0),
    "multipurpose_hall": (25.0, 40.0), "auditorium": (30.0, 50.0),
    "admin_office": (10.0, 12.0), "faculty_room": (12.0, 15.0),
    # Hospitality / hostel
    "hostel_room": (10.0, 12.0), "warden_room": (10.0, 12.0),
    "common_room": (15.0, 18.0), "dining_hall": (25.0, 30.0), "guest_room": (12.0, 14.0),
    "laundry": (8.0, 12.0), "housekeeping": (8.0, 10.0),
    # Healthcare
    "consultation_room": (10.0, 12.0), "treatment_room": (12.0, 14.0),
    "procedure_room": (12.0, 14.0), "pharmacy": (8.0, 10.0), "nurse_station": (8.0, 10.0),
    "ward": (20.0, 25.0), "diagnostic_lab": (18.0, 24.0),
    # Retail / restaurant
    "showroom": (25.0, 30.0), "cashier": (6.0, 10.0), "stock_room": (10.0, 12.0),
    "restaurant_dining": (25.0, 35.0), "restaurant_kitchen": (15.0, 20.0),
    "wash_area": (8.0, 10.0),
    # Industrial / warehouse
    "storage_hall": (30.0, 40.0), "loading_bay": (15.0, 25.0),
    "production_hall": (30.0, 50.0), "utility_room": (10.0, 12.0),
    "warehouse_office": (10.0, 12.0),
    # Banking / community / apartment
    "teller_area": (15.0, 20.0), "vault": (10.0, 12.0),
    "stage": (20.0, 30.0), "apartment_unit": (20.0, 30.0),
    # User-defined space (name and size come from the request's custom_rooms)
    "custom_room": (10.0, 12.0),
}

MIN_DIMS = {
    # Residential
    "master_bedroom": (9.5, 10.0), "bedroom": (9.0, 9.0), "bathroom": (4.0, 5.5),
    "kitchen": (7.0, 7.0), "living_room": (9.0, 10.0), "dining_room": (7.0, 8.0),
    "pooja_room": (3.5, 4.0), "study_room": (6.0, 7.0), "store_room": (4.0, 5.0),
    "balcony": (4.0, 6.0), "parking": (8.0, 10.0), "garden": (5.0, 5.0),
    "staircase": (6.0, 5.0),
    # Shared/common
    "reception": (8.0, 10.0), "lobby": (8.0, 12.0), "waiting_area": (8.0, 10.0),
    "corridor_lobby": (6.0, 10.0), "security_room": (5.0, 6.0),
    "pantry": (5.0, 6.0), "toilet": (4.0, 5.5), "restroom": (6.0, 8.0), "service_room": (6.0, 8.0),
    # Office
    "office_cabin": (7.0, 8.0), "open_work_area": (15.0, 18.0), "meeting_room": (8.0, 10.0),
    "conference_room": (10.0, 14.0), "manager_office": (8.0, 10.0), "director_office": (9.0, 10.0),
    "server_room": (6.0, 8.0), "hr_room": (7.0, 8.0), "break_room": (6.0, 8.0),
    # Education
    "classroom": (15.0, 18.0), "principal_office": (8.0, 10.0), "staff_room": (9.0, 12.0),
    "library": (12.0, 15.0), "computer_lab": (14.0, 18.0), "science_lab": (14.0, 18.0),
    "multipurpose_hall": (18.0, 24.0), "auditorium": (24.0, 40.0), "admin_office": (8.0, 10.0),
    "faculty_room": (9.0, 12.0),
    # Hospitality
    "hostel_room": (8.0, 10.0), "warden_room": (8.0, 10.0), "common_room": (12.0, 14.0),
    "dining_hall": (18.0, 24.0), "guest_room": (9.0, 11.0), "laundry": (6.0, 8.0), "housekeeping": (6.0, 8.0),
    # Healthcare
    "consultation_room": (8.0, 10.0), "treatment_room": (9.0, 10.0), "procedure_room": (9.0, 10.0),
    "pharmacy": (6.0, 8.0), "nurse_station": (6.0, 8.0), "ward": (16.0, 20.0), "diagnostic_lab": (14.0, 18.0),
    # Retail/restaurant
    "showroom": (18.0, 20.0), "cashier": (5.0, 7.0), "stock_room": (8.0, 10.0),
    "restaurant_dining": (18.0, 24.0), "restaurant_kitchen": (12.0, 15.0), "wash_area": (6.0, 8.0),
    # Industrial
    "storage_hall": (20.0, 25.0), "loading_bay": (10.0, 18.0), "production_hall": (25.0, 35.0),
    "utility_room": (8.0, 10.0), "warehouse_office": (8.0, 10.0),
    # Banking/community/apartment
    "teller_area": (10.0, 15.0), "vault": (8.0, 10.0), "stage": (15.0, 20.0), "apartment_unit": (16.0, 20.0),
    "custom_room": (5.0, 6.0),
}

ROOM_LABELS = {
    "master_bedroom": "Master Bedroom", "bedroom": "Bedroom", "bathroom": "Bathroom", "kitchen": "Kitchen",
    "living_room": "Living Room", "dining_room": "Dining Room", "pooja_room": "Pooja Room", "study_room": "Study Room",
    "store_room": "Store Room", "balcony": "Balcony", "parking": "Parking", "garden": "Garden", "staircase": "Staircase",
    "reception": "Reception", "lobby": "Lobby", "waiting_area": "Waiting Area", "corridor_lobby": "Lobby / Circulation",
    "security_room": "Security Room", "pantry": "Pantry", "toilet": "Toilet", "restroom": "Restroom", "service_room": "Service Room",
    "office_cabin": "Office Cabin", "open_work_area": "Open Work Area", "meeting_room": "Meeting Room", "conference_room": "Conference Room",
    "manager_office": "Manager Office", "director_office": "Director Office", "server_room": "Server Room", "hr_room": "HR Room", "break_room": "Break Room",
    "classroom": "Classroom", "principal_office": "Principal Office", "staff_room": "Staff Room", "library": "Library",
    "computer_lab": "Computer Lab", "science_lab": "Science Lab", "multipurpose_hall": "Multipurpose Hall", "auditorium": "Auditorium",
    "admin_office": "Admin Office", "faculty_room": "Faculty Room", "hostel_room": "Hostel Room", "warden_room": "Warden Room",
    "common_room": "Common Room", "dining_hall": "Dining Hall", "guest_room": "Guest Room", "laundry": "Laundry", "housekeeping": "Housekeeping",
    "consultation_room": "Consultation Room", "treatment_room": "Treatment Room", "procedure_room": "Procedure Room", "pharmacy": "Pharmacy",
    "nurse_station": "Nurse Station", "ward": "Ward", "diagnostic_lab": "Diagnostic Lab", "showroom": "Showroom", "cashier": "Cashier",
    "stock_room": "Stock Room", "restaurant_dining": "Dining Hall", "restaurant_kitchen": "Kitchen", "wash_area": "Wash Area",
    "storage_hall": "Storage Hall", "loading_bay": "Loading Bay", "production_hall": "Production Hall", "utility_room": "Utility Room",
    "warehouse_office": "Office", "teller_area": "Teller Area", "vault": "Vault", "stage": "Stage", "apartment_unit": "Apartment Unit",
    "custom_room": "Custom Room",
}

# Preferred Vastu zones are useful as arrangement guidance where Vastu is enabled.
# Non-residential spaces use broad, practical zones rather than pretending there is
# one universal rule for every institutional building.
PREFERRED_ZONES = {
    "master_bedroom": "SW", "bedroom": "NW", "bathroom": "NW", "kitchen": "SE", "living_room": "N", "dining_room": "E",
    "pooja_room": "NE", "study_room": "NE", "store_room": "W", "balcony": "N", "parking": "NW", "garden": "N", "staircase": "S",
    "reception": "N", "lobby": "N", "waiting_area": "NE", "security_room": "NW", "pantry": "SE", "toilet": "NW", "restroom": "NW",
    "service_room": "NW", "office_cabin": "W", "open_work_area": "N", "meeting_room": "E", "conference_room": "E", "manager_office": "SW",
    "director_office": "SW", "server_room": "SE", "hr_room": "W", "break_room": "SE", "classroom": "N", "principal_office": "SW", "staff_room": "W",
    "library": "NE", "computer_lab": "SE", "science_lab": "SE", "multipurpose_hall": "N", "auditorium": "N", "admin_office": "W", "faculty_room": "W",
    "hostel_room": "W", "warden_room": "SW", "common_room": "N", "dining_hall": "E", "guest_room": "W", "laundry": "NW", "housekeeping": "NW",
    "consultation_room": "E", "treatment_room": "E", "procedure_room": "E", "pharmacy": "SE", "nurse_station": "N", "ward": "N", "diagnostic_lab": "SE",
    "showroom": "N", "cashier": "SE", "stock_room": "SW", "restaurant_dining": "N", "restaurant_kitchen": "SE", "wash_area": "NW",
    "storage_hall": "SW", "loading_bay": "W", "production_hall": "W", "utility_room": "NW", "warehouse_office": "SW", "teller_area": "E", "vault": "SW",
    "stage": "S", "apartment_unit": "W", "custom_room": "W",
}

# Explicit acceptable Vastu zones.  The previous implementation treated every
# direction as acceptable, which made the reported Vastu result meaningless.
# These are traditional planning conventions, not building-code requirements.
_ACCEPTABLE_BY_GROUP = {
    "master_bedroom": ["SW", "S", "W"],
    "bedroom": ["NW", "W", "S", "SW", "N"],
    "bathroom": ["NW", "W", "SE", "S"],
    "toilet": ["NW", "W", "SE", "S"],
    "restroom": ["NW", "W", "SE", "S"],
    "kitchen": ["SE", "NW"],
    "living_room": ["N", "NE", "E"],
    "dining_room": ["W", "E", "N", "S", "NW"],
    "pooja_room": ["NE", "N", "E"],
    "study_room": ["NE", "N", "E", "W"],
    "store_room": ["SW", "W", "S"],
    "balcony": ["N", "E", "NE"],
    "parking": ["NW", "SE", "N", "E"],
    "garden": ["N", "E", "NE"],
    "staircase": ["S", "SW", "W", "SE"],
    "reception": ["N", "E", "NE", "W"],
    "lobby": ["N", "E", "NE", "W"],
    "waiting_area": ["N", "E", "NE", "W"],
    "security_room": ["NW", "W", "S"],
    "pantry": ["SE", "NW"],
    "service_room": ["NW", "W", "SE", "S"],
    "office_cabin": ["W", "N", "E"],
    "open_work_area": ["N", "E", "W"],
    "meeting_room": ["E", "N", "W", "NE"],
    "conference_room": ["E", "N", "W", "NE"],
    "manager_office": ["SW", "W", "S"],
    "director_office": ["SW", "W", "S"],
    "server_room": ["SE", "S", "W"],
    "hr_room": ["W", "N", "E"],
    "break_room": ["SE", "E", "NW"],
    "classroom": ["N", "E", "W", "NE", "NW"],
    "principal_office": ["SW", "W", "S"],
    "staff_room": ["W", "N", "E"],
    "library": ["NE", "N", "E", "W"],
    "computer_lab": ["SE", "E", "N", "W"],
    "science_lab": ["SE", "E", "N", "W"],
    "multipurpose_hall": ["N", "E", "W", "NE"],
    "auditorium": ["N", "E", "W", "NE"],
    "admin_office": ["W", "N", "E"],
    "faculty_room": ["W", "N", "E"],
    "hostel_room": ["W", "NW", "N", "S"],
    "warden_room": ["SW", "W", "S"],
    "common_room": ["N", "E", "W", "NE"],
    "dining_hall": ["E", "W", "S"],
    "guest_room": ["NW", "W", "N", "S"],
    "laundry": ["NW", "W", "SE", "S"],
    "housekeeping": ["NW", "W", "SE", "S"],
    "consultation_room": ["E", "NE", "N", "W"],
    "treatment_room": ["E", "N", "W", "NE"],
    "procedure_room": ["E", "N", "W", "NE"],
    "pharmacy": ["SE", "E", "N"],
    "nurse_station": ["N", "E", "W"],
    "ward": ["N", "E", "W"],
    "diagnostic_lab": ["SE", "E", "N", "W"],
    "showroom": ["N", "E", "NE", "W"],
    "cashier": ["SE", "E", "N"],
    "stock_room": ["SW", "W", "S"],
    "restaurant_dining": ["N", "E", "W", "NE"],
    "restaurant_kitchen": ["SE", "NW"],
    "wash_area": ["NW", "W", "SE", "S"],
    "storage_hall": ["SW", "W", "S"],
    "loading_bay": ["W", "NW", "SE"],
    "production_hall": ["W", "S", "SE", "E"],
    "utility_room": ["NW", "W", "SE", "S"],
    "warehouse_office": ["SW", "W", "N"],
    "teller_area": ["E", "N", "W"],
    "vault": ["SW", "W", "S"],
    "stage": ["S", "W", "E"],
    "apartment_unit": ["W", "NW", "N", "E", "S"],
}
_ACCEPTABLE_BY_GROUP["custom_room"] = ["W", "NW", "N", "E", "S", "SW"]
ACCEPTABLE_ZONES = {k: v for k, v in _ACCEPTABLE_BY_GROUP.items()}
for _k, _v in PREFERRED_ZONES.items():
    ACCEPTABLE_ZONES.setdefault(_k, [_v])

BUILDING_TITLES = {
    "1bhk": "1 BHK House", "2bhk": "2 BHK House", "3bhk": "3 BHK House", "4bhk": "4 BHK House",
    "duplex": "Duplex House", "villa": "Villa", "studio": "Studio Apartment", "farmhouse": "Farmhouse",
    "office": "Office / Company", "company": "Office / Company", "school": "School", "college": "College",
    "hostel": "Hostel", "hospital": "Hospital", "clinic": "Clinic", "hotel": "Hotel", "restaurant": "Restaurant",
    "retail": "Retail / Shop", "mall": "Retail / Mall", "warehouse": "Warehouse", "factory": "Factory / Industrial",
    "community_hall": "Community Hall", "bank": "Bank", "apartment_block": "Apartment Block", "custom": "Custom Building",
}
ALIASES = {"company": "office", "commercial": "office", "shop": "retail", "shopping_mall": "mall", "industrial": "factory", "apartment": "apartment_block"}

RESIDENTIAL = {"1bhk", "2bhk", "3bhk", "4bhk", "duplex", "villa", "studio", "farmhouse", "custom"}

# Baseline plot sizes keep default presets feasible. Users can override them.
DEFAULT_PLOTS = {
    "1bhk": (25, 40), "2bhk": (30, 40), "3bhk": (30, 50), "4bhk": (40, 60), "duplex": (40, 60),
    "villa": (50, 80), "studio": (25, 35), "farmhouse": (60, 80), "office": (60, 100), "school": (100, 150),
    "college": (150, 200), "hostel": (60, 100), "hospital": (100, 150), "clinic": (40, 60), "hotel": (100, 150),
    "restaurant": (40, 60), "retail": (40, 60), "mall": (100, 150), "warehouse": (80, 120), "factory": (120, 200),
    "community_hall": (80, 120), "bank": (50, 80), "apartment_block": (100, 150), "custom": (40, 60),
}

DEFAULT_SPACE_COUNT = {
    "office": 8, "school": 8, "college": 12, "hostel": 10, "hospital": 6, "clinic": 3, "hotel": 12,
    "restaurant": 1, "retail": 1, "mall": 4, "warehouse": 1, "factory": 1, "community_hall": 1, "bank": 4,
    "apartment_block": 4,
}


def canonical_type(value: str | None) -> str:
    raw = str(value or "custom").strip().lower().replace(" ", "_").replace("-", "_")
    return ALIASES.get(raw, raw if raw in BUILDING_TITLES else "custom")


def property_title(value: str | None) -> str:
    return BUILDING_TITLES.get(canonical_type(value), "Custom Building")


def _repeat(room_type: str, count: int) -> list[str]:
    return [room_type] * max(0, int(count))


def _residential_program(building_type: str, bedrooms: int, bathrooms: int, kitchens: int) -> list[str]:
    bedrooms = max(1, int(bedrooms))
    rooms = ["master_bedroom"] + _repeat("bedroom", bedrooms - 1)
    rooms += _repeat("bathroom", max(1, bathrooms))
    rooms += _repeat("kitchen", max(1, kitchens))
    rooms += ["living_room", "dining_room"]
    if building_type == "studio":
        rooms = ["master_bedroom", "bathroom", "kitchen", "living_room"]
    if building_type == "farmhouse":
        rooms += ["garden"]
    if building_type in {"duplex", "villa"}:
        rooms += ["staircase", "balcony"]
    return rooms


def build_room_list(req) -> list[str]:
    """Create an explicit space program from the request."""
    b = canonical_type(getattr(req, "building_type", "custom"))
    count = max(1, int(getattr(req, "space_count", 0) or getattr(req, "bedrooms", 1)))
    baths = max(1, int(getattr(req, "bathrooms", 1)))
    kitchens = max(1, int(getattr(req, "kitchens", 1)))
    rooms: list[str]
    if b in RESIDENTIAL:
        rooms = _residential_program(b, getattr(req, "bedrooms", count), baths, kitchens)
        # Respect explicit toggles for common residential features.
        if not getattr(req, "living_room", True): rooms = [r for r in rooms if r != "living_room"]
        if not getattr(req, "dining_room", True): rooms = [r for r in rooms if r != "dining_room"]
        for field, rt in (("pooja_room", "pooja_room"), ("study_room", "study_room"), ("store_room", "store_room"), ("balcony", "balcony"), ("garden", "garden")):
            if getattr(req, field, False): rooms.append(rt)
        # Duplex/villa explicitly need a staircase when multi-floor.
        if getattr(req, "staircase", True) and "staircase" not in rooms and int(getattr(req, "floors", 1)) > 1:
            rooms.append("staircase")
    elif b == "office":
        rooms = ["reception", "open_work_area", "conference_room", "meeting_room", "manager_office", "director_office"]
        rooms += _repeat("office_cabin", max(1, count))
        rooms += ["pantry", "server_room", "hr_room", "break_room"]
        rooms += _repeat("restroom", baths)
    elif b == "school":
        rooms = ["reception", "principal_office", "staff_room", "library", "computer_lab", "science_lab", "multipurpose_hall"]
        rooms += _repeat("classroom", count)
        rooms += _repeat("restroom", baths)
    elif b == "college":
        rooms = ["reception", "admin_office", "faculty_room", "library", "computer_lab", "science_lab", "auditorium"]
        rooms += _repeat("classroom", count)
        rooms += _repeat("restroom", baths)
    elif b == "hostel":
        rooms = ["reception", "warden_room", "common_room", "dining_hall", "kitchen", "laundry", "housekeeping"]
        rooms += _repeat("hostel_room", count) + _repeat("restroom", baths)
    elif b == "hospital":
        rooms = ["reception", "waiting_area", "nurse_station", "pharmacy", "diagnostic_lab", "treatment_room"]
        rooms += _repeat("consultation_room", count) + _repeat("restroom", baths) + ["staircase"]
    elif b == "clinic":
        rooms = ["reception", "waiting_area", "procedure_room", "pharmacy"]
        rooms += _repeat("consultation_room", count) + _repeat("restroom", baths)
    elif b == "hotel":
        rooms = ["reception", "lobby", "dining_hall", "kitchen", "laundry", "housekeeping", "staircase"]
        rooms += _repeat("guest_room", count) + _repeat("restroom", baths)
    elif b == "restaurant":
        rooms = ["reception", "restaurant_dining", "restaurant_kitchen", "pantry", "wash_area"] + _repeat("restroom", baths)
    elif b in {"retail", "mall"}:
        rooms = ["reception", "showroom", "cashier", "stock_room", "manager_office"] + _repeat("showroom", count - 1) + _repeat("restroom", baths)
    elif b == "warehouse":
        rooms = ["storage_hall", "loading_bay", "warehouse_office", "utility_room"] + _repeat("restroom", baths)
    elif b == "factory":
        rooms = ["production_hall", "warehouse_office", "storage_hall", "loading_bay", "utility_room"] + _repeat("restroom", baths)
    elif b == "community_hall":
        rooms = ["lobby", "multipurpose_hall", "stage", "kitchen", "store_room", "parking"] + _repeat("restroom", baths)
    elif b == "bank":
        rooms = ["reception", "lobby", "teller_area", "manager_office", "vault", "server_room"] + _repeat("office_cabin", count) + _repeat("restroom", baths)
    elif b == "apartment_block":
        rooms = ["reception", "lobby", "staircase"] + _repeat("apartment_unit", count)
        rooms += _repeat("parking", 1) + _repeat("restroom", baths)
    else:
        rooms = _residential_program("custom", getattr(req, "bedrooms", count), baths, kitchens)

    # Parking is a toggle, while staircase is structurally required for multi-floor plans.
    if getattr(req, "parking", False) and b not in {"community_hall", "apartment_block"}:
        rooms.append("parking")
    if getattr(req, "floors", 1) > 1 and "staircase" not in rooms and getattr(req, "staircase", True):
        rooms.append("staircase")
    return rooms


def validate_program(room_list: list[str]) -> dict:
    counts = Counter(room_list)
    unknown = sorted({r for r in room_list if r not in ROOM_DIMS})
    return {"room_count": len(room_list), "counts": dict(counts), "unknown_types": unknown, "valid": not unknown}


def algorithm_trace(building_type: str, vastu: bool, floors: int) -> list[dict]:
    return [
        {"step": 1, "algorithm": "Input Validation & Space Programming", "purpose": "Validate plot, counts, floors and convert building type to an explicit space program."},
        {"step": 2, "algorithm": "Decision Tree Requirement Analysis", "purpose": "Classify compact/balanced/spacious/luxury planning needs from plot and requirements."},
        {"step": 3, "algorithm": "Rule-Based Vastu Arrangement" if vastu else "Rule-Based Functional Zoning", "purpose": "Assign preferred/acceptable directional zones before geometry placement."},
        {"step": 4, "algorithm": "Dimension-First Layout Generation", "purpose": "Place spaces using target/minimum dimensions, setbacks and building-type priorities."},
        {"step": 5, "algorithm": "Constraint & Collision Checking", "purpose": "Reject overlaps, boundary violations and minimum-dimension violations."},
        {"step": 6, "algorithm": "Architectural Dimension Annotation", "purpose": "Render room, building-footprint and overall plot dimensions in feet and inches."},
        {"step": 7, "algorithm": "3D-Ready Geometry Generation", "purpose": "Reuse validated room coordinates for the interactive multi-floor 3D view."},
        {"step": 8, "algorithm": "Random Forest / Gradient Boosting Cost Estimation", "purpose": "Estimate construction cost with model and formula cross-checks."},
        {"step": 9, "algorithm": "Preliminary Civil Structure & Bylaw Checks", "purpose": "Provide indicative structural sizing, materials and setback/FAR checks."},
    ]
