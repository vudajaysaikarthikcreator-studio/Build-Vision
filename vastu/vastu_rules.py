"""Rule-based Vastu arrangement rules for residential and common institutional spaces."""

from planning.building_programs import PREFERRED_ZONES, ACCEPTABLE_ZONES

ZONE_GRID = (("NW", "N", "NE"), ("W", "C", "E"), ("SW", "S", "SE"))
COLORS = {
    "master_bedroom":"#DCE6F2", "bedroom":"#E8EEF6", "bathroom":"#EAF4F4", "kitchen":"#F5E8D8",
    "living_room":"#E6F0E1", "dining_room":"#F2E8D5", "pooja_room":"#F8E6C8", "study_room":"#E4E0F2",
    "store_room":"#ECECEC", "balcony":"#E0F2EE", "parking":"#F0F0F0", "garden":"#E5F1E1", "staircase":"#E8E8E8",
}

def room_color(room_type: str) -> str:
    return COLORS.get(room_type, "#E8EEF6")
