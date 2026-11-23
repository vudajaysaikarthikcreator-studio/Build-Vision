from dataclasses import dataclass
from typing import List, Optional, Tuple
from .vastu_rules import PREFERRED_ZONES, ACCEPTABLE_ZONES
from planning.building_programs import ROOM_LABELS

@dataclass
class RoomAssignment:
    room_type: str
    label: str
    preferred_zone: str
    zone: str
    matched: bool
    # Optional per-room overrides (custom sizes / fixed floor). The standard
    # engine ignores them; the guaranteed engine and the pipeline honour them.
    dims: Optional[Tuple[float, float]] = None
    min_dims: Optional[Tuple[float, float]] = None
    fixed_floor: Optional[int] = None
    custom: bool = False
    fixed_zone: bool = False               # zone was chosen by the user, never re-assigned
    hint_zones: Optional[list] = None      # usual Vastu zones for a user-named space

# Common directional-planning guidance for repeated spaces. These are ordered
# preferences, not a universal religious/engineering standard; the geometry
# engine tries the first zone, then the next suitable zone for repeated rooms.
VASTU_ZONE_SEQUENCES = {
    # Traditional residential priorities used by the active placement engine.
    # Repeated rooms may share a zone; a second bathroom should not be pushed
    # into SE merely because NW/W is already occupied.
    "master_bedroom": ["SW"],
    "bedroom": ["NW", "S", "W", "NW"],
    "bathroom": ["NW", "W", "SE", "S"],
    "toilet": ["NW", "W", "SE", "S"],
    "restroom": ["NW", "W", "SE", "S"],
    "office_cabin": ["W", "NW", "N", "E"],
    "classroom": ["N", "E", "W", "NE", "NW", "SE"],
    "hostel_room": ["W", "NW", "N", "SW"],
    "guest_room": ["W", "NW", "N", "SW"],
    "consultation_room": ["E", "NE", "N", "W"],
    "showroom": ["N", "E", "W", "NE"],
    "apartment_unit": ["W", "NW", "N", "E", "SW"],
    "meeting_room": ["E", "N", "W", "NE"],
    "parking": ["NW", "N", "W", "NE"],
}


def _label(room_type: str, index: int, total: int) -> str:
    base = ROOM_LABELS.get(room_type, room_type.replace('_',' ').title())
    numbered = {
        "bedroom", "bathroom", "kitchen", "office_cabin", "classroom", "hostel_room", "guest_room",
        "consultation_room", "restroom", "showroom", "apartment_unit", "meeting_room", "toilet",
    }
    if room_type in numbered and total > 1:
        return f"{base} {index + 1}"
    return base


def _preferred_for_index(room_type: str, index: int) -> str:
    sequence = VASTU_ZONE_SEQUENCES.get(room_type)
    if sequence:
        return sequence[min(index, len(sequence) - 1)]
    return PREFERRED_ZONES.get(room_type, 'C')


def assign_rooms(room_list: List[str], vastu_enabled: bool=True, plot_area: float|None=None) -> List[RoomAssignment]:
    counts={}
    zone_counts={z:0 for row in (("NW","N","NE"),("W","C","E"),("SW","S","SE")) for z in row}
    result=[]
    for room_type in room_list:
        idx=counts.get(room_type,0); counts[room_type]=idx+1
        base_preferred=_preferred_for_index(room_type, idx)
        if not vastu_enabled:
            preferred='C'
            zone='C'
            matched=True
        else:
            preferred=base_preferred
            candidates=ACCEPTABLE_ZONES.get(room_type,[preferred])
            ordered=[z for z in VASTU_ZONE_SEQUENCES.get(room_type, []) if z in candidates]
            ordered += [z for z in candidates if z not in ordered]
            # The indexed preference returned above is authoritative. Repeated
            # spaces are allowed to share a directional zone; the geometry
            # engine separates their rectangles without changing the rule.
            zone=preferred if preferred in candidates else (ordered[0] if ordered else preferred)
            matched=(zone==preferred)
        zone_counts[zone]=zone_counts.get(zone,0)+1
        result.append(RoomAssignment(room_type,_label(room_type,idx,room_list.count(room_type)),preferred,zone,matched))
    return result
