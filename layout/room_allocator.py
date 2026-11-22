from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional, Tuple

from planning.building_programs import ROOM_DIMS

TARGET_AREA = {k: v[0] * v[1] for k, v in ROOM_DIMS.items()}

@dataclass
class RoomAllocation:
    room_type: str
    label: str
    preferred_zone: str
    zone: str
    matched: bool
    target_area: float
    dims: Optional[Tuple[float, float]] = None
    min_dims: Optional[Tuple[float, float]] = None
    fixed_floor: Optional[int] = None
    custom: bool = False
    fixed_zone: bool = False
    hint_zones: Optional[list] = None
    plan_scale: float = 1.0          # linear growth of the reference size, set by layout/sizing.py


def allocate(assignments) -> List[RoomAllocation]:
    out = []
    for a in assignments:
        dims = getattr(a, "dims", None)
        area = dims[0] * dims[1] if dims else TARGET_AREA.get(a.room_type, 70)
        out.append(RoomAllocation(a.room_type, a.label, a.preferred_zone, a.zone, a.matched, area,
                                  dims, getattr(a, "min_dims", None), getattr(a, "fixed_floor", None),
                                  bool(getattr(a, "custom", False)), bool(getattr(a, "fixed_zone", False)),
                                  getattr(a, "hint_zones", None)))
    return out

def group_by_zone(allocations):
    grouped = defaultdict(list)
    for a in allocations:
        grouped[a.zone].append(a)
    return dict(grouped)
