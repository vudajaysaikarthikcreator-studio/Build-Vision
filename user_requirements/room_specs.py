"""Custom room specifications supplied by the user.

``custom_rooms`` entries (all keys optional except ``type`` or ``name``)::

    {"type": "bedroom", "name": "Guest Bedroom", "count": 2,
     "width_ft": 12, "length_ft": 14,          # E-W x N-S size you want
     "min_width_ft": 10, "min_length_ft": 12,  # optional hard minimums
     "zone": "NW",                             # optional Vastu zone override
     "floor": 1}                               # optional 0-based floor

``room_size_overrides`` changes the size of every room of a built-in type::

    {"bedroom": {"width_ft": 12, "length_ft": 14}}
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from planning.building_programs import ROOM_DIMS, MIN_DIMS, ROOM_LABELS

VALID_ZONES = {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "C"}
_ZONE16 = {"NNE": "N", "ENE": "E", "ESE": "E", "SSE": "S", "SSW": "S", "WSW": "W", "WNW": "W", "NNW": "N",
           "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "CENTER": "C", "CENTRE": "C",
           "NORTHEAST": "NE", "NORTHWEST": "NW", "SOUTHEAST": "SE", "SOUTHWEST": "SW"}

# Friendly names -> built-in types (so "puja", "hall", "toilet" behave like the real thing).
TYPE_ALIASES = {
    "puja": "pooja_room", "pooja": "pooja_room", "prayer_room": "pooja_room", "temple": "pooja_room",
    "hall": "living_room", "drawing_room": "living_room", "living": "living_room", "lounge": "living_room",
    "master": "master_bedroom", "master_bed": "master_bedroom", "bed": "bedroom", "bedroom": "bedroom",
    "toilet": "bathroom", "washroom": "bathroom", "bath": "bathroom", "wc": "bathroom",
    "dining": "dining_room", "study": "study_room", "office_room": "study_room", "store": "store_room",
    "storage": "store_room", "garage": "parking", "car_parking": "parking", "lawn": "garden",
    "stairs": "staircase", "stair": "staircase", "terrace": "balcony", "cabin": "office_cabin",
    "waiting": "waiting_area", "lab": "science_lab", "canteen": "dining_hall", "kitchenette": "pantry",
}
# Where an unknown, user-named space usually goes in Vastu practice.
NAME_ZONE_HINTS = [
    (("gym", "fitness", "exercise"), ["SE", "W", "S"]),
    (("yoga", "meditation", "prayer"), ["NE", "E", "N"]),
    (("theatre", "theater", "cinema", "media"), ["W", "SE", "S"]),
    (("servant", "maid", "driver", "guard"), ["NW", "SE", "W"]),
    (("library", "reading"), ["NE", "N", "W"]),
    (("guest",), ["NW", "W", "N"]),
    (("play", "game", "kids", "child"), ["W", "NW", "N"]),
    (("wash", "laundry", "utility"), ["NW", "W", "SE"]),
    (("bar", "party"), ["W", "NW"]),
]


class RoomSpecError(ValueError):
    """A custom room entry is invalid."""


@dataclass
class CustomRoom:
    room_type: str
    label: str
    dims: Optional[Tuple[float, float]]        # (length_ns_ft, width_ew_ft) or None = type default
    min_dims: Optional[Tuple[float, float]]
    zone: Optional[str]
    floor: Optional[int]
    hint_zones: Optional[List[str]] = None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


def _clean_name(name) -> str:
    text = re.sub(r"[<>&\"'`]", "", str(name or "")).strip()
    return re.sub(r"\s+", " ", text)[:40]


def _dim(value, what: str, lo=3.0, hi=300.0) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise RoomSpecError(f"{what} must be a number of feet.")
    if not (lo <= f <= hi):
        raise RoomSpecError(f"{what} must be between {lo:g} ft and {hi:g} ft (got {f:g}).")
    return f


def normalize_zone(zone) -> Optional[str]:
    if zone in (None, "", "auto"):
        return None
    z = str(zone).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    z = _ZONE16.get(z, z)
    if z not in VALID_ZONES:
        raise RoomSpecError(f"Unknown zone '{zone}'. Use N, NE, E, SE, S, SW, W, NW or C (centre).")
    return z


def parse_custom_rooms(entries: Optional[Sequence[dict]], flex: float = 0.85,
                       max_total: int = 120) -> List[CustomRoom]:
    """Validate and expand ``custom_rooms`` into one CustomRoom per physical room."""
    out: List[CustomRoom] = []
    for i, e in enumerate(entries or [], 1):
        if not isinstance(e, dict):
            raise RoomSpecError(f"Custom room #{i} must be an object with type/name and sizes.")
        raw_type = _slug(e.get("type") or e.get("room_type") or "")
        name = _clean_name(e.get("name") or e.get("label"))
        if not raw_type and not name:
            raise RoomSpecError(f"Custom room #{i} needs a 'type' or a 'name'.")
        rtype = TYPE_ALIASES.get(raw_type, raw_type)
        if rtype not in ROOM_DIMS or rtype == "custom_room":
            probe = _slug(name or raw_type)
            rtype = TYPE_ALIASES.get(probe, probe)
            if rtype not in ROOM_DIMS or rtype == "custom_room":
                rtype = "custom_room"
        label_base = name or ROOM_LABELS.get(rtype, raw_type.replace("_", " ").title())
        try:
            count = int(e.get("count", 1))
        except (TypeError, ValueError):
            raise RoomSpecError(f"'{label_base}': count must be a whole number.")
        if not (1 <= count <= 50):
            raise RoomSpecError(f"'{label_base}': count must be between 1 and 50.")
        wd = _dim(e.get("width_ft", e.get("width")), f"'{label_base}' width")
        ln = _dim(e.get("length_ft", e.get("length")), f"'{label_base}' length")
        if (wd is None) != (ln is None):
            base = ROOM_DIMS.get(rtype, ROOM_DIMS["custom_room"])
            if wd is None:
                wd = base[1] if ln is None else max(3.0, round(base[0] * base[1] / ln, 1))
            else:
                ln = max(3.0, round(base[0] * base[1] / wd, 1))
        mw = _dim(e.get("min_width_ft"), f"'{label_base}' min width")
        ml = _dim(e.get("min_length_ft"), f"'{label_base}' min length")
        dims = (ln, wd) if wd and ln else None
        mins = None
        if dims:
            mins = (max(3.0, round((ml or ln * flex), 2)), max(3.0, round((mw or wd * flex), 2)))
            mins = (min(mins[0], ln), min(mins[1], wd))
        elif mw or ml:
            base = MIN_DIMS.get(rtype, MIN_DIMS["custom_room"])
            mins = (ml or base[0], mw or base[1])
        zone = normalize_zone(e.get("zone"))
        floor = e.get("floor")
        if floor not in (None, ""):
            try:
                floor = int(floor)
            except (TypeError, ValueError):
                raise RoomSpecError(f"'{label_base}': floor must be a whole number (0 = ground floor).")
            if not (0 <= floor <= 4):
                raise RoomSpecError(f"'{label_base}': floor must be between 0 and 4.")
        else:
            floor = None
        hints = None
        if rtype == "custom_room" and not zone:
            low = (name or raw_type).lower()
            for keys, zs in NAME_ZONE_HINTS:
                if any(k in low for k in keys):
                    hints = zs
                    break
        for n in range(count):
            label = label_base if count == 1 else f"{label_base} {n + 1}"
            out.append(CustomRoom(rtype, label, dims, mins, zone, floor, hints))
        if len(out) > max_total:
            raise RoomSpecError(f"Too many custom rooms (limit {max_total}).")
    return out


def parse_size_overrides(overrides: Optional[Dict[str, dict]], flex: float = 0.85) -> Dict[str, dict]:
    """{'bedroom': {'width_ft': 12, 'length_ft': 14}} -> {'bedroom': {'dims':(14,12),'min_dims':(...)}}"""
    out: Dict[str, dict] = {}
    for key, v in (overrides or {}).items():
        t = TYPE_ALIASES.get(_slug(key), _slug(key))
        if t not in ROOM_DIMS:
            raise RoomSpecError(f"Unknown room type '{key}' in room_size_overrides.")
        if not isinstance(v, dict):
            raise RoomSpecError(f"room_size_overrides['{key}'] must contain width_ft and length_ft.")
        wd = _dim(v.get("width_ft", v.get("width")), f"{key} width")
        ln = _dim(v.get("length_ft", v.get("length")), f"{key} length")
        if wd is None or ln is None:
            raise RoomSpecError(f"room_size_overrides['{key}'] needs both width_ft and length_ft.")
        mw = _dim(v.get("min_width_ft"), f"{key} min width")
        ml = _dim(v.get("min_length_ft"), f"{key} min length")
        mins = (min(ln, max(3.0, ml or ln * flex)), min(wd, max(3.0, mw or wd * flex)))
        out[t] = {"dims": (ln, wd), "min_dims": mins}
    return out
