"""
backend/pipeline.py

Runs the full BuildVision pipeline end to end, exactly in the order the
brief specifies (section 19):

  requirements -> input validation -> decision tree -> vastu arrangement
  -> constraint layout generation -> collision/boundary check
  -> 2D blueprint -> 3D-ready layout data -> cost estimation

Routes call `run_pipeline()`; nothing here talks to FastAPI or the
database directly, so it can also be exercised from a script or tests.
"""

import time
import uuid
from typing import TYPE_CHECKING

from ml.predict import (
    predict_planning_type, predict_planning_type_ensemble,
    predict_cost, predict_cost_ensemble,
)
from vastu.vastu_arrangement import assign_rooms
from vastu.vastu_rules import room_color
from layout.room_allocator import allocate
from layout.sizing import apply_plot_scaling
from layout.layout_generator import generate_layout, MIN_DIMS
from layout.safe_layout import safe_generate_layout, LayoutInfeasible
from vastu.vastu_engine import audit_layout, entrance_advice
from user_requirements import normalize_requirements, RequirementError
from blueprint.blueprint_generator import generate_floor_blueprints
from civil.structural_engineering import generate_structural_report
from planning.building_programs import build_room_list as program_room_list, canonical_type, property_title, validate_program, algorithm_trace, RESIDENTIAL, MIN_DIMS as _TYPE_MIN_DIMS
from copy import copy

if TYPE_CHECKING:
    from backend.schemas import HouseRequirementIn

RATE_TABLE = {"basic": 1350, "standard": 1800, "premium": 2450}


class InfeasibleError(ValueError):
    """The requested spaces cannot fit on the plot (message is user-facing)."""


def _build_room_list(req: "HouseRequirementIn"):
    """Standard program for the building type (empty in custom-rooms-only mode)."""
    rooms = [] if getattr(req, "custom_rooms_only", False) else program_room_list(req)
    program = validate_program(rooms)
    if not program["valid"]:
        raise ValueError("Unsupported space type(s): " + ", ".join(program["unknown_types"]))
    return rooms


def build_assignments(req: "HouseRequirementIn", room_list=None):
    """Vastu-assign the standard program plus any custom rooms / size overrides."""
    room_list = _build_room_list(req) if room_list is None else room_list
    specs = list(getattr(req, "_custom_specs", None) or [])
    overrides = getattr(req, "_size_overrides", None) or {}
    plot_area = req.plot_length * req.plot_width
    types = list(room_list) + [c.room_type for c in specs]
    assignments = assign_rooms(types, vastu_enabled=req.vastu_preference, plot_area=plot_area)
    notes = str(getattr(req, "design_notes", "") or "").lower()
    wants_attached_master_bath = (
        req.vastu_preference and
        str(getattr(req, "building_type", "custom") or "custom").lower() in RESIDENTIAL and
        ("attached toilet" in notes or "attached bathroom" in notes or "master bedroom with attached" in notes)
    )
    if wants_attached_master_bath:
        first_bath = next((a for a in assignments if a.room_type == "bathroom"), None)
        if first_bath is not None:
            # A bathroom immediately beside the SW master is a practical
            # adjacency preference. South is a configured acceptable service
            # zone, while NW/W remains the default for other bathrooms.
            first_bath.preferred_zone = "S"
            first_bath.zone = "S"
            first_bath.matched = True
            first_bath.fixed_zone = True
    for a in assignments[:len(room_list)]:
        ov = overrides.get(a.room_type)
        if ov:
            a.dims, a.min_dims = tuple(ov["dims"]), tuple(ov["min_dims"])
    # Industrial sheds are sized by the plot, not by a fixed 30x40 ft default: the main hall
    # takes ~15% of the plot (unless the user gave an explicit size for it).
    _bt = canonical_type(getattr(req, "building_type", "custom"))
    if _bt in {"warehouse", "factory"}:
        _hall = "storage_hall" if _bt == "warehouse" else "production_hall"
        for a in assignments[:len(room_list)]:
            if a.room_type == _hall and a.room_type not in overrides and not getattr(a, "dims", None):
                area = max(plot_area * 0.15, 1200.0)
                depth = round((area / 1.6) ** 0.5, 1)
                a.dims = (depth, round(area / depth, 1))
                a.min_dims = (round(depth * 0.7, 1), round(area / depth * 0.7, 1))
                break
    for a, c in zip(assignments[len(room_list):], specs):
        a.label = c.label
        a.custom = True
        a.dims = tuple(c.dims) if c.dims else None
        a.min_dims = tuple(c.min_dims) if c.min_dims else None
        a.fixed_floor = c.floor
        if req.vastu_preference:
            if c.zone:
                a.zone = a.preferred_zone = c.zone
                a.matched, a.fixed_zone = True, True
            elif c.hint_zones:
                a.preferred_zone = a.zone = c.hint_zones[0]
                a.hint_zones, a.matched = list(c.hint_zones), True
    return assignments


# Smallest area (sqft) each room type can realistically be built at. Used
# only for the feasibility check below — the real sizing comes from the
# plot-scaled reference table in vastu/vastu_rules.py.
MIN_ROOM_AREA = {
    "master_bedroom": 120, "bedroom": 90, "bathroom": 35, "kitchen": 60,
    "living_room": 120, "dining_room": 80, "pooja_room": 16, "study_room": 55,
    "store_room": 30, "balcony": 30, "parking": 120, "garden": 40,
    "staircase": 45,
}
DEFAULT_MIN_ROOM_AREA = 60

# Share of each floor lost to walls, corridors and stairs.
CIRCULATION_OVERHEAD = 0.15


def check_feasibility(req: "HouseRequirementIn", room_list, assignments=None) -> dict:
    """Can the requested rooms actually fit on this plot?

    The layout generator will always return something, but on a badly
    over-packed plot (say 14 rooms on 500 sqft) rooms end up overlapping
    no matter how the geometry is nudged. Catching that here means the
    user is told to drop a room or add a floor, instead of being handed a
    plan with collisions in it.
    """
    plot_area = req.plot_length * req.plot_width
    # Use the same minimum room dimensions as the geometry engine so the
    # feasibility check and the actual planner agree. For multi-floor homes,
    # the requested spaces are distributed across the requested levels.
    usable_area = plot_area * (1 - CIRCULATION_OVERHEAD) * max(1, int(req.floors))
    if assignments is not None:
        room_list = list(assignments)
        required = sum(
            (a.min_dims[0] * a.min_dims[1]) if getattr(a, "min_dims", None)
            else MIN_DIMS.get(a.room_type, (4.0, 5.0))[0] * MIN_DIMS.get(a.room_type, (4.0, 5.0))[1]
            for a in assignments
        )
    else:
        required = sum(
            MIN_DIMS.get(rt, (4.0, 5.0))[0] * MIN_DIMS.get(rt, (4.0, 5.0))[1]
            for rt in room_list
        )
    ratio = required / usable_area if usable_area else float("inf")

    if ratio > 1.0:
        level = "error"
        message = (
            f"These {len(room_list)} rooms need at least {required:,.0f} sq.ft, but a "
            f"{req.plot_length:.0f}x{req.plot_width:.0f} ft plot gives about "
            f"{usable_area:,.0f} sq.ft of cumulative usable floor area across {req.floors} floor(s) after an allowance for walls and circulation. "
            "Remove some rooms or use a larger plot."
        )
    elif ratio > 0.82:
        level = "warning"
        message = (
            f"This plot is tight: the requested rooms need roughly {required:,.0f} of the "
            f"{usable_area:,.0f} sq.ft available. Rooms will be close to their minimum "
            "sizes and the layout may need compromises."
        )
    else:
        level, message = "ok", None

    return {
        "level": level,
        "message": message,
        "required_min_area": round(required, 1),
        "usable_area": round(usable_area, 1),
        "utilisation": round(ratio, 3),
        "room_count": len(room_list),
    }


def validate_requirements(req: "HouseRequirementIn"):
    errors = []
    if req.plot_length <= 0 or req.plot_width <= 0:
        errors.append("Plot dimensions must be positive.")
    if req.bedrooms < 1 and not getattr(req, "custom_rooms_only", False):
        errors.append("At least one bedroom is required.")
    room_list = _build_room_list(req)
    if not room_list and not getattr(req, "_custom_specs", None):
        errors.append("At least one room must be requested.")
    if errors:
        raise ValueError("; ".join(errors))
    return room_list


def effective_budget(req: "HouseRequirementIn", category: str|None=None) -> float:
    """Return a safe planning budget. Frontend may send 0 to mean 'auto budget'."""
    cat=category or req.construction_category
    rate=RATE_TABLE.get(cat,RATE_TABLE['standard'])
    plot_area=req.plot_length*req.plot_width
    if req.budget and req.budget>0: return float(req.budget)
    return float(plot_area*req.floors*rate*0.78)


def _distribute_assignments(assignments, floors: int, building_type: str = "custom"):
    """Distribute spaces across floors with type-specific functional zoning and balancing."""
    floors = max(1, min(int(floors), 5))
    buckets = [[] for _ in range(floors)]
    b = canonical_type(building_type)
    pinned = [a for a in assignments if getattr(a, "fixed_floor", None) is not None]
    assignments = [a for a in assignments if getattr(a, "fixed_floor", None) is None]
    if floors == 1:
        buckets[0].extend(assignments)
    else:
        ground_singletons = {
            "parking", "garden", "staircase", "reception", "lobby", "security_room", "loading_bay",
            "living_room", "waiting_area", "restaurant_dining", "open_work_area", "conference_room",
            "meeting_room", "manager_office", "director_office", "principal_office", "staff_room",
            "admin_office", "faculty_room", "library", "computer_lab", "science_lab", "nurse_station",
            "pharmacy", "diagnostic_lab", "treatment_room", "procedure_room", "restaurant_kitchen",
            "pantry", "kitchen", "dining_room", "dining_hall", "laundry", "housekeeping", "warden_room",
            "teller_area", "vault", "server_room", "stock_room", "warehouse_office", "utility_room",
            "production_hall", "storage_hall", "multipurpose_hall", "stage", "showroom",
        }
        repeated_limits = {
            "duplex": {"bathroom": 1},
            "villa": {"bathroom": 1},
            "school": {"classroom": 3, "restroom": 2},
            "college": {"classroom": 3, "restroom": 2},
            "hospital": {"consultation_room": 2, "restroom": 2},
            "clinic": {"consultation_room": 2, "restroom": 2},
            "retail": {"showroom": 1},
            "mall": {"showroom": 2},
            "apartment_block": {"apartment_unit": 0},
        }.get(b, {})
        # Spaces that are intentionally upper-floor/repeated and should be balanced.
        distributed_types = {
            "bedroom", "hostel_room", "guest_room", "office_cabin", "apartment_unit",
            "classroom", "consultation_room", "restroom", "bathroom", "showroom",
        }
        upper_index = 0
        ground_seen = {}
        counts = {}
        for a in assignments:
            rt = a.room_type
            counts[rt] = counts.get(rt, 0) + 1
            # Duplex/villa service bathrooms are deliberately stacked toward the
            # SE/S side on upper floors. Keeping one bathroom on the entry level
            # avoids forcing multiple W/NW bathrooms into an upper-floor west band
            # that is too narrow to hold them without crossing the Vastu grid.
            if rt == "bathroom" and b in {"duplex", "villa"} and counts[rt] > 1:
                z = "SE" if counts[rt] in {2, 3} else "S"
                setattr(a, "preferred_zone", z)
                setattr(a, "zone", z)
                setattr(a, "matched", True)
            if rt == "bedroom" and b in {"duplex", "villa"} and counts[rt] >= 3:
                # On a compact upper floor, the third secondary bedroom is more
                # safely kept in the north band than left as a singleton W room,
                # which would geometrically resolve to the centre column. North is
                # treated as an acceptable secondary-bedroom zone; the master stays SW.
                setattr(a, "preferred_zone", "N")
                setattr(a, "zone", "N")
                setattr(a, "matched", True)
            if rt in distributed_types:
                limit = repeated_limits.get(rt, 0)
                # Residential bedrooms stay above the entry floor except the master.
                if rt == "bedroom" and b in {"1bhk","2bhk","3bhk","4bhk","duplex","villa","farmhouse","custom"}:
                    # Duplex / villa: one guest (or parents') bedroom stays on the entry floor, as in real houses,
                    # so the ground floor is not just living + dining + kitchen.
                    if b in {"duplex", "villa"} and counts[rt] == 1 and sum(1 for x in assignments if x.room_type == "bedroom") >= 3:
                        buckets[0].append(a); continue
                    buckets[1 + (upper_index % (floors - 1))].append(a); upper_index += 1; continue
                # Special repeated institutional spaces may keep a few on ground.
                if limit > 0 and ground_seen.get(rt, 0) < limit:
                    buckets[0].append(a); ground_seen[rt] = ground_seen.get(rt, 0) + 1
                else:
                    buckets[1 + (upper_index % (floors - 1))].append(a); upper_index += 1
            elif rt in ground_singletons:
                # A second kitchen in a house (duplex/villa/joint family) belongs on the upper floor.
                if rt == "kitchen" and b in {"duplex", "villa", "farmhouse", "custom", "4bhk", "3bhk"} and counts[rt] > 1:
                    buckets[1 + (upper_index % (floors - 1))].append(a); upper_index += 1
                # Apartment units are never entry-floor spaces; they belong on upper floors.
                elif rt == "apartment_unit":
                    buckets[1 + (upper_index % (floors - 1))].append(a); upper_index += 1
                else:
                    buckets[0].append(a)
            else:
                buckets[1 + (upper_index % (floors - 1))].append(a); upper_index += 1

        # Move rooms between upper floors until each requested level has useful content.
        for fi in range(1, floors):
            if buckets[fi]:
                continue
            donor = max(range(1, floors), key=lambda idx: len(buckets[idx]))
            if len(buckets[donor]) > 1:
                buckets[fi].append(buckets[donor].pop())

    for a in pinned:                                   # user-pinned floors (custom rooms)
        buckets[min(max(int(a.fixed_floor), 0), floors - 1)].append(a)
    for fi, bucket in enumerate(buckets):
        for a in bucket:
            setattr(a, "floor", fi)
    return buckets


def _colorize(layout):
    for r in layout.get('rooms',[]): r['color']=room_color(r['room_type'])
    return layout


def _floor_layouts(req, assignments):
    # Bath roles first (attached to a bedroom / one common washroom), so floors keep them together:
    # an attached bath stays on its bedroom's floor and the common washroom sits on the staircase floor.
    try:
        from layout.residential_planner import apply_bath_roles, RESIDENTIAL_PLANNER_TYPES
        _is_res = canonical_type(req.building_type) in RESIDENTIAL_PLANNER_TYPES
    except Exception:  # noqa: BLE001
        _is_res = False
    if _is_res:
        apply_bath_roles(assignments)
    buckets=_distribute_assignments(assignments,req.floors,req.building_type)
    if _is_res and len(buckets) > 1:
        def _floor_of(a):
            return next((i for i, b in enumerate(buckets) if a in b), 0)
        for bath in [a for a in assignments if a.room_type in ("bathroom", "toilet")]:
            owner = next((x for x in assignments if getattr(x, "attached_bath", None) == bath.label), None)
            src = _floor_of(bath)
            dst = _floor_of(owner) if owner is not None else (_floor_of(next((x for x in assignments if x.room_type == "staircase"), bath)))
            if dst != src and len(buckets[src]) > 1:
                buckets[src].remove(bath); buckets[dst].append(bath)
    # Every upper floor needs its own staircase (the program only lists one, which the
    # distributor puts on the ground floor, leaving the upper floors unreachable).
    if len(buckets) > 1:
        _stair = next((a for b in buckets for a in b if a.room_type == "staircase"), None)
        if _stair is not None:
            from dataclasses import replace as _dc_replace
            for _bi in range(1, len(buckets)):
                if buckets[_bi] and not any(a.room_type == "staircase" for a in buckets[_bi]):
                    buckets[_bi].append(_dc_replace(_stair))
    # Vastu-aware entrance position: when the user leaves the door at the
    # generic centre position, use the traditional best sector for the selected
    # facing. Explicit left/right choices remain authoritative.
    selected_door_position = str(getattr(req, "main_door_position", "center") or "center").lower()
    if bool(getattr(req, "vastu_preference", False)) and selected_door_position == "center":
        selected_door_position = entrance_advice(
            getattr(req, "main_entrance_direction", None) or getattr(req, "road_direction", "North")
        )["recommended_door_position"]
    # Plot-aware sizing per floor.  An upper floor is never scaled to cover more than the entry floor, so every
    # floor fits inside the same outer walls (stacked staircase, aligned bays).
    from layout.sizing import COVERAGE, target_area_of
    _site_t = ('parking', 'garden', 'loading_bay')
    cov_floor = [None] * len(buckets)
    area_target = None
    _cov0 = COVERAGE.get(canonical_type(req.building_type))
    if _cov0:
        _plot_area = req.plot_length * req.plot_width
        _ground = None
        for _fi, _b in enumerate(buckets):
            _al = allocate(_b)
            _c = _cov0 if _fi == 0 else min(_cov0, (_ground or _plot_area) * 0.92 / _plot_area)
            apply_plot_scaling(_al, req.plot_length, req.plot_width, req.building_type, coverage=_c)
            _tot = sum(target_area_of(x) for x in _al if x.room_type not in _site_t)
            if _fi == 0:
                _ground = _tot
            else:
                _hall = ('staircase', 'multipurpose_hall', 'auditorium', 'dining_hall', 'stage', 'common_room', 'open_work_area')
                _need = sum(target_area_of(x) for x in _al if x.room_type not in _site_t + _hall)
                area_target = max(area_target or 0.0, _need)
            cov_floor[_fi] = _c
    results=[]
    stair_ref=None                                   # ground-floor staircase every upper floor must stack over
    for fi,bucket in enumerate(buckets):
        # If a higher floor has no explicit rooms, keep a tiny staircase/access
        # space only when needed; otherwise generate an empty validated plate.
        if not bucket:
            raise LayoutInfeasible(f"Floor {fi + 1} has no rooms. Reduce the number of floors or add rooms.")
        _allocs = allocate(bucket)
        apply_plot_scaling(_allocs, req.plot_length, req.plot_width, req.building_type, coverage=cov_floor[fi])
        _src = {a.label: a for a in bucket}
        for _al in _allocs:
            _a = _src.get(_al.label)
            if _a is not None:
                _al.attached_to = getattr(_a, "attached_to", None)
                _al.common = bool(getattr(_a, "common", False))
                _al.attached_bath = getattr(_a, "attached_bath", None)
        if stair_ref is not None:
            for _al in _allocs:
                if _al.room_type == "staircase":
                    _al.zone = _al.preferred_zone = stair_ref["zone"]
                    _al.matched = True
                    _al.fixed_zone = True            # keep the stair in the same zone cell as the floor below
        layout=_colorize(safe_generate_layout(
            req.plot_length, req.plot_width, _allocs, floors=1, stair_ref=stair_ref,
            road_direction=req.road_direction, building_type=req.building_type,
            wall_width_ft=req.wall_width_ft,
            room_size_details=req.room_size_details,
            appearance=req.appearance,
            main_door_position=selected_door_position,
            area_target=(area_target if fi == 0 else None),
            parking_position=req.parking_position,
            staircase_position=req.staircase_position,
            custom_mode=bool(getattr(req, "custom_mode", False)),
            main_entrance_direction=getattr(req, "main_entrance_direction", req.road_direction),
            hall_shape=getattr(req, "hall_shape", "auto"),
            setback_ft=getattr(req, "setback_ft", None),
            setbacks=getattr(req, "setbacks", None),
        ))
        for r in layout['rooms']: r['floor']=fi
        if fi == 0 and len(buckets) > 1:
            _st = [r for r in layout['rooms'] if r['room_type'] == 'staircase']
            _env = layout.get('building_envelope')
            if _st and _env:
                _g = max(_st, key=lambda r: r['w'] * r['h'])
                from vastu.vastu_engine import zone9_of_point
                stair_ref = {"stair": {k: _g[k] for k in ("x", "y", "w", "h")},
                             "env": {k: _env[k] for k in ("x", "y", "w", "h")},
                             "box": ({k: layout["planning_envelope"][k] for k in ("x", "y", "w", "h")}
                                     if layout.get("planning_envelope") else None),
                             "zone": zone9_of_point(_g['x'] + _g['w'] / 2.0, _g['y'] + _g['h'] / 2.0, _env),
                             "bay_frame": layout.get("bay_frame")}
        layout['floor']=fi
        layout['building_type']=canonical_type(req.building_type)
        layout['vastu_enabled']=bool(req.vastu_preference)
        layout['main_door_position']=selected_door_position
        layout['title']=f"Ground Floor" if fi==0 else f"{'First' if fi==1 else 'Second' if fi==2 else 'Third' if fi==3 else 'Fourth'} Floor"
        results.append(layout)
    return results


def run_pipeline(req: "HouseRequirementIn", project_id: str = None) -> dict:
    project_id=project_id or f"proj_{uuid.uuid4().hex[:12]}"
    # Requirements module: plot lengths/angles/bearing -> engine-ready plot, custom rooms, entrance.
    req,req_report=normalize_requirements(req)
    room_list=validate_requirements(req)
    assignments=build_assignments(req,room_list)
    feasibility=check_feasibility(req,room_list,assignments)
    if feasibility['level']=='error': raise InfeasibleError(feasibility['message'])
    plot_area=req.plot_length*req.plot_width
    family_size=req.family_size or (req.bedrooms+1)
    category=req.construction_category
    building_type = canonical_type(req.building_type)
    per_bed_budget=effective_budget(req,category)/max(req.bedrooms,1)
    if per_bed_budget<700000 and category=='premium': category='standard'
    planning_budget=effective_budget(req,category)

    planning_type,proba=predict_planning_type(plot_area=plot_area,plot_length=req.plot_length,plot_width=req.plot_width,bedrooms=req.bedrooms,bathrooms=req.bathrooms,floors=req.floors,budget=planning_budget,parking_required=req.parking,family_size=family_size)
    planning_type_challenger,proba_challenger=predict_planning_type_ensemble(plot_area=plot_area,plot_length=req.plot_length,plot_width=req.plot_width,bedrooms=req.bedrooms,bathrooms=req.bathrooms,floors=req.floors,budget=planning_budget,parking_required=req.parking,family_size=family_size)

    try:
        floor_layouts=_floor_layouts(req,assignments)
    except LayoutInfeasible as exc:
        raise InfeasibleError(str(exc)) from exc
    # One consistent Vastu score for every engine: the share of (non-site) rooms whose zone is acceptable.
    # (The engines' own raw numbers mean different things - a distance sum vs a 0..1 fit - and the old
    # code summed them over floors, which showed scores such as 151%.)
    _SITE = {'parking', 'garden', 'loading_bay'}
    for fl in floor_layouts:
        _pool = [r for r in fl['rooms'] if str(r.get('room_type', '')).lower() not in _SITE]
        _ok = sum(1 for r in _pool if r.get('vastu_acceptable'))
        fl['vastu_raw_score'] = fl.get('vastu_score')
        fl['vastu_acceptable_count'] = _ok
        fl['vastu_score'] = round(_ok / max(len(_pool), 1), 4)
    all_rooms=[r for fl in floor_layouts for r in fl['rooms']]
    collisions=sum(len(fl['collisions']) for fl in floor_layouts)
    boundary=sum(len(fl['boundary_violations']) for fl in floor_layouts)
    vastu_match_count=sum(int(fl.get('vastu_match_count',0)) for fl in floor_layouts)
    vastu_acceptable_count=sum(int(fl.get('vastu_acceptable_count',0)) for fl in floor_layouts)
    _pool_n=sum(len([r for r in fl['rooms'] if str(r.get('room_type','')).lower() not in _SITE]) for fl in floor_layouts)
    vastu_score=round(min(1.0, vastu_acceptable_count/max(_pool_n,1)),4)
    vastu_match_percent=round(100.0*vastu_match_count/max(len(all_rooms),1),1)
    vastu_violations=[]
    vastu_advisories=[]
    # Core placement rules are enforced as hard checks. Dining/store/balcony and
    # similar secondary spaces remain advisory because traditional Vastu sources
    # differ on their exact placement and compact plots may require compromises.
    vastu_hard_types={'master_bedroom','bedroom','bathroom','toilet','restroom','kitchen','living_room','pooja_room','study_room','staircase'} if building_type in RESIDENTIAL else set()
    for fl in floor_layouts:
        for room in fl.get('rooms',[]):
            if room.get('room_type') in {'parking','garden','loading_bay'}:
                continue
            if req.vastu_preference and not room.get('vastu_acceptable', False):
                item={
                    'floor': fl.get('floor',0),
                    'room': room.get('label'),
                    'preferred_zone': room.get('preferred_zone'),
                    'actual_zone': room.get('zone'),
                    'message': f"{room.get('label')} is outside its configured traditional Vastu acceptable zones."
                }
                if room.get('room_type') in vastu_hard_types:
                    vastu_violations.append(item)
                else:
                    vastu_advisories.append(item)

    effective_door_position = str((floor_layouts[0].get("main_door_position") if floor_layouts else None)
                                  or getattr(req, "main_door_position", "center"))
    vastu_audit=audit_layout(floor_layouts,road_direction=req.road_direction,door_position=effective_door_position,
                             north_offset_deg=getattr(req,"north_offset_deg",0.0),vastu_enabled=bool(req.vastu_preference),
                             building_type=building_type,entrance_direction=req.main_entrance_direction)
    layout_notes=[n for fl in floor_layouts for n in (fl.get("layout_notes") or [])]

    circulation_factor=0.12 if plot_area<1200 else 0.16 if plot_area<2500 else 0.20
    builtup_area=0.0
    for fl in floor_layouts:
        builtup_area += sum(r['w']*r['h'] for r in fl['rooms'])*(1+circulation_factor)
    rate=RATE_TABLE.get(category,RATE_TABLE['standard'])
    bath_factor=1+max(0,req.bathrooms-2)*0.015
    total=builtup_area*rate*bath_factor
    cost={'plot_area':round(plot_area,1),'builtup_area':round(builtup_area,1),'floors':req.floors,'num_rooms':len(all_rooms),'rate':rate,'material':round(total*0.6,0),'labour':round(total*0.4,0),'total':round(total,0),
          'model_predicted_total':round(predict_cost(plot_area=plot_area,builtup_area=builtup_area,floors=req.floors,num_rooms=len(all_rooms),num_bathrooms=req.bathrooms,category=category),0),
          'model_predicted_total_challenger':round(predict_cost_ensemble(plot_area=plot_area,builtup_area=builtup_area,floors=req.floors,num_rooms=len(all_rooms),num_bathrooms=req.bathrooms,category=category),0)}

    # Generate a validated image for each real floor layout.
    floor_plans=generate_floor_blueprints(floor_layouts[0] if floor_layouts else {'plot':{'length':req.plot_length,'width':req.plot_width},'rooms':[]},req.main_entrance_direction,project_id,floors=req.floors,floor_layouts=floor_layouts)
    blueprint_path=floor_plans[0]['path']

    # Preliminary civil-engineering estimate: footing/column/beam/slab
    # sizing, material quantities, staircase and setback/FAR compliance.
    # See civil/structural_engineering.py; failures here must never break
    # plan generation, so they degrade to a warning instead of a 500.
    try:
        structural=generate_structural_report(
            plot_length=req.plot_length,plot_width=req.plot_width,floors=req.floors,
            builtup_area_sqft=builtup_area,category=category,soil_type=req.soil_type,
            rooms=all_rooms,has_staircase=req.staircase,
        )
        structural_warnings=[]
        if not structural['far_compliance']['within_far_limit']:
            structural_warnings.append('Proposed built-up area exceeds the typical permissible FAR/FSI for this plot size.')
        if not structural['far_compliance']['within_typical_floor_limit']:
            structural_warnings.append('Requested floor count exceeds the typical permissible floors for this plot size.')
        if structural.get('staircase') and not structural['staircase']['compliant']:
            structural_warnings.extend(structural['staircase']['issues'])
    except Exception as exc:  # noqa: BLE001
        structural=None
        structural_warnings=[f"Structural/compliance estimate unavailable: {exc}"]

    return {
        "project_id": project_id,
        "name": req.project_name or f"{property_title(building_type)} - {req.plot_length:.0f}x{req.plot_width:.0f} Plot",
        "building_type": building_type,
        "planning_type": planning_type,
        "planning_type_confidence": proba,
        "planning_type_challenger": planning_type_challenger,
        "planning_type_challenger_confidence": proba_challenger,
        "category": category,
        "plot": {**(floor_layouts[0]["plot"] if floor_layouts else {"length": req.plot_length, "width": req.plot_width, "area": plot_area}),
                 "gross_area": req_report["plot"]["area_sqft"], "shape": req_report["plot"]["mode"],
                 "edges_ft": req_report["plot"]["edge_lengths_ft"], "corner_angles_deg": req_report["plot"]["corner_angles_deg"],
                 "polygon": req_report["plot"]["polygon_true_rel"]},
        "floors": req.floors,
        "vastu_enabled": req.vastu_preference,
        "vastu_match_count": vastu_match_count,
        "vastu_acceptable_count": vastu_acceptable_count,
        "vastu_score": vastu_score,
        "vastu_match_percent": vastu_match_percent,
        "main_door_position": effective_door_position,
        "vastu_violations": vastu_violations,
        "vastu_advisories": vastu_advisories,
        "entrance": req.main_entrance_direction,
        "rooms": all_rooms,
        "floor_layouts": floor_layouts,
        "collisions": collisions,
        "boundary_violations": boundary,
        "cost": cost,
        "blueprint_path": blueprint_path,
        "floor_plans": [{"floor": f["floor"], "title": f["title"], "filename": f["filename"], "engine": f.get("engine"), "warning": f.get("warning")} for f in floor_plans],
        "blueprint_engine": floor_plans[0].get("engine") if floor_plans else None,
        "feasibility": feasibility,
        "structural": structural,
        "planning_budget_used": round(planning_budget, 0),
        "algorithm_trace": algorithm_trace(building_type, bool(req.vastu_preference), req.floors),
        "vastu_audit": vastu_audit,
        "requirements_report": req_report,
        "warnings": list(dict.fromkeys(req_report["warnings"]+req_report["adjustments"]+layout_notes)) + ([f"Budget was auto-estimated at Rs. {planning_budget:,.0f} because no budget was supplied."] if not req.budget else []) + structural_warnings,
    }
