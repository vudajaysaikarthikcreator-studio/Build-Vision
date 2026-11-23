"""Hall-centred residential planner: per-room doors, attached / under-stair washrooms, any BHK count."""
from types import SimpleNamespace

from backend.pipeline import run_pipeline

_DEFAULTS = dict(
    building_type="custom", space_count=1, plot_length=60, plot_width=50, bedrooms=3, bathrooms=2, kitchens=1,
    living_room=True, dining_room=True, pooja_room=False, study_room=False, store_room=False, parking=True,
    floors=1, staircase=True, balcony=False, garden=False, budget=0, road_direction="North",
    main_entrance_direction="North", vastu_preference=True, construction_category="standard", family_size=None,
    project_name=None, soil_type="Medium clay / sandy clay", wall_width_ft=0.5, room_size_details=True,
    appearance="technical", main_door_position="center", parking_position="road_side",
    staircase_position="road_side", design_notes=None, setback_ft=None, setbacks=None, custom_mode=False,
    custom_rooms_only=False, hall_shape="auto", north_offset_deg=0.0,
)


def _req(**kw):
    d = dict(_DEFAULTS)
    d.update(kw)
    return SimpleNamespace(**d)


def _inside(c, p, tol=0.06):
    return (c["x"] >= p["x"] - tol and c["y"] >= p["y"] - tol and
            c["x"] + c["w"] <= p["x"] + p["w"] + tol and c["y"] + c["h"] <= p["y"] + p["h"] + tol)


def _check_floor(fl):
    rooms = fl["rooms"]
    by_label = {r["label"]: r for r in rooms}
    hall = {r["label"] for r in rooms if r.get("open_group") == "hall"}
    assert hall, "plan has a hall group"
    for r in rooms:
        if r["room_type"] in ("parking", "garden") or r.get("open_group"):
            continue
        spec = r.get("door_spec")
        assert spec, f"{r['label']} has no door of its own"
        if r.get("inside_of"):
            parent = by_label[r["inside_of"]]
            assert _inside(r, parent), f"{r['label']} is not inside {parent['label']}"
        if r["room_type"] in ("master_bedroom", "bedroom", "kitchen"):
            assert spec["to"] in hall, f"{r['label']} must open onto the hall, not {spec['to']}"
    # attached washrooms: inside the bedroom, door from the bedroom only
    for r in rooms:
        if r.get("contained_kind") == "attached_bath":
            assert by_label[r["inside_of"]]["room_type"] in ("master_bedroom", "bedroom")
            assert r["door_to"] == [r["inside_of"]]


def test_every_bhk_gets_n_bedrooms_hall_and_own_doors():
    for n, plot in [(1, (40, 25)), (2, (45, 40)), (3, (60, 40)), (4, (70, 50)), (5, (80, 60))]:
        res = run_pipeline(_req(building_type=f"{n}bhk", bedrooms=n, bathrooms=min(n, 3), plot_length=plot[0],
                                plot_width=plot[1]), f"pytest_bhk_{n}")
        fl = res["floor_layouts"][0]
        beds = [r for r in fl["rooms"] if r["room_type"] in ("master_bedroom", "bedroom")]
        assert len(beds) == n
        assert res["collisions"] == 0 and res["boundary_violations"] == 0
        if fl.get("engine") == "residential":
            _check_floor(fl)


def test_all_four_entrances():
    for e in ("North", "East", "South", "West"):
        res = run_pipeline(_req(building_type="3bhk", bedrooms=3, bathrooms=2, plot_length=60, plot_width=50,
                                road_direction=e, main_entrance_direction=e), f"pytest_ent_{e}")
        assert res["collisions"] == 0
        fl = res["floor_layouts"][0]
        if fl.get("engine") == "residential":
            _check_floor(fl)


def test_requested_hall_shape_is_honoured_when_possible():
    for shape in ("l_shape", "u_shape"):
        res = run_pipeline(_req(building_type="4bhk", bedrooms=4, bathrooms=3, plot_length=80, plot_width=60,
                                hall_shape=shape), f"pytest_shape_{shape}")
        fl = res["floor_layouts"][0]
        assert fl.get("engine") == "residential"
        assert fl["hall_shape"] == shape


def test_common_washroom_is_under_the_stairs_on_two_floors():
    res = run_pipeline(_req(building_type="3bhk", bedrooms=3, bathrooms=3, floors=2, plot_length=60, plot_width=40),
                       "pytest_stairs")
    ground = res["floor_layouts"][0]
    stairs = next(r for r in ground["rooms"] if r["room_type"] == "staircase")
    under = [r for r in ground["rooms"] if r.get("contained_kind") == "under_stair_bath"]
    assert under and under[0]["inside_of"] == stairs["label"]
    assert _inside(under[0], stairs)
    upper = res["floor_layouts"][1]
    assert all(r.get("contained_kind") == "attached_bath" for r in upper["rooms"] if r["room_type"] == "bathroom")
