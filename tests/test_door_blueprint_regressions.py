import os
from pathlib import Path

os.environ.setdefault("BLUEPRINT_RENDERER", "local")

from blueprint.door_engine import build_doors, validate_doors
from blueprint.local_renderer import render_local_blueprint
from blueprint.wall_engine import _room_sides
from vastu.vastu_engine import audit_layout


def _rooms():
    return [
        {"room_type": "living_room", "label": "Living Room", "zone": "NE", "door_side": "E",
         "x": 10.0, "y": 20.0, "w": 12.0, "h": 18.0},
        {"room_type": "bedroom", "label": "Bedroom", "zone": "SW", "door_side": "E",
         "x": 22.0, "y": 20.0, "w": 12.0, "h": 18.0},
    ]


def test_room_side_axes_match_cardinal_directions():
    sides = _room_sides(_rooms()[0])
    assert sides["S"] == (10.0, 20.0, 10.0, 38.0)
    assert sides["N"] == (22.0, 20.0, 22.0, 38.0)
    assert sides["W"] == (10.0, 20.0, 22.0, 20.0)
    assert sides["E"] == (10.0, 38.0, 22.0, 38.0)


def test_main_door_is_on_requested_real_exterior_wall():
    rooms = _rooms()
    doors = build_doors(
        rooms, entrance_direction="East", plot_length=50, plot_width=60,
        building_envelope={"x": 10, "y": 20, "w": 24, "h": 18},
        main_door_position="right",
    )
    mains = [d for d in doors if d["is_main_entrance"]]
    assert len(mains) == 1
    assert mains[0]["room"] == "Living Room"
    assert mains[0]["wall_code"] == "E"
    assert mains[0]["is_exterior"] is True
    assert mains[0]["position"] > 0
    assert validate_doors(rooms, doors)["passed"] is True


def test_upper_floor_has_no_main_entrance():
    rooms = _rooms()
    doors = build_doors(
        rooms, entrance_direction="North", plot_length=50, plot_width=60,
        building_envelope={"x": 10, "y": 20, "w": 24, "h": 18},
        main_door_position="center", allow_main_entrance=False,
    )
    assert not [d for d in doors if d["is_main_entrance"]]
    assert validate_doors(rooms, doors, expect_main_entrance=False)["passed"] is True


def test_vastu_audit_uses_main_entrance_not_road_side():
    fl = [{
        "floor": 0,
        "building_envelope": {"x": 0, "y": 0, "w": 30, "h": 40},
        "rooms": [{"room_type": "living_room", "label": "Living Room", "x": 5, "y": 20, "w": 10, "h": 10}],
    }]
    result = audit_layout(
        fl, road_direction="North", entrance_direction="East", door_position="center",
        vastu_enabled=True, building_type="4bhk",
    )
    assert result["entrance"]["facing"] == "East"


def test_renderer_uses_authoritative_doors_and_upper_floor_access_only(tmp_path: Path):
    rooms = _rooms()
    layout = {
        "plot": {"length": 50, "width": 60},
        "rooms": rooms,
        "building_envelope": {"x": 10, "y": 20, "w": 24, "h": 18},
        "floor": 1,
        "vastu_enabled": True,
        "wall_width_ft": 0.5,
        "main_door_position": "center",
        "appearance": "technical",
        "room_size_details": True,
    }
    out = tmp_path / "first_floor.png"
    render_local_blueprint(layout, "Internal staircase access", str(out), "FIRST FLOOR")
    assert out.exists() and out.stat().st_size > 0
    assert not layout.get("render_warnings")


def test_uploaded_sai_3bhk_vastu_and_door_regression():
    from types import SimpleNamespace
    from backend.pipeline import run_pipeline

    req = SimpleNamespace(
        project_name="sai", building_type="3bhk", space_count=3,
        plot_length=40, plot_width=60, bedrooms=3, bathrooms=2, kitchens=1,
        living_room=True, dining_room=True, pooja_room=False, study_room=False,
        store_room=False, parking=True, floors=1, staircase=True, balcony=False,
        garden=False, budget=2500000, road_direction="North",
        main_entrance_direction="North", vastu_preference=True,
        construction_category="standard", family_size=4,
        soil_type="Medium clay / sandy clay", wall_width_ft=0.5,
        room_size_details=True, appearance="technical",
        main_door_position="center", parking_position="road_side",
        staircase_position="road_side",
        design_notes=("Living room near entrance\n"
                      "Kitchen in East side\n"
                      "Master bedroom with attached toilet"),
        setback_ft=3, north_offset_deg=0.0, custom_mode=False,
        custom_rooms_only=False,
    )
    result = run_pipeline(req, project_id="regression_uploaded_sai")
    assert result["collisions"] == 0
    assert result["boundary_violations"] == 0
    assert result["vastu_violations"] == []
    assert result["main_door_position"] == "left"
    assert result["vastu_audit"]["entrance"]["door_zone16"] == "NNE"
    assert result["vastu_audit"]["entrance"]["grade"] == "ideal"

    floor = result["floor_layouts"][0]
    indoor = [r for r in floor["rooms"] if r["room_type"] not in {"parking", "garden", "loading_bay"}]
    parking = next(r for r in floor["rooms"] if r["room_type"] == "parking")
    for room in indoor:
        ox = min(room["x"] + room["w"], parking["x"] + parking["w"]) - max(room["x"], parking["x"])
        oy = min(room["y"] + room["h"], parking["y"] + parking["h"]) - max(room["y"], parking["y"])
        assert max(0.0, ox) * max(0.0, oy) == 0.0

    master = next(r for r in indoor if r["room_type"] == "master_bedroom")
    baths = [r for r in indoor if r["room_type"] == "bathroom"]
    # Attached washroom is carved INSIDE the master bedroom (door only from the bedroom).
    attached = [b for b in baths if b.get("inside_of") == master["label"]]
    assert attached
    b = attached[0]
    assert b["x"] >= master["x"] - 0.05 and b["y"] >= master["y"] - 0.05
    assert b["x"] + b["w"] <= master["x"] + master["w"] + 0.05 and b["y"] + b["h"] <= master["y"] + master["h"] + 0.05
    assert b["door_to"] == [master["label"]]

    doors = build_doors(
        floor["rooms"], entrance_direction="North", plot_length=40, plot_width=60,
        building_envelope=floor["building_envelope"],
        main_door_position=result["main_door_position"], allow_main_entrance=True,
    )
    mains = [d for d in doors if d["is_main_entrance"]]
    assert len(mains) == 1
    assert mains[0]["wall_code"] == "N"
    assert mains[0]["room"] == "Living Room"
    assert validate_doors(floor["rooms"], doors, expect_main_entrance=True)["passed"]
