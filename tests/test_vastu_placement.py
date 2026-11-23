"""Regression checks for active Vastu placement on representative plans."""

import os
os.environ.setdefault("BLUEPRINT_RENDERER", "local")

from backend.pipeline import run_pipeline
from backend.schemas import HouseRequirementIn
from blueprint.door_engine import build_doors


def _by_label(result):
    return {r["label"]: r for r in result["rooms"]}


def test_1bhk_core_vastu_zones():
    req = HouseRequirementIn(
        building_type="1bhk", space_count=1, plot_length=25, plot_width=40,
        bedrooms=1, bathrooms=1, kitchens=1, floors=1, budget=0,
        parking=False, staircase=True, vastu_preference=True,
    )
    result = run_pipeline(req, project_id="test_vastu_1bhk")
    rooms = _by_label(result)
    assert rooms["Master Bedroom"]["zone"] == "SW"
    assert rooms["Kitchen"]["zone"] == "SE"
    assert rooms["Living Room"]["zone"] in {"N", "NE"}
    assert rooms["Bathroom"]["vastu_acceptable"] is True
    assert result["vastu_violations"] == []
    assert result["collisions"] == 0
    assert result["boundary_violations"] == 0


def test_4bhk_core_vastu_zones():
    req = HouseRequirementIn(
        building_type="4bhk", space_count=4, plot_length=40, plot_width=60,
        bedrooms=4, bathrooms=3, kitchens=1, floors=1, budget=0,
        parking=False, staircase=True, vastu_preference=True,
        road_direction="East", main_entrance_direction="East",
    )
    result = run_pipeline(req, project_id="test_vastu_4bhk")
    rooms = _by_label(result)
    assert rooms["Master Bedroom"]["zone"] == "SW"
    assert rooms["Kitchen"]["zone"] == "SE"
    assert rooms["Living Room"]["zone"] in {"N", "NE"}
    assert rooms["Bedroom 1"]["zone"] in {"NW", "N"}
    assert rooms["Dining Room"]["vastu_acceptable"] is True
    assert result["vastu_violations"] == []
    assert result["collisions"] == 0
    assert result["boundary_violations"] == 0
    for floor in result["floor_layouts"]:
        assert floor.get("no_unplanned_gaps") is True
        assert floor.get("gap_report", {}).get("max_internal_gap_ft", 0.0) <= 0.20


if __name__ == "__main__":
    test_1bhk_core_vastu_zones()
    test_4bhk_core_vastu_zones()
    print("VASTU CORE PLACEMENT TESTS PASSED")


def test_main_entrance_follows_requested_direction():
    req = HouseRequirementIn(
        building_type="4bhk", space_count=4, plot_length=100, plot_width=150,
        bedrooms=4, bathrooms=3, kitchens=1, floors=1, budget=0,
        parking=False, staircase=True, vastu_preference=True,
        road_direction="East", main_entrance_direction="East",
    )
    result = run_pipeline(req, project_id="test_vastu_entrance")
    doors = build_doors(result["floor_layouts"][0]["rooms"], "East", 100, 150)
    mains = [d for d in doors if d.get("is_main_entrance")]
    assert len(mains) == 1
    assert mains[0]["wall_code"] == "E"
    assert mains[0]["room"] == "Living Room"
