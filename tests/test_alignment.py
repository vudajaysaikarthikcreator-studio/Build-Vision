"""Regression tests for orthogonal room/wall alignment."""
import os
os.environ.setdefault("BLUEPRINT_RENDERER", "local")

from backend.pipeline import run_pipeline
from backend.schemas import HouseRequirementIn


def _run(building_type, L, W, bedrooms, baths, floors, road):
    req = HouseRequirementIn(
        building_type=building_type, space_count=bedrooms, plot_length=L, plot_width=W,
        bedrooms=bedrooms, bathrooms=baths, kitchens=1, floors=floors, budget=0,
        parking=False, staircase=True, vastu_preference=True, road_direction=road,
        main_entrance_direction=road,
    )
    return run_pipeline(req, project_id=f"alignment_{building_type}")


def test_residential_alignment():
    for bt, L, W, b, baths, floors, road in [
        ("1bhk", 25, 40, 1, 1, 1, "South"),
        ("3bhk", 40, 60, 3, 2, 1, "East"),
        ("4bhk", 40, 60, 4, 3, 1, "East"),
    ]:
        result = _run(bt, L, W, b, baths, floors, road)
        assert result["collisions"] == 0
        assert result["boundary_violations"] == 0
        for floor in result["floor_layouts"]:
            report = floor.get("alignment_report", {})
            assert report.get("total_edges", 0) > 0
            assert floor.get("no_unplanned_gaps") is True


def test_institutional_alignment():
    result = _run("school", 100, 150, 8, 4, 2, "North")
    assert result["collisions"] == 0
    assert result["boundary_violations"] == 0
    for floor in result["floor_layouts"]:
        assert floor.get("alignment_report", {}).get("total_edges", 0) > 0


if __name__ == "__main__":
    test_residential_alignment()
    test_institutional_alignment()
    print("ROOM ALIGNMENT TESTS PASSED")
