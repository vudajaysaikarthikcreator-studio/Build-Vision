from backend.pipeline import run_pipeline
from backend.schemas import HouseRequirementIn


def _check(building_type, plot_length, plot_width, bedrooms, bathrooms, floors=1):
    req = HouseRequirementIn(
        building_type=building_type,
        space_count=bedrooms,
        plot_length=plot_length,
        plot_width=plot_width,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        kitchens=1,
        floors=floors,
        budget=0,
        parking=False,
        staircase=True,
        vastu_preference=True,
        road_direction="East",
        main_entrance_direction="East",
    )
    result = run_pipeline(req, project_id=f"vastu_complete_{building_type}")
    assert result["vastu_violations"] == []
    assert sum(len(f["collisions"]) for f in result["floor_layouts"]) == 0
    assert sum(len(f["boundary_violations"]) for f in result["floor_layouts"]) == 0
    return result


def test_common_residential_vastu_plans():
    _check("1bhk", 25, 40, 1, 1)
    _check("2bhk", 30, 40, 2, 2)
    _check("3bhk", 30, 50, 3, 3)
    _check("4bhk", 40, 60, 4, 3)
    _check("4bhk", 100, 150, 4, 3)
    _check("duplex", 40, 60, 4, 3, floors=2)
    _check("villa", 50, 80, 4, 4, floors=2)
