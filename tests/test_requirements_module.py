"""Requirements module: plot lengths/angles, custom rooms, Vastu audit, never-fail layouts."""
import os, sys
os.environ.setdefault("BLUEPRINT_RENDERER", "local")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import pytest
from test_pipeline_lightweight import make_req
from backend.pipeline import run_pipeline, InfeasibleError
from user_requirements import resolve_plot, parse_custom_rooms, PlotError, RoomSpecError, RequirementError


def test_rectangle_and_rotation():
    assert resolve_plot(plot_length=50, plot_width=30, plot_bearing_deg=90)["plot_length"] == 30.0


def test_quadrilateral_angles_and_bad_corner():
    r = resolve_plot(plot_sides={"north": 40, "south": 40, "east": 60, "west": 60},
                     plot_angles={"NW": 100, "NE": 80, "SE": 100, "SW": 80})
    assert r["mode"] == "quadrilateral" and r["buildable"]["area_sqft"] < 2400
    with pytest.raises(PlotError):
        resolve_plot(plot_sides={"north": 40, "south": 40, "east": 60, "west": 60}, plot_angles={"SW": 190})


def test_traverse_and_road_bearing():
    r = resolve_plot(plot_traverse=[{"length": 50, "bearing": 90}, {"length": 40, "bearing": 180},
                                    {"length": 50, "bearing": 270}, {"length": 40, "bearing": 0}], road_bearing_deg=200)
    assert r["road_direction"] == "South" and r["plot_width"] == 50.0


def test_custom_room_validation():
    assert len(parse_custom_rooms([{"name": "Guest Bedroom", "type": "bedroom", "count": 2, "width_ft": 12, "length_ft": 14}])) == 2
    with pytest.raises(RoomSpecError):
        parse_custom_rooms([{"type": "bedroom", "zone": "up"}])


@pytest.mark.parametrize("bt,count", [("2bhk", 2), ("3bhk", 3), ("restaurant", 1), ("retail", 1)])
def test_previously_failing_types_now_generate(bt, count):
    from planning.building_programs import DEFAULT_PLOTS
    L, W = DEFAULT_PLOTS[bt]
    for road in ("North", "South", "East", "West"):
        r = run_pipeline(make_req(building_type=bt, plot_length=L, plot_width=W, bedrooms=count, space_count=count,
                                  road_direction=road, main_entrance_direction=road, budget=0), f"t_{bt}_{road}")
        assert r["collisions"] == 0 and r["boundary_violations"] == 0


def test_custom_rooms_sizes_zone_and_audit():
    r = run_pipeline(make_req(building_type="3bhk", plot_length=45, plot_width=60, bedrooms=3, budget=0,
                              custom_rooms=[{"name": "Home Gym", "width_ft": 10, "length_ft": 12},
                                            {"type": "puja", "zone": "NE"}],
                              room_size_overrides={"kitchen": {"width_ft": 10, "length_ft": 11}}), "t_custom")
    labels = [x["label"] for x in r["rooms"]]
    assert "Home Gym" in labels and r["collisions"] == 0 and r["vastu_audit"]["enabled"]


def test_impossible_request_is_a_clear_refusal():
    with pytest.raises((InfeasibleError, RequirementError)):
        run_pipeline(make_req(building_type="college", plot_length=30, plot_width=30, budget=0), "t_bad")
