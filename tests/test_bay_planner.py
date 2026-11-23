"""Regression test: villa / duplex / farmhouse / hostel / school plans must be regular, not random.

Run:  python tests/test_bay_planner.py
"""
import os
import sys

os.environ.setdefault("BLUEPRINT_RENDERER", "local")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline import run_pipeline
from backend.schemas import HouseRequirementIn
from planning.building_programs import DEFAULT_PLOTS

CASES = {"duplex": (4, 3, 2, 2), "villa": (5, 4, 2, 2), "farmhouse": (3, 2, 1, 1),
         "hostel": (10, 6, 1, 2), "school": (8, 4, 1, 2)}
SITE = {"parking", "garden", "loading_bay"}
NOT_ROOM_SHAPED = {"staircase", "corridor_lobby", "balcony"}


def check(building_type):
    c, b, k, f = CASES[building_type]
    L, W = DEFAULT_PLOTS[building_type]
    req = HouseRequirementIn(building_type=building_type, space_count=c, plot_length=L, plot_width=W, bedrooms=c,
                             bathrooms=b, kitchens=k, floors=f, budget=0, parking=True, staircase=True,
                             vastu_preference=True)
    res = run_pipeline(req, project_id=f"t_bay_{building_type}")
    assert res["collisions"] == 0 and res["boundary_violations"] == 0
    stairs = []
    for fl in res["floor_layouts"]:
        assert fl["engine"] == "bay", f"{building_type} floor {fl['floor']} fell back to {fl['engine']}"
        rooms = [r for r in fl["rooms"] if r["room_type"] not in SITE and not r.get("inside_of")]
        # the building fills a sensible part of the plot (the old planners left 3-30% of it used)
        area = sum(r["w"] * r["h"] for r in rooms)
        if fl["floor"] == 0:
            assert area / (L * W) >= 0.25, f"{building_type}: building covers only {area / (L * W):.0%} of the plot"
        # one tiled block: no empty pockets
        x0, y0 = min(r["x"] for r in rooms), min(r["y"] for r in rooms)
        x1, y1 = max(r["x"] + r["w"] for r in rooms), max(r["y"] + r["h"] for r in rooms)
        assert abs(area - (x1 - x0) * (y1 - y0)) < 1.0, f"{building_type}: plan is not a solid block"
        # every real room is room-shaped
        for r in rooms:
            if r["room_type"] in NOT_ROOM_SHAPED:
                continue
            ratio = max(r["w"], r["h"]) / min(r["w"], r["h"])
            assert ratio <= 3.2, f"{building_type}: {r['label']} is {r['w']:.0f}x{r['h']:.0f} ft"
        # every room has its own way in (planner-assigned door, main entrance room, or open hall)
        for r in fl["rooms"]:
            if r["room_type"] in SITE or r["room_type"] == "corridor_lobby":
                continue
            assert r.get("door_spec") or r.get("open_group"), f"{building_type}: {r['label']} has no door"
        st = [r for r in rooms if r["room_type"] == "staircase"]
        if st:
            stairs.append((round(st[0]["x"], 1), round(st[0]["y"], 1), round(st[0]["w"], 1), round(st[0]["h"], 1)))
    assert len(set(stairs)) <= 1, f"{building_type}: staircases do not stack: {stairs}"


if __name__ == "__main__":
    for t in CASES:
        check(t)
        print("PASS", t)
    print("ALL BAY-PLANNER CHECKS PASSED")
