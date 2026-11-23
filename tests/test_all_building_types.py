"""Regression test for the BuildVision multi-building planning engine."""
from collections import Counter
import os

os.environ.setdefault("BLUEPRINT_RENDERER", "local")

from backend.pipeline import run_pipeline
from backend.schemas import HouseRequirementIn
from layout import quality
from planning.building_programs import DEFAULT_PLOTS, build_room_list

CASES = {
    "1bhk": (1, 1, 1, 1, False), "2bhk": (2, 2, 1, 1, True), "3bhk": (3, 2, 1, 1, True),
    "4bhk": (4, 3, 1, 1, True), "duplex": (4, 3, 2, 2, True), "villa": (5, 4, 2, 2, True),
    "studio": (1, 1, 1, 1, False), "farmhouse": (3, 2, 1, 1, True), "office": (8, 3, 1, 2, True),
    "school": (8, 4, 1, 2, True), "college": (12, 6, 1, 3, True), "hostel": (10, 6, 1, 2, True),
    "hospital": (6, 6, 1, 3, True), "clinic": (3, 2, 1, 1, True), "hotel": (12, 8, 1, 3, True),
    "restaurant": (1, 3, 1, 1, True), "retail": (1, 2, 1, 1, True), "mall": (4, 6, 1, 3, True),
    "warehouse": (1, 2, 1, 1, True), "factory": (1, 4, 1, 1, True), "community_hall": (1, 6, 1, 1, True),
    "bank": (4, 3, 1, 2, True), "apartment_block": (4, 4, 1, 4, True),
}


def main():
    for building_type, (count, baths, kitchens, floors, parking) in CASES.items():
        L, W = DEFAULT_PLOTS[building_type]
        req = HouseRequirementIn(
            building_type=building_type, space_count=count, plot_length=L, plot_width=W,
            bedrooms=count, bathrooms=baths, kitchens=kitchens, floors=floors,
            budget=0, parking=parking, staircase=True, vastu_preference=True,
        )
        expected = build_room_list(req)
        result = run_pipeline(req, project_id=f"regression_{building_type}")
        actual = [r["room_type"] for r in result["rooms"]]
        dimension_violations = sum(len(f.get("dimension_violations", [])) for f in result["floor_layouts"])
        # Derived circulation (passage / lobby) may be added by the planner, and every upper
        # floor carries its own staircase; everything the programme asked for must be present.
        exp, act = Counter(expected), Counter(actual)
        for derived in ("corridor_lobby", "lobby"):
            if act[derived] > exp[derived]:
                act[derived] = exp[derived]
        if floors > 1 and exp["staircase"]:
            assert act["staircase"] == floors, (building_type, "staircase per floor", act["staircase"])
            act["staircase"] = exp["staircase"]
        assert exp == act, (building_type, exp, act)
        # Staircases must stack: every upper-floor stair lies over the ground-floor stair.
        if floors > 1:
            def _stair(fl):
                st = [r for r in fl["rooms"] if r["room_type"] == "staircase"]
                return max(st, key=lambda r: r["w"] * r["h"])
            g = _stair(result["floor_layouts"][0])
            for fl in result["floor_layouts"][1:]:
                u = _stair(fl)
                ox = max(0.0, min(g["x"] + g["w"], u["x"] + u["w"]) - max(g["x"], u["x"]))
                oy = max(0.0, min(g["y"] + g["h"], u["y"] + u["h"]) - max(g["y"], u["y"]))
                assert ox * oy >= 0.85 * min(g["w"] * g["h"], u["w"] * u["h"]), (building_type, "staircase not stacked", fl.get("floor"))
        for f in result["floor_layouts"]:
            q = quality.evaluate(f, building_type)
            assert q["penalty"] <= 8.0, (building_type, f.get("floor"), q["issues"][:4])
        assert result["collisions"] == 0, (building_type, result["collisions"])
        assert result["boundary_violations"] == 0, (building_type, result["boundary_violations"])
        assert dimension_violations == 0, (building_type, dimension_violations)
        max_gap = max((f.get('gap_report', {}).get('max_internal_gap_ft', 0.0) for f in result['floor_layouts']), default=0.0)
        assert max_gap <= 1.5, (building_type, max_gap)
        assert len(result["floor_plans"]) == floors, (building_type, len(result["floor_plans"]))
        assert len(result["algorithm_trace"]) >= 8, building_type
        print(f"PASS {building_type}: {len(actual)} spaces / {floors} floor(s)")
    print(f"ALL {len(CASES)} BUILDING TYPES PASSED")


if __name__ == "__main__":
    main()
