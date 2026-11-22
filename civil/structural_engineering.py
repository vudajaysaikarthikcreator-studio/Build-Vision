"""
civil/structural_engineering.py

Preliminary civil-engineering and building-bylaw algorithms that sit
alongside the AI layout/cost pipeline. Everything here is a documented
thumb rule (the kind an architect uses at the concept stage, before a
structural consultant runs the actual RCC design) backed by CSV lookup
tables in datasets/civil/, so the numbers can be retuned per
state/municipality without touching code.

Public entry point: generate_structural_report(...) — called from
backend/pipeline.py and backend/routes/projects.py.

Nothing here should be read as a substitute for a licensed structural
engineer's calculations or the locally applicable building bylaws.
"""

import csv
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(HERE, "..", "datasets", "civil")

# Rough dead + live load allowance for a residential RCC framed structure,
# per sqft of built-up area, per floor. Used only to turn "built-up area"
# into an approximate load for footing/column pre-sizing — not a
# substitute for a proper structural load takedown.
LOAD_PER_SQFT_PER_FLOOR_KN = 1.35
FACTOR_OF_SAFETY = 1.5


def _load_csv(name):
    path = os.path.join(DATASETS_DIR, name)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def _soil_table():
    return _load_csv("soil_bearing_capacity.csv")


@lru_cache(maxsize=None)
def _member_table():
    return _load_csv("structural_member_thumb_rules.csv")


@lru_cache(maxsize=None)
def _column_table():
    return _load_csv("column_footing_reference.csv")


@lru_cache(maxsize=None)
def _material_table():
    return _load_csv("material_quantity_per_sqft.csv")


@lru_cache(maxsize=None)
def _setback_table():
    return _load_csv("setback_rules_by_plot_size.csv")


@lru_cache(maxsize=None)
def _far_table():
    return _load_csv("far_by_plot_size.csv")


@lru_cache(maxsize=None)
def _staircase_table():
    return {row["parameter"]: row for row in _load_csv("staircase_dimension_standards.csv")}


def list_soil_types():
    """Soil types the frontend can offer in a dropdown."""
    return [row["soil_type"] for row in _soil_table()]


def get_soil_bearing_capacity(soil_type: str) -> dict:
    for row in _soil_table():
        if row["soil_type"].strip().lower() == (soil_type or "").strip().lower():
            return row
    # Default to the "medium clay / sandy clay" row — a conservative,
    # commonly-assumed default when no soil test is available yet.
    return _soil_table()[2]


def _tier_row(table, area_key_min, area_key_max, value):
    for row in table:
        if float(row[area_key_min]) <= value < float(row[area_key_max]):
            return row
    return table[-1]


# ---------------------------------------------------------------- footing
def recommend_footing(plot_length: float, plot_width: float, floors: int,
                       soil_type: str = "Medium clay / sandy clay",
                       num_columns: int | None = None) -> dict:
    """
    Pre-size a typical isolated RCC footing for this plot/floor count.

    Very approximate: spreads (plot area x floors x load/sqft) over an
    assumed column grid to get a load-per-column, then sizes a square
    footing so that load / footing_area <= safe bearing capacity.
    """
    plot_area = max(plot_length * plot_width, 1.0)
    soil = get_soil_bearing_capacity(soil_type)
    sbc_kn_m2 = float(soil["safe_bearing_capacity_kn_m2"])

    # Assume one column roughly every 200-250 sqft of plate on a typical
    # residential grid, minimum 4 (the four plot corners).
    columns = num_columns or max(4, round(plot_area / 220))
    total_load_kn = plot_area * floors * LOAD_PER_SQFT_PER_FLOOR_KN * FACTOR_OF_SAFETY
    load_per_column_kn = total_load_kn / columns

    # sqft -> sqm for the SBC check (SBC table is in kN/m2)
    required_area_m2 = load_per_column_kn / sbc_kn_m2 if sbc_kn_m2 else 0
    side_m = max(required_area_m2, 0.81) ** 0.5  # 0.9m x 0.9m minimum practical footing
    side_ft = round(side_m * 3.281, 2)

    return {
        "soil_type": soil["soil_type"],
        "safe_bearing_capacity_kn_m2": sbc_kn_m2,
        "assumed_columns": columns,
        "estimated_load_per_column_kn": round(load_per_column_kn, 1),
        "footing_type": "Isolated RCC footing" if sbc_kn_m2 >= 100 else "Raft / combined footing recommended",
        "recommended_footing_size_ft": f"{side_ft} x {side_ft}",
        "foundation_advice": soil["foundation_advice"],
    }


# ---------------------------------------------------------------- column
def recommend_column(floors: int, load_per_column_kn: float | None = None) -> dict:
    """
    Picks the smallest standard column size that supports both the floor
    count and, when known (from recommend_footing's load estimate), the
    approximate axial load per column — so the two never disagree the way
    a floors-only lookup can on a small, heavily-loaded plot.
    """
    rows = _column_table()
    by_floor = [r for r in rows if int(r["max_floors_supported"]) >= floors] or [rows[-1]]

    chosen = by_floor[0]
    if load_per_column_kn is not None:
        for row in by_floor:
            if float(row["safe_load_per_column_kn"]) >= load_per_column_kn:
                chosen = row
                break
        else:
            # None of the floor-adequate sizes carry the load: step up
            # through the full table by load capacity alone.
            for row in rows:
                if float(row["safe_load_per_column_kn"]) >= load_per_column_kn:
                    chosen = row
                    break
            else:
                chosen = rows[-1]

    return {
        "recommended_size_mm": chosen["column_size_mm"],
        "recommended_size_in": chosen["column_size_in"],
        "safe_axial_load_kn": float(chosen["safe_load_per_column_kn"]),
        "max_floors_supported": int(chosen["max_floors_supported"]),
        "typical_use": chosen["typical_use"],
    }


# ---------------------------------------------------------------- beam / slab
def _member_rule(member: str, span_ft: float) -> dict | None:
    for row in _member_table():
        if row["member"] == member and float(row["span_ft_min"]) <= span_ft <= float(row["span_ft_max"]):
            return row
    return None


def recommend_beam(span_ft: float) -> dict:
    row = _member_rule("RCC Beam", span_ft) or _member_table()[2]
    depth_in = round(max(span_ft * (0.8 if span_ft <= 10 else 0.9 if span_ft <= 16 else 1.0), 9), 1)
    width_in = round(max(depth_in / 2, 9), 1)
    return {"span_ft": round(span_ft, 1), "depth_in": depth_in, "width_in": width_in,
            "condition": row["condition"], "notes": row["notes"]}


def recommend_slab_thickness(span_ft: float) -> dict:
    if span_ft <= 10:
        divisor, kind = 26, "one-way"
    elif span_ft <= 16:
        divisor, kind = 30, "two-way"
    else:
        divisor, kind = 32, "continuous"
    thickness_in = round(max(span_ft * 12 / divisor, 4.0), 2)
    return {"span_ft": round(span_ft, 1), "slab_type": kind, "thickness_in": thickness_in}


# ---------------------------------------------------------------- staircase
def check_staircase(riser_mm: float = 175, tread_mm: float = 270,
                     width_mm: float = 1000, flight_span_ft: float = 10) -> dict:
    std = _staircase_table()
    issues = []

    def _range_check(name, value, key):
        lo, hi = float(std[key]["min_value"]), float(std[key]["max_value"])
        if not (lo <= value <= hi):
            issues.append(f"{name} {value:.0f} mm is outside the comfortable {lo:.0f}-{hi:.0f} mm range.")
        return lo, hi

    _range_check("Riser", riser_mm, "riser")
    _range_check("Tread", tread_mm, "tread")
    _range_check("Flight width", width_mm, "flight_width")

    combo = 2 * riser_mm + tread_mm
    lo, hi = float(std["riser_tread_rule_2r_plus_t"]["min_value"]), float(std["riser_tread_rule_2r_plus_t"]["max_value"])
    if not (lo <= combo <= hi):
        issues.append(f"(2 x riser) + tread = {combo:.0f} mm, outside the comfortable {lo:.0f}-{hi:.0f} mm range.")

    waist_thickness_in = round(flight_span_ft * 12 / 20, 2)

    return {
        "riser_mm": riser_mm, "tread_mm": tread_mm, "flight_width_mm": width_mm,
        "two_riser_plus_tread_mm": combo,
        "waist_slab_thickness_in": waist_thickness_in,
        "compliant": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------- materials
def estimate_material_quantities(builtup_area_sqft: float, category: str = "standard") -> dict:
    row = next((r for r in _material_table() if r["category"] == category), _material_table()[1])
    cement_bags = builtup_area_sqft * float(row["cement_bags_per_sqft"])
    steel_kg = builtup_area_sqft * float(row["steel_kg_per_sqft"])
    return {
        "category": row["category"],
        "cement_bags": round(cement_bags),
        "steel_kg": round(steel_kg),
        "steel_tons": round(steel_kg / 1000, 2),
        "sand_cft": round(builtup_area_sqft * float(row["sand_cft_per_sqft"])),
        "aggregate_cft": round(builtup_area_sqft * float(row["aggregate_cft_per_sqft"])),
        "bricks": round(builtup_area_sqft * float(row["bricks_per_sqft"])),
        "paint_liters": round(builtup_area_sqft * float(row["paint_liters_per_100sqft"]) / 100, 1),
    }


# ---------------------------------------------------------------- bylaws
def check_setback_compliance(plot_length_ft: float, plot_width_ft: float) -> dict:
    plot_area = plot_length_ft * plot_width_ft
    row = _tier_row(_setback_table(), "plot_area_sqft_min", "plot_area_sqft_max", plot_area)
    front, rear, side = float(row["front_setback_ft"]), float(row["rear_setback_ft"]), float(row["side_setback_ft"])
    max_coverage_pct = float(row["max_ground_coverage_pct"])

    buildable_length = max(plot_length_ft - 2 * side, 0)
    buildable_width = max(plot_width_ft - front - rear, 0)
    buildable_footprint = buildable_length * buildable_width
    max_allowed_footprint = plot_area * max_coverage_pct / 100

    return {
        "plot_area_sqft": round(plot_area, 1),
        "front_setback_ft": front, "rear_setback_ft": rear, "side_setback_ft": side,
        "max_ground_coverage_pct": max_coverage_pct,
        "buildable_footprint_after_setbacks_sqft": round(buildable_footprint, 1),
        "max_footprint_by_coverage_sqft": round(max_allowed_footprint, 1),
        "effective_max_ground_floor_footprint_sqft": round(min(buildable_footprint, max_allowed_footprint), 1),
    }


def check_far_compliance(plot_length_ft: float, plot_width_ft: float, floors: int, builtup_area_sqft: float) -> dict:
    plot_area = plot_length_ft * plot_width_ft
    row = _tier_row(_far_table(), "plot_area_sqft_min", "plot_area_sqft_max", plot_area)
    far = float(row["permissible_far"])
    max_floors_typical = int(row["max_floors_typical"])
    permissible_builtup = plot_area * far
    proposed_far = builtup_area_sqft / plot_area if plot_area else 0

    return {
        "plot_area_sqft": round(plot_area, 1),
        "permissible_far": far,
        "permissible_total_builtup_sqft": round(permissible_builtup, 1),
        "proposed_builtup_sqft": round(builtup_area_sqft, 1),
        "proposed_far": round(proposed_far, 2),
        "within_far_limit": builtup_area_sqft <= permissible_builtup,
        "max_floors_typical": max_floors_typical,
        "within_typical_floor_limit": floors <= max_floors_typical,
    }


# ---------------------------------------------------------------- aggregate report
def generate_structural_report(plot_length: float, plot_width: float, floors: int,
                                builtup_area_sqft: float, category: str = "standard",
                                soil_type: str = "Medium clay / sandy clay",
                                rooms: list | None = None,
                                has_staircase: bool = True) -> dict:
    """
    One-call aggregator used by backend/pipeline.py and
    backend/routes/projects.py. `rooms` (list of dicts with w/h in feet,
    as produced by layout_generator) is used, if given, to pick a
    realistic governing span for the beam/slab pre-sizing instead of a
    flat guess.
    """
    if rooms:
        governing_span_ft = max((max(r["w"], r["h"]) for r in rooms), default=12)
    else:
        governing_span_ft = max(plot_length, plot_width) / 3

    footing = recommend_footing(plot_length, plot_width, floors, soil_type)
    result = {
        "disclaimer": "Preliminary civil-engineering thumb-rule estimates for concept "
                      "planning only. Get a licensed structural engineer's design and "
                      "confirm local building bylaws before construction.",
        "footing": footing,
        "column": recommend_column(floors, footing["estimated_load_per_column_kn"]),
        "beam": recommend_beam(governing_span_ft),
        "slab": recommend_slab_thickness(governing_span_ft),
        "material_estimate": estimate_material_quantities(builtup_area_sqft, category),
        "setback_compliance": check_setback_compliance(plot_length, plot_width),
        "far_compliance": check_far_compliance(plot_length, plot_width, floors, builtup_area_sqft),
    }
    if has_staircase:
        result["staircase"] = check_staircase()
    return result
