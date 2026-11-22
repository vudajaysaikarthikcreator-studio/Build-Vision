"""
Generates larger, better-balanced synthetic training datasets for the
requirement-planning and cost-estimation models (ml/predict.py).

Why these exist alongside the original *_999.csv files:
  - The original requirement_planning_*_999.csv files sample `budget`
    almost independently of plot size, and the *_planning_rule* formula
    they were labelled with is budget-dominated (budget/3_000_000 term),
    so ~90% of rows end up labelled "Luxury" in two of the three files.
    That's a real class-imbalance problem for the DecisionTreeClassifier/
    GradientBoostingClassifier trained in ml/predict.py: it sees very few
    Compact/Balanced examples to learn from.
  - This script keeps the exact same label formula (so the new data is
    100% consistent with the app's own planning-type semantics) but
    samples budget so it actually correlates with plot size/bedrooms,
    which is both more realistic and yields a roughly-even class split.
  - Cost-estimation rows are generated from the same cost formula
    ml/predict.py uses as its non-ML fallback, plus regional/labour noise,
    matching the style of the original data but at greater volume and
    with wider, more realistic combinations of inputs.

Run: python3 ml/datasets/_generate_expanded_datasets.py
(Deterministic: fixed random seeds per file so results are reproducible.)
"""
import csv
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROWS_PER_FILE = 2500

CATEGORY_RATE = {"basic": 1350.0, "standard": 1800.0, "premium": 2450.0}


def planning_rule(area, bedrooms, floors, budget):
    score = area / 1800.0 + bedrooms * 0.35 + floors * 0.15 + budget / 3_000_000.0
    if score < 2.0:
        return "Compact"
    if score < 3.2:
        return "Balanced"
    if score < 4.6:
        return "Spacious"
    return "Luxury"


def budget_for_target_score(area, bedrooms, floors, target_score, rng):
    """Invert the planning_rule formula to pick a budget that lands the
    row's score near `target_score`, then add noise. This is what actually
    fixes the class imbalance: instead of sampling budget independently of
    everything else, we choose it so every class gets a fair share of rows.
    """
    base = area / 1800.0 + bedrooms * 0.35 + floors * 0.15
    needed = (target_score - base) * 3_000_000.0
    needed = max(150_000.0, needed)
    noise = rng.uniform(0.85, 1.18)
    return round(max(150_000.0, needed * noise), -3)


def gen_requirement_planning(name, plot_area_range, plot_aspect_range, seed):
    rng = random.Random(seed)
    rows = []
    # Target scores spanning all four bands evenly, then jittered so the
    # boundary itself isn't razor-sharp (real decision trees need some
    # near-boundary examples to learn a robust split).
    band_targets = {
        "Compact": (0.6, 1.9),
        "Balanced": (2.0, 3.1),
        "Spacious": (3.2, 4.5),
        "Luxury": (4.6, 7.5),
    }
    labels_cycle = list(band_targets.keys()) * (ROWS_PER_FILE // 4 + 1)
    rng.shuffle(labels_cycle)
    for i in range(ROWS_PER_FILE):
        target_label = labels_cycle[i]
        lo, hi = band_targets[target_label]
        target_score = rng.uniform(lo, hi)

        plot_area = rng.uniform(*plot_area_range)
        aspect = rng.uniform(*plot_aspect_range)
        plot_length = round((plot_area * aspect) ** 0.5, 1)
        plot_width = round(plot_area / max(plot_length, 1.0), 1)
        bedrooms = rng.randint(1, 6)
        bathrooms = max(1, bedrooms - rng.randint(0, 1))
        floors = rng.randint(1, 4)
        family_size = max(1, bedrooms * 2 - rng.randint(0, 2))
        parking_required = 1 if rng.random() < 0.72 else 0

        budget = budget_for_target_score(plot_area, bedrooms, floors, target_score, rng)
        label = planning_rule(plot_area, bedrooms, floors, budget)

        rows.append({
            "plot_area": int(round(plot_area)),
            "plot_length": plot_length,
            "plot_width": plot_width,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "budget": int(budget),
            "parking_required": parking_required,
            "family_size": family_size,
            "planning_type": label,
        })

    path = BASE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "plot_area", "plot_length", "plot_width", "bedrooms", "bathrooms",
            "floors", "budget", "parking_required", "family_size", "planning_type",
        ])
        writer.writeheader()
        writer.writerows(rows)
    from collections import Counter
    return path, Counter(r["planning_type"] for r in rows)


def gen_cost_estimation(name, plot_area_range, category_weights, seed):
    rng = random.Random(seed)
    rows = []
    categories = list(category_weights.keys())
    weights = list(category_weights.values())
    for _ in range(ROWS_PER_FILE):
        plot_area = rng.uniform(*plot_area_range)
        floors = rng.randint(1, 4)
        # Realistic FAR-ish ground coverage per floor rather than fully
        # independent builtup_area, so num_rooms/builtup_area stay coherent.
        coverage = rng.uniform(0.35, 0.62)
        builtup_area = round(plot_area * coverage * floors, 1)
        num_rooms = max(2, round(builtup_area / rng.uniform(180, 260)))
        num_bathrooms = max(1, min(num_rooms, round(num_rooms * rng.uniform(0.28, 0.45))))
        category = rng.choices(categories, weights=weights, k=1)[0]
        rate = CATEGORY_RATE[category]

        formula = builtup_area * rate * (1 + max(0, num_bathrooms - 2) * 0.015)
        # Regional/labour-cost noise, matching the ~1-8% spread seen in the
        # original dataset.
        noise = rng.uniform(0.94, 1.09)
        cost = round(formula * noise, 0)

        rows.append({
            "plot_area": int(round(plot_area)),
            "builtup_area": builtup_area,
            "floors": floors,
            "num_rooms": int(num_rooms),
            "num_bathrooms": int(num_bathrooms),
            "category": category,
            "cost": cost,
        })

    path = BASE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "plot_area", "builtup_area", "floors", "num_rooms",
            "num_bathrooms", "category", "cost",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    summaries = []

    p, dist = gen_requirement_planning(
        "requirement_planning_urban_2500.csv", (600, 3500), (0.9, 1.6), seed=101)
    summaries.append((p.name, dist))
    p, dist = gen_requirement_planning(
        "requirement_planning_rural_2500.csv", (1500, 9000), (0.7, 2.2), seed=202)
    summaries.append((p.name, dist))
    p, dist = gen_requirement_planning(
        "requirement_planning_mixed_2500.csv", (600, 9000), (0.6, 2.4), seed=303)
    summaries.append((p.name, dist))

    gen_cost_estimation(
        "cost_estimation_construction_2500.csv", (800, 7000),
        {"basic": 0.34, "standard": 0.44, "premium": 0.22}, seed=404)
    gen_cost_estimation(
        "cost_estimation_city_adjusted_2500.csv", (600, 6000),
        {"basic": 0.22, "standard": 0.40, "premium": 0.38}, seed=505)
    gen_cost_estimation(
        "cost_estimation_budget_mixed_2500.csv", (800, 8000),
        {"basic": 0.45, "standard": 0.35, "premium": 0.20}, seed=606)

    for name, dist in summaries:
        print(name, dict(dist))


if __name__ == "__main__":
    main()
