# BuildVision ML / Rule Datasets

This folder contains reproducible project datasets used by the BuildVision ML and Vastu layers.
The datasets are synthetic/engineering-reference data and are not a substitute for architectural, structural, or quantity-surveyor data.

Files: 15

- **Requirement-planning datasets** (Decision Tree / Gradient Boosting planning-type recommendation):
  - `requirement_planning_{urban,rural,mixed}_999.csv` — original, 999 rows each.
  - `requirement_planning_{urban,rural,mixed}_2500.csv` — added later, 2,500 rows each.
    Budget is sampled so it actually correlates with plot size/bedrooms/floors (rather
    than independently of them), which both keeps rows architecturally realistic *and*
    gives the model a much better class balance across Compact/Balanced/Spacious/Luxury
    than the original files alone (which skew 84-93% "Luxury" because independently-
    sampled large budgets dominate the labeling formula). Both the `_999` and `_2500`
    files are loaded together by `ml/predict.py`.
  - Regenerate with `python3 ml/datasets/_generate_expanded_datasets.py` (deterministic,
    fixed seeds).

- **Cost-estimation datasets** (Random Forest / Gradient Boosting cost estimation):
  - `cost_estimation_{construction,city_adjusted,budget_mixed}_999.csv` — original, 999 rows each.
  - `cost_estimation_{construction,city_adjusted,budget_mixed}_2500.csv` — added later, 2,500
    rows each, generated from the same cost formula `ml/predict.py` uses as its non-ML
    fallback, plus regional/labour noise in the same ~1-9% spread as the original data.
    Both `_999` and `_2500` files are loaded together.

- **Vastu-arrangement datasets** (rule-support/reference data for room-zone arrangement):
  - `vastu_arrangement_{balanced,east_facing,north_facing}_999.csv` — 999 rows each.
    Reference-only; not currently loaded by any code path.
