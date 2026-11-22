# Reference datasets

These 20 CSVs are small, hand-checkable reference/rule tables — one per
ML/rule component in the brief — supplied alongside the trained models.
They're intentionally tiny (a handful of illustrative rows each), so they
are **not** used to retrain any model (the real training data is
`ml/dataset.csv` and `ml/cost_dataset.csv`, 1,200–1,500 rows each). Treat
these as documentation / sanity-check tables you can compare the running
code against:

| File | What it documents | Compare against |
|---|---|---|
| `01_layout_classification.csv` | plot size → layout type (1/2/3BHK) examples | `ml/dataset.csv`, `ml/train_decision_tree.py` |
| `02_cost_regression.csv` | plot/room counts → cost examples | `ml/cost_dataset.csv`, `ml/train_random_forest.py` |
| `03_room_recommendation.csv` | zone + layout type → recommended room | `vastu/vastu_rules.py` |
| `04_vastu_arrangement_rules.csv` | room/zone preference + conflict score | `vastu/vastu_rules.py: VASTU_RULES` |
| `05_layout_score_regression.csv` | per-criterion match flags → layout score | `layout/collision_checker.py` |
| `06_layout_validation_classification.csv` | per-criterion flags → Valid/Invalid | `layout/collision_checker.py` |
| `07_similarity_knn_dataset.csv` | plot/room profile → nearest layout type | (not yet implemented as an endpoint) |
| `08_layout_clustering_features.csv` | project profile features for clustering | (not yet implemented as an endpoint) |
| `09_requirement_label_dataset.csv` | free-text requirement → layout label | (not yet implemented — would front `ml/train_decision_tree.py`) |
| `10_blueprint_generation_features.csv` | plot/room profile → target room areas | `vastu/vastu_rules.py: ROOM_META` |
| `11_room_area_requirements.csv` | room → min/max sqft, category | `vastu/vastu_rules.py: ROOM_META` |
| `12_room_adjacency_rules.csv` | room pairs → preferred/avoid adjacency | `layout/collision_checker.py` |
| `13_plot_size_layout_dataset.csv` | plot size → layout type | `ml/dataset.csv` |
| `14_budget_layout_dataset.csv` | budget tier → recommended plot/layout | `backend/pipeline.py: RATE_TABLE` |
| `15_staircase_rules.csv` | staircase zone preference | `vastu/vastu_rules.py: VASTU_RULES['staircase']` |
| `16_parking_arrangement_rules.csv` | parking zone preference | `vastu/vastu_rules.py: VASTU_RULES['parking']` |
| `17_open_space_rules.csv` | open space/garden zone preference | `vastu/vastu_rules.py: VASTU_RULES['garden']` |
| `18_layout_validation_dataset.csv` | full-plan zone assignment → pass/fail score | `layout/collision_checker.py` |
| `19_blueprint_room_features.csv` | room → wall/door/window/ventilation flags | `blueprint/blueprint_generator.py` |
| `20_final_layout_labels.csv` | full requirement profile → final layout label | `ml/train_decision_tree.py` |
| `21_room_area_by_plot_size.csv` | 10 room types x 5 plot-size bands (50 rows) → realistic target sqft per room, scaled to plot size | `vastu/vastu_rules.py: room_area()`, actively loaded at runtime (not just documentation) |

Unlike 01-20, **`21_room_area_by_plot_size.csv` is live data**: `vastu/vastu_rules.room_area(room_type, plot_area)`
loads it and looks up the band matching the requested plot's area, so a
600 sqft plot and a 3000 sqft plot asking for "2 bedrooms" get differently
sized bedrooms instead of both being forced to the same fixed constant.
This is also what fixed the "every room comes out the same size"
blueprint bug — see `layout/layout_generator.py: _row_col_weights()`,
which now sizes each Vastu zone row/column by the *total target area* of
the rooms assigned to it instead of splitting the plot into flat equal
thirds.

A few of these (07, 08, 09) describe components — KNN similarity search,
clustering, free-text requirement labeling — that aren't wired up as API
endpoints in this codebase yet. They're kept here as a spec for adding
those later; ping if you want them implemented as actual endpoints.
