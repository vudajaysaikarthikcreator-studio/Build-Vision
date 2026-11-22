# Civil Engineering & Building Bylaw Reference Datasets

These CSVs back `civil/structural_engineering.py`. They are **thumb-rule /
preliminary-design reference tables**, the same kind used for early-stage
estimation by architects and civil engineers before a structural
consultant does the detailed RCC design. They are not a substitute for a
licensed structural engineer's calculations or the locally applicable
building bylaw / development control regulations, which vary by state and
municipality.

| File | Used for |
|---|---|
| `soil_bearing_capacity.csv` | Safe bearing capacity (SBC) by soil type -> footing sizing |
| `structural_member_thumb_rules.csv` | Span-based beam / slab / lintel / waist-slab pre-sizing |
| `column_footing_reference.csv` | RCC column size vs safe axial load vs floors supported |
| `material_quantity_per_sqft.csv` | Cement / steel / sand / aggregate / brick / paint quantity per sqft, by finish category |
| `setback_rules_by_plot_size.csv` | Typical front/rear/side setback and max ground coverage % by plot size |
| `far_by_plot_size.csv` | Typical permissible Floor Area Ratio (FAR/FSI) and floor count by plot size |
| `staircase_dimension_standards.csv` | Riser/tread/width/headroom comfort & safety ranges |

All values are loaded at runtime (not hard-coded in Python), so they can be
retuned per state/municipality by editing the CSVs — no code change needed.
