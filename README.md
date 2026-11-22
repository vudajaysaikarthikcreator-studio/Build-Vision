# BuildVision blueprint image dataset

**240 labelled floor-plan images · 23 building types · 2,117 room annotations** (about 10 MB).

| What | Where |
|---|---|
| Images (PNG, ~1100 px wide) | `images/<building_type>/<plan_id>_f<floor>.png` |
| Everything in one file (images + rooms) | `index.json` |
| One row per image | `annotations.csv` |
| One row per room (feet **and** pixel boxes) | `rooms.csv` |
| COCO object-detection labels (64 room categories) | `coco_annotations.json` |
| Photo-style samples (simulated phone photo / blue cyanotype) | `images_photo_style/` + `photo_style_manifest.csv` |
| Your own real photos go here | `images/real/` (+ optional `real_labels.csv`) |

## Important: what the images are
The sandbox this was built in has no internet, so **no real photographs were downloaded**. The images are
floor plans drawn by *this project's own* planner + renderer (planning -> Vastu -> layout -> blueprint), so
every label is exact. They cover residential (studio, 1-4 BHK, duplex, villa, farmhouse) and 15 non-residential
types (office, school, college, hostel, hospital, clinic, hotel, restaurant, retail, mall, warehouse, factory,
community hall, bank, apartment block) with 3-4 plot sizes and all four road sides.
`images_photo_style/` has *simulated* photos (paper texture, uneven light, perspective, blur, or blue cyanotype) -
useful for robustness, not a substitute for real scans.

## Columns
`annotations.csv`: image_id, file, source (`synthetic`|`real`), building_type, plan_id, floor, floors,
plot_length_ft (N-S), plot_width_ft (E-W), plot_area_sqft, road_direction, bedrooms, bathrooms, kitchens, parking,
vastu_score (share of rooms in an acceptable Vastu zone, 0-1), engine, width_px, height_px, scale_px_per_ft, rooms_count.

`rooms.csv`: image_id, room_index, room_type, label, zone, x_ft, y_ft, w_ft, h_ft, area_sqft,
xmin_px, ymin_px, xmax_px, ymax_px, vastu_acceptable.
Plot coordinates: `x` runs south->north, `y` west->east; the image has north at the top.

## Use it
```python
from backend.blueprint_dataset import list_images, similar, BlueprintDataset

similar("3bhk", plot_length=30, plot_width=40, bedrooms=3)        # nearest reference blueprints
list_images(building_type="school", floor=0)["items"]
ds = BlueprintDataset(building_type="villa")                      # works as a torch Dataset (len + getitem)
img, ann = ds[0]                                                  # PIL image, labels
```
* API (FastAPI, port 8000): `GET /api/blueprint-dataset`, `/stats`, `/similar?building_type=3bhk&plot_length=30&plot_width=40`,
  `/{image_id}`; images at `/static/blueprint-dataset/<file>`.
* Gallery: open `frontend/dataset.html` (also linked from the planner top bar). Click an image to see the room boxes.
* Training: `coco_annotations.json` works with any COCO loader (Detectron2, torchvision, YOLO converters).

## Add real blueprint photos
1. Copy images into `images/real/` (jpg/png/webp).
2. Optional labels: copy `real_labels.example.csv` to `real_labels.csv`, one row per file.
3. `python datasets/blueprints/build_index.py` - they appear in the gallery and API with `source = "real"`.

Public floor-plan datasets exist (for example CubiCasa5K and RPLAN); download them yourself, check their licences,
and drop the images into `images/real/` the same way.

## Regenerate / extend
```
python datasets/blueprints/generate_blueprint_dataset.py          # resumable; --only school hospital; --max-plans 20
python datasets/blueprints/build_index.py                         # CSV / JSON / COCO
python datasets/blueprints/augment_photo_style.py --per-type 3    # more photo-style samples
```
Edit `RESIDENTIAL` / `COMMERCIAL` at the top of `generate_blueprint_dataset.py` to add plot sizes or types.
The generator always uses the local Pillow renderer (never Gemini), so results are free and reproducible.
