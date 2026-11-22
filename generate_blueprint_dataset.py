"""
Generate the BuildVision blueprint IMAGE dataset from the project's own planner + renderer.

Every image is a floor-plan drawing produced by the real pipeline (planning -> Vastu -> layout ->
blueprint renderer) for a different building type / plot size / road side / floor, together with
exact labels: plot size, room rectangles (feet and pixels), room types, Vastu zone, scores.

    python datasets/blueprints/generate_blueprint_dataset.py                 # everything (resumable)
    python datasets/blueprints/generate_blueprint_dataset.py --only school hospital
    python datasets/blueprints/generate_blueprint_dataset.py --max-plans 20  # quick sample
    python datasets/blueprints/build_index.py                                # (re)build CSV / JSON / COCO

Needs the project requirements (pillow, numpy, scipy, pydantic ...). It never calls Gemini: the
renderer is forced to the local Pillow renderer so the dataset is reproducible and free.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
os.environ["BLUEPRINT_RENDERER"] = "local"            # reproducible, offline, no API cost

from PIL import Image                                  # noqa: E402

IMAGES_DIR = os.path.join(HERE, "images")
RECORDS = os.path.join(HERE, "_records.jsonl")
MAX_WIDTH_PX = 1100                                    # saved image width (the renderer's own is ~1500-1900)
ROADS = ["North", "East", "South", "West"]

# building_type -> (plots (length N-S, width E-W) in ft, bedrooms/space_count, bathrooms, kitchens, floors)
RESIDENTIAL = {
    "studio":    ([(15, 25), (20, 30), (25, 30)], 1, 1, 1, 1),
    "1bhk":      ([(20, 30), (25, 40), (30, 40)], 1, 1, 1, 1),
    "2bhk":      ([(25, 40), (30, 40), (30, 50)], 2, 2, 1, 1),
    "3bhk":      ([(30, 40), (30, 50), (40, 60)], 3, 3, 1, 1),
    "4bhk":      ([(40, 60), (50, 60), (50, 80)], 4, 4, 1, 1),
    "duplex":    ([(30, 50), (40, 60), (50, 80)], 4, 4, 2, 2),
    "villa":     ([(40, 60), (50, 80), (60, 90)], 4, 5, 2, 2),
    "farmhouse": ([(60, 90), (80, 120), (100, 150)], 3, 3, 1, 1),
}
# non-residential: (space_count, bathrooms, kitchens, floors) - plots come from planning.building_programs.DEFAULT_PLOTS
COMMERCIAL = {
    "office": (8, 4, 1, 2), "school": (8, 4, 1, 2), "college": (8, 6, 1, 3), "hostel": (12, 6, 1, 2),
    "hospital": (10, 6, 1, 3), "clinic": (4, 2, 1, 1), "hotel": (12, 12, 1, 3), "restaurant": (1, 2, 1, 1),
    "retail": (4, 2, 1, 1), "mall": (8, 6, 1, 3), "warehouse": (1, 2, 1, 1), "factory": (1, 4, 1, 1),
    "community_hall": (1, 4, 1, 1), "bank": (6, 3, 1, 2), "apartment_block": (4, 4, 1, 4),
}


def build_recipes(only=None):
    from planning.building_programs import DEFAULT_PLOTS
    recipes, n = [], 0
    for bt, (plots, beds, baths, kit, floors) in RESIDENTIAL.items():
        for pi, (L, W) in enumerate(plots):
            for ri, road in enumerate(ROADS):
                n += 1
                recipes.append(dict(building_type=bt, plot_length=L, plot_width=W, road=road, space_count=beds,
                                    bedrooms=beds, bathrooms=baths, kitchens=kit, floors=floors,
                                    parking=((pi + ri) % 4 != 3)))
    for ti, (bt, (cnt, baths, kit, floors)) in enumerate(COMMERCIAL.items()):
        L0, W0 = DEFAULT_PLOTS[bt]
        plots = [(L0, W0), (round(L0 * 1.25 / 5) * 5, round(W0 * 1.25 / 5) * 5)]
        for pi, (L, W) in enumerate(plots):
            for ri in range(2):
                road = ROADS[(ti + pi + ri * 2) % 4]
                recipes.append(dict(building_type=bt, plot_length=L, plot_width=W, road=road, space_count=cnt,
                                    bedrooms=cnt, bathrooms=baths, kitchens=kit, floors=floors, parking=True))
    seen = {}
    for r in recipes:
        base = f"{r['building_type']}_{int(r['plot_length'])}x{int(r['plot_width'])}_{r['road'][0]}_{r['floors']}f"
        seen[base] = seen.get(base, 0) + 1
        r["plan_id"] = base if seen[base] == 1 else f"{base}_{seen[base]}"
    if only:
        recipes = [r for r in recipes if r["building_type"] in set(only)]
    return recipes


def done_plans():
    ids = set()
    if os.path.exists(RECORDS):
        with open(RECORDS, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["plan_id"])
                except Exception:      # noqa: BLE001
                    pass
    return ids


def pixel_transform(plot_length: float, plot_width: float, factor: float):
    """Feet -> pixel mapping used by blueprint/local_renderer.py (north at the top, x = N-S, y = E-W)."""
    scale = max(5.0, min(12.0, 1050.0 / max(plot_length, plot_width)))
    return scale * factor, 150.0 * factor, 115.0 * factor


def room_bbox_px(room: dict, L: float, scale: float, ox: float, oy: float):
    x0 = ox + room["y"] * scale
    x1 = ox + (room["y"] + room["h"]) * scale
    y0 = oy + (L - (room["x"] + room["w"])) * scale
    y1 = oy + (L - room["x"]) * scale
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def generate_plan(recipe: dict) -> dict:
    from backend.schemas import HouseRequirementIn
    from backend.pipeline import run_pipeline

    req = HouseRequirementIn(
        building_type=recipe["building_type"], space_count=recipe["space_count"], plot_length=recipe["plot_length"],
        plot_width=recipe["plot_width"], bedrooms=recipe["bedrooms"], bathrooms=recipe["bathrooms"],
        kitchens=recipe["kitchens"], floors=recipe["floors"], budget=0, parking=recipe["parking"], staircase=True,
        vastu_preference=True, road_direction=recipe["road"], main_entrance_direction=recipe["road"])
    pid = "bpds_" + recipe["plan_id"]
    res = run_pipeline(req, project_id=pid)
    out_dir = os.path.join(ROOT, "blueprint", "output")
    folder = os.path.join(IMAGES_DIR, recipe["building_type"])
    os.makedirs(folder, exist_ok=True)
    images = []
    for fp, layout in zip(res["floor_plans"], res["floor_layouts"]):
        src = os.path.join(out_dir, fp["filename"])
        with Image.open(src) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            factor = min(1.0, MAX_WIDTH_PX / w0)
            if factor < 1.0:
                im = im.resize((round(w0 * factor), round(h0 * factor)), Image.LANCZOS)
            image_id = f"{recipe['plan_id']}_f{fp['floor']}"
            rel = f"{recipe['building_type']}/{image_id}.png"
            # 256-colour palette PNG: line drawings keep every detail but are ~3x smaller
            im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
                os.path.join(IMAGES_DIR, rel), "PNG", optimize=True)
            width_px, height_px = im.size
        plot = layout["plot"]
        L, W = float(plot["length"]), float(plot["width"])
        scale, ox, oy = pixel_transform(L, W, factor)
        rooms = []
        for i, r in enumerate(layout["rooms"]):
            rooms.append({
                "index": i, "room_type": r.get("room_type"), "label": r.get("label"), "zone": r.get("zone"),
                "x_ft": round(r["x"], 2), "y_ft": round(r["y"], 2), "w_ft": round(r["w"], 2), "h_ft": round(r["h"], 2),
                "area_sqft": round(r["w"] * r["h"], 1), "bbox_px": room_bbox_px(r, L, scale, ox, oy),
                "vastu_acceptable": bool(r.get("vastu_acceptable")) if r.get("vastu_acceptable") is not None else None,
            })
        images.append({
            "image_id": image_id, "file": rel, "floor": fp["floor"], "floor_title": fp["title"],
            "width_px": width_px, "height_px": height_px, "scale_px_per_ft": round(scale, 4),
            "origin_px": [round(ox, 2), round(oy, 2)], "plot_length_ft": L, "plot_width_ft": W,
            "engine": layout.get("engine"), "vastu_score": layout.get("vastu_score"),     # share of rooms in an acceptable zone
            "stair_stacked": layout.get("stair_stacked"), "rooms": rooms,
        })
        try:
            os.remove(src)
        except OSError:
            pass
    for extra in (f"{pid}.png",):
        try:
            os.remove(os.path.join(out_dir, extra))
        except OSError:
            pass
    return {
        "plan_id": recipe["plan_id"], "building_type": recipe["building_type"], "floors": recipe["floors"],
        "plot_length_ft": recipe["plot_length"], "plot_width_ft": recipe["plot_width"],
        "plot_area_sqft": recipe["plot_length"] * recipe["plot_width"], "road_direction": recipe["road"],
        "bedrooms": recipe["bedrooms"], "bathrooms": recipe["bathrooms"], "kitchens": recipe["kitchens"],
        "parking": recipe["parking"], "vastu_score_plan": res.get("vastu_score"),
        "collisions": res.get("collisions"), "boundary_violations": res.get("boundary_violations"),
        "builtup_area_sqft": (res.get("cost") or {}).get("builtup_area"), "images": images,
        "source": "synthetic: BuildVision planner + local renderer",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="building types to generate (default: all)")
    ap.add_argument("--max-plans", type=int, default=0, help="stop after N new plans (0 = no limit)")
    ap.add_argument("--time-budget", type=int, default=0, help="stop after N seconds (resume later)")
    a = ap.parse_args()
    recipes = build_recipes(a.only)
    done = done_plans()
    todo = [r for r in recipes if r["plan_id"] not in done]
    print(f"{len(recipes)} plans in recipe, {len(done)} already done, {len(todo)} to do", flush=True)
    t0, made, failed = time.time(), 0, 0
    for r in todo:
        if a.max_plans and made >= a.max_plans:
            break
        if a.time_budget and time.time() - t0 > a.time_budget:
            break
        try:
            rec = generate_plan(r)
        except Exception as e:          # noqa: BLE001 - infeasible plot/program combinations are skipped
            failed += 1
            print(f"  skip {r['plan_id']}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            with open(os.path.join(HERE, "_skipped.log"), "a", encoding="utf-8") as f:
                f.write(f"{r['plan_id']}\t{type(e).__name__}\t{str(e)[:200]}\n")
            continue
        with open(RECORDS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        made += 1
        print(f"  [{made}] {r['plan_id']}  ({len(rec['images'])} image(s), {time.time() - t0:.0f}s)", flush=True)
    print(f"done: {made} new plans, {failed} skipped, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
