"""
Build the machine-readable files of the blueprint dataset from what is on disk.

    python datasets/blueprints/build_index.py

Reads   _records.jsonl                (written by generate_blueprint_dataset.py - labelled synthetic plans)
        images/real/*.jpg|png|webp    (OPTIONAL: your own real blueprint photos / scans)
        real_labels.csv               (OPTIONAL labels for those: file,building_type,floors,plot_length_ft,
                                       plot_width_ft,bedrooms,bathrooms,notes)
Writes  index.json  annotations.csv  rooms.csv  coco_annotations.json
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "images")
RECORDS = os.path.join(HERE, "_records.jsonl")
REAL_DIR = os.path.join(IMAGES, "real")
REAL_LABELS = os.path.join(HERE, "real_labels.csv")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def _read_records():
    plans = {}
    if os.path.exists(RECORDS):
        with open(RECORDS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    plans[rec["plan_id"]] = rec              # latest wins
    return list(plans.values())


def _real_images():
    labels = {}
    if os.path.exists(REAL_LABELS):
        with open(REAL_LABELS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["file"].strip()] = row
    out = []
    if os.path.isdir(REAL_DIR):
        try:
            from PIL import Image
        except ImportError:                                  # pragma: no cover
            Image = None
        for name in sorted(os.listdir(REAL_DIR)):
            if os.path.splitext(name)[1].lower() not in IMG_EXT:
                continue
            w = h = None
            if Image is not None:
                try:
                    with Image.open(os.path.join(REAL_DIR, name)) as im:
                        w, h = im.size
                except Exception:                            # noqa: BLE001
                    pass
            lab = labels.get(name) or labels.get(f"real/{name}") or {}

            def num(k):
                try:
                    return float(lab.get(k)) if lab.get(k) not in (None, "") else None
                except ValueError:
                    return None
            out.append({
                "image_id": "real_" + os.path.splitext(name)[0], "file": f"real/{name}", "source": "real",
                "building_type": (lab.get("building_type") or "unknown").strip().lower(), "plan_id": None,
                "floor": 0, "floor_title": None, "floors": int(num("floors") or 1),
                "plot_length_ft": num("plot_length_ft"), "plot_width_ft": num("plot_width_ft"),
                "plot_area_sqft": None, "road_direction": None, "bedrooms": int(num("bedrooms") or 0) or None,
                "bathrooms": int(num("bathrooms") or 0) or None, "kitchens": None, "parking": None,
                "vastu_score": None, "engine": None, "width_px": w, "height_px": h, "scale_px_per_ft": None,
                "origin_px": None, "rooms_count": 0, "rooms": [], "notes": lab.get("notes")})
    return out


def build():
    images = []
    for plan in _read_records():
        for im in plan["images"]:
            if not os.path.exists(os.path.join(IMAGES, im["file"])):
                continue
            row = {
                "image_id": im["image_id"], "file": im["file"], "source": "synthetic",
                "building_type": plan["building_type"], "plan_id": plan["plan_id"], "floor": im["floor"],
                "floor_title": im["floor_title"], "floors": plan["floors"],
                "plot_length_ft": plan["plot_length_ft"], "plot_width_ft": plan["plot_width_ft"],
                "plot_area_sqft": plan["plot_area_sqft"], "road_direction": plan["road_direction"],
                "bedrooms": plan["bedrooms"], "bathrooms": plan["bathrooms"], "kitchens": plan["kitchens"],
                "parking": plan["parking"], "vastu_score": im.get("vastu_score"), "engine": im.get("engine"),
                "width_px": im["width_px"], "height_px": im["height_px"], "scale_px_per_ft": im["scale_px_per_ft"],
                "origin_px": im["origin_px"], "rooms_count": len(im["rooms"]), "rooms": im["rooms"]}
            images.append(row)
    images += _real_images()
    images.sort(key=lambda r: (r["source"], r["building_type"], r["plot_area_sqft"] or 0, r["image_id"]))

    categories = sorted({r["room_type"] for im in images for r in im["rooms"] if r.get("room_type")})
    by_type = Counter(im["building_type"] for im in images)
    summary = {
        "images": len(images), "synthetic": sum(1 for i in images if i["source"] == "synthetic"),
        "real": sum(1 for i in images if i["source"] == "real"), "building_types": len(by_type),
        "by_building_type": dict(sorted(by_type.items())), "room_categories": len(categories),
        "room_annotations": sum(len(i["rooms"]) for i in images)}
    index = {
        "name": "BuildVision Blueprint Image Dataset", "version": "1.0",
        "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
        "description": "Floor-plan images with labelled rooms. 'synthetic' images are rendered by this project's "
                       "own planner (not photographs); 'real' images are whatever you add under images/real/.",
        "image_base": "images/", "summary": summary, "categories": categories, "images": images}
    with open(os.path.join(HERE, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))

    cols = ["image_id", "file", "source", "building_type", "plan_id", "floor", "floors", "plot_length_ft",
            "plot_width_ft", "plot_area_sqft", "road_direction", "bedrooms", "bathrooms", "kitchens", "parking",
            "vastu_score", "engine", "width_px", "height_px", "scale_px_per_ft", "rooms_count"]
    with open(os.path.join(HERE, "annotations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for im in images:
            w.writerow([im.get(c) for c in cols])
    with open(os.path.join(HERE, "rooms.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "room_index", "room_type", "label", "zone", "x_ft", "y_ft", "w_ft", "h_ft",
                    "area_sqft", "xmin_px", "ymin_px", "xmax_px", "ymax_px", "vastu_acceptable"])
        for im in images:
            for r in im["rooms"]:
                b = r["bbox_px"]
                w.writerow([im["image_id"], r["index"], r["room_type"], r["label"], r["zone"], r["x_ft"], r["y_ft"],
                            r["w_ft"], r["h_ft"], r["area_sqft"], b[0], b[1], b[2], b[3], r.get("vastu_acceptable")])
    cat_id = {c: i + 1 for i, c in enumerate(categories)}
    coco = {"info": {"description": index["name"], "version": index["version"]},
            "licenses": [], "categories": [{"id": i, "name": c} for c, i in cat_id.items()],
            "images": [], "annotations": []}
    ann_id = 1
    for n, im in enumerate(images, start=1):
        if not im["rooms"]:
            continue
        coco["images"].append({"id": n, "file_name": im["file"], "width": im["width_px"], "height": im["height_px"],
                               "building_type": im["building_type"], "floor": im["floor"]})
        for r in im["rooms"]:
            x0, y0, x1, y1 = r["bbox_px"]
            coco["annotations"].append({"id": ann_id, "image_id": n, "category_id": cat_id[r["room_type"]],
                                        "bbox": [x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)],
                                        "area": round((x1 - x0) * (y1 - y0), 1), "iscrowd": 0})
            ann_id += 1
    with open(os.path.join(HERE, "coco_annotations.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, separators=(",", ":"))
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps(s, indent=2))
