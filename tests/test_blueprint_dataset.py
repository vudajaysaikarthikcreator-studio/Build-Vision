"""Sanity checks for datasets/blueprints (run: python tests/test_blueprint_dataset.py)."""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend import blueprint_dataset as ds   # noqa: E402


def main():
    idx = ds.load_index()
    images = idx["images"]
    assert len(images) >= 200, "dataset looks empty - run datasets/blueprints/generate_blueprint_dataset.py"
    types = {e["building_type"] for e in images}
    assert len(types) >= 23, types
    for e in images:
        path = os.path.join(ds.IMAGES_DIR, e["file"])
        assert os.path.exists(path), ("missing image", e["file"])
        for r in e["rooms"]:
            x0, y0, x1, y1 = r["bbox_px"]
            assert 0 <= x0 < x1 <= e["width_px"] + 1 and 0 <= y0 < y1 <= e["height_px"] + 1, (e["image_id"], r["label"], r["bbox_px"])
        if e["source"] == "synthetic":
            assert 0.0 <= (e["vastu_score"] or 0.0) <= 1.0, (e["image_id"], e["vastu_score"])
    coco = json.load(open(os.path.join(ds.DATASET_DIR, "coco_annotations.json"), encoding="utf-8"))
    assert len(coco["annotations"]) == sum(len(e["rooms"]) for e in images)
    s = ds.similar("3bhk", 30, 40, 3, 1, "North", 0, 3)
    assert s and s[0]["building_type"] == "3bhk" and s[0]["distance"] <= s[-1]["distance"]
    assert ds.list_images(building_type="school")["total"] > 0
    assert ds.get_image(images[0]["image_id"])["rooms"]
    print(f"OK: {len(images)} images, {len(types)} building types, {len(coco['annotations'])} room boxes")


if __name__ == "__main__":
    main()
