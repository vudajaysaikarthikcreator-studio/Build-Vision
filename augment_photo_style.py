"""
Make PHOTO-LIKE copies of the rendered blueprints (for training models that must read photographed or
scanned plans).  These are simulated: paper texture, uneven light, slight rotation/perspective, blur,
JPEG compression - or the classic white-on-blue "cyanotype" look.  They are NOT real photographs.

    python datasets/blueprints/augment_photo_style.py                    # one sample per building type
    python datasets/blueprints/augment_photo_style.py --per-type 3       # more samples
    python datasets/blueprints/augment_photo_style.py --all              # every image (large!)

Output: images_photo_style/<building_type>/<image_id>_<style>.jpg  +  photo_style_manifest.csv
The room boxes of the source image are NOT valid after rotation/perspective; use these for
classification / retrieval / OCR-style training, or re-detect boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import zlib

import numpy as np
from PIL import Image, ImageFilter, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images_photo_style")


def _perspective_coeffs(src, dst):
    rows, vec = [], []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); vec.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); vec.append(v)
    return np.linalg.solve(np.array(rows, dtype=float), np.array(vec, dtype=float))


def photo_style(img: Image.Image, style: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if style == "cyanotype":
        g = ImageOps.invert(img.convert("L"))
        img = ImageOps.colorize(g, black=(12, 50, 128), white=(232, 242, 255))
    # paper / print texture + uneven light
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    grain = nrng.normal(0, 5.0, (h, w, 1)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.3, 0.6)
    light = 1.0 - 0.28 * (((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)[..., None]
    tint = np.array([1.0, 0.985, 0.94], dtype=np.float32) if style == "paper" else np.array([0.97, 1.0, 1.0], dtype=np.float32)
    arr = np.clip(arr * light * tint + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    # camera pose: small perspective + rotation on a darker table background
    j = 0.035
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(rng.uniform(0, j) * w, rng.uniform(0, j) * h), (w - rng.uniform(0, j) * w, rng.uniform(0, j) * h),
           (w - rng.uniform(0, j) * w, h - rng.uniform(0, j) * h), (rng.uniform(0, j) * w, h - rng.uniform(0, j) * h)]
    img = img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(dst, src), Image.BICUBIC, fillcolor=(58, 52, 46))
    img = img.rotate(rng.uniform(-3.0, 3.0), resample=Image.BICUBIC, expand=False, fillcolor=(58, 52, 46))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    scale = rng.uniform(0.75, 0.95)                     # phone-camera softness
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    with open(os.path.join(HERE, "index.json"), encoding="utf-8") as f:
        images = [e for e in json.load(f)["images"] if e["source"] == "synthetic"]
    chosen, per = [], {}
    for e in images:
        k = per.get(e["building_type"], 0)
        if a.all or (k < a.per_type and e["floor"] == 0 and e["image_id"].endswith("_f0")):
            chosen.append(e)
            per[e["building_type"]] = k + 1
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for n, e in enumerate(chosen):
        rng = random.Random(zlib.crc32(e["image_id"].encode()))
        style = "cyanotype" if n % 2 else "paper"
        src = os.path.join(HERE, "images", e["file"])
        out_dir = os.path.join(OUT, e["building_type"])
        os.makedirs(out_dir, exist_ok=True)
        name = f"{e['image_id']}_{style}.jpg"
        with Image.open(src) as im:
            photo_style(im, style, rng).save(os.path.join(out_dir, name), "JPEG", quality=80, optimize=True)
        rows.append([f"images_photo_style/{e['building_type']}/{name}", e["image_id"], e["building_type"], style])
    with open(os.path.join(HERE, "photo_style_manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source_image_id", "building_type", "style"])
        w.writerows(rows)
    print(f"{len(rows)} photo-style images written to {OUT}")


if __name__ == "__main__":
    main()
