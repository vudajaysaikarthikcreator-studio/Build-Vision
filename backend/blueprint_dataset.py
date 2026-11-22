"""
Access layer for the blueprint image dataset in datasets/blueprints/ (see its README.md).

Pure Python (no FastAPI, no torch) so it works in the API, in notebooks and in training scripts:

    from backend.blueprint_dataset import list_images, similar, BlueprintDataset

    similar("3bhk", plot_length=30, plot_width=40, bedrooms=3)       # nearest reference blueprints
    ds = BlueprintDataset(building_type="school")                    # PIL image + labels per item
    img, ann = ds[0]
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterator, List, Optional, Tuple

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "blueprints"))
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
INDEX_PATH = os.path.join(DATASET_DIR, "index.json")
IMAGE_URL_PREFIX = "/static/blueprint-dataset/"        # mounted in backend/main.py

_cache: Dict[str, object] = {"mtime": None, "data": None}


def load_index() -> dict:
    """index.json, re-read automatically when build_index.py rewrites it."""
    if not os.path.exists(INDEX_PATH):
        return {"images": [], "summary": {"images": 0}, "categories": [], "missing": True}
    mtime = os.path.getmtime(INDEX_PATH)
    if _cache["data"] is None or _cache["mtime"] != mtime:
        with open(INDEX_PATH, encoding="utf-8") as f:
            _cache["data"] = json.load(f)
        _cache["mtime"] = mtime
    return _cache["data"]            # type: ignore[return-value]


def _public(entry: dict, with_rooms: bool = False) -> dict:
    out = {k: v for k, v in entry.items() if k != "rooms"}
    out["image_url"] = IMAGE_URL_PREFIX + entry["file"]
    if with_rooms:
        out["rooms"] = entry.get("rooms", [])
    return out


def list_images(building_type: Optional[str] = None, floors: Optional[int] = None, floor: Optional[int] = None,
                bedrooms: Optional[int] = None, source: Optional[str] = None, road_direction: Optional[str] = None,
                min_area: Optional[float] = None, max_area: Optional[float] = None, q: Optional[str] = None,
                limit: int = 48, offset: int = 0) -> dict:
    items = load_index().get("images", [])
    bt = building_type.strip().lower() if building_type else None
    res = []
    for e in items:
        if bt and e["building_type"] != bt:
            continue
        if floors is not None and e.get("floors") != floors:
            continue
        if floor is not None and e.get("floor") != floor:
            continue
        if bedrooms is not None and e.get("bedrooms") != bedrooms:
            continue
        if source and e.get("source") != source:
            continue
        if road_direction and (e.get("road_direction") or "").lower() != road_direction.lower():
            continue
        area = e.get("plot_area_sqft")
        if min_area is not None and (area is None or area < min_area):
            continue
        if max_area is not None and (area is None or area > max_area):
            continue
        if q and q.lower() not in (e["image_id"] + " " + e["building_type"]).lower():
            continue
        res.append(e)
    total = len(res)
    page = res[max(offset, 0): max(offset, 0) + max(limit, 1)]
    return {"total": total, "offset": offset, "limit": limit, "items": [_public(e) for e in page]}


def get_image(image_id: str) -> Optional[dict]:
    for e in load_index().get("images", []):
        if e["image_id"] == image_id:
            return _public(e, with_rooms=True)
    return None


def stats() -> dict:
    idx = load_index()
    out = dict(idx.get("summary", {}))
    out.update({"name": idx.get("name"), "version": idx.get("version"), "generated": idx.get("generated"),
                "categories": idx.get("categories", []), "description": idx.get("description")})
    return out


def _log_ratio(a: Optional[float], b: Optional[float]) -> float:
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return abs(math.log(a / b))


def similar(building_type: Optional[str] = None, plot_length: Optional[float] = None,
            plot_width: Optional[float] = None, bedrooms: Optional[int] = None, floors: Optional[int] = None,
            road_direction: Optional[str] = None, floor: Optional[int] = 0, top_k: int = 6) -> List[dict]:
    """Reference blueprints closest to a requirement (same type first, then plot size / bedrooms / floors)."""
    items = load_index().get("images", [])
    bt = building_type.strip().lower() if building_type else None
    pool = [e for e in items if (not bt or e["building_type"] == bt) and (floor is None or e.get("floor") == floor)]
    if not pool:                                   # unknown type: fall back to any type
        pool = [e for e in items if floor is None or e.get("floor") == floor]
    area = plot_length * plot_width if plot_length and plot_width else None
    ratio = plot_length / plot_width if plot_length and plot_width else None
    scored: List[Tuple[float, dict]] = []
    for e in pool:
        d = 0.0
        if area:
            d += 2.0 * _log_ratio(area, e.get("plot_area_sqft"))
        if ratio and e.get("plot_length_ft") and e.get("plot_width_ft"):
            d += 0.7 * _log_ratio(ratio, e["plot_length_ft"] / e["plot_width_ft"])
        if bedrooms is not None and e.get("bedrooms") is not None:
            d += 0.5 * abs(bedrooms - e["bedrooms"])
        if floors is not None and e.get("floors") is not None:
            d += 1.0 * abs(floors - e["floors"])
        if road_direction and e.get("road_direction") and e["road_direction"].lower() != road_direction.lower():
            d += 0.15
        scored.append((d, e))
    scored.sort(key=lambda t: (t[0], t[1]["image_id"]))
    out = []
    for d, e in scored[: max(top_k, 1)]:
        item = _public(e)
        item["distance"] = round(d, 3)
        out.append(item)
    return out


class BlueprintDataset:
    """Iterable dataset: item -> (PIL.Image RGB, annotation dict). Works as a torch Dataset (len + getitem)."""

    def __init__(self, building_type: Optional[str] = None, source: Optional[str] = None,
                 floor: Optional[int] = None, transform=None):
        self.entries = [e for e in load_index().get("images", [])
                        if (not building_type or e["building_type"] == building_type.lower())
                        and (not source or e.get("source") == source)
                        and (floor is None or e.get("floor") == floor)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        from PIL import Image
        e = self.entries[i]
        img = Image.open(os.path.join(IMAGES_DIR, e["file"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, e

    def __iter__(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]
