"""
/api/blueprint-dataset - browse the blueprint image dataset (datasets/blueprints) and find reference
blueprints similar to a requirement. Images are served from /static/blueprint-dataset/<file>.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend import blueprint_dataset as ds

router = APIRouter(prefix="/api/blueprint-dataset", tags=["blueprint-dataset"])


@router.get("")
def list_dataset(building_type: Optional[str] = None, floors: Optional[int] = None, floor: Optional[int] = None,
                 bedrooms: Optional[int] = None, source: Optional[str] = Query(None, description="synthetic | real"),
                 road_direction: Optional[str] = None, min_area: Optional[float] = None,
                 max_area: Optional[float] = None, q: Optional[str] = None,
                 limit: int = Query(48, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Filtered, paged list of dataset images (without room boxes)."""
    return ds.list_images(building_type=building_type, floors=floors, floor=floor, bedrooms=bedrooms, source=source,
                          road_direction=road_direction, min_area=min_area, max_area=max_area, q=q,
                          limit=limit, offset=offset)


@router.get("/stats")
def dataset_stats():
    return ds.stats()


@router.get("/similar")
def similar_blueprints(building_type: Optional[str] = None, plot_length: Optional[float] = None,
                       plot_width: Optional[float] = None, bedrooms: Optional[int] = None,
                       floors: Optional[int] = None, road_direction: Optional[str] = None,
                       floor: Optional[int] = 0, top_k: int = Query(6, ge=1, le=24)):
    """Reference blueprints closest to the given requirement."""
    return {"items": ds.similar(building_type, plot_length, plot_width, bedrooms, floors, road_direction,
                                floor, top_k)}


@router.get("/{image_id}")
def dataset_image(image_id: str):
    """One image with all its room annotations (feet and pixels)."""
    item = ds.get_image(image_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset image '{image_id}'")
    return item
