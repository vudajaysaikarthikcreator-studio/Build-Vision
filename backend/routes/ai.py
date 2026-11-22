from fastapi import APIRouter, HTTPException

from ai.openai_service import analyze_house_requirements, openai_status
from backend.schemas import HouseRequirementIn
from backend.pipeline import effective_budget, validate_requirements
from user_requirements import normalize_requirements, RequirementError
from ml.predict import predict_planning_type, predict_planning_type_ensemble

router = APIRouter(prefix="/api/ai", tags=["OpenAI AI"])


@router.get("/status")
def status():
    return openai_status()


@router.post("/analyze")
def analyze(req: HouseRequirementIn):
    try:
        req, _report = normalize_requirements(req)
    except RequirementError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rooms = validate_requirements(req)
    area = req.plot_length * req.plot_width
    family_size = req.family_size or (req.bedrooms + 1)
    budget = effective_budget(req)
    primary, _ = predict_planning_type(
        plot_area=area,
        plot_length=req.plot_length,
        plot_width=req.plot_width,
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        floors=req.floors,
        budget=budget,
        parking_required=req.parking,
        family_size=family_size,
    )
    challenger, _ = predict_planning_type_ensemble(
        plot_area=area,
        plot_length=req.plot_length,
        plot_width=req.plot_width,
        bedrooms=req.bedrooms,
        bathrooms=req.bathrooms,
        floors=req.floors,
        budget=budget,
        parking_required=req.parking,
        family_size=family_size,
    )
    try:
        return analyze_house_requirements(
            plot_length=req.plot_length,
            plot_width=req.plot_width,
            bedrooms=req.bedrooms,
            bathrooms=req.bathrooms,
            kitchens=req.kitchens,
            floors=req.floors,
            parking=req.parking,
            vastu_enabled=req.vastu_preference,
            entrance=req.main_entrance_direction,
            road_direction=req.road_direction,
            budget=budget,
            planning_type=primary,
            rooms=rooms,
        ) | {
            "ml": {"primary": primary, "challenger": challenger},
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
