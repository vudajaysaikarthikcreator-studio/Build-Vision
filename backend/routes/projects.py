"""
backend/routes/projects.py

Implements the AI-pipeline endpoints from the brief's section 17:

    POST /api/analyze-requirements
    POST /api/vastu-arrangement
    POST /api/generate-layout
    POST /api/generate-blueprint
    POST /api/generate-3d
    POST /api/cost-estimation
    POST /api/projects/generate     (runs the whole pipeline in one call
                                      and persists it — this is what the
                                      House Planner's "Generate House Plan"
                                      button should call)
    GET  /api/blueprint/{id}
    GET  /api/report/{id}
    GET  /api/market-reference       (real ₹/sq.ft resale/rental benchmarks —
                                       see data/build_market_reference.py —
                                       shown for comparison only; it never
                                       feeds the cost-estimation model)
"""

import csv
import json
import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.schemas import HouseRequirementIn, GeneratePlanOut
from backend.database import get_db
from backend import models
from backend.pipeline import (run_pipeline, validate_requirements, check_feasibility, RATE_TABLE, effective_budget,
                              _floor_layouts, build_assignments, InfeasibleError)
from layout.safe_layout import LayoutInfeasible
from user_requirements import normalize_requirements, RequirementError
from vastu.vastu_engine import audit_layout, entrance_advice
from ml.predict import (
    predict_planning_type, predict_planning_type_ensemble,
    predict_cost, predict_cost_ensemble,
)
from vastu.vastu_arrangement import assign_rooms
from vastu.vastu_rules import room_color
from layout.room_allocator import allocate
from layout.layout_generator import generate_layout
from layout.safe_layout import safe_generate_layout
from blueprint.blueprint_generator import (
    generate_floor_blueprints, FLOOR_TITLES,
    OUTPUT_DIR as BLUEPRINT_DIR,
)
from reports.report_generator import generate_report, generate_3d_visualization
from civil.structural_engineering import generate_structural_report, list_soil_types
from planning.building_programs import property_title

router = APIRouter(prefix="/api", tags=["pipeline"])


def _prep(req: HouseRequirementIn):
    """Normalise plot geometry / custom rooms; turn bad input into a clear HTTP 400."""
    try:
        return normalize_requirements(req)
    except RequirementError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/resolve-plot")
def resolve_plot_route(req: HouseRequirementIn):
    """Preview how lengths, angles and bearings resolve into a buildable block + Vastu entrance guidance."""
    _, report = _prep(req)
    return report


# ---------------------------------------------------------------- step 1-2
@router.post("/analyze-requirements")
def analyze_requirements(req: HouseRequirementIn):
    """
    Runs both the primary Decision Tree (matches the brief, fully
    inspectable) and the Gradient Boosting challenger (higher measured
    accuracy — see ml/train_requirement_ensemble.py) and returns both, so
    the frontend can show "AI recommendation" without hiding which model
    said what.
    """
    req, _report = _prep(req)
    room_list = validate_requirements(req)
    plot_area = req.plot_length * req.plot_width
    family_size = req.family_size or (req.bedrooms + 1)
    planning_type, proba = predict_planning_type(
        plot_area=plot_area, plot_length=req.plot_length, plot_width=req.plot_width,
        bedrooms=req.bedrooms, bathrooms=req.bathrooms, floors=req.floors,
        budget=effective_budget(req), parking_required=req.parking, family_size=family_size,
    )
    planning_type_challenger, proba_challenger = predict_planning_type_ensemble(
        plot_area=plot_area, plot_length=req.plot_length, plot_width=req.plot_width,
        bedrooms=req.bedrooms, bathrooms=req.bathrooms, floors=req.floors,
        budget=effective_budget(req), parking_required=req.parking, family_size=family_size,
    )
    return {
        "planning_type": planning_type,
        "confidence": proba,
        "model": "DecisionTreeClassifier (primary)",
        "planning_type_challenger": planning_type_challenger,
        "confidence_challenger": proba_challenger,
        "model_challenger": "GradientBoostingClassifier (challenger, ~99.6% test accuracy vs 98.3%)",
        "building_type": req.building_type,
        "required_room_count": len(room_list),
        "rooms": room_list,
        "feasibility": check_feasibility(req, room_list, build_assignments(req, room_list)),
        "requirements_report": _report,
    }


# ---------------------------------------------------------------- step 3
@router.post("/vastu-arrangement")
def vastu_arrangement(req: HouseRequirementIn):
    req, _report = _prep(req)
    room_list = validate_requirements(req)
    assignments = build_assignments(req, room_list)
    return {
        "vastu_enabled": req.vastu_preference,
        "entrance": entrance_advice(req.main_entrance_direction),
        "arrangement": [
            {"room_type": a.room_type, "label": a.label, "preferred_zone": a.preferred_zone,
             "zone": a.zone, "matched": a.matched}
            for a in assignments
        ],
    }


# ---------------------------------------------------------------- step 4-5
@router.post("/generate-layout")
def generate_layout_route(req: HouseRequirementIn):
    req,_report=_prep(req)
    room_list=validate_requirements(req)
    assignments=build_assignments(req,room_list)
    try:
        floor_layouts=_floor_layouts(req,assignments)
    except LayoutInfeasible as e:
        raise HTTPException(status_code=400, detail=str(e))
    ground=floor_layouts[0]
    return {**ground,"floors":req.floors,"floor_layouts":floor_layouts}


# ---------------------------------------------------------------- step 6
@router.post("/generate-blueprint")
def generate_blueprint_route(req: HouseRequirementIn, project_id: str="preview"):
    req,_report=_prep(req)
    room_list=validate_requirements(req)
    assignments=build_assignments(req,room_list)
    try:
        floor_layouts=_floor_layouts(req,assignments)
    except LayoutInfeasible as e:
        raise HTTPException(status_code=400, detail=str(e))
    floor_plans=generate_floor_blueprints(floor_layouts[0],req.main_entrance_direction,project_id,floors=req.floors,floor_layouts=floor_layouts)
    path=floor_plans[0]['path']
    return {'blueprint_url':f'/api/blueprint/{project_id}','file':os.path.basename(path),'floor_plans':[{'floor':fp['title'].title(),'image_url':f"/api/blueprint/{project_id}/floor/{fp['floor']}",'engine':fp.get('engine'),'warning':fp.get('warning')} for fp in floor_plans]}


# ---------------------------------------------------------------- step 6b (3D-ready data)
@router.post("/generate-3d")
def generate_3d_route(req: HouseRequirementIn):
    req,_report=_prep(req)
    room_list=validate_requirements(req)
    assignments=build_assignments(req,room_list)
    try:
        floor_layouts=_floor_layouts(req,assignments)
    except LayoutInfeasible as e:
        raise HTTPException(status_code=400, detail=str(e))
    from blueprint.model3d import build_floor_model
    models=[]
    for fi,fl in enumerate(floor_layouts):
        try:
            m=build_floor_model(fl, req.main_entrance_direction, is_ground_floor=(fi==0))
        except Exception:  # noqa: BLE001 - the viewer still works from room boxes
            m={'walls':[],'doors':[],'windows':[],'envelope':None}
        m['floor']=fi
        models.append(m)
    return {'plot':floor_layouts[0]['plot'],'floors':req.floors,
            'floor_height_ft':10.5,'wall_height_ft':9.5,
            'wall_width_ft':req.wall_width_ft,
            'appearance':req.appearance,
            'rooms':[r for fl in floor_layouts for r in fl['rooms']],
            'floor_models':models,
            'floor_layouts':floor_layouts}


# ---------------------------------------------------------------- step 7
@router.post("/cost-estimation")
def cost_estimation_route(req: HouseRequirementIn):
    req, _report = _prep(req)
    room_list = validate_requirements(req)
    plot_area = req.plot_length * req.plot_width
    assignments = build_assignments(req, room_list)
    allocations = allocate(assignments)
    try:
        layout = safe_generate_layout(req.plot_length, req.plot_width, allocations, floors=req.floors,
                                      road_direction=req.road_direction, building_type=req.building_type,
                                      custom_mode=bool(req.custom_mode), setback_ft=req.setback_ft, setbacks=req.setbacks,
                                      main_entrance_direction=req.main_entrance_direction,
                                      hall_shape=getattr(req, 'hall_shape', 'auto'))
    except LayoutInfeasible as e:
        raise HTTPException(status_code=400, detail=str(e))

    circulation_factor = 0.12 if plot_area < 1200 else 0.16 if plot_area < 2500 else 0.20
    room_area_sum = sum(r["w"] * r["h"] for r in layout["rooms"])
    builtup_area = room_area_sum * (1 + circulation_factor) * req.floors
    rate = RATE_TABLE.get(req.construction_category, RATE_TABLE["standard"])
    bath_factor = 1 + max(0, req.bathrooms - 2) * 0.015
    total = builtup_area * rate * bath_factor
    model_total = predict_cost(
        plot_area=plot_area, builtup_area=builtup_area, floors=req.floors,
        num_rooms=len(layout["rooms"]), num_bathrooms=req.bathrooms,
        category=req.construction_category,
    )
    model_total_challenger = predict_cost_ensemble(
        plot_area=plot_area, builtup_area=builtup_area, floors=req.floors,
        num_rooms=len(layout["rooms"]), num_bathrooms=req.bathrooms,
        category=req.construction_category,
    )

    return {
        "plot_area": round(plot_area, 1),
        "builtup_area": round(builtup_area, 1),
        "rate": rate,
        "material": round(total * 0.6, 0),
        "labour": round(total * 0.4, 0),
        "total": round(total, 0),
        "model_predicted_total": round(model_total, 0),
        "model": "RandomForestRegressor (primary)",
        "model_predicted_total_challenger": round(model_total_challenger, 0),
        "model_challenger": "GradientBoostingRegressor (challenger, R2 0.990 vs 0.981, lower MAE)",
        "disclaimer": "This is an approximate estimation and not a professional construction quotation." + (" Budget 0 means the backend auto-estimated a planning budget." if not req.budget else ""),
    }


def _persist_project(db: Session, pid: str, req: HouseRequirementIn, result: dict,
                     user_id, warnings: list) -> bool:
    """Write the generated plan to MySQL. Returns False (and records a
    warning) instead of raising if the database is unavailable."""
    try:
        db.add(models.HouseRequirement(
            plan_id=pid, user_id=user_id,
            plot_length=result["plot"]["length"], plot_width=result["plot"]["width"],
            bedrooms=req.bedrooms, bathrooms=req.bathrooms, kitchens=req.kitchens,
            floors=req.floors, budget=req.budget, road_direction=(result.get("requirements_report") or {}).get("plot",{}).get("road_direction") or req.road_direction,
            main_entrance_direction=result["entrance"],
            vastu_preference=req.vastu_preference, construction_category=req.construction_category,
            raw_json=req.model_dump_json(),
        ))
        db.add(models.Layout(
            plan_id=pid, plot_length=result["plot"]["length"], plot_width=result["plot"]["width"],
            floors=req.floors, planning_type=result["planning_type"],
            collisions=result["collisions"], boundary_violations=result["boundary_violations"],
            passed_checks=(result["collisions"] == 0 and result["boundary_violations"] == 0),
        ))
        for r in result["rooms"]:
            db.add(models.Room(
                plan_id=pid, room_type=r["room_type"], label=r["label"], zone=r["zone"],
                x=r["x"], y=r["y"], w=r["w"], h=r["h"], door_side=r["door_side"],
            ))
            db.add(models.VastuArrangement(
                plan_id=pid, room_type=r["room_type"], label=r["label"],
                preferred_zone=r.get("preferred_zone"), assigned_zone=r["zone"],
                matched=r.get("matched"),
            ))
        db.add(models.Blueprint(
            plan_id=pid, image_path=result["blueprint_path"], entrance_direction=req.main_entrance_direction,
        ))
        c = result["cost"]
        db.add(models.CostEstimation(
            plan_id=pid, plot_area=c["plot_area"], builtup_area=c["builtup_area"],
            rate_per_sqft=c["rate"], material_cost=c["material"], labour_cost=c["labour"],
            total_cost=c["total"], model_predicted_total=c["model_predicted_total"],
            model_predicted_total_challenger=c.get("model_predicted_total_challenger"),
        ))
        db.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        warnings.append(
            "The plan was generated but not saved to MySQL "
            f"({exc}). Check that MySQL is running in XAMPP — see /api/diagnostics."
        )
        return False


# ---------------------------------------------------------------- step 7b (civil engineering)
@router.get("/soil-types")
def soil_types():
    """Soil types the frontend can offer in a dropdown for the structural estimate."""
    return {"soil_types": list_soil_types()}


@router.post("/structural-estimate")
def structural_estimate_route(req: HouseRequirementIn):
    """
    Preliminary civil-engineering estimate: footing/column/beam/slab
    pre-sizing, material quantities, staircase compliance and
    setback/FAR building-bylaw compliance — see
    civil/structural_engineering.py. Can be called standalone (e.g. to
    let the user try different soil types) or read from the `structural`
    key already included in /api/projects/generate's response.
    """
    req, _report = _prep(req)
    room_list = validate_requirements(req)
    plot_area = req.plot_length * req.plot_width
    assignments = build_assignments(req, room_list)
    try:
        floor_layouts = _floor_layouts(req, assignments)
    except LayoutInfeasible as e:
        raise HTTPException(status_code=400, detail=str(e))
    all_rooms = [r for fl in floor_layouts for r in fl["rooms"]]

    circulation_factor = 0.12 if plot_area < 1200 else 0.16 if plot_area < 2500 else 0.20
    builtup_area = sum(r["w"] * r["h"] for fl in floor_layouts for r in fl["rooms"]) * (1 + circulation_factor)

    structural = generate_structural_report(
        plot_length=req.plot_length, plot_width=req.plot_width, floors=req.floors,
        builtup_area_sqft=builtup_area, category=req.construction_category,
        soil_type=req.soil_type, rooms=all_rooms, has_staircase=req.staircase,
    )
    return structural


# ---------------------------------------------------------------- full pipeline + persistence
@router.post("/projects/generate", response_model=GeneratePlanOut)
def generate_project(req: HouseRequirementIn, user_id: int = None, db: Session = Depends(get_db)):
    try:
        result = run_pipeline(req)
    except (RequirementError, InfeasibleError, LayoutInfeasible) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pid = result["project_id"]
    warnings = list(result.get("warnings") or [])

    # Persistence is best-effort: a MySQL hiccup must not throw away a
    # plan that has already been generated. The plan is returned either
    # way and the response says whether it was saved.
    saved = _persist_project(db, pid, req, result, user_id, warnings)

    floor_plans = [
        {"floor": fp["title"].title(),
         "image_url": f"/api/blueprint/{pid}/floor/{fp['floor']}",
         "engine": fp.get("engine"), "warning": fp.get("warning")}
        for fp in result.get("floor_plans", [])
    ]
    c = result["cost"]

    return {
        "project_id": pid,
        "name": result["name"],
        "planning_type": result["planning_type"],
        "planning_type_challenger": result.get("planning_type_challenger"),
        "category": result["category"],
        "building_type": result.get("building_type", req.building_type),
        "plot": result["plot"],
        "floors": result["floors"],
        "vastu_enabled": result["vastu_enabled"],
        "entrance": result["entrance"],
        "rooms": result["rooms"],
        "collisions": result["collisions"],
        "boundary_violations": result["boundary_violations"],
        "cost": c,
        "blueprint_url": f"/api/blueprint/{pid}",
        "floor_plans": floor_plans,
        "blueprint_engine": result.get("blueprint_engine"),
        "feasibility": result.get("feasibility"),
        "structural": result.get("structural"),
        "saved_to_database": saved,
        "warnings": warnings,
        "algorithm_trace": result.get("algorithm_trace", []),
        "vastu_audit": result.get("vastu_audit"),
        "requirements_report": result.get("requirements_report"),
        "vastu_match_count": result.get("vastu_match_count", 0),
        "vastu_acceptable_count": result.get("vastu_acceptable_count", 0),
        "vastu_score": result.get("vastu_score", 0.0),
        "vastu_match_percent": result.get("vastu_match_percent", 0.0),
    }


# ---------------------------------------------------------------- fetch artifacts
@router.get("/blueprint/{project_id}")
def get_blueprint(project_id: str):
    """Ground-floor blueprint image (unchanged path, kept for existing frontends)."""
    path = os.path.join(BLUEPRINT_DIR, f"{project_id}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Blueprint not found. Generate the project first.")
    return FileResponse(path, media_type="image/png")


_FLOOR_SLUGS = {0: "ground_floor", 1: "first_floor", 2: "second_floor",
                3: "third_floor", 4: "fourth_floor"}


@router.get("/blueprint/{project_id}/floor/{floor_index}")
def get_blueprint_floor(project_id: str, floor_index: int):
    """
    Additional per-floor blueprint image for multi-floor projects (brief
    section 8/12), e.g. GET /api/blueprint/proj_abc123/floor/1 for the
    First Floor. Falls back to the ground-floor image if a project was
    generated before this endpoint existed and only has the single file.
    """
    slug = _FLOOR_SLUGS.get(floor_index, f"floor{floor_index}")
    path = os.path.join(BLUEPRINT_DIR, f"{project_id}_{slug}.png")
    if not os.path.exists(path):
        fallback = os.path.join(BLUEPRINT_DIR, f"{project_id}.png")
        if floor_index == 0 and os.path.exists(fallback):
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail="Floor blueprint not found. Generate the project first.")
    return FileResponse(path, media_type="image/png")


@router.get("/report/{project_id}")
def get_report(project_id: str, db: Session = Depends(get_db)):
    layout_row = db.query(models.Layout).filter_by(plan_id=project_id).first()
    if not layout_row:
        raise HTTPException(status_code=404, detail="Project not found. Generate it first via /api/projects/generate.")

    rooms = db.query(models.Room).filter_by(plan_id=project_id).all()
    vastu_rows = db.query(models.VastuArrangement).filter_by(plan_id=project_id).all()
    vastu_by_room = {v.room_type: v for v in vastu_rows}
    cost_row = db.query(models.CostEstimation).filter_by(plan_id=project_id).first()
    bp_row = db.query(models.Blueprint).filter_by(plan_id=project_id).first()
    req_row = db.query(models.HouseRequirement).filter_by(plan_id=project_id).first()

    saved_req = {}
    if req_row and req_row.raw_json:
        try:
            saved_req = json.loads(req_row.raw_json)
        except Exception:
            saved_req = {}
    saved_type = saved_req.get("building_type", "custom")
    project = {
        "id": project_id,
        "name": f"{property_title(saved_type)} Plan",
        "building_type": saved_type,
        "plot": {"length": layout_row.plot_length, "width": layout_row.plot_width,
                 "area": layout_row.plot_length * layout_row.plot_width},
        "floors": layout_row.floors,
        "planning_type": layout_row.planning_type,
        "category": req_row.construction_category if req_row else "standard",
        "vastu_enabled": req_row.vastu_preference if req_row else True,
        "entrance": bp_row.entrance_direction if bp_row else "North",
        "rooms": [
            {"label": r.label, "room_type": r.room_type, "zone": r.zone,
             "preferred_zone": vastu_by_room.get(r.room_type).preferred_zone if vastu_by_room.get(r.room_type) else "-",
             "matched": vastu_by_room.get(r.room_type).matched if vastu_by_room.get(r.room_type) else None,
             "x": r.x, "y": r.y, "w": r.w, "h": r.h, "door_side": r.door_side or "S",
             "floor": int(getattr(r, "floor", saved_req.get("floor", 0) if isinstance(saved_req, dict) else 0) or 0)}
            for r in rooms
        ],
        "cost": {
            "builtup_area": cost_row.builtup_area, "rate": cost_row.rate_per_sqft,
            "material": cost_row.material_cost, "labour": cost_row.labour_cost,
            "total": cost_row.total_cost,
        } if cost_row else {},
    }

    # Recompute the preliminary structural/compliance estimate for the PDF.
    # It isn't persisted (like `feasibility`), so it's rebuilt from the
    # saved requirement + room rows; soil type falls back to the default
    # since it isn't stored in house_requirements yet.
    if cost_row:
        try:
            project["structural"] = generate_structural_report(
                plot_length=layout_row.plot_length, plot_width=layout_row.plot_width,
                floors=layout_row.floors, builtup_area_sqft=cost_row.builtup_area,
                category=project["category"], rooms=[{"w": r.w, "h": r.h} for r in rooms],
                has_staircase=any(r.room_type == "staircase" for r in rooms),
            )
        except Exception:  # noqa: BLE001
            project["structural"] = None

    image_path = bp_row.image_path if bp_row else None

    # Discover any additional per-floor images generated alongside the
    # ground floor (brief section 8) so the PDF report can include every
    # floor, not just the ground-floor image.
    floor_images = []
    for floor_idx, slug in _FLOOR_SLUGS.items():
        candidate = os.path.join(BLUEPRINT_DIR, f"{project_id}_{slug}.png")
        if os.path.exists(candidate):
            floor_images.append({"title": FLOOR_TITLES.get(floor_idx, slug.replace("_", " ").title()),
                                  "path": candidate})

    # Build the same floor-by-floor geometry used by the interactive Three.js
    # viewer, then render a static isometric PNG for the backend-generated PDF.
    three_d_path = None
    try:
        saved_req_model = HouseRequirementIn.model_validate(saved_req) if saved_req else None
        if saved_req_model is not None:
            assignments = assign_rooms(
                validate_requirements(saved_req_model),
                vastu_enabled=saved_req_model.vastu_preference,
                plot_area=saved_req_model.plot_length * saved_req_model.plot_width,
            )
            report_floor_layouts = _floor_layouts(saved_req_model, assignments)
            if report_floor_layouts:
                project["building_envelope"] = report_floor_layouts[0].get("building_envelope")
                project["main_door_position"] = report_floor_layouts[0].get(
                    "main_door_position", saved_req_model.main_door_position
                )
            else:
                project["main_door_position"] = saved_req_model.main_door_position
            project["entrance"] = saved_req_model.main_entrance_direction
            project["vastu_audit"] = audit_layout(
                report_floor_layouts,
                road_direction=saved_req_model.road_direction,
                door_position=project["main_door_position"],
                north_offset_deg=getattr(saved_req_model, "north_offset_deg", 0.0),
                vastu_enabled=bool(saved_req_model.vastu_preference),
                building_type=saved_req_model.building_type,
                entrance_direction=saved_req_model.main_entrance_direction,
            )
        else:
            report_floor_layouts = [{
                "floor": 0,
                "rooms": [
                    {**r, "floor": 0}
                    for r in project["rooms"]
                ],
                "plot": project["plot"],
            }]
        three_d_path = generate_3d_visualization(project, report_floor_layouts)
    except Exception as exc:  # noqa: BLE001
        # A PDF should still be downloadable even if static 3D rendering is
        # unavailable on the server. The frontend Export as PDF path captures
        # the live Three.js canvas directly.
        print(f"[BuildVision] Static 3D PDF visualization skipped: {exc}")

    pdf_path = generate_report(
        project, image_path, floor_images=floor_images,
        three_d_image_path=three_d_path,
    )
    return FileResponse(pdf_path, media_type="application/pdf",
                         filename=os.path.basename(pdf_path))


# ---------------------------------------------------------------- market reference (real data)
MARKET_REFERENCE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "market_reference.csv"
)


@router.get("/market-reference")
def market_reference():
    """
    Real ₹/sq.ft resale & rental benchmarks aggregated from the user-supplied
    Pune/Bangalore listings and the monthly locality price index — see
    data/build_market_reference.py for how it's built and why it's kept
    separate from the cost-estimation model's training data.

    Returns 404 with instructions if the reference hasn't been built yet
    (run `python data/build_market_reference.py` from the project root).
    """
    if not os.path.exists(MARKET_REFERENCE_CSV):
        raise HTTPException(
            status_code=404,
            detail="Market reference not built yet. Run: python data/build_market_reference.py",
        )
    with open(MARKET_REFERENCE_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {"disclaimer": "Real resale/rental market benchmarks, for comparison only — "
                           "not used by the cost-estimation model.", "rows": rows}
