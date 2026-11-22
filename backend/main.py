"""
backend/main.py

FastAPI application entrypoint. Wires up:
  - CORS (so the existing static index.html, served by Apache/XAMPP on a
    different port, can call this API from the browser)
  - the pipeline router (analyze / vastu / layout / blueprint / 3d / cost
    / full-generate / blueprint & report retrieval)
  - the optional auth router
  - a static mount so generated blueprint PNGs are also reachable directly
  - table creation for the new pipeline tables on startup (additive only
    — never touches the existing `users` / `plans` tables)

Run:
    uvicorn backend.main:app --reload --port 8000

from the project root (with the venv from requirements.txt active).
"""

import os
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import engine, Base, DATABASE_SAFE_URL, initialize_database
from backend import models  # noqa: F401  (registers models on Base before create_all)
from backend.routes import projects, auth, ai, blueprint_dataset
from backend.blueprint_dataset import IMAGES_DIR as BLUEPRINT_DATASET_IMAGES

app = FastAPI(
    title="BuildVision AI Engine",
    description="Decision-tree requirement analysis, rule-based Vastu arrangement, "
                "constraint-based layout generation, 2D blueprint + 3D data, and "
                "random-forest cost estimation — the AI/ML backend for the "
                "BuildVision house planner.",
    version="1.0.0",
)

# NOTE: "*" together with allow_credentials=True is rejected by browsers.
# The app doesn't use cookies across origins, so credentials stay off and
# any origin (http://localhost, http://127.0.0.1, file://, ...) can call
# the API from XAMPP.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(projects.router)
app.include_router(auth.router)
app.include_router(ai.router)
app.include_router(blueprint_dataset.router)

BLUEPRINT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "blueprint", "output")
os.makedirs(BLUEPRINT_OUTPUT, exist_ok=True)
app.mount("/static/blueprints", StaticFiles(directory=BLUEPRINT_OUTPUT), name="blueprints")

# Blueprint image dataset (datasets/blueprints/images) for the gallery and /api/blueprint-dataset
os.makedirs(BLUEPRINT_DATASET_IMAGES, exist_ok=True)
app.mount("/static/blueprint-dataset", StaticFiles(directory=BLUEPRINT_DATASET_IMAGES), name="blueprint-dataset")


DB_READY = {"ok": False, "error": None}


def _warm_ml_models():
    """Train the decision-tree/random-forest/gradient-boosting models once,
    in a background thread, right after the server comes up.

    ml/predict.py lazily trains these from the bundled CSVs on first use and
    then caches them (functools.lru_cache) for the life of the process. With
    the expanded datasets that first training pass takes several seconds; if
    it happens on the user's first "Generate" click instead of at startup,
    it reads to the browser as exactly the kind of slow/hanging request that
    used to surface as "Generation failed. Failed to fetch." Warming them
    here means the first real request never pays that cost.
    """
    try:
        from ml.predict import _planning_models, _cost_models
        _planning_models()
        _cost_models()
        print("[BuildVision] ML models warmed up and cached.")
    except Exception as exc:  # noqa: BLE001
        # Never let a warm-up problem block startup or plan generation;
        # ml/predict.py trains lazily (and falls back to formulas) anyway.
        print("[BuildVision] ML warm-up skipped:", exc)


@app.on_event("startup")
def on_startup():
    import threading
    threading.Thread(target=_warm_ml_models, daemon=True).start()

    # Creates ONLY the new tables this service owns (house_requirements,
    # rooms, layouts, blueprints, vastu_arrangements, cost_estimations) if
    # they don't already exist. It never touches `users` or `plans` —
    # those stay owned by database/schema.sql / the PHP app.
    #
    # If MySQL isn't running yet this must NOT kill the server: without
    # this guard uvicorn exits, the browser gets ERR_CONNECTION_REFUSED
    # and the UI just says "Failed to fetch". Plan generation itself
    # doesn't need the database, so we start anyway and report the
    # problem through /health and /api/diagnostics.
    try:
        initialize_database()
        DB_READY["ok"] = True
        DB_READY["error"] = None
    except Exception as exc:  # noqa: BLE001
        DB_READY["ok"] = False
        DB_READY["error"] = str(exc)
        print("[BuildVision] WARNING: database not available -", exc)
        print("[BuildVision] The AI engine will still generate plans; "
              "saving to MySQL is skipped until the database is reachable.")


@app.get("/")
def root():
    return {
        "service": "BuildVision AI Engine",
        "docs": "/docs",
        "pipeline_endpoint": "POST /api/projects/generate",
    }


@app.get("/health")
def health():
    return {"status": "ok", "database": "ok" if DB_READY["ok"] else "unavailable"}


@app.get("/api/diagnostics")
def diagnostics():
    """One call that tells you exactly what is and isn't configured —
    open http://127.0.0.1:8000/api/diagnostics in the browser."""
    from blueprint.blueprint_generator import gemini_status
    from ai.openai_service import openai_status

    db = {"connected": DB_READY["ok"], "error": DB_READY["error"], "url": DATABASE_SAFE_URL}
    if not DB_READY["ok"]:
        try:
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            DB_READY["ok"], DB_READY["error"] = True, None
            db = {"connected": True, "error": None, "url": DATABASE_SAFE_URL}
        except Exception as exc:  # noqa: BLE001
            db["error"] = str(exc)
    return {"status": "ok", "openai": openai_status(), "gemini": gemini_status(), "ml": {"available": True, "models": ["DecisionTreeClassifier", "GradientBoostingClassifier", "RandomForestRegressor", "GradientBoostingRegressor"]}, "database": db}
