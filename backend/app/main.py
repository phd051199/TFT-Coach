from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import load_catalog
from .optimizer import HybridCoach
from .schemas import CoachRequest


app = FastAPI(title="HexCoach TFT Set 18 API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

catalog = load_catalog()
coach = HybridCoach(catalog)
ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "dist"

if (DIST_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data": coach.data_status(), "model": coach.model_status()}


@app.post("/api/coach/recommend")
def recommend(request: CoachRequest) -> dict:
    return coach.recommend(
        request.level,
        request.ownedChampionIds,
        request.components,
        request.targetCompId,
        previous_level=request.previousLevel,
        previous_comp_id=request.previousCompId,
        previous_owned_ids=request.previousOwnedChampionIds,
        previous_components=request.previousComponents,
        previous_item_plan=[row.model_dump() for row in request.previousItemPlan],
    )


@app.post("/api/admin/reload")
def reload_runtime() -> dict:
    coach.reload()
    return {"ok": True, "data": coach.data_status(), "model": coach.model_status()}


@app.get("/")
def frontend_index() -> FileResponse:
    index = DIST_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="Frontend build missing. Run `npm run build`.")
    return FileResponse(index)


@app.get("/{path:path}")
def frontend_spa(path: str) -> FileResponse:
    # API paths must never silently fall through to the SPA. This keeps bad API URLs visible
    # as real 404s instead of returning index.html with HTTP 200.
    if path == "api" or path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    candidate = (DIST_DIR / path).resolve()
    try:
        candidate.relative_to(DIST_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found") from exc
    if candidate.is_file():
        return FileResponse(candidate)
    index = DIST_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="Frontend build missing. Run `npm run build`.")
    return FileResponse(index)

