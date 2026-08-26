from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "data": coach.data_status(), "model": coach.model_status()}


@app.post("/api/coach/recommend")
def recommend(request: CoachRequest) -> dict:
    return coach.recommend(request.level, request.ownedChampionIds, request.components)


@app.post("/api/admin/reload")
def reload_runtime() -> dict:
    coach.reload()
    return {"ok": True, "data": coach.data_status(), "model": coach.model_status()}

