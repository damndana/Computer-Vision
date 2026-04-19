"""
FastAPI inference: POST /analyze_meal

Run locally:
  export DATABASE_URL=...
  export GOOGLE_GEMINI_API_KEY=...
  python -m meal_pipeline.embedding_generator   # first-time index build
  uvicorn meal_pipeline.api:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from meal_pipeline.meal_retriever import MealRetriever
from meal_pipeline.orchestrator import analyze_meal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_retriever: MealRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    logger.info("Loading CLIP + FAISS retriever (first request may download CLIP weights)…")
    r = MealRetriever()
    r.ensure_loaded()
    _retriever = r
    logger.info("Meal retriever ready.")
    yield
    _retriever = None


app = FastAPI(title="Nutristeppe Meal Pipeline", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "retriever": "loaded" if _retriever else "not_ready"}


@app.post("/analyze_meal")
async def analyze_meal_endpoint(
    image: UploadFile = File(..., description="JPEG/PNG image of the meal"),
    portion_grams: float = Form(..., description="Total grams (split across dishes if multi-plate)"),
) -> JSONResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    if portion_grams <= 0 or portion_grams > 20000:
        raise HTTPException(status_code=400, detail="portion_grams must be in (0, 20000]")

    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    try:
        result = analyze_meal(img, float(portion_grams), _retriever)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("analyze_meal failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Public response: match spec (meal_name + macros + portion + confidence)
    public = {
        "dishes": [
            {
                "meal_name": d["meal_name"],
                "portion_grams": d["portion_grams"],
                "calories": d["calories"],
                "protein": d["protein"],
                "fat": d["fat"],
                "carbs": d["carbs"],
                "confidence": d["confidence"],
            }
            for d in result.get("dishes", [])
        ]
    }
    return JSONResponse(public)
