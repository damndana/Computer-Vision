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
import os
import threading
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
_retriever_lock = threading.Lock()


def _ensure_retriever_loaded() -> None:
    """
    Lazy-load CLIP+FAISS. This avoids slow startup (and deploy-time 502)
    by deferring heavy model loading until first request.
    """
    global _retriever
    if _retriever is None:
        _retriever = MealRetriever()
    # Ensure only one concurrent load
    with _retriever_lock:
        assert _retriever is not None
        _retriever.ensure_loaded()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    _retriever = MealRetriever()
    # Optional eager load (slower startup, but first request faster)
    if os.environ.get("MEAL_PIPELINE_EAGER_LOAD", "0") == "1":
        logger.info("Eager-loading CLIP + FAISS retriever at startup…")
        _ensure_retriever_loaded()
        logger.info("Meal retriever ready.")
    else:
        logger.info("Lazy-loading enabled (CLIP+FAISS loads on first request).")
    yield
    _retriever = None


app = FastAPI(title="Nutristeppe Meal Pipeline", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, str]:
    loaded = bool(getattr(_retriever, "_model", None)) and bool(getattr(_retriever, "_index", None))
    return {"status": "ok", "retriever": "loaded" if loaded else "not_ready"}


@app.post("/analyze_meal")
async def analyze_meal_endpoint(
    image: UploadFile = File(..., description="JPEG/PNG image of the meal"),
    portion_grams: float = Form(..., description="Total grams (split across dishes if multi-plate)"),
) -> JSONResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Service initializing")
    if portion_grams <= 0 or portion_grams > 20000:
        raise HTTPException(status_code=400, detail="portion_grams must be in (0, 20000]")

    raw = await image.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    try:
        _ensure_retriever_loaded()
        assert _retriever is not None
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
