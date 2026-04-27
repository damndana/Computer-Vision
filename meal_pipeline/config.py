"""Central configuration for the meal recognition pipeline (env-driven)."""
from __future__ import annotations

import os
import pathlib

# Paths: optional MEAL_PIPELINE_DATA_DIR (Dokploy/Docker often sets /app/data).
_REPO = pathlib.Path(__file__).resolve().parents[1]
_DATA = pathlib.Path(
    os.environ.get(
        "MEAL_PIPELINE_DATA_DIR",
        str(_REPO / "data"),
    )
)

CLIP_MODEL_NAME = os.environ.get(
    "CLIP_MODEL_NAME", "sentence-transformers/clip-ViT-B-32"
)
FAISS_INDEX_PATH = pathlib.Path(
    os.environ.get("FAISS_INDEX_PATH", str(_DATA / "meals_faiss.index"))
)
MEAL_IDS_PATH = pathlib.Path(
    os.environ.get("MEAL_IDS_PATH", str(_DATA / "meal_faiss_ids.npy"))
)
# gemini-2.0-flash may be unavailable for new users; default to a newer Flash model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "20"))

# Reuse Nutristeppe table name from main app
MEAL_TABLE = os.environ.get("MEAL_INFO_TABLE") or os.environ.get(
    "NUTRISTEPPE_TABLE", "database_nutristeppe"
)
