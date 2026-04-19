"""
Image or text → CLIP embedding → FAISS top-k meal IDs.

Loads model + index lazily (heavy); reuse one instance in FastAPI lifespan.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from meal_pipeline import config
from meal_pipeline.vector_index import load_index, l2_normalize_rows, search

logger = logging.getLogger(__name__)


class MealRetriever:
    def __init__(self) -> None:
        self._model = None
        self._index = None
        self._meal_ids: Optional[np.ndarray] = None

    def ensure_loaded(self) -> None:
        if self._model is not None and self._index is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._index, self._meal_ids = load_index(
            config.FAISS_INDEX_PATH, config.MEAL_IDS_PATH
        )
        if self._index is None or self._meal_ids is None:
            raise FileNotFoundError(
                f"FAISS index not found. Run: python -m meal_pipeline.embedding_generator "
                f"(expected {config.FAISS_INDEX_PATH})"
            )
        logger.info("Loading CLIP model: %s", config.CLIP_MODEL_NAME)
        self._model = SentenceTransformer(
            config.CLIP_MODEL_NAME, device=os.environ.get("CLIP_DEVICE", "cpu")
        )

    @property
    def model(self):
        self.ensure_loaded()
        assert self._model is not None
        return self._model

    def retrieve_for_image(self, image: Image.Image, k: int) -> List[Tuple[int, float]]:
        """CLIP image encoder → top-k (meal_id, similarity score)."""
        self.ensure_loaded()
        assert self._index is not None and self._meal_ids is not None
        img = image.convert("RGB")
        v = self._model.encode(
            [img],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        v = np.asarray(v, dtype=np.float32)
        scores, ids = search(self._index, v, self._meal_ids, k)
        out: List[Tuple[int, float]] = []
        for j in range(ids.shape[1]):
            out.append((int(ids[0, j]), float(scores[0, j])))
        return out

    def retrieve_for_text(self, text: str, k: int) -> List[Tuple[int, float]]:
        """CLIP text encoder (e.g. dish description from multi-meal detector)."""
        self.ensure_loaded()
        assert self._index is not None and self._meal_ids is not None
        v = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        v = np.asarray(v, dtype=np.float32)
        scores, ids = search(self._index, v, self._meal_ids, k)
        out: List[Tuple[int, float]] = []
        for j in range(ids.shape[1]):
            out.append((int(ids[0, j]), float(scores[0, j])))
        return out
