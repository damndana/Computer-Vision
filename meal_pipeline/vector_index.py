"""
FAISS vector index for CLIP embeddings (cosine via normalized inner product).

Build with embedding_generator; load at API startup for sub-ms search on CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """In-place L2 normalize each row (float32)."""
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    faiss.normalize_L2(x)
    return x


def build_ip_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Inner product on L2-normalized vectors == cosine similarity."""
    x = l2_normalize_rows(vectors)
    dim = x.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x)
    logger.info("FAISS IndexFlatIP: %s vectors, dim=%s", x.shape[0], dim)
    return index


def save_index(
    index: faiss.IndexFlatIP, meal_ids: np.ndarray, index_path: Path, ids_path: Path
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(str(ids_path), np.asarray(meal_ids, dtype=np.int64))
    logger.info("Saved FAISS index to %s, ids to %s", index_path, ids_path)


def load_index(
    index_path: Path, ids_path: Path
) -> Tuple[Optional[faiss.Index], Optional[np.ndarray]]:
    if not index_path.is_file() or not ids_path.is_file():
        logger.warning("FAISS index or id file missing: %s %s", index_path, ids_path)
        return None, None
    index = faiss.read_index(str(index_path))
    meal_ids = np.load(str(ids_path))
    logger.info("Loaded FAISS index: %s vectors", index.ntotal)
    return index, meal_ids


def search(
    index: faiss.Index,
    query_vectors: np.ndarray,
    meal_ids: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (scores, meal_id_rows) each shape (n_queries, k).
    query_vectors must be float32; will be L2-normalized.
    """
    q = l2_normalize_rows(query_vectors)
    k = min(int(k), index.ntotal)
    if k <= 0:
        return np.zeros((q.shape[0], 0), np.float32), np.zeros((q.shape[0], 0), np.int64)
    scores, indices = index.search(q, k)
    mapped = meal_ids[indices]
    return scores, mapped
