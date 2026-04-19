"""
Offline job: encode all meals with CLIP (text) and build FAISS index + id map.

Run once (or after DB updates):
  python -m meal_pipeline.embedding_generator

Optional: persist vectors to PostgreSQL table meal_embeddings (BYTEA).
"""
from __future__ import annotations

import argparse
import logging
import os
import struct
from typing import List, Tuple

import numpy as np
import psycopg2
from tqdm import tqdm

from meal_pipeline import config
from meal_pipeline.db_meals import load_all_meals_df, meal_table_name, rich_text_for_meal
from meal_pipeline.vector_index import build_ip_index, l2_normalize_rows, save_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_embeddings_table(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meal_embeddings (
            meal_id BIGINT PRIMARY KEY,
            embedding_dim INTEGER NOT NULL,
            embedding_vector BYTEA NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
    conn.commit()


def _write_pg_embeddings(
    conn, meal_ids: List[int], vectors: np.ndarray
) -> None:
    """Store float32 vectors as raw bytes (optional)."""
    _ensure_embeddings_table(conn)
    dim = vectors.shape[1]
    cur = conn.cursor()
    for mid, row in zip(meal_ids, vectors):
        buf = struct.pack(f"{dim}f", *row.astype(np.float32).tolist())
        cur.execute(
            """
            INSERT INTO meal_embeddings (meal_id, embedding_dim, embedding_vector)
            VALUES (%s, %s, %s)
            ON CONFLICT (meal_id) DO UPDATE SET
                embedding_dim = EXCLUDED.embedding_dim,
                embedding_vector = EXCLUDED.embedding_vector,
                updated_at = NOW();
            """,
            (int(mid), int(dim), psycopg2.Binary(buf)),
        )
    conn.commit()
    logger.info("Upserted %s rows into meal_embeddings", len(meal_ids))


def generate_embeddings(
    batch_size: int = 64,
    write_pg: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    df = load_all_meals_df()
    if df.empty:
        raise RuntimeError("No meals in database — nothing to embed.")

    if "id" not in df.columns:
        raise RuntimeError("Meal table must have an 'id' column (BIGSERIAL).")

    texts: List[str] = []
    ids: List[int] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        texts.append(rich_text_for_meal(d))
        ids.append(int(row["id"]))

    logger.info("Loading CLIP model: %s", config.CLIP_MODEL_NAME)
    model = SentenceTransformer(config.CLIP_MODEL_NAME, device=os.environ.get("CLIP_DEVICE", "cpu"))

    logger.info("Encoding %s meal texts…", len(texts))
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="CLIP encode"):
        chunk = texts[i : i + batch_size]
        v = model.encode(
            chunk,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embs.append(np.asarray(v, dtype=np.float32))
    vectors = np.vstack(embs)
    vectors = l2_normalize_rows(vectors)
    meal_ids = np.asarray(ids, dtype=np.int64)

    index = build_ip_index(vectors)
    save_index(index, meal_ids, config.FAISS_INDEX_PATH, config.MEAL_IDS_PATH)

    if write_pg:
        url = os.environ.get("DATABASE_URL")
        if not url:
            logger.warning("write_pg requested but DATABASE_URL missing — skipping PG store.")
        else:
            conn = psycopg2.connect(url)
            try:
                _write_pg_embeddings(conn, ids, vectors)
            finally:
                conn.close()

    logger.info(
        "Done. Index: %s | Table: %s | meals=%s",
        config.FAISS_INDEX_PATH,
        meal_table_name(),
        len(ids),
    )
    return vectors, meal_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Build CLIP text embeddings + FAISS for all meals.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--write-pg", action="store_true", help="Also upsert meal_embeddings table.")
    args = p.parse_args()
    generate_embeddings(batch_size=args.batch_size, write_pg=args.write_pg)


if __name__ == "__main__":
    main()
