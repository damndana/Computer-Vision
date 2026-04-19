"""PostgreSQL helpers: load meal rows compatible with Nutristeppe schema."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2


def pg_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    return psycopg2.connect(url)


def meal_table_name() -> str:
    return (
        os.environ.get("MEAL_INFO_TABLE")
        or os.environ.get("NUTRISTEPPE_TABLE", "database_nutristeppe")
    ).replace('"', "")


def load_all_meals_df() -> pd.DataFrame:
    """Load full meal catalog for offline embedding build."""
    conn = pg_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not set")
    safe = meal_table_name()
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{safe}"', conn)
    finally:
        conn.close()
    df.columns = df.columns.str.strip()
    return df


def fetch_meals_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Return meal_id -> row dict for nutrition and prompts."""
    if not ids:
        return {}
    conn = pg_conn()
    if conn is None:
        raise RuntimeError("DATABASE_URL is not set")
    safe = meal_table_name()
    uniq = sorted({int(i) for i in ids})
    placeholders = ",".join(["%s"] * len(uniq))
    sql = f'SELECT * FROM "{safe}" WHERE id IN ({placeholders})'
    try:
        df = pd.read_sql_query(sql, conn, params=uniq)
    finally:
        conn.close()
    df.columns = df.columns.str.strip()
    out: Dict[int, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        rid = int(row.get("id", row.get("meal_id", -1)))
        if rid < 0:
            continue
        out[rid] = row.to_dict()
    return out


def rich_text_for_meal(row: Dict[str, Any]) -> str:
    """CLIP text embedding: rich description (not name alone)."""
    parts: List[str] = []
    name = str(row.get("name", "") or "").strip()
    name_en = str(row.get("name_en", "") or "").strip()
    ing = str(row.get("ingredients", "") or "").strip()
    steps = str(row.get("steps", "") or "").strip()
    if name:
        parts.append(name)
    if name_en and name_en.lower() != name.lower():
        parts.append(name_en)
    if ing:
        parts.append(ing)
    if steps:
        parts.append(steps)
    text = ". ".join(parts)
    return text if text else name or "unknown meal"
