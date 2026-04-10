"""
PostgreSQL access for Nutristeppe dish data and experiment results.
Falls back to local CSV when DATABASE_URL is unset (local dev).
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import Json

# Defaults match production PostgreSQL (database: computer_vision).
# Override: NUTRISTEPPE_TABLE / RESULTS_TABLE or MEAL_INFO_TABLE / MEAL_RESULTS_TABLE.
MEAL_INFO_TABLE = os.environ.get("MEAL_INFO_TABLE") or os.environ.get(
    "NUTRISTEPPE_TABLE", "database_nutristeppe"
)
MEAL_RESULTS_TABLE = os.environ.get("MEAL_RESULTS_TABLE") or os.environ.get(
    "RESULTS_TABLE", "results"
)
APP_USERS_TABLE = os.environ.get("APP_USERS_TABLE", "app_users")


def _pg_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        return psycopg2.connect(url)
    except Exception:
        return None


def _normalize_pg_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "kcal_portion" in df.columns and "kilocalories_portion" not in df.columns:
        df["kilocalories_portion"] = pd.to_numeric(df["kcal_portion"], errors="coerce").fillna(0)
    if "name_en" not in df.columns:
        df["name_en"] = ""
    df["name_en"] = df["name_en"].fillna("").astype(str)
    # DB column is health_index; keep alias for any code expecting index_health
    if "health_index" in df.columns and "index_health" not in df.columns:
        df["index_health"] = pd.to_numeric(df["health_index"], errors="coerce")
    numeric_cols = [
        "kilocalories", "protein", "fat", "carbohydrate",
        "fiber", "sugar_mg", "salt_total_mg", "saturated_fat_mg",
        "serving_size_g", "kilocalories_portion", "calculated_kcal",
        "health_index", "index_health",
        "kcal_portion",
        "protein_portion", "fat_portion", "carbohydrate_portion",
        "salt_total_mg_portion", "sugar_mg_portion", "fiber_portion",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "name" in df.columns:
        df["search_text"] = df["name"].fillna("").astype(str).str.lower()
    return df


def load_database_pg() -> Optional[pd.DataFrame]:
    conn = _pg_conn()
    if conn is None:
        return None
    safe_meal = MEAL_INFO_TABLE.replace('"', "")
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{safe_meal}"', conn)
        return _normalize_pg_frame(df)
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def ensure_app_users_table(conn) -> None:
    safe = APP_USERS_TABLE.replace('"', "")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{safe}" (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    conn.commit()


def _ensure_results_user_id_column(conn) -> None:
    safe = MEAL_RESULTS_TABLE.replace('"', "")
    with conn.cursor() as cur:
        cur.execute(f'ALTER TABLE "{safe}" ADD COLUMN IF NOT EXISTS user_id INTEGER;')
    conn.commit()


def _ensure_results_kcal_columns(conn) -> None:
    safe = MEAL_RESULTS_TABLE.replace('"', "")
    with conn.cursor() as cur:
        cur.execute(
            f'ALTER TABLE "{safe}" ADD COLUMN IF NOT EXISTS kcal_user_portion DOUBLE PRECISION;'
        )
        cur.execute(
            f'ALTER TABLE "{safe}" ADD COLUMN IF NOT EXISTS kcal_gemini_portion DOUBLE PRECISION;'
        )
    conn.commit()


def create_app_user(user_name: str) -> Optional[int]:
    conn = _pg_conn()
    if conn is None:
        return None
    name = (user_name or "").strip()
    if not name:
        return None
    try:
        ensure_app_users_table(conn)
        safe = APP_USERS_TABLE.replace('"', "")
        with conn.cursor() as cur:
            cur.execute(
                f'INSERT INTO "{safe}" (user_name) VALUES (%s) RETURNING id',
                (name,),
            )
            row = cur.fetchone()
        conn.commit()
        return int(row[0]) if row else None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_app_user(user_id: int) -> Optional[Dict[str, Any]]:
    conn = _pg_conn()
    if conn is None:
        return None
    try:
        ensure_app_users_table(conn)
        safe = APP_USERS_TABLE.replace('"', "")
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT id, user_name FROM "{safe}" WHERE id = %s',
                (user_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"id": int(row[0]), "user_name": str(row[1])}
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_database_csv(base_path: pathlib.Path) -> pd.DataFrame:
    csv_path = base_path / "2April.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return _normalize_pg_frame(df)


def ensure_results_table(conn) -> None:
    safe = MEAL_RESULTS_TABLE.replace('"', "")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{safe}" (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                user_id INTEGER,
                user_name TEXT NOT NULL,
                user_dish_name TEXT,
                user_portion DOUBLE PRECISION,
                gemini_dish_name TEXT,
                gemini_portion DOUBLE PRECISION,
                matched_db_dish TEXT,
                matched_db_name_en TEXT,
                algorithm_results JSONB,
                verification_status BOOLEAN,
                verification_detail JSONB,
                image_jpeg BYTEA,
                kcal_user_portion DOUBLE PRECISION,
                kcal_gemini_portion DOUBLE PRECISION
            );
            """
        )
    conn.commit()


def save_result_row(
    user_name: str,
    user_dish_name: str,
    user_portion: float,
    gemini_dish_name: str,
    gemini_portion: float,
    matched_db_dish: str,
    matched_db_name_en: str,
    algorithm_results: Dict[str, Any],
    verification_status: bool,
    verification_detail: Dict[str, Any],
    image_jpeg: Optional[bytes],
    user_id: Optional[int] = None,
    kcal_user_portion: Optional[float] = None,
    kcal_gemini_portion: Optional[float] = None,
) -> bool:
    conn = _pg_conn()
    if conn is None:
        return False
    try:
        ensure_results_table(conn)
        _ensure_results_user_id_column(conn)
        _ensure_results_kcal_columns(conn)
        safe = MEAL_RESULTS_TABLE.replace('"', "")
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO "{safe}"
                (user_name, user_id, user_dish_name, user_portion, gemini_dish_name, gemini_portion,
                 matched_db_dish, matched_db_name_en, algorithm_results, verification_status,
                 verification_detail, image_jpeg, kcal_user_portion, kcal_gemini_portion)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_name,
                    user_id,
                    user_dish_name,
                    user_portion,
                    gemini_dish_name,
                    gemini_portion,
                    matched_db_dish,
                    matched_db_name_en,
                    Json(algorithm_results),
                    verification_status,
                    Json(verification_detail),
                    psycopg2.Binary(image_jpeg) if image_jpeg else None,
                    kcal_user_portion,
                    kcal_gemini_portion,
                ),
            )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_all_results(limit: int = 200) -> List[Dict[str, Any]]:
    conn = _pg_conn()
    if conn is None:
        return []
    try:
        ensure_results_table(conn)
        _ensure_results_user_id_column(conn)
        _ensure_results_kcal_columns(conn)
        safe = MEAL_RESULTS_TABLE.replace('"', "")
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, user_id, user_name, user_dish_name, user_portion,
                       gemini_dish_name, gemini_portion, matched_db_dish, matched_db_name_en,
                       algorithm_results, verification_status, verification_detail, image_jpeg,
                       kcal_user_portion, kcal_gemini_portion
                FROM "{safe}"
                ORDER BY created_at DESC NULLS LAST, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for tup in cur.fetchall():
                rows.append(dict(zip(cols, tup)))
        return rows
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
