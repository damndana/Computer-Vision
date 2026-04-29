import io
import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from PIL import Image

from database import (
    create_app_user,
    get_app_user,
    load_database_csv,
    load_database_pg,
    save_result_row,
)
from theme import image_wide, inject_theme, render_sidebar_nav

load_dotenv()

st.set_page_config(
    page_title="Nutristeppe — проверка блюда",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme()
render_sidebar_nav()

LS_USER_ID = "nutristeppe_uid"
LS_USER_NAME = "nutristeppe_user_name"


def _persist_user_id_in_browser(uid: int) -> None:
    try:
        from streamlit_js_eval import streamlit_js_eval

        streamlit_js_eval(
            js_expressions=f'localStorage.setItem("{LS_USER_ID}", "{int(uid)}")',
            want_output=False,
            key="persist_uid",
        )
    except ImportError:
        pass


def _persist_user_name_in_browser(name: str) -> None:
    try:
        from streamlit_js_eval import streamlit_js_eval

        esc = json.dumps(name)
        streamlit_js_eval(
            js_expressions=f'localStorage.setItem("{LS_USER_NAME}", {esc})',
            want_output=False,
            key="persist_uname",
        )
    except ImportError:
        pass


def _clear_browser_user_storage() -> None:
    try:
        from streamlit_js_eval import streamlit_js_eval

        streamlit_js_eval(
            js_expressions=(
                f'localStorage.removeItem("{LS_USER_ID}"); '
                f'localStorage.removeItem("{LS_USER_NAME}");'
            ),
            want_output=False,
            key="clear_user_ls",
        )
    except ImportError:
        pass


def try_hydrate_user_from_browser() -> None:
    """Restore user from localStorage + DB (same browser returns without asking name)."""
    if st.session_state.get("user_name", "").strip():
        return
    try:
        from streamlit_js_eval import streamlit_js_eval
    except ImportError:
        return

    if os.environ.get("DATABASE_URL"):
        uid_raw = streamlit_js_eval(
            js_expressions=f'localStorage.getItem("{LS_USER_ID}")',
            key="hydrate_uid",
        )
        if uid_raw is not None and str(uid_raw).strip().isdigit():
            uid = int(str(uid_raw).strip())
            row = get_app_user(uid)
            if row:
                st.session_state.user_id = row["id"]
                st.session_state.user_name = row["user_name"]
                st.rerun()
            streamlit_js_eval(
                js_expressions=f'localStorage.removeItem("{LS_USER_ID}")',
                want_output=False,
                key="drop_stale_uid",
            )
        return

    name_raw = streamlit_js_eval(
        js_expressions=f'localStorage.getItem("{LS_USER_NAME}")',
        key="hydrate_name",
    )
    if name_raw is not None and str(name_raw).strip():
        st.session_state.user_name = str(name_raw).strip()
        st.session_state.user_id = None
        st.rerun()


# -----------------------------------------------------------------------------
# Text normalization & synonyms (RU / EN)
# -----------------------------------------------------------------------------

DISH_SYNONYMS = {
    "картошка": "картофель",
    "картошки": "картофель",
    "помидор": "томат",
    "помидоры": "томат",
    "огурец": "огурцы",
    "макароны": "макаронные изделия",
    "паста": "макаронные изделия",
    "ризотто": "рис",
    "ceasar": "цезарь",
    "caesar": "цезарь",
    "pizza": "пицца",
    "burger": "бургер",
    "salad": "салат",
}


def normalize_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def apply_synonyms(text_norm: str) -> str:
    out = text_norm
    for a, b in DISH_SYNONYMS.items():
        if a in out:
            out = out.replace(a, b)
    return out


def get_gemini_api_key() -> Optional[str]:
    for k in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def compress_image_bytes(image: Image.Image, max_side: int = 1280, quality: int = 82) -> bytes:
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _meal_photo_bytes_set(img: Image.Image) -> None:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    st.session_state["meal_photo_bytes"] = buf.getvalue()


def _meal_photo_image_get() -> Optional[Image.Image]:
    raw = st.session_state.get("meal_photo_bytes")
    if not raw:
        return None
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _clear_meal_photo() -> None:
    st.session_state.pop("meal_photo_bytes", None)
    st.session_state.pop("meal_file_uploader", None)


# -----------------------------------------------------------------------------
# Search engine (fuzzy, token, Levenshtein-style, hybrid; RU + EN)
# -----------------------------------------------------------------------------


class DishSearchEngine:
    def __init__(self, database: pd.DataFrame):
        self.database = database

    @staticmethod
    def _row_names(row: pd.Series) -> Tuple[str, str]:
        ru = normalize_text(str(row.get("name", "") or ""))
        en = normalize_text(str(row.get("name_en", "") or ""))
        return ru, en

    def _score_query_vs_names(self, q: str, ru: str, en: str) -> float:
        scores: List[float] = []
        for name in (ru, en):
            if not name:
                continue
            scores.extend(
                [
                    float(fuzz.ratio(q, name)),
                    float(fuzz.partial_ratio(q, name)),
                    float(fuzz.token_sort_ratio(q, name)),
                    float(fuzz.token_set_ratio(q, name)),
                ]
            )
        return max(scores) if scores else 0.0

    def _invalid_query(self, query: str) -> bool:
        n = normalize_text(query)
        return not n or n in ("неизвестно", "unknown")

    def search_by_fuzzy_matching(self, query: str, threshold: float = 52.0) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            s = self._score_query_vs_names(q, ru, en)
            if s >= threshold:
                d = row.to_dict()
                d["score"] = s
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def search_by_token_matching(self, query: str, min_jaccard: float = 0.28) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        q_tokens = set(q.split())
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            best = 0.0
            for name in (ru, en):
                if not name:
                    continue
                nt = set(name.split())
                if not q_tokens or not nt:
                    continue
                inter = len(q_tokens & nt)
                union = len(q_tokens | nt)
                j = inter / union if union else 0.0
                best = max(best, j)
            if best >= min_jaccard:
                d = row.to_dict()
                d["score"] = best * 100.0
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def search_by_levenshtein(self, query: str, threshold: float = 48.0) -> pd.DataFrame:
        if self._invalid_query(query):
            return pd.DataFrame()
        q = apply_synonyms(normalize_text(query))
        rows_out: List[Dict[str, Any]] = []
        for idx, row in self.database.iterrows():
            ru, en = self._row_names(row)
            s = self._score_query_vs_names(q, ru, en)
            if s >= threshold:
                d = row.to_dict()
                d["score"] = s
                d["index"] = idx
                rows_out.append(d)
        df = pd.DataFrame(rows_out)
        if not df.empty:
            df = df.sort_values("score", ascending=False)
        return df

    def hybrid_search(self, query: str, top_n: int = 5) -> pd.DataFrame:
        parts = [
            self.search_by_fuzzy_matching(query, threshold=48),
            self.search_by_token_matching(query, min_jaccard=0.22),
            self.search_by_levenshtein(query, threshold=45),
        ]
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True)
        if "id" in combined.columns:
            key = "id"
        elif "bls_code" in combined.columns:
            key = "bls_code"
        else:
            key = "name"
        if key in combined.columns:
            combined = combined.sort_values("score", ascending=False).drop_duplicates(subset=[key], keep="first")
        else:
            combined = combined.sort_values("score", ascending=False).drop_duplicates(subset=["name"], keep="first")
        return combined.sort_values("score", ascending=False).head(top_n)

    def top_candidates_for_seed(self, seed: str, top_n: int = 20) -> pd.DataFrame:
        """Broad text retrieval for constrained visual selection (no embeddings in-app)."""
        if self._invalid_query(seed):
            return pd.DataFrame()
        wide_parts = [
            self.search_by_fuzzy_matching(seed, threshold=52).head(top_n * 2),
            self.search_by_fuzzy_matching(seed, threshold=42).head(top_n * 2),
            self.search_by_fuzzy_matching(seed, threshold=35).head(top_n * 3),
            self.search_by_token_matching(seed, min_jaccard=0.22).head(top_n * 2),
            self.search_by_token_matching(seed, min_jaccard=0.15).head(top_n * 3),
            self.search_by_token_matching(seed, min_jaccard=0.10).head(top_n * 4),
            self.search_by_levenshtein(seed, threshold=45).head(top_n * 2),
            self.search_by_levenshtein(seed, threshold=38).head(top_n * 3),
            self.search_by_levenshtein(seed, threshold=32).head(top_n * 4),
        ]
        parts = [p for p in wide_parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, ignore_index=True)
        if "id" in combined.columns:
            dedup_key = "id"
        elif "bls_code" in combined.columns:
            dedup_key = "bls_code"
        else:
            dedup_key = "name"
        combined = combined.sort_values("score", ascending=False).drop_duplicates(
            subset=[dedup_key], keep="first"
        )
        return combined.sort_values("score", ascending=False).head(top_n)

    def algorithm_bundle(self, query: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Primary hybrid table + JSON-safe summaries for all methods."""
        hybrid = self.hybrid_search(query, top_n=5)
        fuzzy = self.search_by_fuzzy_matching(query).head(5)
        token = self.search_by_token_matching(query).head(5)
        lev = self.search_by_levenshtein(query).head(5)

        def pack(df: pd.DataFrame) -> List[Dict[str, Any]]:
            if df.empty:
                return []
            out = []
            for _, r in df.iterrows():
                out.append(
                    {
                        "name": str(r.get("name", "")),
                        "name_en": str(r.get("name_en", "") or ""),
                        "score": float(r.get("score", 0)),
                    }
                )
            return out

        bundle = {
            "hybrid_top5": pack(hybrid),
            "fuzzy_top5": pack(fuzzy),
            "token_top5": pack(token),
            "levenshtein_top5": pack(lev),
        }
        return hybrid, bundle


# -----------------------------------------------------------------------------
# Verification (similarity thresholds, not strict equality)
# -----------------------------------------------------------------------------

USER_GEMINI_MIN = float(os.environ.get("VERIFY_USER_GEMINI_MIN", "72"))
GEMINI_DB_MIN = float(os.environ.get("VERIFY_GEMINI_DB_MIN", "68"))
MAX_PLATE_DISHES = max(1, min(10, int(os.environ.get("MAX_PLATE_DISHES", "10"))))


def verification_scores(
    user_dish: str,
    gemini_dish: str,
    db_name: str,
    db_name_en: str,
) -> Dict[str, float]:
    u = apply_synonyms(normalize_text(user_dish))
    g = apply_synonyms(normalize_text(gemini_dish))
    db_ru = normalize_text(db_name)
    db_en = normalize_text(db_name_en or "")
    user_gemini = max(
        float(fuzz.token_sort_ratio(u, g)),
        float(fuzz.partial_ratio(u, g)),
    )
    gemini_db = 0.0
    for dn in (db_ru, db_en):
        if dn:
            gemini_db = max(
                gemini_db,
                float(fuzz.token_sort_ratio(g, dn)),
                float(fuzz.partial_ratio(g, dn)),
            )
    user_db = 0.0
    for dn in (db_ru, db_en):
        if dn:
            user_db = max(
                user_db,
                float(fuzz.token_sort_ratio(u, dn)),
                float(fuzz.partial_ratio(u, dn)),
            )
    return {
        "user_vs_gemini": user_gemini,
        "gemini_vs_db": gemini_db,
        "user_vs_db": user_db,
    }


def is_verified(user_dish: str, gemini_dish: str, db_name: str, db_name_en: str) -> Tuple[bool, Dict[str, Any]]:
    s = verification_scores(user_dish, gemini_dish, db_name, db_name_en)
    ok = s["user_vs_gemini"] >= USER_GEMINI_MIN and s["gemini_vs_db"] >= GEMINI_DB_MIN
    detail = {
        "thresholds": {"user_vs_gemini_min": USER_GEMINI_MIN, "gemini_vs_db_min": GEMINI_DB_MIN},
        "scores": s,
        "verified": ok,
    }
    return ok, detail


def is_verified_smart(
    user_dish: str, gemini_dish: str, db_name: str, db_name_en: str
) -> Tuple[bool, Dict[str, Any]]:
    """If the user left the dish name empty (photo-only flow), only check Gemini vs DB."""
    if not (user_dish or "").strip():
        s = verification_scores("", gemini_dish, db_name, db_name_en)
        ok = s["gemini_vs_db"] >= GEMINI_DB_MIN
        detail = {
            "thresholds": {
                "mode": "photo_only",
                "gemini_vs_db_min": GEMINI_DB_MIN,
            },
            "scores": s,
            "verified": ok,
        }
        return ok, detail
    return is_verified(user_dish, gemini_dish, db_name, db_name_en)


def clip_index_files_exist() -> bool:
    """True when CLIP+FAISS index files from meal_pipeline are present."""
    try:
        from meal_pipeline import config as mp_cfg

        return bool(mp_cfg.FAISS_INDEX_PATH.is_file() and mp_cfg.MEAL_IDS_PATH.is_file())
    except Exception:
        return False


@st.cache_resource
def _clip_meal_retriever_cached():
    try:
        from meal_pipeline.meal_retriever import MealRetriever

        r = MealRetriever()
        r.ensure_loaded()
        return r
    except Exception:
        return None


def candidates_df_from_clip_image(
    image: Image.Image, database: pd.DataFrame, k: int = 20
) -> pd.DataFrame:
    """Top-k meals by CLIP image similarity (same order as FAISS)."""
    ret = _clip_meal_retriever_cached()
    if ret is None or database.empty or "id" not in database.columns:
        return pd.DataFrame()
    hits = ret.retrieve_for_image(image.convert("RGB"), k)
    rows_out: List[Dict[str, Any]] = []
    for mid, sc in hits:
        m = database[database["id"] == mid]
        if m.empty:
            continue
        d = m.iloc[0].to_dict()
        d["score"] = float(max(0.0, min(1.0, sc))) * 100.0
        rows_out.append(d)
    return pd.DataFrame(rows_out)


def row_candidate_id(row: pd.Series) -> int:
    """Stable integer id for prompt / JSON (DB id preferred, else dataframe index from search)."""
    rid = row.get("id")
    if rid is not None and not (isinstance(rid, float) and pd.isna(rid)):
        try:
            return int(rid)
        except (TypeError, ValueError):
            pass
    idx = row.get("index")
    if idx is not None and not (isinstance(idx, float) and pd.isna(idx)):
        try:
            return int(idx)
        except (TypeError, ValueError):
            pass
    h = abs(hash(str(row.get("name", "")))) % (10**9 - 1)
    return h if h > 0 else 1


def _candidate_description(row: pd.Series) -> str:
    chunks: List[str] = []
    for col in ("ingredients", "description", "steps", "recipe"):
        v = row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if s:
            chunks.append(s)
    return (" ".join(chunks))[:420]


def build_candidate_list(df: pd.DataFrame) -> str:
    lines: List[str] = []
    for _, row in df.iterrows():
        cid = row_candidate_id(row)
        name_ru = str(row.get("name", "") or "")
        name_en = str(row.get("name_en", "") or "")
        desc = _candidate_description(row)
        lines.append(f"{cid} | {name_ru} | {name_en}\ndescription: {desc}")
    return "\n\n".join(lines)


MEAL_SELECTION_PROMPT_TEMPLATE = """You are a professional food recognition assistant.

The photo may show ONE or SEVERAL distinct foods on the same plate or frame. You must analyze ALL clearly separable dishes (main, side, salad, bread, sauce bowl if it is a named dish in the list, etc.), up to {MAX_DISHES} separate entries.

IMPORTANT RULES:

1. For EACH detected dish you MUST choose ONLY from the provided candidate dishes (by id).
2. Do NOT invent new dish names.
3. If a particular visible food does not match any candidate well, use one entry with selected_meal_id null and selected_meal_name "unknown" for that food.
4. Focus on: main ingredients, cooking style, visible structure, sauce presence, shape (pasta, salad, soup, etc.).
5. Ignore plate decorations, lighting, or camera angle.
6. Do not duplicate the same physical food twice. Different foods → different entries.
7. Order entries: put the MAIN dish first when possible; then sides / salads / extras.
8. If two candidates are visually similar, prefer the one whose description ingredients match what is clearly visible.

Confidence guidelines:
0.9–1.0 = almost certain
0.7–0.9 = likely match
0.4–0.7 = possible match
0–0.4 = low confidence

--------------------------------------------------

DATABASE CANDIDATE MEALS:

{CANDIDATE_LIST}

Each candidate contains:
- id
- name_ru
- name_en
- description (ingredients or short description)

--------------------------------------------------

TASK:

1. Analyze the image carefully.
2. List each DISTINCT dish you see.
3. For each, pick the single best matching candidate (or unknown).

--------------------------------------------------

Return ONLY valid JSON in the following format:

{
  "meals": [
    {
      "selected_meal_id": integer or null,
      "selected_meal_name": "string",
      "confidence": number between 0 and 1,
      "visible_ingredients": ["ingredient1","ingredient2"],
      "reasoning": "short explanation",
      "role": "main" | "side" | "drink" | "other"
    }
  ]
}

If the image shows only one dish, return "meals" with exactly one object.

If nothing on the plate matches any candidate at all, return:
{ "meals": [{ "selected_meal_id": null, "selected_meal_name": "unknown", "confidence": 0, "visible_ingredients": [], "reasoning": "no candidate matches the image", "role": "main" }] }

Return ONLY JSON. No extra text.
"""


def normalize_visual_selection_to_meals(obj: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept new multi format {meals:[...]} or legacy single-object selection."""
    if not obj or not isinstance(obj, dict):
        return []
    raw_list = obj.get("meals")
    if raw_list is None and "selected_meals" in obj:
        raw_list = obj["selected_meals"]
    if isinstance(raw_list, list) and raw_list:
        out: List[Dict[str, Any]] = []
        for x in raw_list:
            if isinstance(x, dict):
                out.append(x)
        return out[:MAX_PLATE_DISHES]
    if "selected_meal_id" in obj or "selected_meal_name" in obj:
        return [obj][:MAX_PLATE_DISHES]
    return []


def allocate_user_portions_by_ai_weights(
    user_total_g: float, ai_portions_g: List[float]
) -> List[float]:
    """Split user-declared total grams across dishes by relative AI portion weights."""
    if not ai_portions_g:
        return []
    weights = [max(0.0, float(x)) for x in ai_portions_g]
    s = sum(weights)
    if s <= 0:
        n = len(ai_portions_g)
        return [float(user_total_g) / n] * n if n else []
    return [float(user_total_g) * (w / s) for w in weights]


def _gemini_generation_config_json() -> Any:
    import google.generativeai as genai

    try:
        return genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
    except Exception:
        pass
    try:
        from google.generativeai.types import GenerationConfig

        return GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
    except Exception:
        return {"temperature": 0.2, "response_mime_type": "application/json"}


def _parse_json_object_from_model_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            try:
                obj = json.loads(m.group())
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def find_candidate_row_by_id(candidates_df: pd.DataFrame, meal_id: Any) -> Optional[pd.Series]:
    if candidates_df.empty or meal_id is None:
        return None
    try:
        mid = int(meal_id)
    except (TypeError, ValueError):
        return None
    for _, row in candidates_df.iterrows():
        if row_candidate_id(row) == mid:
            return row
    return None


def resolve_meal_selection_slot(
    candidates_df: pd.DataFrame, sel: Dict[str, Any]
) -> Tuple[bool, Optional[pd.Series], str, str, str]:
    """unknown flag, row, display name, matched RU, matched EN."""
    sid = sel.get("selected_meal_id")
    sname_raw = str(sel.get("selected_meal_name", "") or "").strip()
    sname_l = normalize_text(sname_raw)
    picked = find_candidate_row_by_id(candidates_df, sid)
    unknown = (
        picked is None
        or sid is None
        or sname_l == normalize_text("unknown")
        or sname_l == normalize_text("неизвестно")
    )
    if unknown:
        return True, None, "unknown", "", ""
    assert picked is not None
    gn = str(picked.get("name", "") or sname_raw)
    return False, picked, gn, str(picked.get("name", "")), str(picked.get("name_en", "") or "")


def reorder_candidate_records(
    candidates_df: pd.DataFrame,
    selected: Optional[pd.Series],
) -> List[Dict[str, Any]]:
    if candidates_df.empty:
        return []
    recs = candidates_df.to_dict("records")
    if selected is None:
        return recs
    sid = row_candidate_id(selected)
    hit: Optional[Dict[str, Any]] = None
    rest: List[Dict[str, Any]] = []
    for r in recs:
        if row_candidate_id(pd.Series(r)) == sid:
            hit = r
        else:
            rest.append(r)
    if hit is None:
        return recs
    return [hit] + rest


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------


class GeminiMealAgent:
    def __init__(self, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def select_meals_from_candidates(
        self, jpeg_bytes: bytes, candidates_df: pd.DataFrame
    ) -> Dict[str, Any]:
        try:
            if candidates_df.empty:
                return {"error": "empty_candidates"}
            clist = build_candidate_list(candidates_df)
            prompt = (
                MEAL_SELECTION_PROMPT_TEMPLATE.replace("{CANDIDATE_LIST}", clist).replace(
                    "{MAX_DISHES}", str(MAX_PLATE_DISHES)
                )
            )
            gcfg = _gemini_generation_config_json()
            response = self.model.generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": jpeg_bytes},
                ],
                generation_config=gcfg,
            )
            text = getattr(response, "text", None) or ""
            parsed = _parse_json_object_from_model_text(text)
            if not parsed:
                return {"error": "invalid_json", "raw": text[:800]}
            meals = normalize_visual_selection_to_meals(parsed)
            if not meals:
                meals = normalize_visual_selection_to_meals(
                    {
                        "selected_meal_id": parsed.get("selected_meal_id"),
                        "selected_meal_name": parsed.get("selected_meal_name"),
                        "confidence": parsed.get("confidence", 0),
                        "visible_ingredients": parsed.get("visible_ingredients", []),
                        "reasoning": parsed.get("reasoning", ""),
                        "role": "main",
                    }
                )
            if not meals:
                return {"error": "no_meals_in_response", "raw": text[:800]}
            return {"meals": meals, "raw": parsed}
        except Exception as e:
            return {"error": str(e)}

    def estimate_portion_jpeg(self, jpeg_bytes: bytes, dish_context_ru: str) -> Dict[str, Any]:
        try:
            dish_context_ru = (dish_context_ru or "").strip() or "блюдо"
            prompt = (
                f"Оцени массу основной порции на фото в граммах. Контекст блюда: {dish_context_ru}.\n"
                'Верни ТОЛЬКО JSON: {"portion_grams": число} — одна оценка.'
            )
            gcfg = _gemini_generation_config_json()
            response = self.model.generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": jpeg_bytes},
                ],
                generation_config=gcfg,
            )
            text = getattr(response, "text", None) or ""
            parsed = _parse_json_object_from_model_text(text)
            if not parsed:
                return {"portion_grams": 0.0, "error": "invalid_json"}
            pg = float(parsed.get("portion_grams", 0) or 0)
            return {"portion_grams": pg}
        except Exception as e:
            return {"portion_grams": 0.0, "error": str(e)}

    def estimate_multi_portions_jpeg(
        self, jpeg_bytes: bytes, dish_labels_ru: List[str]
    ) -> Dict[str, Any]:
        """One vision call: grams on plate for each numbered dish."""
        labels = [str(x).strip() or "блюдо" for x in dish_labels_ru]
        if not labels:
            return {"portion_grams_list": [], "error": "empty_labels"}
        if len(labels) == 1:
            r = self.estimate_portion_jpeg(jpeg_bytes, labels[0])
            pg = float(r.get("portion_grams", 0) or 0)
            return {"portion_grams_list": [pg], "error": r.get("error")}
        try:
            lines = "\n".join(f"{i}. {lbl}" for i, lbl in enumerate(labels))
            n = len(labels)
            prompt = (
                "The photo shows a plate or tray with several distinct foods. "
                "Estimate how many grams of EACH numbered item are visible on the photo "
                "(only that item's portion, not the whole plate).\n\n"
                f"{lines}\n\n"
                f'Return ONLY JSON: {{"portions":[{{"index":0,"portion_grams":120}},...]}} '
                f"with exactly {n} objects, indices 0 through {n - 1}."
            )
            gcfg = _gemini_generation_config_json()
            response = self.model.generate_content(
                [
                    prompt,
                    {"mime_type": "image/jpeg", "data": jpeg_bytes},
                ],
                generation_config=gcfg,
            )
            text = getattr(response, "text", None) or ""
            parsed = _parse_json_object_from_model_text(text)
            if not parsed:
                return {"portion_grams_list": [], "error": "invalid_json", "raw": text[:600]}
            arr = parsed.get("portions")
            if not isinstance(arr, list):
                return {"portion_grams_list": [], "error": "bad_shape"}
            by_idx: Dict[int, float] = {}
            for item in arr:
                if not isinstance(item, dict):
                    continue
                try:
                    ix = int(item.get("index", -1))
                except (TypeError, ValueError):
                    continue
                by_idx[ix] = float(item.get("portion_grams", 0) or 0)
            out_list: List[float] = []
            for i in range(n):
                g = by_idx.get(i, 0.0)
                out_list.append(g)
            if all(x <= 0 for x in out_list):
                return {"portion_grams_list": [], "error": "zero_portions"}
            return {"portion_grams_list": out_list}
        except Exception as e:
            return {"portion_grams_list": [], "error": str(e)}

    def verify_ingredient_consistency(
        self,
        dish_ru: str,
        dish_en: str,
        ingredients_expected: str,
        visible_ingredients: List[str],
    ) -> Dict[str, Any]:
        try:
            vis = ", ".join(visible_ingredients or [])
            prompt = (
                "Is this dish consistent with the visible ingredients?\n\n"
                f"Dish (RU): {dish_ru}\n"
                f"Dish (EN): {dish_en}\n"
                f"Ingredients expected (from reference): {ingredients_expected}\n"
                f"Visible ingredients (from image analysis): {vis}\n\n"
                'Return ONLY JSON: {"consistent": true or false, "confidence": number between 0 and 1}'
            )
            gcfg = _gemini_generation_config_json()
            response = self.model.generate_content(prompt, generation_config=gcfg)
            text = getattr(response, "text", None) or ""
            parsed = _parse_json_object_from_model_text(text)
            if not parsed:
                return {"consistent": True, "confidence": 0.5, "error": "invalid_json"}
            return {
                "consistent": bool(parsed.get("consistent", True)),
                "confidence": float(parsed.get("confidence", 0) or 0),
            }
        except Exception as e:
            return {"consistent": True, "confidence": 0.0, "error": str(e)}


# -----------------------------------------------------------------------------
# Nutrition (optional display)
# -----------------------------------------------------------------------------


def nutrition_for_row(row: pd.Series, portion_g: float) -> Dict[str, float]:
    base = row.get("serving_size_g", 100)
    if pd.isna(base) or float(base) == 0:
        base = 100.0
    ratio = float(portion_g) / float(base)
    return {
        "kilocalories": float(row.get("kilocalories", 0) or 0) * ratio,
        "protein_g": float(row.get("protein", 0) or 0) * ratio / 1000.0,
        "fat_g": float(row.get("fat", 0) or 0) * ratio / 1000.0,
        "carb_g": float(row.get("carbohydrate", 0) or 0) * ratio / 1000.0,
    }


@st.cache_data(ttl=120)
def load_database() -> pd.DataFrame:
    base = pathlib.Path(__file__).parent
    df = load_database_pg()
    if df is not None and not df.empty:
        return df
    return load_database_csv(base)


@st.cache_resource
def _db_rows_by_id_cached() -> Dict[int, Dict[str, Any]]:
    """
    Fast lookup cache for CLIP/FAISS hit -> meal row.
    Avoids repeated expensive dataframe filtering like: db[db["id"] == mid]
    """
    df = load_database()
    if df is None or df.empty or "id" not in df.columns:
        return {}
    # Ensure stable int keys and plain dict values
    out: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        try:
            rid = int(r.get("id"))
        except Exception:
            continue
        out[rid] = r.to_dict()
    return out


def ensure_session():
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False


def render_match_cards(
    records: List[Dict[str, Any]],
    user_portion_g: float,
    gemini_portion_g: float,
    first_label: str = "Лучшее совпадение",
    show_kcal_lines: bool = True,
):
    if not records:
        st.warning("Нет близких совпадений в базе.")
        return
    for i, r in enumerate(records):
        name = str(r.get("name", ""))
        en = str(r.get("name_en", "") or "")
        sc = float(r.get("score", 0))
        extra = f" · {en}" if en else ""
        cls = "card card-best" if i == 0 else "card"
        label = first_label if i == 0 else "Другой вариант"
        ser = pd.Series(r)
        nut_u = nutrition_for_row(ser, user_portion_g)
        nut_g = nutrition_for_row(ser, gemini_portion_g) if gemini_portion_g > 0 else None
        ref_kcal = float(r.get("kilocalories", 0) or 0)
        ref_g = float(r.get("serving_size_g", 100) or 100)
        if ref_g <= 0:
            ref_g = 100.0
        kcal_ai_line = ""
        if show_kcal_lines and nut_g is not None:
            kcal_ai_line = (
                f'<br/><span class="muted">Ккал по базе на порцию ИИ ({gemini_portion_g:.0f} г): '
                f'<strong>{nut_g["kilocalories"]:.0f}</strong></span>'
            )
        kcal_user_line = ""
        if show_kcal_lines:
            kcal_user_line = (
                f'<br/><span class="muted">Ккал по базе на вашу порцию ({user_portion_g:.0f} г): '
                f'<strong>{nut_u["kilocalories"]:.0f}</strong></span>'
            )
        st.markdown(
            f'<div class="{cls}"><strong>{label}</strong><br/>{name}{extra}<br/>'
            f'<span class="muted">Уверенность {sc:.0f}%</span>'
            f"{kcal_user_line}"
            f"{kcal_ai_line}"
            f'<br/><span class="muted">Эталон в базе: {ref_kcal:.0f} ккал / {ref_g:.0f} г</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_meal_result_single(payload: Dict[str, Any]):
    verified = payload["verified"]
    user_dish = payload["user_dish"]
    user_portion = payload["user_portion"]
    gemini_name = payload["gemini_name"]
    gemini_portion = payload["gemini_portion"]
    hybrid_records: List[Dict[str, Any]] = payload["hybrid_records"]
    dishes = payload["dishes"]
    primary = payload["primary"]
    algo_json = payload.get("algo_json") or {}
    selection_detail: Dict[str, Any] = payload.get("selection_detail") or {}

    st.divider()
    st.subheader("Итог")
    badge = (
        '<span class="badge-ok">Подтверждено</span>'
        if verified
        else '<span class="badge-bad">Не подтверждено</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    if payload.get("photo_only"):
        st.caption(
            "Режим только по фото: подтверждение — по согласованию ответа ИИ со строкой базы "
            f"(порог ≈ {GEMINI_DB_MIN:.0f}%)."
        )
    else:
        st.caption(
            "Подтверждение: ваше название близко к ответу ИИ, а ответ ИИ согласуется с выбранной строкой базы "
            f"(пороги схожести ≈ {USER_GEMINI_MIN:.0f}% / {GEMINI_DB_MIN:.0f}%)."
        )

    st.markdown("**Порция**" if payload.get("photo_only") else "**Вы указали**")
    st.markdown(f'<div class="card">{user_dish} · {user_portion:.0f} г</div>', unsafe_allow_html=True)
    st.markdown("**ИИ**")
    ai_conf = 0.0
    try:
        ai_conf = float((primary or {}).get("confidence", 0) or 0)
    except Exception:
        ai_conf = 0.0
    st.markdown(
        f'<div class="card">{gemini_name}<br/>'
        f'<span class="muted">Уверенность ИИ: {ai_conf * 100.0:.0f}%</span><br/>'
        f'{gemini_portion:.0f} г</div>',
        unsafe_allow_html=True,
    )
    if selection_detail.get("reasoning"):
        st.markdown(
            f'<div class="card muted"><strong>Почему так</strong><br/>'
            f'{selection_detail.get("reasoning", "")}</div>',
            unsafe_allow_html=True,
        )
    vis = selection_detail.get("visible_ingredients") or []
    if isinstance(vis, list) and vis:
        st.caption("На фото (по ИИ): " + ", ".join(str(x) for x in vis))
    iv = selection_detail.get("ingredient_verify")
    if isinstance(iv, dict) and iv.get("confidence") is not None:
        c = float(iv.get("confidence", 0) or 0)
        ok = bool(iv.get("consistent", True))
        st.caption(
            f"Сверка с описанием блюда в базе: {'согласуется' if ok else 'сомнительно'} "
            f"(уверенность {c:.2f})."
        )

    st.markdown("**Калории из базы**")
    if hybrid_records:
        ser0 = pd.Series(hybrid_records[0])
        n_u = nutrition_for_row(ser0, user_portion)
        n_g = (
            nutrition_for_row(ser0, gemini_portion) if gemini_portion > 0 else None
        )
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                f"На вашу порцию ({user_portion:.0f} г)",
                f"{n_u['kilocalories']:.0f} ккал",
            )
        with m2:
            if n_g is not None:
                st.metric(
                    f"На порцию ИИ ({gemini_portion:.0f} г)",
                    f"{n_g['kilocalories']:.0f} ккал",
                )
            else:
                st.metric("На порцию ИИ", "—")
    else:
        st.warning("Нет совпадения в базе — калории из справочника не посчитаны.")

    st.markdown("**Совпадения в базе**")
    fl = (
        "Выбрано по фото (из кандидатов)"
        if payload.get("constrained_visual_pick")
        else "Лучшее совпадение"
    )
    render_match_cards(hybrid_records, user_portion, gemini_portion, first_label=fl)

    if hybrid_records:
        nut = nutrition_for_row(pd.Series(hybrid_records[0]), user_portion)
        nut_ai = (
            nutrition_for_row(pd.Series(hybrid_records[0]), gemini_portion)
            if gemini_portion > 0
            else None
        )
        macro = (
            f'<div class="card muted"><strong>БЖУ по лучшему совпадению</strong><br/>'
            f'Ваша порция: {nut["kilocalories"]:.0f} ккал · Б {nut["protein_g"]:.1f} г · '
            f'Ж {nut["fat_g"]:.1f} г · У {nut["carb_g"]:.1f} г'
        )
        if nut_ai is not None:
            macro += (
                f'<br/>Порция ИИ: {nut_ai["kilocalories"]:.0f} ккал · '
                f'Б {nut_ai["protein_g"]:.1f} г · Ж {nut_ai["fat_g"]:.1f} г · '
                f'У {nut_ai["carb_g"]:.1f} г'
            )
        macro += "</div>"
        st.markdown(macro, unsafe_allow_html=True)

    if len(dishes) > 1:
        with st.expander("Другие объекты на фото"):
            for d in dishes:
                if d is primary:
                    continue
                st.write(f"- {d['dish_name']} (~{d['portion_grams']:.0f} g)")

    if algo_json:
        with st.expander("Детали подбора (алгоритмы)"):
            st.json(algo_json)


def render_meal_result_multi(payload: Dict[str, Any], meal_items: List[Dict[str, Any]]):
    verified = payload["verified"]
    user_dish = payload["user_dish"]
    user_portion = payload["user_portion"]
    gemini_name = payload["gemini_name"]
    gemini_portion = payload["gemini_portion"]
    hybrid_records: List[Dict[str, Any]] = payload.get("hybrid_records") or []
    algo_json = payload.get("algo_json") or {}

    st.divider()
    st.subheader("Итог")
    badge = (
        '<span class="badge-ok">Подтверждено</span>'
        if verified
        else '<span class="badge-bad">Не подтверждено</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    if payload.get("photo_only"):
        st.caption(
            "Только фото: для каждого блюда проверяется согласование ответа ИИ со строкой базы "
            f"(порог ≈ {GEMINI_DB_MIN:.0f}%). "
            "Зелёная галочка — если **все** пункты прошли проверку."
        )
    else:
        st.caption(
            "Для каждого блюда: ваше название сравнивается с ответом ИИ по этому пункту, "
            "ответ ИИ — со строкой базы "
            f"(пороги ≈ {USER_GEMINI_MIN:.0f}% / {GEMINI_DB_MIN:.0f}%). "
            "Зелёная галочка — только если **все** распознанные блюда прошли проверку."
        )

    st.markdown(
        "**Порция (на всё фото)**" if payload.get("photo_only") else "**Вы указали (на всё фото)**"
    )
    st.markdown(f'<div class="card">{user_dish} · {user_portion:.0f} г</div>', unsafe_allow_html=True)
    st.markdown("**ИИ — несколько блюд**")
    st.markdown(
        f'<div class="card">{gemini_name} · суммарно ~{gemini_portion:.0f} г (оценка по фото)</div>',
        unsafe_allow_html=True,
    )

    total_u = 0.0
    total_g = 0.0
    for idx, it in enumerate(meal_items):
        role = str(it.get("role") or "other")
        gname = str(it.get("gemini_name") or "—")
        try:
            ai_conf = float(it.get("ai_confidence", 0) or 0)
        except Exception:
            ai_conf = 0.0
        gport = float(it.get("gemini_portion") or 0)
        ualloc = float(it.get("user_portion_allocated") or 0)
        v_ok = bool(it.get("verified"))
        badge_i = "✅" if v_ok else "❌"
        st.markdown(f"##### {badge_i} Блюдо {idx + 1} · {role}")
        st.markdown(
            f'<div class="card"><strong>{gname}</strong><br/>'
            f'<span class="muted">Уверенность ИИ: {ai_conf * 100.0:.0f}%</span><br/>'
            f"Порция ИИ: ~{gport:.0f} г · ваша доля от общей граммовки: ~{ualloc:.0f} г</div>",
            unsafe_allow_html=True,
        )
        sd = it.get("selection_detail") or {}
        if sd.get("reasoning"):
            st.markdown(
                f'<div class="card muted">{sd.get("reasoning", "")}</div>',
                unsafe_allow_html=True,
            )
        vis = sd.get("visible_ingredients") or []
        if isinstance(vis, list) and vis:
            st.caption("На фото (по ИИ): " + ", ".join(str(x) for x in vis))
        iv = sd.get("ingredient_verify")
        if isinstance(iv, dict) and iv.get("confidence") is not None:
            c = float(iv.get("confidence", 0) or 0)
            ok = bool(iv.get("consistent", True))
            st.caption(
                f"Сверка с описанием в базе: {'согласуется' if ok else 'сомнительно'} ({c:.2f})."
            )
        row_dict = it.get("nutrition_row_dict")
        if isinstance(row_dict, dict) and row_dict:
            ser = pd.Series(row_dict)
            nu = nutrition_for_row(ser, ualloc)
            ng = nutrition_for_row(ser, gport) if gport > 0 else None
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"Ккал (ваша доля ~{ualloc:.0f} г)", f"{nu['kilocalories']:.0f}")
            with c2:
                if ng is not None:
                    st.metric(f"Ккал (порция ИИ ~{gport:.0f} г)", f"{ng['kilocalories']:.0f}")
                else:
                    st.metric("Ккал (порция ИИ)", "—")
            total_u += float(nu["kilocalories"])
            if ng is not None:
                total_g += float(ng["kilocalories"])
        else:
            st.info("Нет строки справочника для этого пункта.")

    st.markdown("**Сумма по распознанным блюдам (ккал)**")
    st.markdown(
        f'<div class="card muted">По вашей граммовке (доли): <strong>{total_u:.0f}</strong> ккал<br/>'
        f"По порциям ИИ: <strong>{total_g:.0f}</strong> ккал</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Кандидаты из базы (по вашему названию)**")
    st.caption(
        "Один общий список кандидатов для снимка; ниже — оценки текстового поиска, без разбивки по каждому блюду."
    )
    render_match_cards(
        hybrid_records,
        user_portion,
        gemini_portion if gemini_portion > 0 else 1.0,
        first_label="Топ по названию",
        show_kcal_lines=False,
    )

    if algo_json:
        with st.expander("Детали подбора (алгоритмы)"):
            st.json(algo_json)


def render_meal_result(payload: Dict[str, Any]):
    items = payload.get("meal_items")
    if isinstance(items, list) and len(items) > 0:
        render_meal_result_multi(payload, items)
        return
    render_meal_result_single(payload)


def main():
    ensure_session()
    try_hydrate_user_from_browser()

    st.title("Проверка блюда")
    st.caption("Фото, что вы реально съели — сравнение с ИИ и базой Nutristeppe.")

    if not st.session_state.user_name.strip():
        name = st.text_input("Ваше имя", placeholder="Как подписывать записи", key="name_input")
        if st.button("Далее", type="primary"):
            if name.strip():
                nm = name.strip()
                if os.environ.get("DATABASE_URL"):
                    new_id = create_app_user(nm)
                    if new_id is not None:
                        st.session_state.user_id = new_id
                        st.session_state.user_name = nm
                        _persist_user_id_in_browser(new_id)
                    else:
                        st.session_state.user_name = nm
                        st.warning("Не удалось сохранить профиль в базе — имя только в этой сессии.")
                else:
                    st.session_state.user_name = nm
                    st.session_state.user_id = None
                    _persist_user_name_in_browser(nm)
                st.rerun()
        return

    st.markdown(
        f'<p class="muted">Вы: <strong>{st.session_state.user_name}</strong></p>',
        unsafe_allow_html=True,
    )
    if st.button("Сменить имя", type="secondary", key="chg"):
        st.session_state.user_name = ""
        st.session_state.user_id = None
        st.session_state.pop("meal_result", None)
        st.session_state.show_camera = False
        _clear_meal_photo()
        _clear_browser_user_storage()
        st.rerun()

    st.subheader("Фото")
    st.caption("Камера не включается сама — только после кнопки ниже (меньше расход батареи и приватность).")

    up = st.file_uploader(
        "Загрузить файл",
        type=["jpg", "jpeg", "png"],
        key="meal_file_uploader",
        help="Фото из галереи или файлов",
    )

    if st.button("Сделать фото камерой", type="secondary", key="btn_open_cam"):
        st.session_state.show_camera = True
        st.rerun()

    if st.session_state.show_camera:
        st.info("Разрешите доступ к камере в браузере. Превью только здесь — после снимка окно камеры закроется.")
        cam = st.camera_input(
            "Камера",
            key="widget_camera_input",
            help="Нажмите «Take Photo» / «Снять», когда готовы.",
        )
        if st.button("Отмена — закрыть камеру", type="secondary", key="btn_close_cam"):
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()
        if cam is not None:
            _meal_photo_bytes_set(Image.open(cam).convert("RGB"))
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()

    has_stored = bool(st.session_state.get("meal_photo_bytes"))
    has_file = up is not None
    if has_stored or has_file:
        if st.button("Удалить фото", type="secondary", key="btn_clear_photo"):
            _clear_meal_photo()
            st.session_state.show_camera = False
            st.session_state.pop("widget_camera_input", None)
            st.rerun()

    img: Optional[Image.Image] = None
    if up is not None:
        img = Image.open(up).convert("RGB")
        st.session_state.pop("meal_photo_bytes", None)
    else:
        img = _meal_photo_image_get()

    if img is not None:
        image_wide(img)

    st.subheader("Порция")
    user_portion = st.number_input("Порция, г (на всё фото)", min_value=0.0, max_value=5000.0, value=200.0, step=10.0)

    photo_pipeline = clip_index_files_exist()
    if photo_pipeline:
        st.success(
            "Режим **только фото**: кандидаты из базы подбираются по CLIP+FAISS — название блюда не обязательно."
        )
    else:
        st.warning(
            "Чтобы не вводить название блюда, на сервере нужны файлы индекса CLIP "
            "(см. README: `python -m meal_pipeline.embedding_generator`). "
            "Пока индекса нет — ниже можно **опционально** указать название для текстового поиска кандидатов."
        )

    with st.expander("Необязательно: как вы называете блюдо (если нет CLIP-индекса)", expanded=not photo_pipeline):
        user_dish = st.text_input(
            "Название для поиска в базе",
            placeholder="оставьте пустым при работе только по фото",
            key="udish",
        )

    st.caption(
        "При нескольких блюдах на фото граммы — на всё фото; ИИ делит порцию между блюдами по оценке веса."
    )

    analyze = st.button("Анализировать", type="primary", key="go")

    if analyze:
        if img is None:
            st.error("Сначала добавьте фото (камера или файл).")
        else:
            api_key = get_gemini_api_key()
            if not api_key:
                st.error("Анализ недоступен: на сервере не задан ключ API.")
            else:
                db = load_database()
                if db.empty:
                    st.error("База блюд пуста или недоступна.")
                else:
                    db_by_id = _db_rows_by_id_cached()
                    jpeg = compress_image_bytes(img)
                    engine = DishSearchEngine(db)
                    use_clip = False
                    if photo_pipeline:
                        _cret = _clip_meal_retriever_cached()
                        if _cret is not None:
                            use_clip = True
                        else:
                            st.warning(
                                "Найдены файлы CLIP+FAISS, но модель не загрузилась. "
                                "Установите зависимости: `pip install -r requirements-api.txt`, "
                                "либо введите название блюда в поле ниже."
                            )
                    if use_clip:
                        # CLIP+FAISS photo-only mode: retrieve candidates per detected dish
                        candidates_df = pd.DataFrame()
                    elif user_dish.strip():
                        candidates_df = engine.top_candidates_for_seed(user_dish.strip(), 20)
                    else:
                        candidates_df = pd.DataFrame()

                    if use_clip:
                        _user_dish_saved = user_dish.strip() or "(только фото)"
                        try:
                            from meal_pipeline import config as mp_cfg
                            from meal_pipeline.gemini_reasoner import GeminiReasoner
                            from meal_pipeline.multi_meal_detector import MultiMealDetector

                            retr = _clip_meal_retriever_cached()
                            if retr is None:
                                raise RuntimeError("clip_retriever_unavailable")

                            with st.spinner("ИИ определяет, сколько блюд на фото…"):
                                detector = MultiMealDetector(api_key, mp_cfg.GEMINI_MODEL)
                                layout = detector.analyze_plate(jpeg)

                            # role heuristic: largest fraction is main
                            dishes_layout = list(layout.dishes or [])
                            if not dishes_layout:
                                dishes_layout = [{"description": "meal on plate", "fraction": 1.0}]
                            main_i = 0
                            best_f = -1.0
                            for i, d in enumerate(dishes_layout):
                                try:
                                    f = float(d.get("fraction", 0) or 0)
                                except (TypeError, ValueError):
                                    f = 0.0
                                if f > best_f:
                                    best_f = f
                                    main_i = i

                            reasoner = GeminiReasoner(api_key, mp_cfg.GEMINI_MODEL)

                            meal_items: List[Dict[str, Any]] = []
                            per_dish_details: List[Dict[str, Any]] = []
                            for i, d in enumerate(dishes_layout[:MAX_PLATE_DISHES]):
                                desc = str(d.get("description", "") or "").strip() or "food item"
                                try:
                                    frac = float(d.get("fraction", 1.0 / max(len(dishes_layout), 1)) or 0)
                                except (TypeError, ValueError):
                                    frac = 1.0 / max(len(dishes_layout), 1)
                                frac = max(0.0, min(1.0, frac))
                                role = "main" if i == main_i else "side"

                                # Candidate retrieval for this dish description
                                hits = retr.retrieve_for_text(desc, 20)
                                ids = [mid for mid, _ in hits]
                                # Build candidates from in-memory db frame (avoid extra DB round trips)
                                cand_rows: List[Dict[str, Any]] = []
                                candidates_df_i_rows: List[Dict[str, Any]] = []
                                for mid, sc in hits:
                                    rr = db_by_id.get(int(mid))
                                    if not rr:
                                        continue
                                    rr["score"] = float(max(0.0, min(1.0, sc))) * 100.0
                                    candidates_df_i_rows.append(rr)
                                    cand_rows.append(
                                        {
                                            "meal_id": int(mid),
                                            "id": int(mid),
                                            "name": str(rr.get("name", "")),
                                            "description": _candidate_description(pd.Series(rr)),
                                            "retrieval_score": float(sc),
                                        }
                                    )
                                candidates_df_i = pd.DataFrame(candidates_df_i_rows)

                                with st.spinner(f"ИИ выбирает блюдо для: {desc[:40]}…"):
                                    picks = reasoner.select_from_candidates(jpeg, cand_rows, slot_hint=desc)
                                if not picks:
                                    sel = {
                                        "selected_meal_id": None,
                                        "selected_meal_name": "unknown",
                                        "confidence": 0.0,
                                        "visible_ingredients": [],
                                        "reasoning": f"No candidate matches this food: {desc}",
                                        "role": role,
                                    }
                                    unk = True
                                    row = None
                                    gname = "unknown"
                                    mru = ""
                                    men = ""
                                else:
                                    p0 = picks[0]
                                    mid = int(p0["meal_id"])
                                    sel = {
                                        "selected_meal_id": mid,
                                        "selected_meal_name": str(p0.get("meal_name", "")),
                                        "confidence": float(p0.get("confidence", 0) or 0),
                                        "visible_ingredients": [],
                                        "reasoning": f"Slot hint: {desc}",
                                        "role": role,
                                    }
                                    unk = False
                                    row = None
                                    if not candidates_df_i.empty:
                                        row = candidates_df_i[candidates_df_i["id"] == mid].head(1)
                                        row = row.iloc[0] if not row.empty else None
                                    if row is None:
                                        # Fallback: look up in db directly
                                        m = db[db["id"] == mid] if "id" in db.columns else pd.DataFrame()
                                        row = m.iloc[0] if not m.empty else None
                                    if row is None:
                                        unk = True
                                        gname = "unknown"
                                        mru = ""
                                        men = ""
                                    else:
                                        row = pd.Series(row)
                                        gname = str(row.get("name", "") or sel["selected_meal_name"])
                                        mru = str(row.get("name", ""))
                                        men = str(row.get("name_en", "") or "")

                                portion_i = float(user_portion) * frac
                                if portion_i <= 0:
                                    portion_i = 0.0

                                verified_i, vdetail_i = is_verified_smart("", gname, mru, men)
                                per_dish_details.append(vdetail_i)

                                rd = row.to_dict() if isinstance(row, pd.Series) else None
                                db_conf = 0.0
                                if isinstance(row, pd.Series):
                                    try:
                                        db_conf = float(row.get("score", 0) or 0) / 100.0
                                    except Exception:
                                        db_conf = 0.0
                                meal_items.append(
                                    {
                                        "role": role,
                                        "gemini_name": gname,
                                        "ai_confidence": float(sel.get("confidence", 0) or 0),
                                        "db_confidence": db_conf,
                                        "gemini_portion": 0.0,
                                        "user_portion_allocated": portion_i,
                                        "matched_name": mru,
                                        "matched_en": men,
                                        "verified": verified_i,
                                        "unknown": unk,
                                        "nutrition_row_dict": rd,
                                        "selection_detail": {
                                            "reasoning": str(sel.get("reasoning", "") or ""),
                                            "visible_ingredients": [],
                                            "ingredient_verify": None,
                                        },
                                        "hybrid_records": candidates_df_i.to_dict("records")
                                        if not candidates_df_i.empty
                                        else [],
                                    }
                                )

                                # Keep candidates of main dish for the bottom list
                                if i == main_i:
                                    candidates_df = candidates_df_i

                            any_known = any(not x.get("unknown") for x in meal_items)
                            verified_overall = any_known and all(
                                bool(x.get("verified")) for x in meal_items if not x.get("unknown")
                            )

                            gemini_name = " · ".join(
                                str(x.get("gemini_name") or "")
                                for x in meal_items
                                if str(x.get("gemini_name") or "").strip() and x.get("gemini_name") != "unknown"
                            ) or "unknown"
                            gemini_portion = float(user_portion)

                            matched_name = " · ".join(
                                str(x.get("matched_name") or "")
                                for x in meal_items
                                if str(x.get("matched_name") or "").strip()
                            )
                            matched_en = " · ".join(
                                str(x.get("matched_en") or "")
                                for x in meal_items
                                if str(x.get("matched_en") or "").strip()
                            )

                            hybrid_df, algo_json = engine.algorithm_bundle(
                                str(candidates_df.iloc[0].get("name", "блюдо")) if not candidates_df.empty else "блюдо"
                            )
                            algo_json = {
                                **algo_json,
                                "multi_plate": len(meal_items) >= 2,
                                "candidate_source": "clip_faiss_per_dish",
                                "photo_only": True,
                                "plate_layout": dishes_layout,
                                "per_dish_verification": per_dish_details,
                            }
                            algo_json["meal_items"] = [
                                {
                                    "role": x["role"],
                                    "gemini_name": x["gemini_name"],
                                    "ai_confidence": float(x.get("ai_confidence", 0) or 0),
                                    "db_confidence": float(x.get("db_confidence", 0) or 0),
                                    "user_portion_allocated": x["user_portion_allocated"],
                                    "matched_name": x["matched_name"],
                                    "matched_en": x["matched_en"],
                                    "verified": x["verified"],
                                    "unknown": x["unknown"],
                                    "reasoning": (x.get("selection_detail") or {}).get("reasoning", ""),
                                }
                                for x in meal_items
                            ]

                            # calories from DB rows per dish (user portion split)
                            ku_acc = 0.0
                            for x in meal_items:
                                rd = x.get("nutrition_row_dict")
                                if not isinstance(rd, dict) or not rd:
                                    continue
                                ku_acc += float(
                                    nutrition_for_row(pd.Series(rd), float(x.get("user_portion_allocated") or 0))[
                                        "kilocalories"
                                    ]
                                )
                            ku: Optional[float] = ku_acc if any_known else None
                            kg: Optional[float] = None

                            saved = save_result_row(
                                user_name=st.session_state.user_name,
                                user_dish_name=_user_dish_saved,
                                user_portion=float(user_portion),
                                gemini_dish_name=gemini_name[:512],
                                gemini_portion=float(gemini_portion),
                                matched_db_dish=matched_name[:512],
                                matched_db_name_en=matched_en[:512],
                                algorithm_results=algo_json,
                                verification_status=verified_overall,
                                verification_detail={
                                    "multi_plate": len(meal_items) >= 2,
                                    "per_dish": per_dish_details,
                                    "verified_overall": verified_overall,
                                },
                                image_jpeg=jpeg,
                                user_id=st.session_state.get("user_id"),
                                kcal_user_portion=ku,
                                kcal_gemini_portion=kg,
                            )
                            if saved:
                                st.caption("Сохранено в истории.")
                            elif os.environ.get("DATABASE_URL"):
                                st.caption("Не удалось сохранить в историю (ошибка БД).")

                            dishes: List[Dict[str, Any]] = [
                                {
                                    "dish_name": str(x.get("gemini_name") or "unknown"),
                                    "portion_grams": float(x.get("user_portion_allocated") or 0),
                                    "confidence": 0.0,
                                }
                                for x in meal_items
                            ]
                            primary = dishes[0] if dishes else {
                                "dish_name": "unknown",
                                "portion_grams": 0.0,
                                "confidence": 0.0,
                            }

                            st.session_state.meal_result = {
                                "verified": verified_overall,
                                "user_dish": _user_dish_saved,
                                "user_portion": float(user_portion),
                                "gemini_name": gemini_name,
                                "gemini_portion": float(gemini_portion),
                                "hybrid_records": candidates_df.to_dict("records") if not candidates_df.empty else [],
                                "dishes": dishes,
                                "primary": primary,
                                "algo_json": algo_json,
                                "constrained_visual_pick": any_known,
                                "meal_items": meal_items,
                                "multi_plate": len(meal_items) >= 2,
                                "photo_only": True,
                            }
                        except Exception as e:
                            st.error(
                                "Ошибка фото-режима CLIP+FAISS (по блюдам). "
                                f"Проверьте зависимости и индекс. Деталь: {e}"
                            )
                    elif candidates_df.empty:
                        if use_clip:
                            st.error(
                                "CLIP+FAISS не вернул кандидатов (проверьте индекс и совпадение id в базе). "
                                "Либо укажите название блюда в блоке выше."
                            )
                        else:
                            st.error(
                                "Нет кандидатов: включите CLIP-индекс на сервере **или** введите название блюда "
                                "в необязательном поле."
                            )
                    else:
                        _user_dish_saved = user_dish.strip() or "(только фото)"
                        with st.spinner("ИИ сравнивает фото со списком из базы…"):
                            agent = GeminiMealAgent(api_key)
                            result = agent.select_meals_from_candidates(jpeg, candidates_df)

                        if result.get("error"):
                            st.error(f"Ошибка модели: {result['error']}")
                            if result.get("raw"):
                                st.code(str(result.get("raw", ""))[:1200])
                        else:
                            meals_raw: List[Dict[str, Any]] = list(result.get("meals") or [])
                            slots: List[Dict[str, Any]] = []
                            for sel in meals_raw:
                                unk, row, gname, mru, men = resolve_meal_selection_slot(
                                    candidates_df, sel
                                )
                                vis_list = sel.get("visible_ingredients")
                                if not isinstance(vis_list, list):
                                    vis_list = []
                                vis_list = [str(x) for x in vis_list if str(x).strip()]
                                slots.append(
                                    {
                                        "sel": sel,
                                        "unknown": unk,
                                        "row": row,
                                        "gemini_name": gname,
                                        "matched_name": mru,
                                        "matched_en": men,
                                        "visible_ingredients": vis_list,
                                        "role": str(sel.get("role") or "other"),
                                    }
                                )

                            labels = [s["gemini_name"] for s in slots if not s["unknown"]]
                            p_est_multi: Dict[str, Any] = {}
                            if labels:
                                with st.spinner("Оценка порций по фото (все блюда)…"):
                                    p_est_multi = agent.estimate_multi_portions_jpeg(jpeg, labels)
                                plist = list(p_est_multi.get("portion_grams_list") or [])
                                if len(plist) != len(labels) or p_est_multi.get("error"):
                                    plist = []
                                    for lbl in labels:
                                        one = agent.estimate_portion_jpeg(jpeg, lbl)
                                        plist.append(
                                            float(one.get("portion_grams", 0) or 0) or 250.0
                                        )
                                for i, g in enumerate(plist):
                                    if g <= 0:
                                        plist[i] = 250.0
                            else:
                                plist = []

                            ki = 0
                            known_ports: List[float] = []
                            for s in slots:
                                if s["unknown"]:
                                    s["gemini_portion"] = 0.0
                                else:
                                    g = float(plist[ki]) if ki < len(plist) else 250.0
                                    s["gemini_portion"] = g
                                    known_ports.append(g)
                                    ki += 1

                            user_allocs = allocate_user_portions_by_ai_weights(
                                float(user_portion), known_ports
                            )
                            ai = 0
                            for s in slots:
                                if s["unknown"]:
                                    s["user_portion_allocated"] = 0.0
                                else:
                                    s["user_portion_allocated"] = (
                                        user_allocs[ai] if ai < len(user_allocs) else 0.0
                                    )
                                    ai += 1

                            for s in slots:
                                s["ingredient_verify"] = None
                                if not s["unknown"] and s["row"] is not None:
                                    ing_exp = _candidate_description(s["row"])
                                    with st.spinner(
                                        f"Проверка описания: "
                                        f"{(s.get('gemini_name') or '…')[:40]}…"
                                    ):
                                        s["ingredient_verify"] = (
                                            agent.verify_ingredient_consistency(
                                                str(s["row"].get("name", "")),
                                                str(s["row"].get("name_en", "") or ""),
                                                ing_exp,
                                                s["visible_ingredients"],
                                            )
                                        )

                            verified_each: List[bool] = []
                            vdetails_each: List[Dict[str, Any]] = []
                            for s in slots:
                                v_ok, v_d = is_verified_smart(
                                    user_dish.strip(),
                                    s["gemini_name"],
                                    s["matched_name"],
                                    s["matched_en"],
                                )
                                verified_each.append(v_ok)
                                vdetails_each.append(v_d)

                            any_known = any(not s["unknown"] for s in slots)
                            verified_overall = (
                                any_known
                                and all(
                                    verified_each[i]
                                    for i, s in enumerate(slots)
                                    if not s["unknown"]
                                )
                            )
                            vdetail = {
                                "multi_plate": len(slots) >= 2,
                                "per_dish": vdetails_each,
                                "verified_overall": verified_overall,
                            }

                            _bundle_seed = user_dish.strip()
                            if not _bundle_seed and not candidates_df.empty:
                                _bundle_seed = str(
                                    candidates_df.iloc[0].get("name", "блюдо") or "блюдо"
                                )
                            if not _bundle_seed:
                                _bundle_seed = "блюдо"
                            hybrid_df, algo_json = engine.algorithm_bundle(_bundle_seed)
                            algo_json = {
                                **algo_json,
                                "visual_selection_raw": result.get("raw"),
                                "candidate_pool_size": int(len(candidates_df)),
                                "portion_multi": p_est_multi,
                                "multi_plate": len(slots) >= 2,
                                "candidate_source": "clip_faiss" if use_clip else "text_seed",
                                "photo_only": not bool(user_dish.strip()),
                            }

                            names_join = " · ".join(
                                s["gemini_name"]
                                for s in slots
                                if s["gemini_name"] and s["gemini_name"] != "unknown"
                            )
                            gemini_name = names_join if names_join else "unknown"
                            gemini_portion = sum(float(s["gemini_portion"] or 0) for s in slots)
                            matched_name = " · ".join(
                                s["matched_name"] for s in slots if s["matched_name"]
                            )
                            matched_en = " · ".join(
                                s["matched_en"] for s in slots if s["matched_en"]
                            )

                            ku_acc = 0.0
                            kg_acc = 0.0
                            for s in slots:
                                if s["unknown"] or s["row"] is None:
                                    continue
                                ku_acc += float(
                                    nutrition_for_row(
                                        s["row"], float(s["user_portion_allocated"])
                                    )["kilocalories"]
                                )
                                if float(s["gemini_portion"]) > 0:
                                    kg_acc += float(
                                        nutrition_for_row(
                                            s["row"], float(s["gemini_portion"])
                                        )["kilocalories"]
                                    )
                            ku: Optional[float] = ku_acc if any_known else None
                            kg: Optional[float] = kg_acc if any_known else None

                            primary_pick: Optional[pd.Series] = None
                            for s in slots:
                                if s["unknown"]:
                                    continue
                                if str(s.get("role") or "") == "main":
                                    primary_pick = s["row"]
                                    break
                            if primary_pick is None:
                                for s in slots:
                                    if not s["unknown"] and s["row"] is not None:
                                        primary_pick = s["row"]
                                        break

                            hybrid_records = reorder_candidate_records(
                                candidates_df, primary_pick
                            )

                            meal_items: List[Dict[str, Any]] = []
                            for j, s in enumerate(slots):
                                rd = (
                                    s["row"].to_dict()
                                    if s["row"] is not None
                                    else None
                                )
                                meal_items.append(
                                    {
                                        "role": s["role"],
                                        "gemini_name": s["gemini_name"],
                                    "ai_confidence": float(s["sel"].get("confidence", 0) or 0),
                                    "db_confidence": float(s["row"].get("score", 0) or 0) / 100.0
                                    if s["row"] is not None
                                    else 0.0,
                                        "gemini_portion": float(s["gemini_portion"] or 0),
                                        "user_portion_allocated": float(
                                            s["user_portion_allocated"] or 0
                                        ),
                                        "matched_name": s["matched_name"],
                                        "matched_en": s["matched_en"],
                                        "verified": verified_each[j],
                                        "unknown": s["unknown"],
                                        "nutrition_row_dict": rd,
                                        "selection_detail": {
                                            "reasoning": str(s["sel"].get("reasoning", "") or ""),
                                            "visible_ingredients": s["visible_ingredients"],
                                            "ingredient_verify": s.get("ingredient_verify"),
                                        },
                                        "hybrid_records": reorder_candidate_records(
                                            candidates_df,
                                            s["row"] if not s["unknown"] else None,
                                        ),
                                    }
                                )

                            algo_json["meal_items"] = [
                                {
                                    "role": x["role"],
                                    "gemini_name": x["gemini_name"],
                                    "ai_confidence": float(x.get("ai_confidence", 0) or 0),
                                    "db_confidence": float(x.get("db_confidence", 0) or 0),
                                    "gemini_portion": x["gemini_portion"],
                                    "user_portion_allocated": x["user_portion_allocated"],
                                    "matched_name": x["matched_name"],
                                    "matched_en": x["matched_en"],
                                    "verified": x["verified"],
                                    "unknown": x["unknown"],
                                    "reasoning": (x.get("selection_detail") or {}).get(
                                        "reasoning", ""
                                    ),
                                }
                                for x in meal_items
                            ]

                            saved = save_result_row(
                                user_name=st.session_state.user_name,
                                user_dish_name=_user_dish_saved,
                                user_portion=float(user_portion),
                                gemini_dish_name=gemini_name[:512],
                                gemini_portion=float(gemini_portion),
                                matched_db_dish=matched_name[:512],
                                matched_db_name_en=matched_en[:512],
                                algorithm_results=algo_json,
                                verification_status=verified_overall,
                                verification_detail=vdetail,
                                image_jpeg=jpeg,
                                user_id=st.session_state.get("user_id"),
                                kcal_user_portion=ku,
                                kcal_gemini_portion=kg,
                            )
                            if saved:
                                st.caption("Сохранено в истории.")
                            elif os.environ.get("DATABASE_URL"):
                                st.caption("Не удалось сохранить в историю (ошибка БД).")

                            dishes: List[Dict[str, Any]] = [
                                {
                                    "dish_name": s["gemini_name"],
                                    "portion_grams": float(s["gemini_portion"] or 0),
                                    "confidence": float(s["sel"].get("confidence", 0) or 0),
                                }
                                for s in slots
                            ]
                            primary = dishes[0] if dishes else {
                                "dish_name": "unknown",
                                "portion_grams": 0.0,
                                "confidence": 0.0,
                            }

                            multi_ui = len(slots) >= 2
                            if multi_ui:
                                st.session_state.meal_result = {
                                    "verified": verified_overall,
                                    "user_dish": _user_dish_saved,
                                    "user_portion": float(user_portion),
                                    "gemini_name": gemini_name,
                                    "gemini_portion": float(gemini_portion),
                                    "hybrid_records": hybrid_records,
                                    "dishes": dishes,
                                    "primary": primary,
                                    "algo_json": algo_json,
                                    "constrained_visual_pick": any_known,
                                    "meal_items": meal_items,
                                    "multi_plate": True,
                                    "photo_only": not bool(user_dish.strip()),
                                }
                            else:
                                s0 = slots[0]
                                sel0 = s0["sel"]
                                st.session_state.meal_result = {
                                    "verified": verified_each[0],
                                    "user_dish": _user_dish_saved,
                                    "user_portion": float(user_portion),
                                    "gemini_name": s0["gemini_name"],
                                    "gemini_portion": float(s0["gemini_portion"] or 0),
                                    "hybrid_records": meal_items[0]["hybrid_records"]
                                    if meal_items
                                    else hybrid_records,
                                    "dishes": dishes,
                                    "primary": primary,
                                    "algo_json": algo_json,
                                    "constrained_visual_pick": not s0["unknown"],
                                    "selection_detail": {
                                        "reasoning": str(sel0.get("reasoning", "") or ""),
                                        "visible_ingredients": s0["visible_ingredients"],
                                        "ingredient_verify": s0.get("ingredient_verify"),
                                    },
                                    "photo_only": not bool(user_dish.strip()),
                                }

                            if not any_known:
                                st.warning(
                                    "ИИ не сопоставил фото ни с одним кандидатом из базы. "
                                    "Попробуйте другой ракурс"
                                    + (" или укажите название в необязательном поле." if not use_clip else ".")
                                )

    res = st.session_state.get("meal_result")
    if res is not None:
        render_meal_result(res)
        if st.button("Скрыть результат", type="secondary", key="dismiss_res"):
            st.session_state.pop("meal_result", None)
            st.rerun()


if __name__ == "__main__":
    main()
