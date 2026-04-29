"""
End-to-end: multi-plate Gemini → CLIP+FAISS per slot → Gemini reasoner → nutrition.

No user-provided dish text; only image + total portion grams.
"""
from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from meal_pipeline import config
from meal_pipeline.db_meals import fetch_meals_by_ids, rich_text_for_meal
from meal_pipeline.gemini_reasoner import GeminiReasoner
from meal_pipeline.meal_retriever import MealRetriever
from meal_pipeline.multi_meal_detector import MultiMealDetector
from meal_pipeline.nutrition_calculator import compute_nutrition_for_portion


def _retrieval_score_for_meal_id(
    candidates: List[Dict[str, Any]], meal_id: int
) -> Optional[float]:
    for c in candidates:
        try:
            if int(c.get("meal_id")) == int(meal_id):
                return float(c.get("retrieval_score", 0.0))
        except Exception:
            continue
    return None


def _get_gemini_key() -> str:
    for k in ("GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    raise RuntimeError("Set GOOGLE_GEMINI_API_KEY (or GEMINI_API_KEY) for the pipeline.")


@lru_cache(maxsize=4)
def _get_gemini_clients(api_key: str, model_name: str) -> tuple[MultiMealDetector, GeminiReasoner]:
    # Reuse initialized Gemini model wrappers across requests to reduce per-request overhead.
    return MultiMealDetector(api_key, model_name), GeminiReasoner(api_key, model_name)


def _pil_to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    max_dim = int(getattr(config, "GEMINI_MAX_IMAGE_DIM", 768) or 768)
    q = int(getattr(config, "GEMINI_JPEG_QUALITY", quality) or quality)
    im = img.convert("RGB")
    # Downscale for faster Gemini inference (bandwidth + server-side compute).
    if max(im.size) > max_dim:
        im.thumbnail((max_dim, max_dim), resample=Image.Resampling.LANCZOS)
    im.save(buf, format="JPEG", quality=q, optimize=True)
    return buf.getvalue()


def _candidates_for_ids(
    retriever: MealRetriever, ids_scores: List[Tuple[int, float]], meals: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mid, sc in ids_scores:
        m = meals.get(mid)
        if not m:
            continue
        rows.append(
            {
                "meal_id": mid,
                "id": mid,
                "name": str(m.get("name", "")),
                "description": rich_text_for_meal(m),
                "retrieval_score": float(sc),
            }
        )
    return rows


def analyze_meal(
    image: Image.Image,
    portion_grams: float,
    retriever: MealRetriever,
) -> Dict[str, Any]:
    """
    Returns API-shaped dict: {"dishes": [...]}

    Latency dominated by Gemini (2+N calls in multi mode); CLIP+FAISS are ms-scale on CPU
    once the SentenceTransformer model is warm.
    """
    api_key = _get_gemini_key()
    jpeg = _pil_to_jpeg(image)
    detector, reasoner = _get_gemini_clients(api_key, config.GEMINI_MODEL)

    layout = detector.analyze_plate(jpeg)
    k = config.TOP_K_RETRIEVAL

    dishes_out: List[Dict[str, Any]] = []

    if len(layout.dishes) <= 1:
        hits = retriever.retrieve_for_image(image, k)
        ids = [h[0] for h in hits]
        meals = fetch_meals_by_ids(ids)
        cands = _candidates_for_ids(retriever, hits, meals)
        if cands:
            top_db = cands[0]
            print(f"Meal name from Database (top-1): {str(top_db.get('name',''))}")
            print(f"Database score (top-1): {float(top_db.get('retrieval_score', 0.0)):.4f}")
        picks = reasoner.select_from_candidates(jpeg, cands, slot_hint=None)
        if not picks and cands:
            top = cands[0]
            picks = [
                {
                    "meal_id": int(top["meal_id"]),
                    "meal_name": str(top["name"]),
                    "confidence": 0.35,
                }
            ]
        for p in picks[:1]:
            mid = int(p["meal_id"])
            row = meals.get(mid) or {}
            ai_name = str(p.get("meal_name") or row.get("name", ""))
            ai_conf = float(p.get("confidence", 0) or 0)
            db_score = _retrieval_score_for_meal_id(cands, mid)
            print(f"Meal name from AI: {ai_name}")
            print(f"AI confidence: {ai_conf:.3f}")
            if db_score is not None:
                print(f"Meal name from Database (matched): {str(row.get('name',''))}")
                print(f"Database score (matched): {float(db_score):.4f}")
            nut = compute_nutrition_for_portion(row, float(portion_grams))
            dishes_out.append(
                {
                    "meal_id": mid,
                    "meal_name": str(p.get("meal_name") or row.get("name", "")),
                    "portion_grams": round(float(portion_grams), 1),
                    "calories": nut["calories"],
                    "protein": nut["protein"],
                    "fat": nut["fat"],
                    "carbs": nut["carbs"],
                    "confidence": round(float(p.get("confidence", 0) or 0), 3),
                }
            )
        return {"dishes": dishes_out, "plate_layout": {"dish_count": 1, "dishes": layout.dishes}}

    # Multi: per-dish text retrieval + reasoner
    slot_payloads: List[Dict[str, Any]] = []
    all_ids: List[int] = []
    for i, slot in enumerate(layout.dishes):
        desc = str(slot.get("description", ""))
        frac = float(slot.get("fraction", 1.0 / len(layout.dishes)))
        part_g = max(1.0, float(portion_grams) * frac)
        hits = retriever.retrieve_for_text(desc, k)
        ids = [h[0] for h in hits]
        all_ids.extend(ids)
        # We'll fetch DB rows once for all slots, then build candidates per slot.
        slot_payloads.append(
            {"slot_idx": i, "desc": desc, "part_g": part_g, "hits": hits, "ids": ids}
        )

    meals_all = fetch_meals_by_ids(all_ids)

    # Now build per-slot candidates using the shared DB cache
    for payload in slot_payloads:
        hits = payload["hits"]
        ids = payload["ids"]
        meals = {mid: meals_all.get(mid) for mid in ids if meals_all.get(mid) is not None}
        cands = _candidates_for_ids(retriever, hits, meals)
        if cands:
            top_db = cands[0]
            print(f"Slot hint: {payload['desc']}")
            print(f"Meal name from Database (top-1): {str(top_db.get('name',''))}")
            print(f"Database score (top-1): {float(top_db.get('retrieval_score', 0.0)):.4f}")
        payload["meals"] = meals
        payload["cands"] = cands

    def _reason_one_slot(payload: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
        local_reasoner = GeminiReasoner(api_key, config.GEMINI_MODEL)
        picks_local = local_reasoner.select_from_candidates(
            jpeg, payload["cands"], slot_hint=payload["desc"]
        )
        return int(payload["slot_idx"]), picks_local

    max_workers = int(getattr(config, "GEMINI_PARALLEL_WORKERS", 3) or 3)
    max_workers = max(1, min(max_workers, len(slot_payloads)))
    picks_by_slot: Dict[int, List[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_reason_one_slot, p) for p in slot_payloads]
        for fut in as_completed(futs):
            slot_idx, picks = fut.result()
            picks_by_slot[slot_idx] = picks

    for payload in slot_payloads:
        desc = str(payload["desc"])
        part_g = float(payload["part_g"])
        meals = payload["meals"]
        cands = payload["cands"]
        picks = picks_by_slot.get(int(payload["slot_idx"]), [])

        if not picks and cands:
            top = cands[0]
            picks = [
                {
                    "meal_id": int(top["meal_id"]),
                    "meal_name": str(top["name"]),
                    "confidence": 0.35,
                }
            ]

        for p in picks[:1]:
            mid = int(p["meal_id"])
            row = meals.get(mid) or {}
            ai_name = str(p.get("meal_name") or row.get("name", ""))
            ai_conf = float(p.get("confidence", 0) or 0)
            db_score = _retrieval_score_for_meal_id(cands, mid)
            print(f"Slot hint: {desc}")
            print(f"Meal name from AI: {ai_name}")
            print(f"AI confidence: {ai_conf:.3f}")
            if db_score is not None:
                print(f"Meal name from Database (matched): {str(row.get('name',''))}")
                print(f"Database score (matched): {float(db_score):.4f}")
            nut = compute_nutrition_for_portion(row, part_g)
            dishes_out.append(
                {
                    "meal_id": mid,
                    "meal_name": str(p.get("meal_name") or row.get("name", "")),
                    "portion_grams": round(part_g, 1),
                    "calories": nut["calories"],
                    "protein": nut["protein"],
                    "fat": nut["fat"],
                    "carbs": nut["carbs"],
                    "confidence": round(float(p.get("confidence", 0) or 0), 3),
                }
            )

    return {
        "dishes": dishes_out,
        "plate_layout": {"dish_count": layout.dish_count, "dishes": layout.dishes},
    }
