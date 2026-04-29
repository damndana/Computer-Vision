"""
End-to-end: multi-plate Gemini → CLIP+FAISS per slot → Gemini reasoner → nutrition.

No user-provided dish text; only image + total portion grams.
"""
from __future__ import annotations

import io
import os
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


def _pil_to_jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
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
    detector = MultiMealDetector(api_key, config.GEMINI_MODEL)
    reasoner = GeminiReasoner(api_key, config.GEMINI_MODEL)

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
            print(
                f"[DB retrieval] top='{top_db.get('name','')}' score={float(top_db.get('retrieval_score', 0.0)):.4f}"
            )
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
            print(f"[AI detected] meal='{ai_name}' confidence={ai_conf:.3f}")
            if db_score is not None:
                print(
                    f"[DB matched] meal='{str(row.get('name',''))}' score={float(db_score):.4f}"
                )
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
    for slot in layout.dishes:
        desc = str(slot.get("description", ""))
        frac = float(slot.get("fraction", 1.0 / len(layout.dishes)))
        part_g = max(1.0, float(portion_grams) * frac)
        hits = retriever.retrieve_for_text(desc, k)
        ids = [h[0] for h in hits]
        meals = fetch_meals_by_ids(ids)
        cands = _candidates_for_ids(retriever, hits, meals)
        if cands:
            top_db = cands[0]
            print(
                f"[DB retrieval] slot='{desc}' top='{top_db.get('name','')}' score={float(top_db.get('retrieval_score', 0.0)):.4f}"
            )
        picks = reasoner.select_from_candidates(jpeg, cands, slot_hint=desc)
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
            print(f"[AI detected] slot='{desc}' meal='{ai_name}' confidence={ai_conf:.3f}")
            if db_score is not None:
                print(
                    f"[DB matched] slot='{desc}' meal='{str(row.get('name',''))}' score={float(db_score):.4f}"
                )
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
