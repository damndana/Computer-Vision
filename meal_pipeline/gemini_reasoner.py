"""
Gemini Flash: final dish choice constrained to FAISS-retrieved candidates.

Supports optional slot hint for multi-plate per-dish reasoning.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        o = json.loads(t)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", t)
        if not m:
            return None
        try:
            o = json.loads(m.group())
            return o if isinstance(o, dict) else None
        except json.JSONDecodeError:
            return None


def _gemini_json_config():
    import google.generativeai as genai

    try:
        return genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
        )
    except Exception:
        try:
            from google.generativeai.types import GenerationConfig

            return GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        except Exception:
            return {"temperature": 0.2, "response_mime_type": "application/json"}


def _build_candidate_block(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        mid = int(r.get("meal_id", r.get("id", -1)))
        name = str(r.get("name", ""))
        desc = str(r.get("description", r.get("details", "")))[:400]
        lines.append(f"{mid} | {name}\n  details: {desc}")
    return "\n".join(lines)


class GeminiReasoner:
    def __init__(self, api_key: str, model_name: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def select_from_candidates(
        self,
        jpeg_bytes: bytes,
        candidates: List[Dict[str, Any]],
        slot_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        candidates: list of dicts with keys meal_id, name, description (rich text snippet).

        Returns list of {meal_id, meal_name, confidence} (usually length 1; may be multiple
        if the model returns several — caller should take first or validate).
        """
        if not candidates:
            return []
        block = _build_candidate_block(candidates)
        hint = ""
        if slot_hint:
            hint = (
                f"\nFocus on this part of the plate only: \"{slot_hint}\". "
                "Pick the candidate that best matches THAT food, not the whole image.\n"
            )
        prompt = f"""You are a food recognition AI.

You are given:
1. A photo of a meal
2. A list of candidate dishes retrieved from a database

Your job is to choose the most visually accurate meal from the candidate list.
{hint}
Rules:

* You must ONLY select dishes from the candidate list (use their meal_id and exact name).
* Do NOT invent new meals.
* If multiple foods appear in the image but this list targets one food, return ONE dish.
* If unsure, choose the closest visual match.

Each candidate line format: meal_id | name

Output JSON format ONLY:
{{
  "dishes":[
    {{"meal_id": <integer>, "meal_name":"<string>", "confidence": <0.0-1.0>}}
  ]
}}

Candidate dishes:
{block}
"""
        resp = self._model.generate_content(
            [
                prompt,
                {"mime_type": "image/jpeg", "data": jpeg_bytes},
            ],
            generation_config=_gemini_json_config(),
        )
        text = getattr(resp, "text", None) or ""
        obj = _parse_json_obj(text)
        if not obj:
            logger.warning("gemini_reasoner: invalid JSON")
            return []
        dishes = obj.get("dishes")
        if not isinstance(dishes, list):
            return []
        out: List[Dict[str, Any]] = []
        allowed_ids = {int(c.get("meal_id", c.get("id", -1))) for c in candidates}
        for d in dishes:
            if not isinstance(d, dict):
                continue
            try:
                mid = int(d.get("meal_id", -1))
            except (TypeError, ValueError):
                continue
            if mid not in allowed_ids:
                continue
            name = str(d.get("meal_name", "")).strip()
            conf = float(d.get("confidence", 0) or 0)
            out.append({"meal_id": mid, "meal_name": name, "confidence": conf})
        return out
