"""
Step 1–2: Gemini decides how many distinct foods are visible and short labels.

Used to route single CLIP-image retrieval vs per-dish CLIP-text retrieval.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlateLayout:
    dish_count: int
    dishes: List[Dict[str, Any]]  # each: description, fraction (0-1)


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
            temperature=0.15,
            response_mime_type="application/json",
        )
    except Exception:
        try:
            from google.generativeai.types import GenerationConfig

            return GenerationConfig(
                temperature=0.15,
                response_mime_type="application/json",
            )
        except Exception:
            return {"temperature": 0.15, "response_mime_type": "application/json"}


class MultiMealDetector:
    def __init__(self, api_key: str, model_name: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def analyze_plate(self, jpeg_bytes: bytes) -> PlateLayout:
        prompt = """You analyze a food photo.

Return ONLY valid JSON:
{
  "dish_count": <integer number of visually separate foods on the plate>,
  "dishes": [
    {
      "name": "<short dish name (prefer Russian if obvious; else English)>",
      "confidence": <number 0-1: how confident you are this name is correct>,
      "description": "<short English phrase: what this food is, e.g. grilled chicken breast>",
      "fraction": <number 0-1: approximate share of total food area/mass on the plate>
    }
  ]
}

Rules:
- If there is only one cohesive meal, dish_count=1 and one dish with fraction 1.0.
- If several sides, list each separately.
- fractions should sum to approximately 1.0 (±0.15).
- descriptions must be concrete food names, not utensils or table.
- confidence is your best estimate from the image only (not based on any database).
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
            logger.warning("multi_meal_detector: bad JSON, defaulting to single dish")
            return PlateLayout(1, [{"description": "meal on plate", "fraction": 1.0}])

        dishes = obj.get("dishes")
        if not isinstance(dishes, list) or not dishes:
            return PlateLayout(1, [{"description": "meal on plate", "fraction": 1.0}])

        cleaned: List[Dict[str, Any]] = []
        for d in dishes[:10]:
            if not isinstance(d, dict):
                continue
            name = str(d.get("name", "")).strip()
            desc = str(d.get("description", "")).strip() or "food item"
            try:
                conf = float(d.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                conf = 0.0
            try:
                frac = float(d.get("fraction", 1.0 / max(len(dishes), 1)))
            except (TypeError, ValueError):
                frac = 1.0 / max(len(dishes), 1)
            cleaned.append(
                {
                    "name": name or desc,
                    "confidence": max(0.0, min(1.0, conf)),
                    "description": desc,
                    "fraction": max(0.05, min(1.0, frac)),
                }
            )

        if not cleaned:
            return PlateLayout(1, [{"description": "meal on plate", "fraction": 1.0}])

        s = sum(float(x["fraction"]) for x in cleaned) or 1.0
        for x in cleaned:
            x["fraction"] = float(x["fraction"]) / s

        return PlateLayout(dish_count=len(cleaned), dishes=cleaned)
