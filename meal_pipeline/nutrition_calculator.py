"""
Nutrition from DB row + portion (grams).

Nutristeppe rows: kilocalories, protein, fat, carbohydrate are totals for
reference serving_size_g (g). Macro nutrients are stored in milligrams in DB.
"""
from __future__ import annotations

from typing import Any, Dict


def compute_nutrition_for_portion(row: Dict[str, Any], portion_grams: float) -> Dict[str, float]:
    """
    Scale reference row to user portion.

    Returns calories (kcal), protein/fat/carbs in grams for that portion.
    """
    ref_g = float(row.get("serving_size_g", 100) or 100)
    if ref_g <= 0:
        ref_g = 100.0
    ratio = float(portion_grams) / ref_g

    kcal = float(row.get("kilocalories", 0) or 0) * ratio
    # DB stores protein/fat/carbohydrate in mg for the reference serving
    protein_g = float(row.get("protein", 0) or 0) * ratio / 1000.0
    fat_g = float(row.get("fat", 0) or 0) * ratio / 1000.0
    carb_g = float(row.get("carbohydrate", 0) or 0) * ratio / 1000.0

    return {
        "calories": round(kcal, 1),
        "protein": round(protein_g, 2),
        "fat": round(fat_g, 2),
        "carbs": round(carb_g, 2),
    }


def per_100g_snapshot(row: Dict[str, Any]) -> Dict[str, float]:
    """Optional diagnostics: kcal and g macros per 100 g of reference profile."""
    ref_g = float(row.get("serving_size_g", 100) or 100)
    if ref_g <= 0:
        ref_g = 100.0
    scale = 100.0 / ref_g
    return {
        "calories_per_100g": round(float(row.get("kilocalories", 0) or 0) * scale, 2),
        "protein_per_100g": round(float(row.get("protein", 0) or 0) * scale / 1000.0, 3),
        "fat_per_100g": round(float(row.get("fat", 0) or 0) * scale / 1000.0, 3),
        "carbs_per_100g": round(float(row.get("carbohydrate", 0) or 0) * scale / 1000.0, 3),
    }
