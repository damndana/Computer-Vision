# Nutritionist-Food-Recognition-Gemini-Pro
Nutritionist-FoodRecognition is an intelligent food recognition system. Analyze images of dishes, identify ingredients, and receive nutritional information.

## CLIP + FAISS + Gemini API (no dish text from user)

1. `pip install -r requirements-api.txt`
2. Set `DATABASE_URL` and `GOOGLE_GEMINI_API_KEY`.
3. Build the index: `python -m meal_pipeline.embedding_generator` (writes `data/meals_faiss.index` + `data/meal_faiss_ids.npy`).
4. Run: `uvicorn meal_pipeline.api:app --host 0.0.0.0 --port 8080`
5. `POST /analyze_meal` with multipart `image` + form field `portion_grams`.

Modules live under `meal_pipeline/` (`embedding_generator.py`, `vector_index.py`, `meal_retriever.py`, `gemini_reasoner.py`, `multi_meal_detector.py`, `nutrition_calculator.py`, `api.py`).
