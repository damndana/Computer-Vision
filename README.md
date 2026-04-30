# Nutritionist-Food-Recognition-Gemini-Pro
Nutritionist-FoodRecognition is an intelligent food recognition system. Analyze images of dishes, identify ingredients, and receive nutritional information.

## CLIP + FAISS + Gemini API (no dish text from user)

1. `pip install -r requirements-api.txt`
2. Set `DATABASE_URL` and `GOOGLE_GEMINI_API_KEY`.
3. Build the index: `python -m meal_pipeline.embedding_generator` (writes `data/meals_faiss.index` + `data/meal_faiss_ids.npy`).
4. Run: `uvicorn meal_pipeline.api:app --host 0.0.0.0 --port 8080`
5. `POST /analyze_meal` with multipart `image` + form field `portion_grams`.

Modules live under `meal_pipeline/` (`embedding_generator.py`, `vector_index.py`, `meal_retriever.py`, `gemini_reasoner.py`, `multi_meal_detector.py`, `nutrition_calculator.py`, `api.py`).

The **Streamlit** main page uses the same `data/*.index` + `data/*.npy` when present: then **название блюда не обязательно** (кандидаты из CLIP+FAISS).

## Documentation

- Integration & architecture guide: `docs/INTEGRATION.md`

**Docker (Dokploy):**

- The app expects CLIP+FAISS artifacts under `MEAL_PIPELINE_DATA_DIR` (default `/app/data`):
  - `meals_faiss.index`
  - `meal_faiss_ids.npy`
- **Production recommendation**: mount a **persistent volume** on `/app/data` so redeploys don’t rebuild / re-download models.
- Startup build behavior is controlled by:
  - `MEAL_PIPELINE_BUILD_INDEX_ON_STARTUP` (default `0`)
    - `0`: start Streamlit immediately; **skip** FAISS build (photo-only mode disabled until artifacts exist)
    - `1`: start Streamlit immediately; build FAISS **in background** (may take minutes on first run)
- Model download cache is stored under `${MEAL_PIPELINE_DATA_DIR}/hf_cache` by default (can be overridden by `HF_HOME`).

If Gemini returns a 404 about a deprecated model, set `GEMINI_MODEL` in the environment (default is `gemini-2.5-flash`).

## Dockerfiles

- `Dockerfile`: **Streamlit UI** (default; serves the website on port `8501`)
- `Dockerfile.api`: **FastAPI API-only** image (serves API on port `8080`)
