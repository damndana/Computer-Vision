## Nutristeppe Computer Vision — Integration Documentation

This repository contains a **meal photo analysis pipeline** that combines:
- **Gemini** (LLM vision) for *image-only dish detection / dish list* (and AI confidence)
- **CLIP (SentenceTransformers)** + **FAISS** for *fast database matching* (and similarity/classification-style score)
- **PostgreSQL** (Nutristeppe meals table + results/history table)
- **Streamlit** UI (human-facing)
- **FastAPI** service (machine-facing, for integration into other apps)

The project is designed so you can run it as:
- **A standalone web UI** (Streamlit), and/or
- **An API service** (FastAPI) called from Nutristeppe or any other product.

---

## Tech stack

### Core ML / Retrieval
- **Gemini**: `google-generativeai`
  - Used in `meal_pipeline/multi_meal_detector.py` to detect dish(es) from a photo.
  - Used in `meal_pipeline/gemini_reasoner.py` (optional path) to pick from DB candidates (slower; can be disabled in UI flow).
- **CLIP embeddings** via **SentenceTransformers**: `sentence-transformers`
  - Model default: `sentence-transformers/clip-ViT-B-32` (configurable via env).
- **FAISS**: `faiss-cpu`
  - Stores normalized embeddings and supports top-k cosine similarity search.

### Backend / UI
- **Streamlit** (UI): `app.py` and `pages/`
- **FastAPI** (API): `meal_pipeline/api.py`
- **Uvicorn**: `uvicorn`

### Data / DB
- **PostgreSQL**: `psycopg2-binary`
  - Meals source table (default): `database_nutristeppe`
  - Results/history table (default): `results`
- **Pandas**: used in UI / some DB loading utilities
- **Pillow**: image decoding/encoding

### Matching / text utilities
- **fuzzywuzzy** + **python-Levenshtein**: text similarity (used in verification utilities)

---

## Repository structure (high level)

- `app.py`: Streamlit “Проверка блюда” main page
- `pages/2_User_Meals.py`: Streamlit “Мои приёмы пищи” history page
- `database.py`: Postgres helpers for meals + results/history
- `meal_pipeline/` (pipeline modules)
  - `config.py`: environment-driven config (paths, model names, etc.)
  - `embedding_generator.py`: offline job to build CLIP text embeddings + FAISS index from DB
  - `meal_retriever.py`: CLIP encode (image/text) + FAISS retrieval
  - `vector_index.py`: FAISS helpers
  - `multi_meal_detector.py`: Gemini plate/dishes detector (AI dish name + AI confidence)
  - `nutrition_calculator.py`: calories/macros scaling from DB row to portion grams
  - `api.py`: FastAPI endpoint(s)
- `scripts/docker-entrypoint.sh`: Dokploy/Docker startup logic
- `sql/`: SQL helpers (optional)

---

## What the pipeline does (conceptually)

### A) Offline step (one-time / periodic): build FAISS index
**Goal**: make DB search fast (milliseconds).

Command:

```bash
python -m meal_pipeline.embedding_generator
```

What it does:
- Loads all meals from PostgreSQL
- Builds a rich text string per meal (name + name_en + ingredients + steps)
- Encodes texts using CLIP text encoder
- L2-normalizes vectors and builds a FAISS IndexFlatIP (cosine similarity)
- Writes 2 artifacts:
  - `meals_faiss.index`
  - `meal_faiss_ids.npy`

These artifacts must be present on the server to enable **photo-only CLIP+FAISS** mode.

### B) Online step: analyze a meal photo

In the current Streamlit photo-only flow, the analysis is:
1) **Gemini** detects dish(es) on the plate:
   - AI dish name (independent of database)
   - AI confidence (0..1)
   - dish fraction allocation (0..1)
2) **CLIP+FAISS** matches each detected dish to the Nutristeppe database:
   - DB top-1 meal name (and optionally `name_en`)
   - DB **similarity**: raw CLIP cosine similarity (FAISS score)
   - DB **classification-style score**: a UI-friendly score derived from retrieval scores (often used as “top-1 confidence”)
3) Calories are computed from the matched DB row and scaled to the user portion grams (and per-dish allocation for multi-dish photos).

---

## Scores explained (important for reports)

### AI confidence (Gemini)
- Comes from Gemini detector output.
- Represents confidence in the *detected dish name* from the image only.

### DB similarity (CLIP cosine)
- The FAISS score (cosine similarity) for top-1 match.
- Measures embedding proximity between the query and DB meal text embedding.
- It is **not** a probability and often sits in ranges like 0.6–0.85 even when the label “looks correct”.

### DB classification-style score
- A UI-friendly score derived from the top-k retrieval distribution (can be implemented as softmax over top-k similarities).
- This behaves more like “top-1 probability” within the retrieved set.

---

## Running locally

### Streamlit UI

```bash
pip install -r requirements.txt
streamlit run app.py
```

For photo-only mode, also install API deps and build the index:

```bash
pip install -r requirements-api.txt
export DATABASE_URL="postgres://..."
export GOOGLE_GEMINI_API_KEY="..."
python -m meal_pipeline.embedding_generator
streamlit run app.py
```

### FastAPI service (for integration)

```bash
pip install -r requirements-api.txt
export DATABASE_URL="postgres://..."
export GOOGLE_GEMINI_API_KEY="..."
python -m meal_pipeline.embedding_generator
uvicorn meal_pipeline.api:app --host 0.0.0.0 --port 8080
```

---

## Deployment (Dokploy / Docker)

### Persistent storage (required)
To avoid rebuilding the FAISS index and re-downloading CLIP on every redeploy:
- Mount a **persistent volume** at:
  - `/app/data`

The service uses:
- `MEAL_PIPELINE_DATA_DIR=/app/data` by default
- FAISS artifacts are expected in `/app/data`
- HuggingFace cache is stored under `/app/data/hf_cache` by default

### Startup behavior
`scripts/docker-entrypoint.sh`:
- Starts Streamlit immediately.
- If FAISS index files are missing, it can:
  - **Skip build** (default) to avoid blocking deploy
  - Or build in the background if enabled

Env var:
- `MEAL_PIPELINE_BUILD_INDEX_ON_STARTUP`:
  - `0` (default): skip build on startup
  - `1`: build in background

### HuggingFace token (recommended)
To avoid rate limits and speed up model downloads:
- Set `HF_TOKEN` in Dokploy env vars (backend service).

### Streamlit watcher
For stability and reduced overhead, the entrypoint runs Streamlit with:
- `--server.fileWatcherType=none`

---

## Environment variables (reference)

### Required (typical production)
- `DATABASE_URL`: Postgres connection string
- `GOOGLE_GEMINI_API_KEY` (or `GEMINI_API_KEY` / `GOOGLE_API_KEY`)

### Strongly recommended
- `MEAL_PIPELINE_DATA_DIR=/app/data` (and mount `/app/data` as a volume)
- `HF_TOKEN`: HuggingFace access token (read-only is enough)

### Models / retrieval
- `CLIP_MODEL_NAME` (default `sentence-transformers/clip-ViT-B-32`)
- `GEMINI_MODEL` (default `gemini-2.5-flash`)
- `TOP_K_RETRIEVAL` (default `20`)

### Tables
- `MEAL_INFO_TABLE` / `NUTRISTEPPE_TABLE` (default `database_nutristeppe`)
- `MEAL_RESULTS_TABLE` / `RESULTS_TABLE` (default `results`)

### Artifact paths (normally derived from `MEAL_PIPELINE_DATA_DIR`)
- `FAISS_INDEX_PATH`
- `MEAL_IDS_PATH`

### Performance / warmup
- `MEAL_PIPELINE_WARMUP_CLIP` (default `1`): warm CLIP model in background on app start
- `MEAL_PIPELINE_BUILD_INDEX_ON_STARTUP` (default `0`): do not block server on deploy

---

## Integration into other projects

### Recommended pattern
Run this repo as a **separate service** and call it from your main app.

Why:
- CV dependencies are heavy (CLIP weights, FAISS, etc.)
- Keeps your main app lean and avoids coupling deployments
- Lets you scale CV independently

### FastAPI contract
The FastAPI entrypoint is `meal_pipeline/api.py`:
- `POST /analyze_meal` (multipart):
  - file field: `image` (jpg/png)
  - form field: `portion_grams` (float)

Response shape may be adjusted depending on your integration needs (single vs multi dish, confidence fields, etc.).

### Suggested response fields for integration
- `ai_detected_name`
- `ai_confidence`
- `db_matched_name`
- `db_score` (classification-style)
- `db_similarity` (cosine)
- `calories`
- `dishes[]` for multi-dish

---

## Performance targets / troubleshooting

### If analysis takes 60–90 seconds
The most common cause is **CLIP cold start** (loading model weights into memory).

Fixes:
- Ensure the server/container is not restarting frequently
- Enable warmup: `MEAL_PIPELINE_WARMUP_CLIP=1`
- Use persistent volume `/app/data` for HF cache and FAISS artifacts

### If deploy returns 502
Most often caused by blocking startup with index build or model download.

Fix:
- Keep `MEAL_PIPELINE_BUILD_INDEX_ON_STARTUP=0`
- Build index once manually and persist `/app/data`

