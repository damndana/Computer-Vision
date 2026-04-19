#!/bin/sh
# Dokploy / Docker: optional CLIP+FAISS build at startup, then Streamlit.
# Logs → stdout (Dokploy).
set -e

PORT="${PORT:-8501}"
DATA_DIR="${MEAL_PIPELINE_DATA_DIR:-/app/data}"
export MEAL_PIPELINE_DATA_DIR="${DATA_DIR}"

mkdir -p "${DATA_DIR}"

IDX="${DATA_DIR}/meals_faiss.index"
IDS="${DATA_DIR}/meal_faiss_ids.npy"

echo "============================================================"
echo "[nutristeppe] $(date -u +"%Y-%m-%dT%H:%M:%SZ") container start"
echo "[nutristeppe] PORT=${PORT}"
echo "[nutristeppe] MEAL_PIPELINE_DATA_DIR=${DATA_DIR}"
echo "[nutristeppe] PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
if [ -n "${DATABASE_URL:-}" ]; then
  echo "[nutristeppe] DATABASE_URL is set (hidden)"
else
  echo "[nutristeppe] WARNING: DATABASE_URL is unset — DB + FAISS build skipped"
fi
if [ -n "${GOOGLE_GEMINI_API_KEY:-}" ] || [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
  echo "[nutristeppe] Gemini API key env is set (hidden)"
else
  echo "[nutristeppe] WARNING: no GOOGLE_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY — analyze will fail"
fi
echo "============================================================"

if [ -f "${IDX}" ] && [ -f "${IDS}" ]; then
  echo "[nutristeppe] FAISS index already present — skipping embedding build"
  echo "[nutristeppe]   ${IDX}"
  echo "[nutristeppe]   ${IDS}"
else
  if [ -z "${DATABASE_URL:-}" ]; then
    echo "[nutristeppe] WARNING: cannot build FAISS index (no DATABASE_URL). Photo-only mode disabled until index exists."
  else
    echo "[nutristeppe] Building CLIP + FAISS index from PostgreSQL (first start may take several minutes)…"
    python -u -m meal_pipeline.embedding_generator
    echo "[nutristeppe] Embedding build finished."
  fi
fi

echo "============================================================"
echo "[nutristeppe] $(date -u +"%Y-%m-%dT%H:%M:%SZ") Streamlit starting"
echo "[nutristeppe] If UI stays on grey skeleton: DevTools → Network → WS → /_stcore/stream must be 101."
echo "============================================================"

exec python -u -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
