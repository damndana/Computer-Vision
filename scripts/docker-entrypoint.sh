#!/bin/sh
# All echo goes to stdout → visible in Dokploy / Docker logs.
set -e

PORT="${PORT:-8501}"

echo "============================================================"
echo "[nutristeppe] $(date -u +"%Y-%m-%dT%H:%M:%SZ") Streamlit starting"
echo "[nutristeppe] PORT=${PORT}"
echo "[nutristeppe] PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-}"
if [ -n "${DATABASE_URL:-}" ]; then
  echo "[nutristeppe] DATABASE_URL is set (hidden)"
else
  echo "[nutristeppe] WARNING: DATABASE_URL is unset — DB features disabled"
fi
if [ -n "${GOOGLE_GEMINI_API_KEY:-}" ] || [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
  echo "[nutristeppe] Gemini API key env is set (hidden)"
else
  echo "[nutristeppe] WARNING: no GOOGLE_GEMINI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY — analyze will fail"
fi
echo "[nutristeppe] If UI stays on grey skeleton: open DevTools → Network → WS →"
echo "[nutristeppe]   /_stcore/stream must be 101. If not, Traefik WebSocket or timeouts."
echo "============================================================"

# -u = unbuffered Python stdout/stderr (shows in docker logs immediately)
exec python -u -m streamlit run app.py \
  --server.port="${PORT}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
