# Nutristeppe Streamlit + CLIP/FAISS meal index (Dokploy / GitHub)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Default data dir for FAISS + id map (override in Dokploy with MEAL_PIPELINE_DATA_DIR if needed)
ENV MEAL_PIPELINE_DATA_DIR=/app/data

# PyTorch / FAISS use OpenMP on CPU
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements-api.txt includes -r requirements.txt (Streamlit stack + CLIP + FAISS + FastAPI extras)
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY . /app

RUN mkdir -p /app/data

EXPOSE 8501

ENTRYPOINT ["/docker-entrypoint.sh"]
