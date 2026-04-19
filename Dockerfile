# Nutristeppe Streamlit + CLIP/FAISS (Dokploy / GitHub)
# Layer order: deps only re-install when requirements*.txt change; app COPY is last.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MEAL_PIPELINE_DATA_DIR=/app/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Cache this layer until requirements change
COPY requirements.txt requirements-api.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-api.txt

# Application code last — small code pushes reuse the pip layer above
COPY . .

RUN mkdir -p /app/data \
    && chmod +x /app/scripts/docker-entrypoint.sh

EXPOSE 8501

# Entrypoint: optional FAISS build at start, then Streamlit (do not replace with bare CMD)
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
