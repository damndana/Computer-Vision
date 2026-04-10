FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY . /app

EXPOSE 8501

# Logs: stdout/stderr → Dokploy. Do not open http://172.18.x.x from your PC.
ENTRYPOINT ["/docker-entrypoint.sh"]
