FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

# Port: Dokploy sets PORT (e.g. 8501). Address 0.0.0.0 so Traefik can reach the container.
# Open the app only via your HTTPS domain in Dokploy — not http://172.18.x.x (Docker internal IP).
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --server.enableXsrfProtection false"]
