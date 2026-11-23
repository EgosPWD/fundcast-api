FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
# Baseline model inside the image: a fresh named volume mounted on /srv/artifacts
# (docker-compose.dokploy.yml) is initialized from it. Locally the bind mount hides it.
COPY artifacts/modelo_rondas.joblib ./artifacts/
RUN mkdir -p data/uploads

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
