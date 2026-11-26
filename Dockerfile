FROM python:3.18-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Serving runtime only (no MLflow, no pytest)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app.py preprocessing.py ./
COPY model/ model/
COPY static/ static/
RUN mkdir -p logs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
