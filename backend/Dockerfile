FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY templates/ ../templates/

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
