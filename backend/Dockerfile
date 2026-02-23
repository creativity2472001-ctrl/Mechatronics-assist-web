FROM python:3.11-slim

WORKDIR /app

# نسخ requirements أولاً (للاستفادة من Docker cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY backend/ .
COPY templates/ /app/templates/

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
