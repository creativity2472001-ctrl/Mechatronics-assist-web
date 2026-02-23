FROM python:3.11-slim

WORKDIR /app

# نسخ المتطلبات وتثبيتها
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# نسخ جميع ملفات backend
COPY backend/ .

# نسخ مجلد templates إلى المسار الصحيح
COPY templates/ ./templates/

# تحقق من الملفات (للتأكيد)
RUN ls -la && ls -la templates/

EXPOSE 5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--log-level", "debug"]
