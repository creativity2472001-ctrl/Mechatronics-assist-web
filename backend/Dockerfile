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

# تشخيص الملفات (للتأكد)
RUN echo "=== ملفات /app ===" && ls -la
RUN echo "=== ملفات templates ===" && ls -la templates/

EXPOSE 5000

# تشغيل Gunicorn مع إعدادات Railway
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --log-level debug
