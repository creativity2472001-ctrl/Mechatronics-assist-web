FROM python:3.11-slim

WORKDIR /app

# نسخ requirements.txt من backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ كل محتويات backend إلى /app
COPY backend/ .

# نسخ templates إلى المكان الصحيح
COPY templates/ /app/templates/

# تأكد من وجود gunicorn
RUN pip install gunicorn

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
