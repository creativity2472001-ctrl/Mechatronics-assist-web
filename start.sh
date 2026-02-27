#!/bin/bash
# سكريبت تشغيل التطبيق على Render

echo "🔥 Starting Mechatronics Assistant v21.0..."
echo "=========================================="
echo "📊 Environment: $ENVIRONMENT"
echo "🌐 Port: $PORT"
echo "=========================================="

# تشغيل التطبيق
uvicorn app:app --host 0.0.0.0 --port $PORT --log-level info
