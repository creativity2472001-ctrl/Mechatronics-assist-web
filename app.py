#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - الإصدار المتوافق مع google-generativeai 0.3.2
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import hashlib
import logging
import sqlite3
import time
import re
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict

import google.generativeai as genai

# ============================================================
# 📊 إعدادات التسجيل
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🔧 إعدادات التطبيق
# ============================================================

class Config:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    RATE_LIMIT = int(os.getenv('RATE_LIMIT', '10'))
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '127.0.0.1')

config = Config()

if not config.GEMINI_API_KEY:
    logger.error("❌ مفتاح Gemini غير موجود")
    if config.ENVIRONMENT == 'production':
        exit(1)

# تهيئة Gemini
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
    logger.info("✅ Gemini configured successfully")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ============================================================
# 💾 نظام الحفظ (SQLite)
# ============================================================

class CacheDB:
    def __init__(self, db_path: str = "cache.db", max_size: int = 1000):
        self.db_path = db_path
        self.max_size = max_size
        self._init_db()
    
    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS solutions (
                        id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access ON solutions(access_count)")
                conn.commit()
            logger.info("✅ SQLite cache initialized")
        except Exception as e:
            logger.error(f"❌ SQLite init error: {e}")
    
    def get(self, question_hash: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.db_path) as db:
                db.row_factory = sqlite3.Row
                cursor = db.execute(
                    "SELECT answer, created FROM solutions WHERE id = ?",
                    (question_hash,)
                )
                row = cursor.fetchone()
                if row:
                    db.execute(
                        "UPDATE solutions SET access_count = access_count + 1, last_access = CURRENT_TIMESTAMP WHERE id = ?",
                        (question_hash,)
                    )
                    db.commit()
                    return {
                        "answer": row["answer"],
                        "saved_date": row["created"]
                    }
        except Exception as e:
            logger.error(f"❌ Cache read error: {e}")
        return None
    
    def set(self, question_hash: str, question: str, answer: str):
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "INSERT OR REPLACE INTO solutions (id, question, answer) VALUES (?, ?, ?)",
                    (question_hash, question[:200], answer)
                )
                db.commit()
                
                # تنظيف إذا زاد الحجم
                cursor = db.execute("SELECT COUNT(*) FROM solutions")
                count = cursor.fetchone()[0]
                if count > self.max_size:
                    db.execute("""
                        DELETE FROM solutions 
                        WHERE id IN (
                            SELECT id FROM solutions 
                            ORDER BY access_count ASC, last_access ASC 
                            LIMIT ?
                        )
                    """, (count - self.max_size,))
                    db.commit()
        except Exception as e:
            logger.error(f"❌ Cache write error: {e}")
    
    def get_stats(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT COUNT(*) FROM solutions")
                total = cursor.fetchone()[0]
                return {"total": total, "max_size": self.max_size}
        except:
            return {"total": 0, "max_size": self.max_size}

cache = CacheDB(max_size=config.CACHE_MAX_SIZE)

# ============================================================
# 🤖 دوال Gemini (بدون Code Execution)
# ============================================================

def ask_gemini(question: str) -> Optional[str]:
    """إرسال سؤال إلى Gemini"""
    if not config.GEMINI_API_KEY:
        return None
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        
        response = model.generate_content(
            f"حل المسألة التالية بالتفصيل مع الخطوات:\n{question}",
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 4096
            }
        )
        
        if response and response.text:
            return response.text.strip()
        
    except Exception as e:
        logger.error(f"Gemini error: {e}")
    
    return None

# ============================================================
# 🚦 Rate Limiting
# ============================================================

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id] = [t for t in self.requests[client_id] if now - t < self.window]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=config.RATE_LIMIT)

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = request.remote_addr or 'unknown'
        if not rate_limiter.is_allowed(client_id):
            return jsonify({
                "success": False,
                "error": "❌ تجاوزت الحد المسموح من الطلبات"
            }), 429
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# 🎯 المسارات الرئيسية
# ============================================================

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return "❌ ملف index.html غير موجود", 500

@app.route('/api/ask', methods=['POST'])
@rate_limit
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "❌ السؤال فارغ"}), 400
        
        question_hash = hashlib.md5(question.encode()).hexdigest()
        
        # البحث في الذاكرة
        cached = cache.get(question_hash)
        if cached:
            return jsonify({
                "success": True,
                "answer": cached["answer"],
                "cached": True
            })
        
        # حل جديد
        answer = ask_gemini(question)
        
        if not answer:
            return jsonify({
                "success": False,
                "error": "❌ لم نتمكن من حل السؤال"
            }), 500
        
        # حفظ الحل
        cache.set(question_hash, question, answer)
        
        return jsonify({
            "success": True,
            "answer": answer,
            "cached": False
        })
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"success": False, "error": "❌ حدث خطأ"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "gemini": bool(config.GEMINI_API_KEY),
        "cache": cache.get_stats()
    })

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔥 MECHATRONICS ASSISTANT - نسخة بسيطة")
    print("="*60)
    print(f"✅ Gemini: {'✅ متصل' if config.GEMINI_API_KEY else '❌ غير متصل'}")
    print(f"✅ Rate Limit: {config.RATE_LIMIT} طلب/دقيقة")
    print(f"✅ Cache: SQLite")
    print("="*60)
    print(f"🌐 http://{config.HOST}:{config.PORT}")
    print("="*60 + "\n")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.ENVIRONMENT == 'development'
    )
