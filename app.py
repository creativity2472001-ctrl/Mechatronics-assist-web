#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - الإصدار النهائي للإنتاج v22.0
مع جميع التحسينات: SQLite محسن، TTL، معالجة متقدمة، Performance
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import hashlib
import logging
import sqlite3
import time
import re
import threading
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

import google.generativeai as genai
from google.generativeai.types import Tool

# ============================================================
# 📊 إعدادات التسجيل (Logging)
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
    CACHE_TTL_DAYS = int(os.getenv('CACHE_TTL_DAYS', '30'))  # 30 يوم صلاحية
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '127.0.0.1')

config = Config()

if not config.GEMINI_API_KEY:
    logger.error("❌ مفتاح Gemini غير موجود")
    if config.ENVIRONMENT == 'production':
        exit(1)
    else:
        logger.warning("⚠️ تشغيل بدون مفتاح في بيئة التطوير")

# تهيئة Gemini
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)
    logger.info("✅ Gemini configured successfully")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ============================================================
# 🚦 Rate Limiting (مع Thread Safety)
# ============================================================

class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        with self.lock:
            now = time.time()
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # تنظيف الطلبات القديمة
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
            logger.warning(f"Rate limit exceeded for {client_id}")
            return jsonify({
                "success": False,
                "error": "❌ تجاوزت الحد المسموح من الطلبات. حاول بعد دقيقة"
            }), 429
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# 💾 نظام الحفظ المتقدم (SQLite مع اتصال دائم)
# ============================================================

class CacheDB:
    def __init__(self, db_path: str = "cache.db", max_size: int = 1000, ttl_days: int = 30):
        self.db_path = db_path
        self.max_size = max_size
        self.ttl_seconds = ttl_days * 24 * 3600
        self.connection = None
        self.lock = threading.Lock()
        self._init_db()
    
    def _get_connection(self):
        """الحصول على اتصال دائم بقاعدة البيانات"""
        if self.connection is None:
            self.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=10
            )
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    @contextmanager
    def _get_cursor(self):
        """سياق آمن للتعامل مع قاعدة البيانات"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    def _init_db(self):
        """تهيئة قاعدة البيانات"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS solutions (
                        id TEXT PRIMARY KEY,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1,
                        last_access TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires ON solutions(expires_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_access ON solutions(access_count)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON solutions(created)")
                
                # حذف الإجابات المنتهية
                cursor.execute("DELETE FROM solutions WHERE expires_at < datetime('now')")
            logger.info("✅ SQLite cache initialized")
        except Exception as e:
            logger.error(f"❌ SQLite init error: {e}")
    
    def _cleanup_old_entries(self):
        """تنظيف الإجابات القديمة"""
        try:
            with self._get_cursor() as cursor:
                # حذف المنتهية
                cursor.execute("DELETE FROM solutions WHERE expires_at < datetime('now')")
                
                # حساب العدد الحالي
                cursor.execute("SELECT COUNT(*) FROM solutions")
                count = cursor.fetchone()[0]
                
                if count > self.max_size:
                    # حذف الأقدم والأقل استخداماً
                    cursor.execute("""
                        DELETE FROM solutions 
                        WHERE id IN (
                            SELECT id FROM solutions 
                            ORDER BY access_count ASC, last_access ASC 
                            LIMIT ?
                        )
                    """, (count - self.max_size,))
                logger.info(f"🧹 Cache cleaned: {count} entries")
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
    
    def get(self, question_hash: str) -> Optional[Dict]:
        """استرجاع حل من الذاكرة"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("""
                    SELECT answer, created, expires_at 
                    FROM solutions 
                    WHERE id = ? AND expires_at > datetime('now')
                """, (question_hash,))
                row = cursor.fetchone()
                
                if row:
                    # تحديث عدد الزيارات
                    cursor.execute("""
                        UPDATE solutions 
                        SET access_count = access_count + 1, last_access = CURRENT_TIMESTAMP 
                        WHERE id = ?
                    """, (question_hash,))
                    logger.info(f"✅ Cache hit: {question_hash[:8]}...")
                    
                    # تنظيف إذا لزم الأمر
                    self._cleanup_old_entries()
                    
                    return {
                        "answer": row["answer"],
                        "saved_date": row["created"],
                        "expires_at": row["expires_at"]
                    }
        except Exception as e:
            logger.error(f"❌ Cache read error: {e}")
        return None
    
    def set(self, question_hash: str, question: str, answer: str):
        """حفظ حل جديد"""
        try:
            expires_at = (datetime.now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            
            with self._get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO solutions 
                    (id, question, answer, expires_at, created, last_access) 
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (question_hash, question[:200], answer, expires_at))
                
                logger.info(f"✅ Cache set: {question_hash[:8]}...")
                
                # تنظيف إذا لزم الأمر
                self._cleanup_old_entries()
                
        except Exception as e:
            logger.error(f"❌ Cache write error: {e}")
    
    def get_stats(self) -> Dict:
        """إحصائيات الذاكرة"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM solutions")
                total = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM solutions WHERE expires_at < datetime('now')")
                expired = cursor.fetchone()[0]
                
                return {
                    "total": total,
                    "expired": expired,
                    "active": total - expired,
                    "max_size": self.max_size,
                    "ttl_days": self.ttl_seconds // 86400
                }
        except:
            return {"total": 0, "active": 0, "max_size": self.max_size}

cache = CacheDB(
    max_size=config.CACHE_MAX_SIZE,
    ttl_days=config.CACHE_TTL_DAYS
)

# ============================================================
# 🧹 معالجة النصوص
# ============================================================

def clean_answer(text: str) -> str:
    """تنظيف النص من الأسطر الفارغة والرموز الزائدة"""
    if not text:
        return ""
    
    # إزالة الأسطر الفارغة المتكررة
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    
    # إزالة المسافات الزائدة
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned

def extract_code_from_response(response) -> Optional[str]:
    """استخراج الكود من رد Gemini (إذا وجد)"""
    try:
        if not response.candidates:
            return None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                if part.function_call.name == "run_python":
                    return part.function_call.args.get("code", "")
        return None
    except:
        return None

def extract_answer_from_response(response) -> Optional[str]:
    """استخراج الإجابة من رد Gemini بأمان"""
    try:
        if not response.candidates:
            logger.error("No candidates in response")
            return None
        
        if not response.candidates[0].content.parts:
            logger.error("No parts in response")
            return None
        
        answer = ""
        for part in response.candidates[0].content.parts:
            if part.text:
                answer += part.text + "\n"
        
        if not answer.strip():
            logger.error("Empty answer")
            return None
        
        return clean_answer(answer)
        
    except Exception as e:
        logger.error(f"Error extracting answer: {e}")
        return None

# ============================================================
# 🤖 دوال Gemini (مع Code Execution)
# ============================================================

def ask_gemini(question: str) -> Optional[str]:
    """إرسال سؤال إلى Gemini مع Code Execution"""
    if not config.GEMINI_API_KEY:
        logger.error("Gemini API key not configured")
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # إعداد أداة Code Execution
            code_execution_tool = Tool(
                function_declarations=[{
                    "name": "run_python",
                    "description": "Execute Python code for mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }]
            )
            
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-001',
                tools=[code_execution_tool]
            )
            
            enhanced_q = f"""
            حل المسألة التالية باستخدام SymPy.
            اكتب الحل مع شرح الخطوات.
            
            السؤال: {question}
            
            مهم جداً:
            1. استخدم SymPy للحسابات
            2. اشرح كل خطوة
            3. قدم النتيجة النهائية
            """
            
            logger.info(f"Sending to Gemini (attempt {attempt+1}): {question[:100]}...")
            
            response = model.generate_content(
                enhanced_q,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 4096
                }
            )
            
            # استخراج الإجابة
            answer = extract_answer_from_response(response)
            if answer:
                logger.info(f"✅ Gemini success on attempt {attempt+1}")
                return answer
            
            # إذا وصلنا هنا، الإجابة فارغة
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Empty response, retrying in {wait_time}s...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Gemini error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
    
    logger.error("All Gemini attempts failed")
    return None

# ============================================================
# 🎯 المسارات الرئيسية
# ============================================================

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return "❌ ملف index.html غير موجود في مجلد templates", 500

@app.route('/api/ask', methods=['POST'])
@rate_limit
def ask():
    """معالجة الأسئلة مع نظام الحفظ"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "❌ السؤال فارغ"}), 400
        
        # إنشاء مفتاح فريد للسؤال
        question_hash = hashlib.md5(question.encode()).hexdigest()
        logger.info(f"Processing question: {question[:100]}...")
        
        # ===== 1️⃣ البحث في الذاكرة =====
        cached = cache.get(question_hash)
        if cached:
            return jsonify({
                "success": True,
                "answer": cached["answer"],
                "cached": True,
                "saved_date": cached["saved_date"]
            })
        
        # ===== 2️⃣ حل جديد باستخدام Gemini =====
        if not config.GEMINI_API_KEY:
            return jsonify({
                "success": False,
                "error": "❌ مفتاح Gemini غير متوفر"
            }), 500
        
        answer = ask_gemini(question)
        
        if not answer:
            return jsonify({
                "success": False,
                "error": "❌ لم نتمكن من حل السؤال حالياً"
            }), 500
        
        # ===== 3️⃣ حفظ الحل الجديد =====
        cache.set(question_hash, question, answer)
        
        return jsonify({
            "success": True,
            "answer": answer,
            "cached": False
        })
        
    except Exception as e:
        logger.exception(f"Unhandled error: {e}")
        return jsonify({
            "success": False,
            "error": "❌ حدث خطأ غير متوقع"
        }), 500

@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """إحصائيات الذاكرة"""
    return jsonify(cache.get_stats())

@app.route('/api/health', methods=['GET'])
def health():
    """فحص صحة التطبيق"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini": bool(config.GEMINI_API_KEY),
        "cache": cache.get_stats(),
        "environment": config.ENVIRONMENT
    })

@app.route('/api/clear', methods=['POST'])
def clear_cache():
    """مسح الذاكرة (للمسؤولين)"""
    try:
        with cache._get_cursor() as cursor:
            cursor.execute("DELETE FROM solutions")
        logger.info("🧹 Cache cleared by admin")
        return jsonify({"success": True, "message": "تم مسح الذاكرة"})
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================
# 🚀 معالجات الأخطاء
# ============================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "❌ المسار غير موجود"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"success": False, "error": "❌ خطأ داخلي في الخادم"}), 500

# ============================================================
# 🔌 إغلاق اتصالات SQLite
# ============================================================

@app.teardown_appcontext
def close_connection(exception):
    """إغلاق اتصال SQLite عند إنهاء التطبيق"""
    if hasattr(cache, 'connection') and cache.connection:
        cache.connection.close()
        logger.info("✅ SQLite connection closed")

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔥 MECHATRONICS ASSISTANT v22.0 - الإصدار النهائي للإنتاج")
    print("="*80)
    print(f"✅ Gemini: {'✅ متصل' if config.GEMINI_API_KEY else '❌ غير متصل'}")
    print(f"✅ Rate Limit: {config.RATE_LIMIT} طلب/دقيقة")
    print(f"✅ Cache: SQLite محسن مع TTL ({config.CACHE_TTL_DAYS} يوم)")
    print(f"✅ اتصال دائم: نعم (check_same_thread=False)")
    print(f"✅ Environment: {config.ENVIRONMENT}")
    print("="*80)
    print(f"🌐 http://{config.HOST}:{config.PORT}")
    print("🔍 Health: /api/health")
    print("📊 Cache Stats: /api/cache/stats")
    print("="*80 + "\n")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.ENVIRONMENT == 'development',
        threaded=True
    )
