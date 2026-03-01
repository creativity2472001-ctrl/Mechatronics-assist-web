"""
Mechatronics Assistant v5.1 - الإصدار المخفي (مصحح)
المستخدم لا يعلم بوجود نماذج متعددة - نظام واحد خلف الكواليس
"""

from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from openai import OpenAI
import hashlib
import sqlite3
import threading
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
from collections import defaultdict, OrderedDict
import queue
from typing import Optional, Dict, Any
import re
import random
import sys
import io

# حل مشكلة Unicode في CMD
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# تحميل المتغيرات من ملف .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', hashlib.sha256(str(time.time()).encode()).hexdigest())

# ============================================================
# 📊 إعدادات التسجيل
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ============================================================
# 🔑 إعدادات الـ APIs (من متغيرات CMD)
# ============================================================
GEMINI_KEY = os.environ.get('GEMINI_KEY')
OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')

if not GEMINI_KEY or not OPENROUTER_KEY or not GITHUB_TOKEN:
    print("\n" + "="*60)
    print("❌ المفاتيح غير موجودة!")
    print("="*60)
    print("👉 ضع المتغيرات في CMD قبل التشغيل:")
    print("   set GEMINI_KEY=AIzaSyBErJLXTIia9hOhEhGNXQM7IB4zqmAbwTI")
    print("   set OPENROUTER_KEY=sk-or-v1-xxxxxxxxxxxx")
    print("   set GITHUB_TOKEN=github_pat_xxxxxxxxxxxx")
    print("="*60 + "\n")
    exit(1)

# تهيئة Gemini
genai.configure(api_key=GEMINI_KEY)
gemini = genai.GenerativeModel('gemini-2.0-flash-001')

# تهيئة OpenRouter
openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    timeout=30.0
)

# تهيئة GitHub Models
github_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
    timeout=30.0
)

# ============================================================
# 💾 LRU Cache
# ============================================================
class LRUCache:
    def __init__(self, maxsize=500):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
    
    def clear(self):
        with self.lock:
            self.cache.clear()

# ============================================================
# 💾 الذاكرة الأبدية
# ============================================================
class EternalMemory:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.conn = sqlite3.connect('eternal_memory.db', check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.query_lock = threading.Lock()
        self.cache = LRUCache(maxsize=500)
        self._init_db()
        self._initialized = True
        logger.info("✅ الذاكرة الأبدية جاهزة")
    
    def _init_db(self):
        with self.query_lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    model TEXT,
                    category TEXT,
                    language TEXT,
                    response_time REAL,
                    timestamp DATETIME,
                    times_used INTEGER DEFAULT 1,
                    positive_ratings INTEGER DEFAULT 0,
                    negative_ratings INTEGER DEFAULT 0,
                    user_session TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS questions_fts USING fts5(
                    question, answer
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    model TEXT PRIMARY KEY,
                    total_calls INTEGER DEFAULT 0,
                    successful_calls INTEGER DEFAULT 0,
                    failed_calls INTEGER DEFAULT 0,
                    total_response_time REAL DEFAULT 0,
                    avg_response_time REAL DEFAULT 0,
                    priority INTEGER DEFAULT 1,  -- 1 = أعلى أولوية
                    last_used TIMESTAMP
                )
            """)
            
            # إدخال أولويات النماذج (يمكن تعديلها يدوياً لاحقاً)
            models = ['gemini', 'deepseek', 'claude', 'llama', 'gpt4o_mini', 'phi3', 'mistral']
            for i, model in enumerate(models):
                self.conn.execute("""
                    INSERT OR IGNORE INTO model_stats (model, priority) VALUES (?, ?)
                """, (model, i + 1))
            
            self.conn.commit()
    
    def find_exact(self, question, session_id=None):
        q_id = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        cached = self.cache.get(q_id)
        if cached:
            with self.query_lock:
                self.conn.execute("""
                    UPDATE questions SET times_used = times_used + 1 WHERE id = ?
                """, (q_id,))
                self.conn.commit()
            return cached
        
        with self.query_lock:
            cur = self.conn.execute("SELECT * FROM questions WHERE id = ?", (q_id,))
            row = cur.fetchone()
            
            if row:
                self.conn.execute("UPDATE questions SET times_used = times_used + 1 WHERE id = ?", (q_id,))
                self.conn.commit()
                result = dict(row)
                self.cache.put(q_id, result)
                return result
        return None
    
    def find_similar(self, question, limit=3):
        with self.query_lock:
            cur = self.conn.execute("""
                SELECT q.question, q.answer, q.times_used, q.positive_ratings, q.negative_ratings
                FROM questions_fts f
                JOIN questions q ON f.rowid = q.rowid
                WHERE f.question MATCH ?
                ORDER BY (q.times_used * 0.4 + (q.positive_ratings - q.negative_ratings) * 0.6) DESC
                LIMIT ?
            """, (question, limit))
            return [dict(row) for row in cur.fetchall()]
    
    def save(self, question, answer, model, category, language='ar', response_time=0, session_id=None):
        q_id = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        with self.query_lock:
            cur = self.conn.execute("SELECT id FROM questions WHERE id = ?", (q_id,))
            if cur.fetchone():
                self.conn.execute("UPDATE questions SET times_used = times_used + 1 WHERE id = ?", (q_id,))
            else:
                self.conn.execute("""
                    INSERT INTO questions 
                    (id, question, answer, model, category, language, response_time, timestamp, user_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (q_id, question[:2000], answer[:20000], model, category, language, 
                      response_time, datetime.now(), session_id))
                
                rowid = self.conn.execute("SELECT rowid FROM questions WHERE id = ?", (q_id,)).fetchone()[0]
                self.conn.execute("INSERT INTO questions_fts (rowid, question, answer) VALUES (?, ?, ?)",
                                (rowid, question[:2000], answer[:20000]))
            
            self.conn.commit()
            self.cache.put(q_id, {
                'id': q_id, 'question': question[:2000], 'answer': answer[:20000],
                'model': model, 'category': category, 'language': language, 'times_used': 1
            })
    
    def rate_answer(self, question_id, rating):
        with self.query_lock:
            if rating > 0:
                self.conn.execute("UPDATE questions SET positive_ratings = positive_ratings + 1 WHERE id = ?", (question_id,))
            else:
                self.conn.execute("UPDATE questions SET negative_ratings = negative_ratings + 1 WHERE id = ?", (question_id,))
            self.conn.commit()
    
    def update_model_stats(self, model, success=True, response_time=0):
        with self.query_lock:
            if success:
                # جلب البيانات الحالية
                cur = self.conn.execute("SELECT successful_calls, total_response_time FROM model_stats WHERE model = ?", (model,))
                row = cur.fetchone()
                
                if row:
                    successful = row[0] + 1
                    total_time = row[1] + response_time
                    avg_time = total_time / successful if successful > 0 else 0
                    
                    self.conn.execute("""
                        UPDATE model_stats SET
                            total_calls = total_calls + 1,
                            successful_calls = successful_calls + 1,
                            total_response_time = total_response_time + ?,
                            avg_response_time = ?,
                            last_used = CURRENT_TIMESTAMP
                        WHERE model = ?
                    """, (response_time, avg_time, model))
                else:
                    self.conn.execute("""
                        INSERT INTO model_stats 
                        (model, total_calls, successful_calls, total_response_time, avg_response_time, last_used)
                        VALUES (?, 1, 1, ?, ?, CURRENT_TIMESTAMP)
                    """, (model, response_time, response_time))
            else:
                self.conn.execute("""
                    UPDATE model_stats SET
                        total_calls = total_calls + 1,
                        failed_calls = failed_calls + 1,
                        last_used = CURRENT_TIMESTAMP
                    WHERE model = ?
                """, (model,))
            self.conn.commit()
    
    def get_best_models(self, limit=3):
        """الحصول على أفضل النماذج حسب الأولوية والإحصائيات"""
        with self.query_lock:
            cur = self.conn.execute("""
                SELECT model FROM model_stats 
                ORDER BY priority ASC, successful_calls DESC, 
                         CASE WHEN avg_response_time IS NULL THEN 999999 ELSE avg_response_time END ASC
                LIMIT ?
            """, (limit,))
            return [row[0] for row in cur.fetchall()]
    
    def get_stats(self):
        """إحصائيات شاملة للذاكرة"""
        with self.query_lock:
            # عدد الأسئلة الكلي
            cur = self.conn.execute("SELECT COUNT(*) as total FROM questions")
            total = cur.fetchone()[0]
            
            # عدد التصنيفات
            cur = self.conn.execute("SELECT COUNT(DISTINCT category) as categories FROM questions")
            categories = cur.fetchone()[0]
            
            # إجمالي الاستخدامات
            cur = self.conn.execute("SELECT SUM(times_used) as uses FROM questions")
            total_uses = cur.fetchone()[0] or 0
            
            # إحصائيات إضافية
            cur = self.conn.execute("SELECT SUM(positive_ratings) as likes FROM questions")
            likes = cur.fetchone()[0] or 0
            
            cur = self.conn.execute("SELECT SUM(negative_ratings) as dislikes FROM questions")
            dislikes = cur.fetchone()[0] or 0
            
            return {
                'total_questions': total,
                'categories': categories,
                'total_uses': total_uses,
                'likes': likes,
                'dislikes': dislikes
            }

memory = EternalMemory()

# ============================================================
# 📋 التخصصات الهندسية
# ============================================================
ENGINEERING_DOMAINS = [
    'ميكاترونكس', 'ميكانيك', 'كهرباء', 'الكترون', 'محركات',
    'PLC', 'تحكم', 'آلي', 'رياضيات', 'فيزياء', 'هندسة',
    'دائرة', 'دارة', 'مكينة', 'قوة', 'حركة', 'طاقة',
    'برمجة', 'اردوينو', 'raspberry', 'sensor', 'motor'
]

def is_engineering(question):
    q = question.lower()
    banned = ['طب', 'بشر', 'حيوان', 'نبات', 'أحياء', 'جغرافيا']
    for word in banned:
        if word in q:
            return False
    for word in ENGINEERING_DOMAINS:
        if word.lower() in q:
            return True
    return False

# ============================================================
# 🤖 النماذج المتاحة (مخفية عن المستخدم)
# ============================================================
class RateLimiter:
    def __init__(self, max_calls_per_minute=60):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
    
    def can_call(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False

MODELS = {
    'gemini': {
        'name': 'Gemini 2.0 Flash',
        'function': lambda q: gemini.generate_content(q).text,
        'rate_limiter': RateLimiter(60)
    },
    'deepseek': {
        'name': 'DeepSeek R1',
        'function': lambda q: openrouter.chat.completions.create(
            model='deepseek/deepseek-r1',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(60)
    },
    'claude': {
        'name': 'Claude 3.5 Sonnet',
        'function': lambda q: openrouter.chat.completions.create(
            model='anthropic/claude-3.5-sonnet',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(50)
    },
    'llama': {
        'name': 'Llama 3.2',
        'function': lambda q: openrouter.chat.completions.create(
            model='meta-llama/llama-3.2-3b-instruct',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(60)
    },
    'gpt4o_mini': {
        'name': 'GPT-4o Mini',
        'function': lambda q: github_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(100)
    },
    'phi3': {
        'name': 'Phi-3 Mini',
        'function': lambda q: github_client.chat.completions.create(
            model='Phi-3-mini-4k-instruct',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(100)
    },
    'mistral': {
        'name': 'Mistral 7B',
        'function': lambda q: github_client.chat.completions.create(
            model='Mistral-7B-Instruct',
            messages=[{'role': 'user', 'content': q}]
        ).choices[0].message.content,
        'rate_limiter': RateLimiter(100)
    }
}

executor = ThreadPoolExecutor(max_workers=3)  # 3 نماذج بالتوازي

# ============================================================
# 🎯 المسارات
# ============================================================
@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = hashlib.sha256(str(time.time()).encode()).hexdigest()
    return render_template('index.html')  # بدون أي خيارات للنماذج

@app.route('/api/ask', methods=['POST'])
def ask():
    start_time = time.time()
    data = request.json
    question = data.get('question', '').strip()
    language = data.get('language', 'ar')
    session_id = session.get('session_id', 'anonymous')
    
    if not question:
        return jsonify({'error': 'السؤال فارغ'}), 400
    
    logger.info(f"📝 سؤال: {question[:100]}...")
    
    # 1️⃣ التحقق الهندسي
    if not is_engineering(question):
        return jsonify({
            'answer': '❌ أنا متخصص في المجالات الهندسية فقط.',
            'category': 'outside_scope'
        })
    
    # 2️⃣ البحث في الذاكرة
    cached = memory.find_exact(question, session_id)
    if cached:
        return jsonify({
            'answer': cached['answer'],
            'from_memory': True
        })
    
    # 3️⃣ البحث عن أسئلة مشابهة
    similar = memory.find_similar(question, limit=2)
    
    # 4️⃣ الحصول على أفضل النماذج (تلقائي)
    best_models = memory.get_best_models(limit=3)
    logger.info(f"🤖 أفضل النماذج: {best_models}")
    
    # 5️⃣ تجربة النماذج بالتوازي
    futures = {}
    for model_name in best_models:
        if model_name not in MODELS:
            continue
        
        model = MODELS[model_name]
        if not model['rate_limiter'].can_call():
            continue
        
        futures[executor.submit(_try_model, model_name, model, question)] = model_name
    
    for future in as_completed(futures, timeout=45):
        model_name = futures[future]
        try:
            result = future.result(timeout=25)
            if result['success']:
                response_time = time.time() - start_time
                memory.update_model_stats(model_name, True, response_time)
                memory.save(question, result['answer'], model_name, 'engineering', 
                          language, response_time, session_id)
                
                logger.info(f"✅ نجح {model_name}")
                return jsonify({
                    'answer': result['answer']
                })
        except Exception as e:
            memory.update_model_stats(model_name, False)
            logger.error(f"❌ فشل {model_name}: {e}")
            continue
    
    # 6️⃣ إذا فشلت كل النماذج
    return jsonify({
        'answer': 'عذراً، لم أتمكن من الإجابة حالياً. حاول مرة أخرى لاحقاً.'
    })

def _try_model(model_name, model, question):
    try:
        answer = model['function'](question)
        return {'success': True, 'answer': answer}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/api/rate', methods=['POST'])
def rate_answer():
    data = request.json
    question_id = data.get('question_id')
    rating = data.get('rating')
    
    if question_id and rating in [1, -1]:
        memory.rate_answer(question_id, rating)
    return jsonify({'success': True})

@app.route('/api/memory/stats', methods=['GET'])
def memory_stats():
    stats = memory.get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔧 MECHATRONICS ASSISTANT v5.1 - الإصدار المخفي (مصحح)")
    print("="*70)
    print("✅ 7 نماذج AI تعمل خلف الكواليس")
    print("✅ المستخدم لا يرى أي خيارات")
    print("✅ تبديل ذكي تلقائي بين النماذج")
    print("✅ أولوية النماذج قابلة للتعديل")
    print("✅ جميع الأخطاء مصححة")
    print("="*70)
    print(f"🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, threaded=True, host='127.0.0.1', port=5000)
