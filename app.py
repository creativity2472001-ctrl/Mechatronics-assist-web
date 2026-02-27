#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - الإصدار النهائي المحسن بالكامل v7.0
يدعم: Async, Redis, XSS Protection, Environment Variables, Docker Ready
"""

import os
import sys
import logging
import traceback
import re
import signal
import json
import time
import hashlib
import asyncio
import html
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List, Union
from functools import lru_cache, wraps
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

from flask import Flask, render_template, request, jsonify, g, render_template_string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
import redis
from redis import Redis

# ============================================================
# 📦 تحميل المتغيرات البيئية مع دعم multiple .env files
# ============================================================

# تحميل .env العام أولاً
load_dotenv('.env')

# ثم تحميل .env.local إذا وجد (للتطوير المحلي)
load_dotenv('.env.local', override=True)

# ثم تحميل .env.production إذا وجد (للإنتاج)
if os.getenv('FLASK_ENV') == 'production':
    load_dotenv('.env.production', override=True)

# ============================================================
# 📊 إعدادات التطبيق من المتغيرات البيئية
# ============================================================

class Config:
    """فئة الإعدادات المركزية من المتغيرات البيئية"""
    
    # التطبيق
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'
    ENV = os.getenv('FLASK_ENV', 'development')
    PORT = int(os.getenv('PORT', '5000'))
    HOST = os.getenv('HOST', '127.0.0.1')
    
    # مفاتيح API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_ENABLED = os.getenv('REDIS_ENABLED', 'true').lower() == 'true'
    
    # Cache
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '200 per day,50 per hour')
    RATE_LIMIT_ASK = os.getenv('RATE_LIMIT_ASK', '10 per minute')
    RATE_LIMIT_EXECUTE = os.getenv('RATE_LIMIT_EXECUTE', '5 per minute')
    
    # Code Execution
    CODE_TIMEOUT = int(os.getenv('CODE_TIMEOUT', '3'))
    CODE_MEMORY_LIMIT = int(os.getenv('CODE_MEMORY_LIMIT', '100'))  # MB
    CODE_MAX_LOOP_ITERATIONS = int(os.getenv('CODE_MAX_LOOP_ITERATIONS', '10000'))
    
    # Security
    MAX_QUESTION_LENGTH = int(os.getenv('MAX_QUESTION_LENGTH', '5000'))
    ALLOWED_DOMAINS = os.getenv('ALLOWED_DOMAINS', 'رياضيات,فيزياء,ميكانيكا,كهرباء,PLC').split(',')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    LOG_JSON = os.getenv('LOG_JSON', 'true').lower() == 'true'
    
    @classmethod
    def is_api_available(cls, api_name: str) -> bool:
        """التحقق من توفر API"""
        key_map = {
            'gemini': cls.GEMINI_API_KEY,
            'deepseek': cls.DEEPSEEK_API_KEY,
            'openrouter': cls.OPENROUTER_API_KEY,
            'github': cls.GITHUB_TOKEN,
        }
        return bool(key_map.get(api_name.lower()))

config = Config()

# ============================================================
# 📊 نظام التسجيل المحسن مع دعم JSON
# ============================================================

class JSONFormatter(logging.Formatter):
    """تنسيق السجلات بتنسيق JSON"""
    
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # إضافة الرسالة
        if hasattr(record, 'msg') and record.msg:
            if isinstance(record.msg, str):
                log_record['message'] = record.msg
            else:
                log_record.update(record.msg)
        
        # إضافة الاستثناء إذا وجد
        if record.exc_info:
            log_record['exception'] = traceback.format_exception(*record.exc_info)
        
        # إضافة أي خصائص إضافية
        if hasattr(record, 'kwargs'):
            log_record.update(record.kwargs)
        
        return json.dumps(log_record, ensure_ascii=False)

class StructuredLogger:
    """مسجل منظم مع دعم JSON"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """إعداد التسجيل"""
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))
        
        # حذف المعالجات الموجودة
        self.logger.handlers.clear()
        
        # معالج الملف
        if config.LOG_FILE:
            file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
            if config.LOG_JSON:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            self.logger.addHandler(file_handler)
        
        # معالج الكونسول
        console_handler = logging.StreamHandler(sys.stdout)
        if config.LOG_JSON:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
        self.logger.addHandler(console_handler)
    
    def _log(self, level: int, message: str, **kwargs):
        """تسجيل مع بيانات إضافية"""
        if config.LOG_JSON:
            # للـ JSON، نخزن البيانات في record
            extra = {'kwargs': kwargs}
            self.logger.log(level, message, extra=extra)
        else:
            # للـ text العادي
            if kwargs:
                extra_info = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
                self.logger.log(level, f"{message} | {extra_info}")
            else:
                self.logger.log(level, message)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

logger = StructuredLogger(__name__)

# ============================================================
# 🔧 إعدادات Flask
# ============================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False

# ============================================================
# 🚦 Rate Limiting مع دعم Redis
# ============================================================

# إعداد مخزن Rate Limiting
if config.REDIS_ENABLED:
    try:
        rate_limit_storage = f"redis://{config.REDIS_URL}"
        logger.info("✅ Rate Limiting using Redis")
    except:
        rate_limit_storage = "memory://"
        logger.warning("⚠️ Rate Limiting using memory (Redis not available)")
else:
    rate_limit_storage = "memory://"
    logger.info("ℹ️ Rate Limiting using memory (as configured)")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
    storage_uri=rate_limit_storage,
    strategy="fixed-window"
)

# ============================================================
# 💾 نظام Cache المتقدم
# ============================================================

class CacheManager:
    """مدير التخزين المؤقت مع دعم Redis والذاكرة المحلية"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.memory_cache_expiry = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()
        self.setup_redis()
    
    def setup_redis(self):
        """محاولة الاتصال بـ Redis"""
        if not config.REDIS_ENABLED:
            logger.info("ℹ️ Redis is disabled by configuration")
            return
        
        try:
            self.redis_client = redis.from_url(
                config.REDIS_URL, 
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def _get_memory_cache(self, key: str) -> Optional[str]:
        """استرجاع من الذاكرة المحلية"""
        with self.lock:
            if key in self.memory_cache:
                expiry = self.memory_cache_expiry.get(key, 0)
                if expiry > time.time():
                    self.cache_hits += 1
                    return self.memory_cache[key]
                else:
                    # حذف المنتهي
                    del self.memory_cache[key]
                    del self.memory_cache_expiry[key]
        return None
    
    def _set_memory_cache(self, key: str, value: str, ttl: int):
        """تخزين في الذاكرة المحلية"""
        with self.lock:
            # التحقق من الحجم
            if len(self.memory_cache) >= config.CACHE_MAX_SIZE:
                # حذف الأقدم
                oldest_key = min(self.memory_cache_expiry.keys(), 
                               key=lambda k: self.memory_cache_expiry[k])
                del self.memory_cache[oldest_key]
                del self.memory_cache_expiry[oldest_key]
            
            self.memory_cache[key] = value
            self.memory_cache_expiry[key] = time.time() + ttl
    
    def get(self, key: str) -> Optional[str]:
        """استرجاع قيمة من cache"""
        # تجربة Redis أولاً
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.cache_hits += 1
                    return value
            except Exception as e:
                logger.error(f"Redis get error", error=str(e))
        
        # الرجوع للذاكرة المحلية
        value = self._get_memory_cache(key)
        if value:
            return value
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: str, ttl: int = None):
        """تخزين قيمة في cache"""
        if ttl is None:
            ttl = config.CACHE_TTL
        
        # تخزين في Redis
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except Exception as e:
                logger.error(f"Redis set error", error=str(e))
        
        # تخزين في الذاكرة المحلية
        self._set_memory_cache(key, value, ttl)
    
    def delete(self, key: str):
        """حذف من cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except:
                pass
        
        with self.lock:
            self.memory_cache.pop(key, None)
            self.memory_cache_expiry.pop(key, None)
    
    def clear(self):
        """مسح كل cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except:
                pass
        
        with self.lock:
            self.memory_cache.clear()
            self.memory_cache_expiry.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """إحصائيات cache"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": round(hit_rate, 2),
            "memory_size": len(self.memory_cache),
            "redis_connected": self.redis_client is not None,
            "max_size": config.CACHE_MAX_SIZE
        }

cache = CacheManager()

def cached(key_prefix: str = "", ttl: int = None):
    """Decorator للتخزين المؤقت"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # إنشاء مفتاح فريد
            key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            key = hashlib.md5(key_data.encode()).hexdigest()
            
            # محاولة استرجاع من cache
            cached_value = cache.get(key)
            if cached_value:
                return json.loads(cached_value)
            
            # تنفيذ الدالة
            result = func(*args, **kwargs)
            
            # تخزين النتيجة
            if result:
                cache.set(key, json.dumps(result, ensure_ascii=False), ttl)
            
            return result
        return wrapper
    return decorator

# ============================================================
# 🔑 نظام المفاتيح مع التحقق
# ============================================================

class APIKeys:
    """إدارة والتحقق من مفاتيح API"""
    
    def __init__(self):
        self.keys = {
            'gemini': config.GEMINI_API_KEY,
            'deepseek': config.DEEPSEEK_API_KEY,
            'openrouter': config.OPENROUTER_API_KEY,
            'github': config.GITHUB_TOKEN,
        }
        self.validate_all()
    
    def validate(self, key_name: str) -> bool:
        """التحقق من مفتاح معين"""
        key = self.keys.get(key_name)
        return bool(key and len(key) > 10)
    
    def validate_all(self):
        """التحقق من جميع المفاتيح"""
        for key_name, key_value in self.keys.items():
            if key_value and len(key_value) > 10:
                logger.info(f"✅ {key_name}: متصل")
            else:
                logger.warning(f"❌ {key_name}: غير متصل")
    
    def get(self, key_name: str) -> Optional[str]:
        return self.keys.get(key_name)
    
    def get_available_apis(self) -> List[str]:
        """قائمة APIs المتاحة"""
        return [name for name in self.keys if self.validate(name)]
    
    def has_any(self) -> bool:
        return bool(self.get_available_apis())

api_keys = APIKeys()

# ============================================================
# ⚠️ الأخطاء المخصصة
# ============================================================

class APIError(Exception):
    """خطأ في API"""
    pass

class SecurityError(Exception):
    """خطأ أمني"""
    pass

class TimeoutError(Exception):
    """خطأ timeout"""
    pass

class ValidationError(Exception):
    """خطأ في التحقق"""
    pass

# ============================================================
# 📝 System Prompts
# ============================================================

SYSTEM_PROMPTS = {
    'default': """
أنت مساعد هندسي متخصص في الرياضيات والفيزياء والميكانيكا والكهرباء والإلكترونيات.

قواعد صارمة:
1. أي عملية رياضية (حل معادلة، تكامل، مشتقة، نهاية) يجب تنفيذها باستخدام Python + SymPy فقط
2. لا تحسب أي شيء ذهنياً أبداً
3. اتبع الخطوات الأربع بدقة:
   - تحليل المسألة
   - تعريف المتغيرات
   - الحل باستخدام SymPy
   - التحقق من النتيجة
""",
    'math': """
أنت خبير رياضيات. استخدم SymPy للحلول الرياضية بدقة.
""",
    'physics': """
أنت خبير فيزياء. استخدم القوانين الفيزيائية بدقة مع الوحدات المناسبة.
"""
}

# ============================================================
# ⚙️ نظام تنفيذ Python الآمن المحسن
# ============================================================

class Domain(Enum):
    """المجالات المدعومة"""
    MATH = "رياضيات"
    PHYSICS = "فيزياء"
    MECHANICS = "ميكانيكا"
    ELECTRICAL = "كهرباء"
    PLC = "PLC"
    UNKNOWN = "غير معروف"

@dataclass
class ExecutionResult:
    """نتيجة تنفيذ الكود"""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: Optional[float] = None

class CodeAnalyzer:
    """محلل الكود للكشف عن الأنماط الخطيرة"""
    
    # الأنماط الخطيرة
    DANGEROUS_PATTERNS = [
        (r'while\s+True|while\s+1\s*:|while\s*\(\s*True\s*\)', 'حلقة لا نهائية'),
        (r'__import__\s*\(', 'استيراد ديناميكي'),
        (r'eval\s*\(|exec\s*\(|compile\s*\(', 'تنفيذ كود ديناميكي'),
        (r'open\s*\(|file\s*\(|os\.remove|os\.unlink', 'عمليات ملفات'),
        (r'__builtins__|globals\s*\(|locals\s*\(|vars\s*\(', 'الوصول للبيئة'),
        (r'os\.|sys\.|subprocess|socket|requests|urllib', 'مكتبات النظام'),
        (r'__[a-zA-Z0-9_]+__', 'الوصول للدوال الخاصة'),
        (r'getattr|setattr|delattr', 'تعديل السمات'),
        (r'__base__|__class__|__mro__', 'الوصول للـ metaclass'),
    ]
    
    def __init__(self, max_iterations: int = 10000):
        self.max_iterations = max_iterations
    
    def analyze(self, code: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        تحليل الكود
        returns: (is_dangerous, reason, details)
        """
        details = {
            'lines': len(code.split('\n')),
            'chars': len(code),
            'has_loops': False,
            'has_functions': False,
            'estimated_iterations': 0
        }
        
        # فحص الأنماط الخطيرة
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return True, f"⚠️ كود خطر: {reason}", details
        
        # فحص الحلقات الكبيرة
        loop_patterns = [
            (r'for\s+\w+\s+in\s+range\s*\(\s*(\d+)\s*\)', 'range loop'),
            (r'for\s+\w+\s+in\s+range\s*\(\s*\w+\s*\)', 'variable range'),
            (r'while\s+[^:]+:', 'while loop'),
        ]
        
        for pattern, loop_type in loop_patterns:
            matches = re.findall(pattern, code)
            if matches:
                details['has_loops'] = True
                if loop_type == 'range loop' and matches[0].isdigit():
                    iterations = int(matches[0])
                    details['estimated_iterations'] = max(
                        details['estimated_iterations'], 
                        iterations
                    )
                    if iterations > self.max_iterations:
                        return True, f"⚠️ حلقة كبيرة جداً ({iterations} > {self.max_iterations})", details
        
        # فحص الحلقات المتداخلة
        nested_loops = len(re.findall(r'for\s+\w+\s+in', code))
        if nested_loops > 3:
            return True, f"⚠️ تداخل حلقات كبير ({nested_loops} مستويات)", details
        
        # فحص الدوال
        if re.search(r'def\s+\w+\s*\(', code):
            details['has_functions'] = True
        
        return False, "", details

class SafeExecutor:
    """منفذ كود Python آمن مع حدود صارمة"""
    
    # المكتبات المسموحة
    ALLOWED_LIBS = {
        "math": __import__("math"),
        "sympy": __import__("sympy"),
        "numpy": __import__("numpy"),
        "cmath": __import__("cmath"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "collections": __import__("collections"),
        "random": __import__("random"),
        "decimal": __import__("decimal"),
        "fractions": __import__("fractions"),
    }
    
    # الدوال المسموحة
    SAFE_BUILTINS = {
        'print': print, 'range': range, 'len': len,
        'int': int, 'float': float, 'str': str,
        'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'bool': bool, 'abs': abs,
        'round': round, 'pow': pow, 'sum': sum,
        'min': min, 'max': max, 'enumerate': enumerate,
        'zip': zip, 'sorted': sorted, 'reversed': reversed,
        'all': all, 'any': any, 'chr': chr, 'ord': ord,
        'hex': hex, 'oct': oct, 'bin': bin,
        'open': None, '__import__': None, 'help': None,
    }
    
    def __init__(self):
        self.analyzer = CodeAnalyzer(max_iterations=config.CODE_MAX_LOOP_ITERATIONS)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def set_resource_limits(self):
        """تحديد حدود الموارد"""
        try:
            # حد الذاكرة
            memory_bytes = config.CODE_MEMORY_LIMIT * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # حد CPU
            resource.setrlimit(resource.RLIMIT_CPU, (config.CODE_TIMEOUT, config.CODE_TIMEOUT + 1))
        except Exception as e:
            logger.warning(f"Could not set resource limits", error=str(e))
    
    def timeout_handler(self, signum, frame):
        """معالج timeout"""
        raise TimeoutError(f"⏱️ تجاوز الوقت المسموح به ({config.CODE_TIMEOUT} ثوان)")
    
    def _execute_sync(self, code: str, env: Dict) -> Tuple[Any, float]:
        """تنفيذ متزامن مع حدود"""
        start_time = time.time()
        
        # تعيين حدود الموارد
        self.set_resource_limits()
        
        # إعداد signal للـ timeout
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(config.CODE_TIMEOUT)
        
        try:
            local_env = {}
            exec(code, env, local_env)
            signal.alarm(0)
            
            result = local_env.get("result", local_env.get("ans", "✅ تم التنفيذ بنجاح"))
            return result, time.time() - start_time
            
        except Exception as e:
            signal.alarm(0)
            raise e
    
    async def execute_async(self, code: str) -> ExecutionResult:
        """تنفيذ الكود بشكل غير متزامن"""
        start_time = time.time()
        
        # تحليل الكود أولاً
        dangerous, reason, details = self.analyzer.analyze(code)
        if dangerous:
            return ExecutionResult(
                success=False,
                error=reason,
                execution_time=time.time() - start_time
            )
        
        # إعداد بيئة التنفيذ
        exec_env = {
            "__builtins__": self.SAFE_BUILTINS,
            **self.ALLOWED_LIBS
        }
        
        try:
            # تنفيذ في ThreadPool
            loop = asyncio.get_event_loop()
            result, exec_time = await loop.run_in_executor(
                self.executor,
                self._execute_sync,
                code,
                exec_env
            )
            
            return ExecutionResult(
                success=True,
                result=self._sanitize_output(str(result)),
                execution_time=exec_time
            )
            
        except TimeoutError as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"❌ خطأ في التنفيذ: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _sanitize_output(self, output: str) -> str:
        """تنظيف المخرجات من الأكواد الضارة"""
        # إزالة أي HTML
        output = html.escape(output)
        # تحديد الطول
        if len(output) > 10000:
            output = output[:10000] + "... (تم اقتطاع النتيجة)"
        return output

safe_executor = SafeExecutor()

# ============================================================
# 🧹 Preprocessing للأسئلة
# ============================================================

def preprocess_question(question: str) -> str:
    """تنظيف وتوحيد السؤال"""
    if not question:
        return ""
    
    q = question.strip()
    
    # استبدال الرموز
    replacements = {
        '×': '*', '÷': '/', '^': '**', '−': '-',
        '＝': '=', '≈': '≈', '≠': '!=',
        '≤': '<=', '≥': '>=', 'π': 'pi',
        '∞': 'oo', '∫': 'integrate', '∑': 'summation',
        '√': 'sqrt', '∛': 'cbrt', '∜': '**0.25',
        '∈': 'in', '∉': 'not in', '∩': '&', '∪': '|',
        '⊂': '<', '⊃': '>', '⊆': '<=', '⊇': '>=',
        '∠': 'angle', '∥': 'parallel', '⊥': 'perp',
        '°': 'degrees', '℃': 'C', '℉': 'F'
    }
    
    for old, new in replacements.items():
        q = q.replace(old, new)
    
    # تنظيف المسافات
    q = ' '.join(q.split())
    
    return q

# ============================================================
# 🧠 نظام كشف المجال المحسن
# ============================================================

class DomainDetector:
    """كاشف المجال مع نظام نقاط متقدم"""
    
    # أنماط المجالات مع النقاط
    DOMAIN_PATTERNS = {
        Domain.MATH: [
            (r'معادلة|equation|solve|حل', 2),
            (r'مشتقة|تكامل|نهاية|diff|integral|limit', 4),
            (r'مصفوفة|matrix|determinant|محدد|inverse|معكوس', 3),
            (r'احتمال|probability|statistics|إحصاء|متوسط|mean', 3),
            (r'sin|cos|tan|log|ln|exp|جيب|جتا|ظا', 3),
            (r'\d+\s*[\+\-\*/]\s*\d+', 1),
            (r'x\^|x\*\*|أس|قوة', 2),
            (r'∫|∑|√|π|∞|∏|∂', 4),
            (r'plot|graph|رسم|بياني|منحنى', 2),
            (r'نظرية|مبرهنة|theorem|proof|برهان', 3),
        ],
        
        Domain.PHYSICS: [
            (r'f\s*=\s*m\s*a|v\s*=\s*d/t|قوة|كتلة|تسارع', 3),
            (r'newton|نيوتن|force|mass', 3),
            (r'9\.8|gravity|جاذبية|ثابت', 2),
            (r'سرعة|velocity|acceleration|عجلة', 3),
            (r'طاقة|energy|work|شغل|قدرة|power', 3),
            (r'ضغط|pressure|كثافة|density|حجم|volume', 2),
            (r'موجة|wave|تردد|frequency|طول|wavelength', 3),
            (r'كهرباء|electricity|مغناطيس|magnetic', 2),
        ],
        
        Domain.MECHANICS: [
            (r'ميكانيكا|mechanics', 4),
            (r'ذراع|lever|رافعة|pulley|بكرة|عتلة', 3),
            (r'عزم|torque|moment|عزم', 3),
            (r'إجهاد|stress|strain|انفعال|مرونة|elastic', 3),
            (r'ترس|gear|belt|سير|chain|سلسلة|كاوتش', 3),
            (r'اهتزاز|vibration|ديناميك|حركة|motion', 3),
            (r'محمل|bearing|عمود|shaft|وصلة|joint', 2),
        ],
        
        Domain.ELECTRICAL: [
            (r'v\s*=\s*i\s*\*?\s*r|ohm|أوم|فولت|volt', 3),
            (r'جهد|voltage|تيار|current|مقاومة|resistance', 3),
            (r'مكثف|capacitor|ملف|inductor|محث', 3),
            (r'تردد|frequency|hertz|هرتز|موجة|wave', 2),
            (r'محول|transformer|rectifier|مقوم|diode|دايود', 3),
            (r'محرك|motor|generator|مولد|دينامو', 3),
            (r'إلكترونيات|electronics|دائرة|circuit|pcb', 3),
        ],
        
        Domain.PLC: [
            (r'ladder|ld|ldi|out|tim|cnt|plc', 4),
            (r'plc|برمجة\s+plc|plc\s+برمجة', 4),
            (r'hmi|opc|scada|سكادا', 3),
            (r'relay|contact|coil|مرحل|كونتاكتور', 3),
            (r'sensor|مستشعر|actuator|مشغل|solenoid|صمام', 3),
            (r'logix|studio 5000|simatic|step 7|tia portal', 4),
            (r'إنفرتر|inverter|vfd|soft starter|سوفت ستارتر', 3),
        ],
    }
    
    def detect(self, question: str) -> Tuple[Domain, float, Dict[str, float]]:
        """كشف المجال مع نسبة الثقة وتفاصيل النقاط"""
        if not question:
            return Domain.UNKNOWN, 0.0, {}
        
        q_lower = question.lower()
        scores = {domain: 0 for domain in Domain}
        details = {}
        
        # حساب النقاط لكل مجال
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            domain_score = 0
            for pattern, points in patterns:
                matches = re.findall(pattern, q_lower, re.IGNORECASE)
                if matches:
                    domain_score += points * len(matches)
            scores[domain] = domain_score
            details[domain.value] = domain_score
        
        # المجال الأكثر ترجيحاً
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]
        total_score = sum(scores.values()) or 1
        
        # حساب نسبة الثقة
        confidence = max_score / total_score if total_score > 0 else 0
        
        return max_domain if max_score >= 3 else Domain.UNKNOWN, confidence, details

domain_detector = DomainDetector()

# ============================================================
# 🤖 دوال الذكاء الاصطناعي المحسنة
# ============================================================

class AIManager:
    """مدير الذكاء الاصطناعي مع دعم متعدد وتشغيل متوازي"""
    
    def __init__(self):
        self.apis = [
            (self.ask_gemini, "Gemini", 3.0),
            (self.ask_deepseek, "DeepSeek", 2.5),
            (self.ask_openrouter, "OpenRouter", 2.5),
            (self.ask_github_models, "GitHub", 2.0),
        ]
        self.timeout = 10.0  # timeout كلي بالثواني
    
    async def ask_gemini(self, question: str) -> Optional[str]:
        """استدعاء Gemini"""
        if not api_keys.validate('gemini'):
            return None
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_keys.get('gemini'))
            model = genai.GenerativeModel('gemini-2.0-flash-001')
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: model.generate_content(question)
            )
            
            return self._sanitize_response(response.text)
            
        except Exception as e:
            logger.error(f"Gemini error", api="Gemini", error=str(e))
            return None
    
    async def ask_deepseek(self, question: str) -> Optional[str]:
        """استدعاء DeepSeek"""
        if not api_keys.validate('deepseek'):
            return None
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=api_keys.get('deepseek'),
                base_url="https://api.deepseek.com/v1",
                timeout=self.timeout
            )
            
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS['default']},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return self._sanitize_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"DeepSeek error", api="DeepSeek", error=str(e))
            return None
    
    async def ask_openrouter(self, question: str) -> Optional[str]:
        """استدعاء OpenRouter"""
        if not api_keys.validate('openrouter'):
            return None
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=api_keys.get('openrouter'),
                base_url="https://openrouter.ai/api/v1",
                timeout=self.timeout
            )
            
            response = await client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS['default']},
                    {"role": "user", "content": question}
                ]
            )
            
            return self._sanitize_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"OpenRouter error", api="OpenRouter", error=str(e))
            return None
    
    async def ask_github_models(self, question: str) -> Optional[str]:
        """استدعاء GitHub Models"""
        if not api_keys.validate('github'):
            return None
        
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url="https://models.github.ai/inference/v1",
                api_key=api_keys.get('github'),
                timeout=self.timeout
            )
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS['default']},
                    {"role": "user", "content": question}
                ]
            )
            
            return self._sanitize_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"GitHub error", api="GitHub", error=str(e))
            return None
    
    def _sanitize_response(self, response: str) -> str:
        """تنظيف الرد من المحتوى الضار"""
        # إزالة HTML
        response = html.escape(response)
        # منع JavaScript
        response = re.sub(r'javascript:', '', response, flags=re.IGNORECASE)
        return response
    
    async def ask_all_parallel(self, question: str) -> Tuple[Optional[str], str]:
        """استدعاء جميع APIs بشكل متوازي مع سباق"""
        # تصفية APIs المتاحة فقط
        available_apis = [(func, name, timeout) 
                         for func, name, timeout in self.apis 
                         if api_keys.validate(name.lower())]
        
        if not available_apis:
            logger.warning("No APIs available")
            return None, ""
        
        # إنشاء المهام
        tasks = []
        for func, name, _ in available_apis:
            task = asyncio.create_task(func(question))
            task.api_name = name  # إضافة اسم API للمهمة
            tasks.append(task)
        
        # سباق المهام
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # إلغاء المهام المتبقية
            for task in pending:
                task.cancel()
            
            # معالجة النتائج
            for task in done:
                try:
                    result = task.result()
                    if result:
                        logger.info(f"API success", api=task.api_name)
                        return result, task.api_name
                except Exception as e:
                    logger.error(f"API failed", api=task.api_name, error=str(e))
            
            return None, ""
            
        except asyncio.TimeoutError:
            logger.error("All APIs timeout")
            for task in tasks:
                task.cancel()
            return None, ""

ai_manager = AIManager()

# ============================================================
# 🎯 نظام معالجة الأسئلة الرئيسي
# ============================================================

class QuestionProcessor:
    """معالج الأسئلة مع دعم Async"""
    
    def __init__(self):
        self.loop = None
        self.lock = threading.Lock()
    
    def get_loop(self):
        """الحصول على event loop للـ thread الحالي"""
        with self.lock:
            if self.loop is None or self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
            return self.loop
    
    async def process_async(self, question: str) -> Dict[str, Any]:
        """معالجة السؤال بشكل غير متزامن"""
        
        # تنظيف السؤال
        cleaned_question = preprocess_question(question)
        
        # التحقق من الطول
        if len(cleaned_question) > config.MAX_QUESTION_LENGTH:
            return {
                "success": False,
                "error": f"❌ السؤال طويل جداً (الحد الأقصى {config.MAX_QUESTION_LENGTH} حرف)",
                "domain": Domain.UNKNOWN.value
            }
        
        # كشف المجال
        domain, confidence, details = domain_detector.detect(question)
        
        # إذا كان المجال غير معروف
        if domain == Domain.UNKNOWN:
            return {
                "success": False,
                "error": "❌ هذا السؤال خارج نطاق التطبيق",
                "domain": domain.value,
                "confidence": confidence,
                "details": details
            }
        
        # البحث في cache
        cache_key = f"answer:{hashlib.md5(cleaned_question.encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return {
                "success": True,
                "answer": cached_result,
                "domain": domain.value,
                "confidence": confidence,
                "cached": True,
                "details": details
            }
        
        # استدعاء AI بشكل متوازي
        answer, api_used = await ai_manager.ask_all_parallel(cleaned_question)
        
        if not answer:
            return {
                "success": False,
                "error": "❌ لم نتمكن من الإجابة حالياً. تأكد من توفر خدمة الإنترنت",
                "domain": domain.value,
                "confidence": confidence,
                "details": details
            }
        
        # تخزين النتيجة
        cache.set(cache_key, answer)
        
        # تسجيل الطلب
        logger.info(
            "Request processed",
            domain=domain.value,
            confidence=confidence,
            api_used=api_used,
            question_length=len(question)
        )
        
        return {
            "success": True,
            "answer": answer,
            "domain": domain.value,
            "confidence": confidence,
            "api_used": api_used,
            "cached": False,
            "details": details
        }
    
    def process(self, question: str) -> Dict[str, Any]:
        """واجهة متزامنة للمعالجة"""
        loop = self.get_loop()
        try:
            return loop.run_until_complete(self.process_async(question))
        except Exception as e:
            logger.error("Process error", error=str(e))
            return {
                "success": False,
                "error": "❌ حدث خطأ في معالجة السؤال",
                "domain": Domain.UNKNOWN.value
            }

question_processor = QuestionProcessor()

# ============================================================
# 🎯 المسارات الرئيسية
# ============================================================

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template_string(INDEX_HTML)

@app.route('/api/ask', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_ASK)
def ask():
    """معالجة الأسئلة"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                "success": False, 
                "error": "❌ السؤال فارغ"
            })
        
        result = question_processor.process(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Unhandled error", error=str(e), traceback=traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "❌ حدث خطأ داخلي"
        }), 500

@app.route('/api/execute', methods=['POST'])
@limiter.limit(config.RATE_LIMIT_EXECUTE)
async def execute_code():
    """تنفيذ كود Python"""
    try:
        data = request.get_json()
        code = data.get('code', '').strip()
        
        if not code:
            return jsonify({
                "success": False,
                "error": "❌ الكود فارغ"
            })
        
        result = await safe_executor.execute_async(code)
        
        return jsonify({
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time
        })
        
    except Exception as e:
        logger.error("Code execution error", error=str(e))
        return jsonify({
            "success": False,
            "error": "❌ حدث خطأ في تنفيذ الكود"
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """حالة التطبيق"""
    return jsonify({
        "status": "running",
        "version": "7.0",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "env": config.ENV,
            "debug": config.DEBUG,
            "cache_ttl": config.CACHE_TTL,
            "rate_limits": {
                "ask": config.RATE_LIMIT_ASK,
                "execute": config.RATE_LIMIT_EXECUTE
            }
        },
        "apis": {
            name: api_keys.validate(name) 
            for name in ['gemini', 'deepseek', 'openrouter', 'github']
        },
        "cache": cache.get_stats(),
        "domains": [d.value for d in Domain if d != Domain.UNKNOWN]
    })

@app.route('/api/domains', methods=['GET'])
def get_domains():
    """قائمة المجالات المدعومة"""
    return jsonify({
        "domains": [domain.value for domain in Domain if domain != Domain.UNKNOWN]
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_question():
    """تحليل السؤال فقط بدون إجابة"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "السؤال فارغ"})
        
        domain, confidence, details = domain_detector.detect(question)
        
        return jsonify({
            "success": True,
            "domain": domain.value,
            "confidence": confidence,
            "details": details,
            "processed": preprocess_question(question)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """معالج تجاوز الحد المسموح"""
    return jsonify({
        "success": False,
        "error": "❌ تجاوزت الحد المسموح من الطلبات. حاول بعد دقيقة"
    }), 429

# ============================================================
# 📄 قالب HTML (مضمن للتبسيط)
# ============================================================

INDEX_HTML = '''
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مساعد الميكاترونكس v7.0</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 900px;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        
        .status-item {
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            backdrop-filter: blur(5px);
        }
        
        .chat-area {
            padding: 30px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        
        .ask-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .ask-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102,126,234,0.4);
        }
        
        .clear-btn {
            background: #f44336;
            color: white;
        }
        
        .clear-btn:hover:not(:disabled) {
            background: #d32f2f;
        }
        
        .analyze-btn {
            background: #4caf50;
            color: white;
        }
        
        .analyze-btn:hover:not(:disabled) {
            background: #388e3c;
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .result-area {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid #e0e0e0;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .domain-badge {
            background: #667eea;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .confidence-badge {
            background: #4caf50;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .api-badge {
            background: #ff9800;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .answer {
            line-height: 1.8;
            white-space: pre-wrap;
            font-size: 1.1em;
            max-height: 500px;
            overflow-y: auto;
            padding: 10px;
            background: white;
            border-radius: 8px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            color: #f44336;
            background: #ffebee;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border-right: 4px solid #f44336;
        }
        
        .info-text {
            color: #666;
            font-size: 0.9em;
            margin-top: 15px;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            background: #f9f9f9;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }
        
        .details-panel {
            margin-top: 15px;
            padding: 10px;
            background: #e8eaf6;
            border-radius: 8px;
            font-size: 0.9em;
        }
        
        .details-panel summary {
            cursor: pointer;
            color: #3f51b5;
            font-weight: bold;
        }
        
        .version-badge {
            background: #9c27b0;
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 مساعد الميكاترونكس <span class="version-badge">v7.0</span></h1>
            <p>نسخة متطورة مع دعم متعدد لمزودي الذكاء الاصطناعي</p>
            <div class="status-bar" id="statusBar">
                <div class="status-item">⏳ جاري تحميل الحالة...</div>
            </div>
        </div>
        
        <div class="chat-area">
            <div class="input-group">
                <textarea id="questionInput" placeholder="اكتب سؤالك هنا... (رياضيات، فيزياء، ميكانيكا، كهرباء، PLC)"></textarea>
            </div>
            
            <div class="button-group">
                <button class="ask-btn" id="askBtn" onclick="askQuestion()">📤 إرسال السؤال</button>
                <button class="analyze-btn" id="analyzeBtn" onclick="analyzeQuestion()">🔍 تحليل فقط</button>
                <button class="clear-btn" id="clearBtn" onclick="clearChat()">🧹 مسح</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div>جاري معالجة سؤالك...</div>
            </div>
            
            <div class="result-area" id="resultArea" style="display: none;">
                <div class="result-header" id="resultHeader"></div>
                <div class="answer" id="answer"></div>
                <details class="details-panel" id="detailsPanel" style="display: none;">
                    <summary>تفاصيل التحليل</summary>
                    <div id="details"></div>
                </details>
            </div>
            
            <div class="error" id="error" style="display: none;"></div>
            
            <div class="info-text">
                💡 يمكنك استخدام الرموز الرياضية: √, ∫, ∑, π, ∞ وغيرها<br>
                ⚡ Ctrl+Enter للإرسال السريع
            </div>
        </div>
        
        <div class="footer">
            Mechatronics Assistant v7.0 | جميع الحقوق محفوظة © 2026
        </div>
    </div>
    
    <script>
        // تحميل حالة APIs عند بدء التشغيل
        window.onload = async function() {
            await loadStatus();
        };
        
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const statusBar = document.getElementById('statusBar');
                statusBar.innerHTML = '';
                
                const apis = [
                    { name: 'Gemini', status: data.apis.gemini },
                    { name: 'DeepSeek', status: data.apis.deepseek },
                    { name: 'OpenRouter', status: data.apis.openrouter },
                    { name: 'GitHub', status: data.apis.github }
                ];
                
                apis.forEach(api => {
                    const item = document.createElement('div');
                    item.className = 'status-item';
                    item.textContent = api.status ? `✅ ${api.name}` : `❌ ${api.name}`;
                    statusBar.appendChild(item);
                });
                
                // إضافة معلومات cache
                const cacheItem = document.createElement('div');
                cacheItem.className = 'status-item';
                cacheItem.textContent = `💾 ${data.cache.hit_rate}%`;
                cacheItem.title = `Cache hits: ${data.cache.hits}, Misses: ${data.cache.misses}`;
                statusBar.appendChild(cacheItem);
                
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        async function askQuestion() {
            await processQuestion(false);
        }
        
        async function analyzeQuestion() {
            await processQuestion(true);
        }
        
        async function processQuestion(analyzeOnly = false) {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('الرجاء كتابة سؤال');
                return;
            }
            
            // إظهار التحميل
            document.getElementById('loading').classList.add('active');
            document.getElementById('resultArea').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('askBtn').disabled = true;
            document.getElementById('analyzeBtn').disabled = true;
            
            try {
                const endpoint = analyzeOnly ? '/api/analyze' : '/api/ask';
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // عرض النتيجة
                    const header = document.getElementById('resultHeader');
                    header.innerHTML = '';
                    
                    const domainBadge = document.createElement('span');
                    domainBadge.className = 'domain-badge';
                    domainBadge.textContent = `📚 ${data.domain}`;
                    header.appendChild(domainBadge);
                    
                    if (data.confidence) {
                        const confidenceBadge = document.createElement('span');
                        confidenceBadge.className = 'confidence-badge';
                        confidenceBadge.textContent = `🎯 ${Math.round(data.confidence * 100)}%`;
                        header.appendChild(confidenceBadge);
                    }
                    
                    if (data.api_used && !analyzeOnly) {
                        const apiBadge = document.createElement('span');
                        apiBadge.className = 'api-badge';
                        apiBadge.textContent = `⚡ ${data.api_used}`;
                        header.appendChild(apiBadge);
                    }
                    
                    if (data.cached) {
                        const cacheBadge = document.createElement('span');
                        cacheBadge.className = 'api-badge';
                        cacheBadge.style.background = '#9c27b0';
                        cacheBadge.textContent = '💾 من المخزن';
                        header.appendChild(cacheBadge);
                    }
                    
                    // عرض الإجابة أو التحليل
                    if (analyzeOnly) {
                        document.getElementById('answer').innerHTML = `
                            <strong>السؤال بعد المعالجة:</strong><br>
                            ${escapeHtml(data.processed)}<br><br>
                            <strong>نتيجة التحليل:</strong><br>
                            ${JSON.stringify(data.details, null, 2)}
                        `;
                    } else {
                        document.getElementById('answer').innerHTML = data.answer.replace(/\\n/g, '<br>');
                    }
                    
                    // عرض التفاصيل إذا وجدت
                    if (data.details) {
                        const detailsDiv = document.getElementById('details');
                        detailsDiv.innerHTML = Object.entries(data.details)
                            .map(([k, v]) => `${k}: ${v} نقطة`)
                            .join('<br>');
                        document.getElementById('detailsPanel').style.display = 'block';
                    } else {
                        document.getElementById('detailsPanel').style.display = 'none';
                    }
                    
                    document.getElementById('resultArea').style.display = 'block';
                    
                } else {
                    // عرض الخطأ
                    document.getElementById('error').innerHTML = escapeHtml(data.error);
                    document.getElementById('error').style.display = 'block';
                }
                
            } catch (error) {
                document.getElementById('error').innerHTML = '❌ حدث خطأ في الاتصال';
                document.getElementById('error').style.display = 'block';
            } finally {
                // إخفاء التحميل
                document.getElementById('loading').classList.remove('active');
                document.getElementById('askBtn').disabled = false;
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function escapeHtml(unsafe) {
            if (!unsafe) return '';
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }
        
        function clearChat() {
            document.getElementById('questionInput').value = '';
            document.getElementById('resultArea').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }
        
        // دعم Enter للزر
        document.getElementById('questionInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                askQuestion();
            }
        });
        
        // تحديث الحالة كل دقيقة
        setInterval(loadStatus, 60000);
    </script>
</body>
</html>
'''

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*90)
    print("🔥 MECHATRONICS ASSISTANT v7.0 - Production Ready")
    print("="*90)
    print("✅ Gemini | DeepSeek | OpenRouter | GitHub")
    print("✅ Async Processing")
    print("✅ Redis Cache")
    print("✅ Rate Limiting")
    print("✅ XSS Protection")
    print("✅ Domain Detection")
    print("✅ Production Ready")
    print("="*90)
    print("🌐 http://127.0.0.1:5000")
    print("="*90 + "\n")
    
    # تشغيل التطبيق
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True
    )
