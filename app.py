#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - الإصدار النهائي لـ Render v15.0
مع جميع التحسينات بناءً على المراجعة الدقيقة
"""

import os
import re
import time
import json
import hashlib
import asyncio
import logging
import signal
import multiprocessing
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import concurrent.futures
import platform
import tempfile
import subprocess
import sys
from pathlib import Path

import sympy as sp
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
import google.generativeai as genai
from pydantic import BaseModel, Field, validator
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ============================================================
# 📊 Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests')
ERROR_COUNT = Counter('http_errors_total', 'Total HTTP errors')
REQUEST_TIME = Histogram('http_request_duration_seconds', 'HTTP request duration')
COMPUTATION_TIME = Histogram('computation_duration_seconds', 'Computation duration')
AI_TIME = Histogram('ai_duration_seconds', 'AI explanation duration')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')
MATH_SUCCESS = Counter('math_success_total', 'Successful math computations')
MATH_FAILURE = Counter('math_failure_total', 'Failed math computations')
SANDBOX_SUCCESS = Counter('sandbox_success_total', 'Successful sandbox executions')
SANDBOX_FAILURE = Counter('sandbox_failure_total', 'Failed sandbox executions')
AI_CALLS = Counter('ai_calls_total', 'Total AI API calls')
AI_DAILY_LIMIT_HITS = Counter('ai_daily_limit_hits_total', 'Daily AI limit hits')

# ============================================================
# 🔧 إعدادات التسجيل (مع stdout فقط لـ Render)
# ============================================================

class RenderLogHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[RenderLogHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# 🔧 إعدادات التطبيق
# ============================================================

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    # حدود الساندبوكس
    MAX_QUESTION_LENGTH = int(os.getenv('MAX_QUESTION_LENGTH', '2000'))
    CODE_TIMEOUT = int(os.getenv('CODE_TIMEOUT', '3'))
    CODE_MEMORY_LIMIT = int(os.getenv('CODE_MEMORY_LIMIT', '64'))
    
    # Cache
    CACHE_TTL = int(os.getenv('CACHE_TTL', '1800'))
    
    # Rate Limiting
    RATE_LIMIT_FREE = os.getenv('RATE_LIMIT_FREE', '5/minute')
    RATE_LIMIT_PREMIUM = os.getenv('RATE_LIMIT_PREMIUM', '20/minute')
    
    # AI Limits
    AI_DAILY_LIMIT_FREE = int(os.getenv('AI_DAILY_LIMIT_FREE', '3'))  # 3 أسئلة/يوم للـ AI
    AI_DAILY_LIMIT_PREMIUM = int(os.getenv('AI_DAILY_LIMIT_PREMIUM', '20'))
    
    # البيئة
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
    ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1,.onrender.com').split(',')
    
    # المصادقة
    DAILY_FREE_LIMIT = int(os.getenv('DAILY_FREE_LIMIT', '10'))
    
    @classmethod
    def check_gemini(cls):
        """التحقق من مفتاح Gemini مع رسالة مناسبة"""
        if not cls.GEMINI_API_KEY:
            if cls.ENVIRONMENT == 'production':
                logger.warning("⚠️ تشغيل بدون Gemini في الإنتاج - بعض الميزات معطلة")
                return False
            else:
                logger.info("ℹ️ تشغيل بدون Gemini في بيئة التطوير")
                return False
        return True

config = Config()
GEMINI_AVAILABLE = config.check_gemini()

# ============================================================
# 🚀 إنشاء التطبيق
# ============================================================

app = FastAPI(
    title="Mechatronics Assistant v15.0",
    description="مساعد ذكي للميكاترونكس - Render Ready",
    version="15.0.0",
    docs_url="/docs" if config.ENVIRONMENT != 'production' else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted Hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)

# ============================================================
# 🔐 API Key Authentication
# ============================================================

api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)

class UserManager:
    def __init__(self):
        self.users = {}
        self.daily_usage = {}
        self.ai_daily_usage = {}  # استخدام منفصل للـ AI
        self.load_users()
    
    def load_users(self):
        # مستخدمين افتراضيين للتجربة
        self.users = {
            "free_test_key": {
                "plan": "free",
                "daily_limit": config.DAILY_FREE_LIMIT,
                "ai_daily_limit": config.AI_DAILY_LIMIT_FREE
            },
            "premium_test_key": {
                "plan": "premium",
                "daily_limit": 100,
                "ai_daily_limit": config.AI_DAILY_LIMIT_PREMIUM
            }
        }
        logger.info(f"✅ Loaded {len(self.users)} test users")
    
    async def verify_key(self, api_key: str) -> Optional[Dict]:
        if not api_key:
            return None
        return self.users.get(api_key)
    
    async def check_rate_limit(self, api_key: str) -> Tuple[bool, str, Dict]:
        user = self.users.get(api_key)
        if not user:
            return False, "مفتاح غير صالح", {}
        
        today = datetime.utcnow().date().isoformat()
        key = f"{api_key}:{today}"
        
        # الحد العام
        current_usage = self.daily_usage.get(key, 0)
        limit = user.get('daily_limit', config.DAILY_FREE_LIMIT)
        
        if current_usage >= limit:
            return False, f"تجاوزت الحد اليومي ({limit} سؤال)", user
        
        return True, "", user
    
    async def increment_usage(self, api_key: str):
        """زيادة استخدام المستخدم"""
        today = datetime.utcnow().date().isoformat()
        key = f"{api_key}:{today}"
        self.daily_usage[key] = self.daily_usage.get(key, 0) + 1
    
    async def check_ai_limit(self, api_key: str) -> bool:
        """التحقق من حد الـ AI اليومي"""
        user = self.users.get(api_key)
        if not user:
            return False
        
        today = datetime.utcnow().date().isoformat()
        key = f"ai:{api_key}:{today}"
        
        current_ai_usage = self.ai_daily_usage.get(key, 0)
        ai_limit = user.get('ai_daily_limit', config.AI_DAILY_LIMIT_FREE)
        
        if current_ai_usage >= ai_limit:
            AI_DAILY_LIMIT_HITS.inc()
            return False
        
        self.ai_daily_usage[key] = current_ai_usage + 1
        return True
    
    def get_remaining(self, api_key: str) -> Optional[int]:
        if not api_key:
            return None
        user = self.users.get(api_key)
        if not user:
            return None
        today = datetime.utcnow().date().isoformat()
        key = f"{api_key}:{today}"
        used = self.daily_usage.get(key, 0)
        return user.get('daily_limit', 0) - used

user_manager = UserManager()

# ============================================================
# 🚦 Rate Limiting
# ============================================================

limiter = Limiter(
    key_func=lambda: "global",
    default_limits=[config.RATE_LIMIT_FREE],
    storage_uri="memory://"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================
# 📊 Prometheus
# ============================================================

Instrumentator().instrument(app).expose(app)

# ============================================================
# 📝 Enums و Dataclasses
# ============================================================

class QuestionType(Enum):
    COMPUTABLE = "رقمي"
    SYMBOLIC = "رمزي"
    ANALYTICAL = "تحليلي"
    COMPLEX = "معقد"
    UNKNOWN = "غير معروف"

class Domain(Enum):
    MATH = "رياضيات"
    PHYSICS = "فيزياء"
    MECHANICS = "ميكانيكا"
    ELECTRICAL = "كهرباء"
    ELECTRONICS = "إلكترونيات"
    PLC = "PLC"
    UNKNOWN = "غير معروف"

@dataclass
class ComputationResult:
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    question_type: Optional[QuestionType] = None
    confidence: float = 0.0

@dataclass
class AnswerResult:
    success: bool
    answer: Optional[str] = None
    domain: Optional[str] = None
    question_type: Optional[str] = None
    confidence: float = 0.0
    error: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False
    explanation: Optional[str] = None
    used_ai: bool = False

# ============================================================
# 📝 Pydantic Models
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_LENGTH)
    
    @validator('question')
    def sanitize_input(cls, v):
        dangerous = ['<', '>', 'script', 'javascript:', 'exec', 'eval', '__import__']
        v_lower = v.lower()
        for d in dangerous:
            if d in v_lower:
                logger.warning(f"Dangerous input detected")
                raise ValueError('محتويات غير مسموح بها')
        return v

class AnswerResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    domain: Optional[str] = None
    question_type: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    time: Optional[float] = None
    cached: bool = False
    explanation: Optional[str] = None
    remaining_quota: Optional[int] = None
    used_ai: bool = False

# ============================================================
# 💾 Memory Cache
# ============================================================

class MemoryCache:
    def __init__(self):
        self.cache = {}
        self.version = "v15.0"
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Dict]:
        cache_key = f"{self.version}:{key}"
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if entry['expires'] > time.time():
                self.hits += 1
                return entry['data']
            else:
                del self.cache[cache_key]
        self.misses += 1
        return None
    
    async def set(self, key: str, data: Dict, ttl: int = config.CACHE_TTL):
        cache_key = f"{self.version}:{key}"
        self.cache[cache_key] = {
            'data': data,
            'expires': time.time() + ttl
        }
        if len(self.cache) > 1000:
            self._cleanup()
    
    def _cleanup(self):
        now = time.time()
        expired = [k for k, v in self.cache.items() if v['expires'] <= now]
        for k in expired:
            del self.cache[k]
        logger.info(f"Cleaned up {len(expired)} expired cache entries")
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

cache = MemoryCache()

# ============================================================
# 🔐 Sandbox بسيط
# ============================================================

class SafeSandbox:
    def __init__(self, timeout: int = 3, memory_limit: int = 64):
        self.timeout = timeout
        self.memory_limit = memory_limit
    
    async def execute(self, code: str) -> ComputationResult:
        start_time = time.time()
        
        try:
            # بناء كود آمن مع متغير result إلزامي
            safe_code = f"""
import sys
import json
try:
    allowed_builtins = {{'abs': abs, 'round': round, 'pow': pow,
        'int': int, 'float': float, 'str': str,
        'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'bool': bool, 'len': len,
        'range': range, 'min': min, 'max': max,
        'sum': sum, 'any': any, 'all': all}}
    
    allowed_modules = {{
        'math': __import__('math'),
        'sympy': __import__('sympy'),
        'numpy': __import__('numpy'),
    }}
    
    local_env = {{}}
    exec_env = {{**allowed_builtins, **allowed_modules}}
    exec('''{code}''', exec_env, local_env)
    
    # التأكد من وجود result
    if 'result' in local_env:
        result = local_env['result']
    elif 'ans' in local_env:
        result = local_env['ans']
    else:
        result = "تم التنفيذ (لم يتم تعريف result)"
    
    print(json.dumps({{'success': True, 'result': str(result)}}))
except Exception as e:
    print(json.dumps({{'success': False, 'error': str(e)}}))
"""
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-c', safe_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                SANDBOX_FAILURE.inc()
                return ComputationResult(
                    success=False,
                    error=f"⏱️ تجاوز الوقت المسموح ({self.timeout} ثوان)",
                    execution_time=time.time() - start_time
                )
            
            if proc.returncode != 0:
                SANDBOX_FAILURE.inc()
                return ComputationResult(
                    success=False,
                    error="خطأ في التنفيذ",
                    execution_time=time.time() - start_time
                )
            
            result = json.loads(stdout.decode())
            if result.get('success'):
                SANDBOX_SUCCESS.inc()
                return ComputationResult(
                    success=True,
                    result=result['result'],
                    execution_time=time.time() - start_time,
                    question_type=QuestionType.COMPLEX,
                    confidence=100.0
                )
            else:
                SANDBOX_FAILURE.inc()
                return ComputationResult(
                    success=False,
                    error=result.get('error', 'خطأ غير معروف'),
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            SANDBOX_FAILURE.inc()
            return ComputationResult(
                success=False,
                error="خطأ في الساندبوكس",
                execution_time=time.time() - start_time
            )

sandbox = SafeSandbox(
    timeout=config.CODE_TIMEOUT,
    memory_limit=config.CODE_MEMORY_LIMIT
)

# ============================================================
# 🧮 Math Solver
# ============================================================

class MathSolver:
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')
    
    def compute(self, question: str) -> ComputationResult:
        start_time = time.time()
        
        with COMPUTATION_TIME.time():
            try:
                # تنظيف
                expr = question.replace('^', '**').replace('×', '*').replace('÷', '/')
                
                # محاولة التحويل
                try:
                    sympy_expr = sp.sympify(expr)
                except:
                    MATH_FAILURE.inc()
                    return ComputationResult(
                        success=False,
                        execution_time=time.time() - start_time
                    )
                
                # أرقام
                if sympy_expr.is_number:
                    result = sympy_expr.evalf()
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = float(result)
                    MATH_SUCCESS.inc()
                    return ComputationResult(
                        success=True,
                        result=str(result),
                        execution_time=time.time() - start_time,
                        question_type=QuestionType.COMPUTABLE,
                        confidence=100.0
                    )
                
                # معادلات
                solutions = sp.solve(sympy_expr, self.x)
                if solutions:
                    MATH_SUCCESS.inc()
                    # تنسيق النتائج
                    formatted = []
                    for sol in solutions:
                        if sol.is_number:
                            if sol.is_integer:
                                formatted.append(str(int(sol)))
                            else:
                                formatted.append(str(float(sol)))
                        else:
                            formatted.append(str(sol))
                    
                    return ComputationResult(
                        success=True,
                        result=f"الحلول: {formatted}",
                        execution_time=time.time() - start_time,
                        question_type=QuestionType.ANALYTICAL,
                        confidence=100.0
                    )
                
                MATH_FAILURE.inc()
                return ComputationResult(
                    success=False,
                    execution_time=time.time() - start_time
                )
                
            except Exception as e:
                MATH_FAILURE.inc()
                return ComputationResult(
                    success=False,
                    error=str(e),
                    execution_time=time.time() - start_time
                )

math_solver = MathSolver()

# ============================================================
# 🤖 AI Manager مع حدود يومية
# ============================================================

class AIManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.available = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-001')
                self.available = True
                logger.info("✅ Gemini configured successfully")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
    
    async def explain(self, question: str, result: str, confidence: float) -> Optional[str]:
        if not self.available:
            return None
        
        AI_CALLS.inc()
        try:
            prompt = f"""السؤال: {question}
الحل: {result}
دقة الحل: {confidence}%

اشرح هذا الحل خطوة بخطوة بطريقة تعليمية مبسطة."""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"AI explain error: {e}")
            return None
    
    async def solve_complex(self, question: str) -> Tuple[Optional[str], float]:
        if not self.available:
            return None, 0.0
        
        AI_CALLS.inc()
        try:
            prompt = f"""حل هذا السؤال بالتفصيل:
{question}

إذا كان السؤال يتطلب حسابات رياضية، استخدم Python مع sympy.
قدم الحل مع الشرح."""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.model.generate_content,
                prompt
            )
            return response.text, 70.0
        except Exception as e:
            logger.error(f"AI solve error: {e}")
            return None, 0.0

ai_manager = AIManager(config.GEMINI_API_KEY)

# ============================================================
# 🎯 Question Processor
# ============================================================

class QuestionProcessor:
    async def process(self, question: str, api_key: str = None) -> AnswerResult:
        start_time = time.time()
        
        with REQUEST_TIME.time():
            try:
                # Math Solver
                computation = math_solver.compute(question)
                
                if computation.success:
                    explanation = await ai_manager.explain(
                        question,
                        computation.result,
                        computation.confidence
                    )
                    return AnswerResult(
                        success=True,
                        answer=computation.result,
                        domain="رياضيات",
                        question_type=computation.question_type.value if computation.question_type else None,
                        confidence=computation.confidence,
                        execution_time=computation.execution_time,
                        explanation=explanation,
                        used_ai=False
                    )
                
                # Sandbox
                logger.info("Attempting sandbox...")
                sandbox_result = await sandbox.execute(question)
                
                if sandbox_result.success:
                    return AnswerResult(
                        success=True,
                        answer=sandbox_result.result,
                        question_type=QuestionType.COMPLEX.value,
                        confidence=sandbox_result.confidence,
                        execution_time=sandbox_result.execution_time,
                        used_ai=False
                    )
                
                # AI - مع التحقق من الحدود اليومية
                if ai_manager.available and api_key:
                    # التحقق من حد الـ AI للمستخدم
                    if await user_manager.check_ai_limit(api_key):
                        logger.info("Using AI (within daily limit)...")
                        ai_answer, ai_confidence = await ai_manager.solve_complex(question)
                        if ai_answer:
                            return AnswerResult(
                                success=True,
                                answer=ai_answer,
                                confidence=ai_confidence,
                                execution_time=time.time() - start_time,
                                used_ai=True
                            )
                    else:
                        logger.info("AI daily limit reached for user")
                        return AnswerResult(
                            success=False,
                            error="❌ تجاوزت الحد اليومي لاستخدام الذكاء الاصطناعي",
                            execution_time=time.time() - start_time,
                            used_ai=False
                        )
                
                return AnswerResult(
                    success=False,
                    error="❌ لم نتمكن من حل السؤال",
                    execution_time=time.time() - start_time
                )
                
            except Exception as e:
                logger.error(f"Processor error: {e}")
                return AnswerResult(
                    success=False,
                    error="❌ حدث خطأ داخلي",
                    execution_time=time.time() - start_time
                )

processor = QuestionProcessor()

# ============================================================
# 🎯 المسارات
# ============================================================

@app.get("/")
async def home():
    REQUEST_COUNT.inc()
    return {
        "name": "Mechatronics Assistant v15.0",
        "status": "running",
        "environment": config.ENVIRONMENT,
        "gemini_available": ai_manager.available,
        "features": ["Math Solver", "Sandbox", "AI Explanation"]
    }

@app.post("/api/ask", response_model=AnswerResponse)
async def ask(
    request: Request,
    question_req: QuestionRequest,
    api_key: str = Depends(api_key_header)
):
    """معالجة الأسئلة"""
    
    # التحقق من المفتاح
    user = await user_manager.verify_key(api_key) if api_key else None
    
    # التحقق من الحد اليومي
    if api_key and user:
        allowed, message, user_data = await user_manager.check_rate_limit(api_key)
        if not allowed:
            raise HTTPException(status_code=429, detail=message)
        await user_manager.increment_usage(api_key)
    
    # معالجة السؤال
    result = await processor.process(question_req.question, api_key)
    
    if not result.success:
        ERROR_COUNT.inc()
    
    return AnswerResponse(
        success=result.success,
        answer=result.answer,
        domain=result.domain,
        question_type=result.question_type,
        confidence=result.confidence,
        error=result.error,
        time=result.execution_time,
        cached=result.cached,
        explanation=result.explanation,
        remaining_quota=user_manager.get_remaining(api_key) if api_key else None,
        used_ai=result.used_ai
    )

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "gemini": ai_manager.available,
        "version": "15.0.0",
        "environment": config.ENVIRONMENT
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "math": {
            "success": MATH_SUCCESS._value.get(),
            "failure": MATH_FAILURE._value.get()
        },
        "sandbox": {
            "success": SANDBOX_SUCCESS._value.get(),
            "failure": SANDBOX_FAILURE._value.get()
        },
        "ai": {
            "calls": AI_CALLS._value.get(),
            "daily_limit_hits": AI_DAILY_LIMIT_HITS._value.get()
        },
        "cache": cache.get_stats(),
        "environment": config.ENVIRONMENT
    }

# ============================================================
# 🚀 Startup
# ============================================================

@app.on_event("startup")
async def startup():
    logger.info(f"✅ Application started on Render")
    logger.info(f"✅ Environment: {config.ENVIRONMENT}")
    logger.info(f"✅ Gemini: {'✅ Available' if ai_manager.available else '❌ Disabled'}")

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    
    print("\n" + "="*80)
    print("🔥 MECHATRONICS ASSISTANT v15.0 - Render Ready")
    print("="*80)
    print(f"✅ Port: {port}")
    print(f"✅ Environment: {config.ENVIRONMENT}")
    print(f"✅ Gemini: {'✅' if ai_manager.available else '❌'}")
    print(f"✅ AI Daily Limits: Free={config.AI_DAILY_LIMIT_FREE}, Premium={config.AI_DAILY_LIMIT_PREMIUM}")
    print(f"✅ Rate Limits: {config.RATE_LIMIT_FREE}")
    print("="*80)
    print(f"🌐 http://0.0.0.0:{port}")
    print("🔍 Health: /api/health")
    print("="*80 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
