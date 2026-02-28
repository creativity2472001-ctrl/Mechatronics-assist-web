#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - الإصدار النهائي للإنتاج v29.0
Math Intent Engine Pro - مع خطوات الحل + الذاكرة الذاتية
"""

from flask import Flask, render_template, request, jsonify
import os
import hashlib
import sqlite3
import logging
import re
import time
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, Tuple, List, Set, Union

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application,
    convert_xor,
    implicit_multiplication
)

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

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# ============================================================
# 🔧 إعدادات التطبيق
# ============================================================

class Config:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    RATE_LIMIT = int(os.getenv('RATE_LIMIT', '10'))
    CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '1000'))
    CACHE_TTL_DAYS = int(os.getenv('CACHE_TTL_DAYS', '30'))
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '127.0.0.1')
    UNANSWERED_DB = 'unanswered.db'

config = Config()

# تهيئة Gemini إذا وجد المفتاح
gemini_model = None
if config.GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=config.GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-001')
        logger.info("✅ Gemini configured")
    except Exception as e:
        logger.error(f"❌ Gemini config error: {e}")

# ============================================================
# 💾 نظام الذاكرة الذاتية المتقدم
# ============================================================

class SelfLearningMemory:
    """
    نظام ذاكرة ذاتي يتعلم من الأسئلة الجديدة
    يقوم بتخزين الأسئلة غير المحلولة وإجاباتها من LLM
    """
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
        logger.info("✅ SelfLearningMemory initialized")
    
    def _init_db(self):
        """تهيئة قاعدة البيانات"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # جدول الذاكرة الرئيسي
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_hash TEXT UNIQUE NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT,
                        steps TEXT,
                        solved_by TEXT DEFAULT 'pending',
                        confidence REAL DEFAULT 0.0,
                        category TEXT,
                        asked_count INTEGER DEFAULT 1,
                        first_asked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_asked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        solved_at TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                
                # جدول الأسئلة غير المحلولة
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS unanswered (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question_hash TEXT UNIQUE NOT NULL,
                        question TEXT NOT NULL,
                        asked_count INTEGER DEFAULT 1,
                        first_asked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_asked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        sent_to_llm BOOLEAN DEFAULT 0,
                        llm_response TEXT,
                        llm_model TEXT,
                        answered_at TIMESTAMP
                    )
                """)
                
                # جدول التعلم التدريجي
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern TEXT NOT NULL,
                        template TEXT NOT NULL,
                        category TEXT,
                        confidence REAL DEFAULT 0.5,
                        used_count INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
                # إنشاء الفهارس
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_hash ON memory(question_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_unanswered_hash ON unanswered(question_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_unanswered_sent ON unanswered(sent_to_llm)")
                
        except Exception as e:
            logger.error(f"❌ Memory DB init error: {e}")
    
    def get_from_memory(self, question: str) -> Optional[Dict]:
        """البحث في الذاكرة عن سؤال سابق"""
        try:
            q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT question, answer, steps, solved_by, confidence, category 
                    FROM memory 
                    WHERE question_hash = ? AND answer IS NOT NULL
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                """, (q_hash,))
                
                row = cursor.fetchone()
                if row:
                    # تحديث عدد المرات
                    conn.execute("""
                        UPDATE memory 
                        SET asked_count = asked_count + 1, last_asked = CURRENT_TIMESTAMP
                        WHERE question_hash = ?
                    """, (q_hash,))
                    conn.commit()
                    
                    logger.info(f"✅ Found in memory: {q_hash[:8]}...")
                    return dict(row)
                    
        except Exception as e:
            logger.error(f"❌ Memory read error: {e}")
        
        return None
    
    def add_to_memory(self, question: str, answer: str, steps: str = None, 
                     solved_by: str = "local", confidence: float = 1.0, 
                     category: str = None):
        """إضافة حل جديد إلى الذاكرة"""
        try:
            q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            expires_at = (datetime.now() + timedelta(days=365)).isoformat()  # سنة صلاحية
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory 
                    (question_hash, question, answer, steps, solved_by, confidence, category, expires_at) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (q_hash, question[:500], answer, steps, solved_by, confidence, category, expires_at))
                conn.commit()
                
                # إذا كان السؤال في unanswered، نعلم أنه تم حله
                conn.execute("""
                    UPDATE unanswered 
                    SET sent_to_llm = 1, llm_response = ?, answered_at = CURRENT_TIMESTAMP
                    WHERE question_hash = ? AND sent_to_llm = 1
                """, (answer, q_hash))
                conn.commit()
                
                logger.info(f"✅ Added to memory: {q_hash[:8]}...")
                
        except Exception as e:
            logger.error(f"❌ Memory write error: {e}")
    
    def add_unanswered(self, question: str):
        """تسجيل سؤال لم يتم حله محلياً"""
        try:
            q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                # التحقق من وجود السؤال
                cursor = conn.execute("SELECT id FROM unanswered WHERE question_hash = ?", (q_hash,))
                if cursor.fetchone():
                    # تحديث عدد المرات
                    conn.execute("""
                        UPDATE unanswered 
                        SET asked_count = asked_count + 1, last_asked = CURRENT_TIMESTAMP
                        WHERE question_hash = ?
                    """, (q_hash,))
                else:
                    # إضافة سؤال جديد
                    conn.execute("""
                        INSERT INTO unanswered (question_hash, question)
                        VALUES (?, ?)
                    """, (q_hash, question[:500]))
                
                conn.commit()
                logger.info(f"📝 Unanswered logged: {q_hash[:8]}...")
                
        except Exception as e:
            logger.error(f"❌ Unanswered write error: {e}")
    
    def get_next_for_llm(self, limit: int = 5) -> List[Dict]:
        """الحصول على الأسئلة التالية لإرسالها إلى LLM"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT question_hash, question, asked_count 
                    FROM unanswered 
                    WHERE sent_to_llm = 0 
                    ORDER BY asked_count DESC, last_asked ASC
                    LIMIT ?
                """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"❌ Get next for LLM error: {e}")
            return []
    
    def mark_sent_to_llm(self, question_hash: str, model: str = "gemini"):
        """تحديث أن السؤال أرسل إلى LLM"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE unanswered 
                    SET sent_to_llm = 1, llm_model = ?
                    WHERE question_hash = ?
                """, (model, question_hash))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Mark sent error: {e}")
    
    def learn_from_pattern(self, question: str, answer: str, category: str):
        """تعلم نمط جديد من الأسئلة المحلولة"""
        try:
            # استخراج نمط مبسط (للتحسين المستقبلي)
            pattern = self._extract_pattern(question)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning (pattern, template, category)
                    VALUES (?, ?, ?)
                """, (pattern, answer[:200], category))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Learning error: {e}")
    
    def _extract_pattern(self, question: str) -> str:
        """استخراج نمط من السؤال (تبسيط)"""
        # إزالة الأرقام
        pattern = re.sub(r'\d+', 'N', question)
        # إزالة المتغيرات
        pattern = re.sub(r'[a-zA-Z]', 'V', pattern)
        return pattern[:100]
    
    def get_stats(self) -> Dict:
        """إحصائيات الذاكرة"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                memory_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM unanswered WHERE sent_to_llm = 0")
                pending_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM unanswered WHERE sent_to_llm = 1")
                sent_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM learning")
                patterns = cursor.fetchone()[0]
                
                return {
                    "memory": memory_count,
                    "pending": pending_count,
                    "sent_to_llm": sent_count,
                    "learned_patterns": patterns
                }
        except:
            return {"memory": 0, "pending": 0, "sent_to_llm": 0, "learned_patterns": 0}

# ============================================================
# 📝 Step-by-Step Solution Generator - مولد خطوات الحل
# ============================================================

class StepByStepSolver:
    """
    مولد خطوات الحل لكل نوع من المسائل
    يعطي شرحاً تفصيلياً مع كل خطوة
    """
    
    def __init__(self):
        self.x = sp.symbols('x')
        self.y = sp.symbols('y')
        self.z = sp.symbols('z')
        
    def solve_with_steps(self, question: str, intent: str, expr_str: str) -> Dict:
        """حل المسألة مع خطوات تفصيلية"""
        
        solvers = {
            'diff': self._derivative_steps,
            'integrate': self._integral_steps,
            'limit': self._limit_steps,
            'solve': self._equation_steps,
            'system': self._system_steps,
            'sum': self._series_steps,
            'root': self._root_steps,
            'factor': self._factor_steps,
            'expand': self._expand_steps,
            'simplify': self._simplify_steps
        }
        
        solver = solvers.get(intent)
        if solver:
            return solver(question, expr_str)
        
        return {"result": "لا توجد خطوات متاحة لهذه المسألة", "steps": []}
    
    # ============================================================
    # خطوات المشتقات
    # ============================================================
    
    def _derivative_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل المشتقات"""
        steps = []
        
        try:
            expr = sp.sympify(expr_str)
            var = list(expr.free_symbols)[0] if expr.free_symbols else self.x
            
            steps.append(f"**المطلوب:** إيجاد مشتقة {expr_str} بالنسبة لـ {var}")
            steps.append(f"**القانون:** d/d{var} [f({var})] = f'({var})")
            
            # تحليل نوع الدالة
            if expr.has(sp.sin):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة تحتوي على sin")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة مشتقة sin: d/dx sin(u) = cos(u) · du/dx")
                inner = self._get_inner_function(expr, sp.sin)
                if inner:
                    steps.append(f"**الخطوة 3:** الدالة الداخلية u = {inner}")
                    
            elif expr.has(sp.cos):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة تحتوي على cos")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة مشتقة cos: d/dx cos(u) = -sin(u) · du/dx")
                
            elif expr.has(sp.exp) or 'exp' in str(expr):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة أُسية")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة مشتقة e^u: d/dx e^u = e^u · du/dx")
                
            elif expr.has(sp.log):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة لوغاريتمية")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة مشتقة ln|u|: d/dx ln|u| = (1/u) · du/dx")
                
            elif expr.is_Pow:
                base, exp = expr.as_base_exp()
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة هي {base}^{exp}")
                if exp.is_number:
                    steps.append(f"**الخطوة 2:** نستخدم قاعدة القوة: d/dx x^n = n·x^(n-1)")
                    steps.append(f"**الخطوة 3:** المشتقة = {exp}·{base}^{exp-1}")
            
            # حساب المشتقة
            derivative = sp.diff(expr, var)
            
            steps.append(f"\n**الخطوة النهائية:**")
            steps.append(f"d/d{var} ({expr_str}) = {derivative}")
            
            return {
                "result": f"**النتيجة النهائية:** {derivative}",
                "steps": steps,
                "answer": str(derivative)
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    def _get_inner_function(self, expr, func):
        """استخراج الدالة الداخلية"""
        for arg in expr.args:
            if arg.has(func):
                for sub_arg in arg.args:
                    return sub_arg
        return None
    
    # ============================================================
    # خطوات التكاملات
    # ============================================================
    
    def _integral_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل التكاملات"""
        steps = []
        
        try:
            expr = sp.sympify(expr_str)
            var = list(expr.free_symbols)[0] if expr.free_symbols else self.x
            
            steps.append(f"**المطلوب:** إيجاد تكامل ∫ {expr_str} d{var}")
            
            # تحليل نوع التكامل
            if expr.is_Pow:
                base, exp = expr.as_base_exp()
                if exp == -1:
                    steps.append(f"**الخطوة 1:** هذه صيغة خاصة: ∫ 1/{base} d{var}")
                    steps.append(f"**الخطوة 2:** نستخدم القاعدة: ∫ 1/u du = ln|u| + C")
                else:
                    steps.append(f"**الخطوة 1:** نستخدم قاعدة تكامل القوة: ∫ x^n dx = x^(n+1)/(n+1) + C")
                    steps.append(f"**الخطوة 2:** n = {exp}, إذن n+1 = {exp+1}")
                    
            elif expr.has(sp.sin):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة تحتوي على sin")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة تكامل sin: ∫ sin(u) du = -cos(u) + C")
                
            elif expr.has(sp.cos):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة تحتوي على cos")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة تكامل cos: ∫ cos(u) du = sin(u) + C")
                
            elif expr.has(sp.exp):
                steps.append(f"**الخطوة 1:** نلاحظ أن الدالة أُسية")
                steps.append(f"**الخطوة 2:** نستخدم قاعدة تكامل e^u: ∫ e^u du = e^u + C")
            
            # حساب التكامل
            integral = sp.integrate(expr, var)
            
            steps.append(f"\n**الخطوة النهائية:**")
            steps.append(f"∫ {expr_str} d{var} = {integral} + C")
            
            return {
                "result": f"**النتيجة النهائية:** {integral} + C",
                "steps": steps,
                "answer": str(integral) + " + C"
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات النهايات
    # ============================================================
    
    def _limit_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل النهايات مع قاعدة لوبيتال"""
        steps = []
        
        try:
            # استخراج النقطة
            point_match = re.search(r'→\s*([\d.]+|∞|inf)', question)
            point = 0
            if point_match:
                p = point_match.group(1)
                if p in ['∞', 'inf']:
                    point = sp.oo
                else:
                    point = float(p)
            
            expr = sp.sympify(expr_str)
            var = list(expr.free_symbols)[0] if expr.free_symbols else self.x
            
            steps.append(f"**المطلوب:** إيجاد نهاية {expr_str} عندما {var} → {point}")
            
            # محاولة التعويض المباشر
            try:
                direct = expr.subs(var, point)
                steps.append(f"**الخطوة 1:** نعوض {var} = {point} مباشرة:")
                steps.append(f"{expr_str} = {direct}")
                
                if direct.is_finite and direct != sp.nan:
                    steps.append(f"**الخطوة 2:** النهاية موجودة وقيمتها {direct}")
                    return {
                        "result": f"**النتيجة:** {direct}",
                        "steps": steps,
                        "answer": str(direct)
                    }
                else:
                    steps.append(f"**الخطوة 2:** التعويض المباشر يعطي كمية غير معينة ({direct})")
                    
                    # التحقق من قابلية تطبيق لوبيتال
                    num, den = expr.as_numer_denom()
                    steps.append(f"**الخطوة 3:** نكتب الدالة ككسر: ({num})/({den})")
                    
                    # تطبيق لوبيتال
                    num_deriv = sp.diff(num, var)
                    den_deriv = sp.diff(den, var)
                    
                    steps.append(f"**الخطوة 4:** نطبق قاعدة لوبيتال (نشتق البسط والمقام):")
                    steps.append(f"البسط بعد الاشتقاق: {num_deriv}")
                    steps.append(f"المقام بعد الاشتقاق: {den_deriv}")
                    
                    # حساب النهاية الجديدة
                    new_limit = sp.limit(num_deriv/den_deriv, var, point)
                    steps.append(f"**الخطوة 5:** النهاية الجديدة = {new_limit}")
                    
                    limit = new_limit
            except:
                limit = sp.limit(expr, var, point)
            
            steps.append(f"\n**الخطوة النهائية:**")
            steps.append(f"lim_{var}→{point} {expr_str} = {limit}")
            
            return {
                "result": f"**النتيجة النهائية:** {limit}",
                "steps": steps,
                "answer": str(limit)
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات حل المعادلات
    # ============================================================
    
    def _equation_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل المعادلات"""
        steps = []
        
        try:
            if '=' not in expr_str:
                return {"result": "ليست معادلة", "steps": []}
            
            left, right = expr_str.split('=')
            left_expr = sp.sympify(left)
            right_expr = sp.sympify(right)
            
            # نقل الكل لطرف واحد
            equation = left_expr - right_expr
            var = list(equation.free_symbols)[0] if equation.free_symbols else self.x
            
            steps.append(f"**المعادلة:** {left} = {right}")
            steps.append(f"**الخطوة 1:** ننقل جميع الحدود لطرف واحد:")
            steps.append(f"{equation} = 0")
            
            # تحليل نوع المعادلة
            if equation.is_polynomial():
                degree = sp.degree(equation, var)
                steps.append(f"**الخطوة 2:** هذه معادلة من الدرجة {degree}")
                
                if degree == 1:
                    steps.append(f"**الخطوة 3:** معادلة خطية، نحلها بعزل {var}")
                    # استخراج المعاملات
                    coeffs = equation.as_poly(var).all_coeffs()
                    if len(coeffs) == 2:
                        a, b = coeffs
                        steps.append(f"المعادلة بالصيغة: {a}x + {b} = 0")
                        steps.append(f"x = -{b}/{a} = {-b/a}")
                    
                elif degree == 2:
                    steps.append(f"**الخطوة 3:** معادلة تربيعية، نستخدم القانون العام")
                    coeffs = equation.as_poly(var).all_coeffs()
                    if len(coeffs) == 3:
                        a, b, c = coeffs
                        steps.append(f"a = {a}, b = {b}, c = {c}")
                        discriminant = b**2 - 4*a*c
                        steps.append(f"**الخطوة 4:** نحسب المميز Δ = b² - 4ac = {discriminant}")
                        
                        if discriminant > 0:
                            steps.append(f"Δ > 0 → حلان حقيقيان")
                            x1 = (-b + sp.sqrt(discriminant)) / (2*a)
                            x2 = (-b - sp.sqrt(discriminant)) / (2*a)
                            steps.append(f"x₁ = (-b + √Δ)/(2a) = {x1}")
                            steps.append(f"x₂ = (-b - √Δ)/(2a) = {x2}")
                        elif discriminant == 0:
                            steps.append(f"Δ = 0 → حل مزدوج")
                            x = -b / (2*a)
                            steps.append(f"x = -b/(2a) = {x}")
                        else:
                            steps.append(f"Δ < 0 → حلان مركبان")
                            real = -b / (2*a)
                            imag = sp.sqrt(-discriminant) / (2*a)
                            steps.append(f"x₁ = {real} + {imag}i")
                            steps.append(f"x₂ = {real} - {imag}i")
            
            # حل المعادلة
            solutions = sp.solve(equation, var)
            
            steps.append(f"\n**الخطوة النهائية:**")
            if len(solutions) == 1:
                steps.append(f"{var} = {solutions[0]}")
            else:
                for i, sol in enumerate(solutions, 1):
                    steps.append(f"{var}_{i} = {sol}")
            
            return {
                "result": f"**الحلول:** {solutions}",
                "steps": steps,
                "answer": str(solutions)
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات حل أنظمة المعادلات
    # ============================================================
    
    def _system_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل أنظمة المعادلات"""
        steps = []
        
        try:
            steps.append("**حل نظام المعادلات:**")
            
            # استخراج المعادلات
            equations = re.findall(r'([^,]+=[^,]+)', question)
            
            if len(equations) < 2:
                return {"result": "لم يتم العثور على معادلتين", "steps": steps}
            
            steps.append(f"المعادلة الأولى: {equations[0]}")
            steps.append(f"المعادلة الثانية: {equations[1]}")
            
            steps.append("\n**طريقة الحل (بالتعويض):**")
            steps.append("1. نعزل أحد المتغيرات من المعادلة الأولى")
            steps.append("2. نعوض في المعادلة الثانية")
            steps.append("3. نحل المعادلة الناتجة")
            steps.append("4. نعوض الناتج لإيجاد المتغير الآخر")
            
            # محاولة الحل باستخدام SymPy
            try:
                vars = set()
                for eq in equations:
                    for c in eq:
                        if c.isalpha() and c not in ['x', 'y']:
                            vars.add(c)
                
                if not vars:
                    vars = {'x', 'y'}
                
                symbols = {v: sp.symbols(v) for v in vars}
                
                eq1 = sp.Eq(*[sp.sympify(part) for part in equations[0].split('=')])
                eq2 = sp.Eq(*[sp.sympify(part) for part in equations[1].split('=')])
                
                solution = sp.solve([eq1, eq2], list(symbols.values()))
                
                steps.append("\n**الحل:**")
                if isinstance(solution, list):
                    for sol in solution:
                        for var, val in sol.items():
                            steps.append(f"{var} = {val}")
                elif isinstance(solution, dict):
                    for var, val in solution.items():
                        steps.append(f"{var} = {val}")
            except:
                steps.append("\n**ملاحظة:** يمكن استخدام طريقة المصفوفات أيضاً")
            
            return {
                "result": "تم إيجاد حل النظام",
                "steps": steps,
                "answer": str(solution) if 'solution' in locals() else "يمكن حله بالطرق المذكورة"
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات المتسلسلات
    # ============================================================
    
    def _series_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حل المتسلسلات"""
        steps = []
        
        try:
            steps.append("**حساب المتسلسلة:**")
            
            # محاولة استخراج حدود المتسلسلة
            sigma_match = re.search(r'Σ\s*[_{]?\s*([a-zA-Z])\s*=\s*(\d+)\s*[}\^]?\s*[\^]?\s*([∞\d]+)?\s*(.+)', question)
            
            if sigma_match:
                var = sigma_match.group(1)
                start = int(sigma_match.group(2))
                end = sigma_match.group(3)
                expr = sigma_match.group(4)
                
                steps.append(f"المتغير: {var}")
                steps.append(f"البداية: {start}")
                steps.append(f"النهاية: {end if end else '∞'}")
                steps.append(f"التعبير: {expr}")
                
                if end and end.isdigit():
                    end_val = int(end)
                    steps.append(f"\n**الخطوات التفصيلية:**")
                    total = 0
                    for i in range(start, end_val + 1):
                        term = expr.replace(var, str(i))
                        try:
                            val = eval(term)
                            steps.append(f"عند {var} = {i}: {term} = {val}")
                            total += val
                        except:
                            steps.append(f"عند {var} = {i}: {term}")
                    steps.append(f"\n**المجموع الكلي = {total}**")
                else:
                    steps.append("\n**متسلسلة لا نهائية:**")
                    steps.append("لحساب المتسلسلات اللانهائية، نستخدم اختبارات التقارب:")
                    steps.append("1. اختبار النسبة")
                    steps.append("2. اختبار الجذر")
                    steps.append("3. اختبار التكامل")
            
            return {
                "result": "نتيجة المتسلسلة",
                "steps": steps,
                "answer": "يمكن حسابها بالطرق المذكورة"
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات الجذور
    # ============================================================
    
    def _root_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات حساب الجذور"""
        steps = []
        
        try:
            # استخراج العدد
            num_match = re.search(r'(\d+)', question)
            if num_match:
                number = int(num_match.group(1))
                
                steps.append(f"**المطلوب:** إيجاد جذر العدد {number}")
                
                # تحليل العدد
                factors = []
                n = number
                i = 2
                while i * i <= n:
                    while n % i == 0:
                        factors.append(i)
                        n //= i
                    i += 1
                if n > 1:
                    factors.append(n)
                
                if factors:
                    steps.append(f"**الخطوة 1:** نحلل العدد {number} إلى عوامله الأولية")
                    steps.append(f"{number} = {' × '.join(map(str, factors))}")
                    
                    # تجميع العوامل المكررة
                    from collections import Counter
                    factor_counts = Counter(factors)
                    
                    steps.append(f"**الخطوة 2:** نجمع العوامل المكررة")
                    root_type = 2
                    if 'تكعيبي' in question:
                        root_type = 3
                    elif 'رباعي' in question:
                        root_type = 4
                    
                    pairs = []
                    for f, count in factor_counts.items():
                        steps.append(f"العامل {f} تكرر {count} مرة")
                        pairs.append(count // root_type)
                    
                    # حساب الجذر
                    result = 1
                    for f, pair in zip(factor_counts.keys(), pairs):
                        if pair > 0:
                            result *= f ** pair
                            steps.append(f"نخرج {f}^{pair} خارج الجذر")
                    
                    remaining = number // (result ** root_type)
                    if remaining > 1:
                        steps.append(f"يتبقى داخل الجذر: {remaining}")
                    
                    steps.append(f"\n**الخطوة 3:** نبسط الجذر")
                    
                    if remaining == 1:
                        steps.append(f"الجذر التبسيط = {result}")
                    else:
                        steps.append(f"الجذر التبسيط = {result} · {get_root_symbol(root_type)}{remaining}")
            
            # حساب الجذر
            if 'تربيعي' in question:
                result = sp.sqrt(number)
            elif 'تكعيبي' in question:
                result = sp.root(number, 3)
            else:
                result = sp.sqrt(number)
            
            steps.append(f"\n**النتيجة النهائية:**")
            steps.append(f"الجذر = {result}")
            if hasattr(result, 'evalf'):
                steps.append(f"القيمة التقريبية = {result.evalf():.6f}")
            
            return {
                "result": f"**النتيجة:** {result}",
                "steps": steps,
                "answer": str(result)
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    # ============================================================
    # خطوات التحليل والنشر والتبسيط
    # ============================================================
    
    def _factor_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات تحليل العبارات"""
        steps = []
        
        try:
            expr = sp.sympify(expr_str)
            
            steps.append(f"**المطلوب:** تحليل {expr_str}")
            
            if expr.is_polynomial():
                steps.append(f"**الخطوة 1:** نبحث عن العامل المشترك الأكبر")
                
                # البحث عن العامل المشترك الأكبر
                terms = expr.as_ordered_terms()
                if len(terms) > 1:
                    # استخراج المعاملات
                    coeffs = [abs(term.as_coeff_Mul()[0]) for term in terms]
                    from math import gcd
                    common_coeff = 1
                    for c in coeffs:
                        if hasattr(c, 'p'):  # إذا كان عدداً نسبياً
                            common_coeff = gcd(common_coeff, c.p)
                    
                    if common_coeff > 1:
                        steps.append(f"المعامل المشترك الأكبر = {common_coeff}")
                    
                    # البحث عن المتغيرات المشتركة
                    var_powers = {}
                    for term in terms:
                        for var in term.free_symbols:
                            power = term.as_poly(var).degree()
                            if var not in var_powers or power < var_powers[var]:
                                var_powers[var] = power
                    
                    if var_powers:
                        steps.append("المتغيرات المشتركة: " + ", ".join([f"{var}^{power}" for var, power in var_powers.items() if power > 0]))
            
            factored = sp.factor(expr)
            steps.append(f"\n**النتيجة النهائية:** {factored}")
            
            return {
                "result": f"**التحليل:** {factored}",
                "steps": steps,
                "answer": str(factored)
            }
            
        except Exception as e:
            return {"result": f"خطأ في الحساب: {e}", "steps": []}
    
    def _expand_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات النشر"""
        steps = []
        try:
            expr = sp.sympify(expr_str)
            steps.append(f"**نشر التعبير:** {expr_str}")
            
            if expr.is_Pow and expr.exp.is_number and expr.exp > 1:
                steps.append(f"**الخطوة 1:** نستخدم نظرية ذات الحدين")
                steps.append(f"(a + b)^{expr.exp} = Σ C({expr.exp}, k) a^{expr.exp-k} b^k")
            
            expanded = sp.expand(expr)
            steps.append(f"**النتيجة:** {expanded}")
            
            return {"result": str(expanded), "steps": steps, "answer": str(expanded)}
        except Exception as e:
            return {"result": f"خطأ: {e}", "steps": []}
    
    def _simplify_steps(self, question: str, expr_str: str) -> Dict:
        """خطوات التبسيط"""
        steps = []
        try:
            expr = sp.sympify(expr_str)
            steps.append(f"**تبسيط التعبير:** {expr_str}")
            
            steps.append("**الخطوة 1:** نجمع الحدود المتشابهة")
            steps.append("**الخطوة 2:** نبسط الكسور إن وجدت")
            steps.append("**الخطوة 3:** نستخدم القوانين الرياضية")
            
            simplified = sp.simplify(expr)
            steps.append(f"**النتيجة:** {simplified}")
            
            return {"result": str(simplified), "steps": steps, "answer": str(simplified)}
        except Exception as e:
            return {"result": f"خطأ: {e}", "steps": []}

def get_root_symbol(root_type):
    """الحصول على رمز الجذر"""
    symbols = {2: '√', 3: '∛', 4: '∜'}
    return symbols.get(root_type, f'{root_type}√')

# ============================================================
# 🧠 Math Intent Engine Pro - النسخة المتكاملة مع الخطوات والذاكرة
# ============================================================

class MathIntentEngine:
    """محرك الرياضيات الذكي - مع خطوات الحل والذاكرة الذاتية"""
    
    def __init__(self):
        self.variables_cache = {}
        
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application, convert_xor)
        )
        
        self.allowed_functions = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'cot': sp.cot, 'sec': sp.sec, 'csc': sp.csc,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'log': sp.log, 'ln': sp.log, 'exp': sp.exp,
            'sqrt': sp.sqrt, 'Abs': sp.Abs,
        }
        
        # المحسنات
        self.root_parser = RootExpressionParser()
        self.step_solver = StepByStepSolver()
        
        # قواعد الكشف
        self.keywords = {
            'solve': ['حل', 'solve', 'معادلة', 'equation', 'أوجد', 'find'],
            'diff': ['مشتقة', 'diff', 'derivative', 'اشتق', 'dy/dx'],
            'integrate': ['تكامل', 'integral', '∫', 'integrate'],
            'limit': ['نهاية', 'limit', 'lim', '→'],
            'sum': ['مجموع', 'sum', 'Σ', 'sigma', 'متسلسلة'],
            'product': ['جداء', 'product', '∏', 'pi'],
            'factor': ['تحليل', 'factor', 'factorize'],
            'expand': ['نشر', 'expand', 'توسيع', 'فك'],
            'simplify': ['تبسيط', 'simplify', 'بسط'],
            'inequality': ['متباينة', 'inequality', '>', '<', '≥', '≤'],
            'root': ['جذر', 'root', '√', '∛', '∜', 'الجذر'],
            'absolute': ['قيمة مطلقة', 'absolute', '|', 'abs'],
            'system': ['نظام', 'system', 'معادلتين']
        }
        
        # قوالب جاهزة
        self.templates = self._build_templates()
        self.intents = self._build_intents()
        
        logger.info("✅ MathIntentEngine v29.0 initialized with steps & memory")
    
    def _build_templates(self):
        """بناء القوالب"""
        templates = {}
        
        # المعادلات
        templates.update({
            'quadratic': {
                'pattern': r'([+-]?\d*\.?\d*)\s*\*?\s*x\^2\s*([+-]\s*\d*\.?\d*)\s*\*?\s*x\s*([+-]\s*\d*\.?\d*)\s*=\s*0',
                'handler': self._template_quadratic,
                'confidence': 1.0
            },
            'linear': {
                'pattern': r'([+-]?\d*\.?\d*)\s*\*?\s*x\s*([+-]\s*\d*\.?\d*)\s*=\s*([+-]?\d*\.?\d*)',
                'handler': self._template_linear,
                'confidence': 1.0
            }
        })
        
        # المشتقات
        templates.update({
            'sin_derivative': {
                'pattern': r'مشتقة\s*sin\s*\(\s*(\d*\.?\d*)\s*\*?\s*x\s*\)',
                'handler': self._template_sin_derivative,
                'confidence': 1.0
            },
            'cos_derivative': {
                'pattern': r'مشتقة\s*cos\s*\(\s*(\d*\.?\d*)\s*\*?\s*x\s*\)',
                'handler': self._template_cos_derivative,
                'confidence': 1.0
            }
        })
        
        # الجذور
        templates.update({
            'root_square': {
                'pattern': r'(?:جذر|الجذر)\s+(?:التربيعي)?\s*(?:للعدد|لعدد|ل)?\s*(\d+(?:\.\d+)?)',
                'handler': self._template_root_square,
                'confidence': 1.0
            },
            'root_cube': {
                'pattern': r'(?:جذر|الجذر)\s+التكعيبي\s*(?:للعدد|لعدد|ل)?\s*(\d+(?:\.\d+)?)',
                'handler': self._template_root_cube,
                'confidence': 1.0
            }
        })
        
        return templates
    
    def _build_intents(self):
        """بناء النوايا"""
        intents = []
        for name, keywords in self.keywords.items():
            handler_name = f"_handle_{name}"
            if hasattr(self, handler_name):
                intents.append((name, keywords, getattr(self, handler_name), 0.95))
            else:
                intents.append((name, keywords, self._handle_generic, 0.90))
        
        intents.append(('calculate', [], self._handle_calculate, 0.98))
        return intents
    
    # ============================================================
    # معالجات القوالب
    # ============================================================
    
    def _template_quadratic(self, match):
        """معالجة المعادلة التربيعية"""
        a, b, c = match.groups()
        a = float(a) if a and a not in '+-' else 1.0
        b = float(b.replace(' ', '')) if b else 0.0
        c = float(c.replace(' ', '')) if c else 0.0
        
        x = sp.symbols('x')
        expr = a*x**2 + b*x + c
        solutions = sp.solve(expr, x)
        
        discriminant = b**2 - 4*a*c
        
        result = f"**المعادلة: {a}x² + {b}x + {c} = 0**\n\n"
        result += f"المميز (Δ) = {discriminant}\n\n"
        
        if discriminant > 0:
            result += f"حلان حقيقيان:\n"
            result += f"x₁ = {solutions[0]}\n"
            result += f"x₂ = {solutions[1]}"
        elif discriminant == 0:
            result += f"حل مزدوج:\nx = {solutions[0]}"
        else:
            result += f"حلان مركبان:\n{solutions[0]}, {solutions[1]}"
        
        return result
    
    def _template_linear(self, match):
        """معالجة المعادلة الخطية"""
        a, b, c = match.groups()
        a = float(a) if a and a not in '+-' else 1.0
        b = float(b.replace(' ', '')) if b else 0.0
        c = float(c.replace(' ', ''))
        
        x_val = (c - b) / a
        return f"**حل المعادلة:**\n{a}x + {b} = {c}\n\nx = {x_val}"
    
    def _template_sin_derivative(self, match):
        """مشتقة sin"""
        k = match.group(1)
        k = float(k) if k else 1.0
        return f"مشتقة sin({k}x) = {k}·cos({k}x)"
    
    def _template_cos_derivative(self, match):
        """مشتقة cos"""
        k = match.group(1)
        k = float(k) if k else 1.0
        return f"مشتقة cos({k}x) = -{k}·sin({k}x)"
    
    def _template_root_square(self, match):
        """جذر تربيعي"""
        num = float(match.group(1))
        result = sp.sqrt(num)
        return f"√{num} = {result}"
    
    def _template_root_cube(self, match):
        """جذر تكعيبي"""
        num = float(match.group(1))
        result = sp.root(num, 3)
        return f"∛{num} = {result}"
    
    def check_templates(self, question: str) -> Tuple[Optional[str], float, str]:
        """التحقق من القوالب"""
        for template_name, template in self.templates.items():
            try:
                match = re.search(template['pattern'], question, re.IGNORECASE | re.UNICODE)
                if match:
                    result = template['handler'](match)
                    return result, template['confidence'], template_name
            except Exception as e:
                continue
        return None, 0.0, None
    
    def safe_parse(self, expr_str: str) -> Optional[sp.Expr]:
        """تحليل آمن للتعبير"""
        try:
            expr_str = expr_str.replace('^', '**').replace(' ', '')
            variables = self._extract_variables(expr_str)
            
            local_dict = {}
            for var in variables:
                if var not in self.variables_cache:
                    self.variables_cache[var] = sp.symbols(var)
                local_dict[var] = self.variables_cache[var]
            
            local_dict.update(self.allowed_functions)
            
            return parse_expr(
                expr_str,
                transformations=self.transformations,
                local_dict=local_dict,
                evaluate=True
            )
        except Exception as e:
            return None
    
    def _extract_variables(self, expr_str: str) -> Set[str]:
        """استخراج المتغيرات"""
        pattern = r'\b[a-zA-Z]\b'
        return set(re.findall(pattern, expr_str))
    
    def detect_intent(self, question: str) -> Tuple[str, float]:
        """كشف نية السؤال"""
        q = question.lower().strip()
        scores = {}
        
        for intent_name, keywords, _, _ in self.intents:
            score = sum(1 for keyword in keywords if keyword in q)
            if score > 0:
                scores[intent_name] = score
        
        if not scores:
            return 'unknown', 0.0
        
        best_intent = max(scores, key=scores.get)
        return best_intent, min(scores[best_intent] / 5.0, 1.0)
    
    def extract_expression(self, question: str, intent: str) -> str:
        """استخراج التعبير الرياضي"""
        q = question
        
        if intent in self.keywords:
            for keyword in self.keywords[intent]:
                q = re.sub(r'\b' + keyword + r'\b', '', q, flags=re.IGNORECASE)
        
        general_words = ['أوجد', 'احسب', 'ما', 'هو', 'قيمة', 'then', 'find', 'value']
        for word in general_words:
            q = re.sub(r'\b' + word + r'\b', '', q, flags=re.IGNORECASE)
        
        return q.strip()
    
    # ============================================================
    # معالجات النوايا مع دعم الخطوات
    # ============================================================
    
    def _handle_diff(self, expr_str: str) -> Optional[str]:
        """معالجة المشتقات مع خطوات"""
        result = self.step_solver.solve_with_steps("", "diff", expr_str)
        return result["result"]
    
    def _handle_integrate(self, expr_str: str) -> Optional[str]:
        """معالجة التكاملات مع خطوات"""
        result = self.step_solver.solve_with_steps("", "integrate", expr_str)
        return result["result"]
    
    def _handle_limit(self, expr_str: str, question: str) -> Optional[str]:
        """معالجة النهايات مع خطوات"""
        result = self.step_solver.solve_with_steps(question, "limit", expr_str)
        return result["result"]
    
    def _handle_solve(self, expr_str: str) -> Optional[str]:
        """معالجة المعادلات مع خطوات"""
        result = self.step_solver.solve_with_steps("", "solve", expr_str)
        return result["result"]
    
    def _handle_root(self, expr_str: str, question: str) -> Optional[str]:
        """معالجة الجذور مع خطوات"""
        # جرب المحلل المتخصص أولاً
        root_result = self.root_parser.parse(question)
        if root_result['success']:
            return root_result['result']
        
        # إذا فشل، استخدم مولد الخطوات
        result = self.step_solver.solve_with_steps(question, "root", expr_str)
        return result["result"]
    
    def _handle_factor(self, expr_str: str) -> Optional[str]:
        """معالجة التحليل"""
        result = self.step_solver.solve_with_steps("", "factor", expr_str)
        return result["result"]
    
    def _handle_expand(self, expr_str: str) -> Optional[str]:
        """معالجة النشر"""
        result = self.step_solver.solve_with_steps("", "expand", expr_str)
        return result["result"]
    
    def _handle_simplify(self, expr_str: str) -> Optional[str]:
        """معالجة التبسيط"""
        result = self.step_solver.solve_with_steps("", "simplify", expr_str)
        return result["result"]
    
    def _handle_sum(self, expr_str: str) -> Optional[str]:
        """معالجة المجموع"""
        expr = self.safe_parse(expr_str)
        if expr is None:
            return None
        return f"**النتيجة:** {expr}"
    
    def _handle_product(self, expr_str: str) -> Optional[str]:
        """معالجة الجداء"""
        expr = self.safe_parse(expr_str)
        if expr is None:
            return None
        return f"**النتيجة:** {expr}"
    
    def _handle_inequality(self, expr_str: str) -> Optional[str]:
        """معالجة المتباينات"""
        expr = self.safe_parse(expr_str)
        if expr is None:
            return None
        return f"**النتيجة:** {expr}"
    
    def _handle_absolute(self, expr_str: str, question: str) -> Optional[str]:
        """معالجة القيمة المطلقة"""
        expr = self.safe_parse(expr_str)
        if expr is None:
            return None
        result = sp.Abs(expr)
        return f"**القيمة المطلقة:** |{expr_str}| = {result}"
    
    def _handle_system(self, expr_str: str, question: str) -> Optional[str]:
        """معالجة أنظمة المعادلات"""
        result = self.step_solver.solve_with_steps(question, "system", expr_str)
        return result["result"]
    
    def _handle_calculate(self, expr_str: str) -> Optional[str]:
        """معالجة العمليات الحسابية البسيطة"""
        expr = self.safe_parse(expr_str)
        if expr is None or not expr.is_number:
            return None
        result = expr.evalf()
        if result.is_integer():
            return f"**النتيجة:** {int(result)}"
        return f"**النتيجة:** {result}"
    
    def _handle_generic(self, expr_str: str) -> Optional[str]:
        """معالج عام"""
        expr = self.safe_parse(expr_str)
        if expr is None:
            return None
        return f"**النتيجة:** {expr}"
    
    def process(self, question: str) -> Tuple[Optional[str], float, str, Dict]:
        """معالجة السؤال مع دعم الخطوات"""
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "question": question[:100]
        }
        
        # 1. التحقق من القوالب الجاهزة
        template_result, template_confidence, template_name = self.check_templates(question)
        if template_result:
            metadata["template"] = template_name
            return template_result, template_confidence, template_name, metadata
        
        # 2. كشف النية
        intent, base_confidence = self.detect_intent(question)
        metadata["intent"] = intent
        
        if intent == 'unknown':
            return None, 0.0, 'unknown', metadata
        
        # 3. استخراج التعبير
        expr_str = self.extract_expression(question, intent)
        metadata["expression"] = expr_str
        
        # 4. تنفيذ المعالج
        for intent_name, _, handler, _ in self.intents:
            if intent_name == intent:
                if intent in ['limit', 'root', 'absolute', 'system']:
                    result = handler(expr_str, question)
                else:
                    result = handler(expr_str)
                
                if result is not None:
                    return result, base_confidence, intent, metadata
                break
        
        return None, base_confidence * 0.5, intent, metadata

# ============================================================
# 🧠 Root Expression Parser (محلل الجذور)
# ============================================================

class RootExpressionParser:
    def __init__(self):
        self.root_patterns = [
            {
                'pattern': r'(الجذر|جذر)\s+(التربيعي|التكعيبي|الرباعي)\s*(?:للعدد|لعدد)?\s*(\d+)',
                'handler': self._handle_root
            },
            {
                'pattern': r'([√∛∜])\s*(\d+)',
                'handler': self._handle_symbol
            },
            {
                'pattern': r'(\d+)\s*\^\s*\(?1/(\d+)\)?',
                'handler': self._handle_power
            }
        ]
    
    def _handle_root(self, match):
        root_type = match.group(2)
        number = int(match.group(3))
        
        root_map = {
            'التربيعي': 2,
            'التكعيبي': 3,
            'الرباعي': 4
        }
        
        n = root_map.get(root_type, 2)
        result = sp.root(number, n)
        
        return {
            'result': result,
            'decimal': float(result.evalf()),
            'root_type': n
        }
    
    def _handle_symbol(self, match):
        symbol = match.group(1)
        number = int(match.group(2))
        
        symbol_map = {
            '√': 2,
            '∛': 3,
            '∜': 4
        }
        
        n = symbol_map.get(symbol, 2)
        result = sp.root(number, n)
        
        return {
            'result': result,
            'decimal': float(result.evalf()),
            'root_type': n
        }
    
    def _handle_power(self, match):
        number = int(match.group(1))
        n = int(match.group(2))
        
        result = sp.root(number, n)
        
        return {
            'result': result,
            'decimal': float(result.evalf()),
            'root_type': n
        }
    
    def format_result(self, result_dict):
        root_symbols = {2: '√', 3: '∛', 4: '∜'}
        symbol = root_symbols.get(result_dict['root_type'], '√')
        
        if result_dict['decimal'].is_integer():
            return f"**النتيجة:** {symbol}{result_dict['result']} = {int(result_dict['decimal'])}"
        return f"**النتيجة:** {symbol}{result_dict['result']} ≈ {result_dict['decimal']:.4f}"
    
    def parse(self, text):
        for pattern_info in self.root_patterns:
            match = re.search(pattern_info['pattern'], text, re.UNICODE)
            if match:
                result_dict = pattern_info['handler'](match)
                return {
                    'success': True,
                    'result': self.format_result(result_dict)
                }
        return {'success': False}

# ============================================================
# 💾 CacheDB (نظام التخزين المؤقت)
# ============================================================

class CacheDB:
    def __init__(self, db_path: str = "cache.db", max_size: int = 1000, ttl_days: int = 30):
        self.db_path = db_path
        self.max_size = max_size
        self.ttl_seconds = ttl_days * 24 * 3600
        self._init_db()
    
    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        id TEXT PRIMARY KEY,
                        question TEXT,
                        answer TEXT,
                        confidence REAL,
                        intent TEXT,
                        metadata TEXT,
                        created TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Cache init error: {e}")
    
    def get(self, key: str) -> Optional[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT answer, confidence, intent, metadata FROM cache WHERE id = ? AND expires_at > datetime('now')",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
        except:
            pass
        return None
    
    def set(self, key: str, question: str, answer: str, confidence: float, intent: str, metadata: Dict = None):
        try:
            expires_at = (datetime.now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            metadata_str = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache 
                    (id, question, answer, confidence, intent, metadata, created, expires_at) 
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?)
                """, (key, question[:200], answer, confidence, intent, metadata_str, expires_at))
                conn.commit()
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def get_stats(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                count = cursor.fetchone()[0]
                return {"total": count}
        except:
            return {"total": 0}

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
            return jsonify({"success": False, "error": "❌ تجاوزت الحد المسموح"}), 429
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# 🤖 دوال المساعدة
# ============================================================

def ask_gemini(question: str) -> Optional[str]:
    if not gemini_model:
        return None
    try:
        response = gemini_model.generate_content(question + "\n\n اشرح الخطوات بالتفصيل")
        return response.text if response else None
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

def ask_deepseek(question: str) -> Optional[str]:
    # يمكن إضافة DeepSeek API هنا
    return None

# ============================================================
# 🚀 تهيئة المحركات
# ============================================================

math_engine = MathIntentEngine()
cache_db = CacheDB()
memory = SelfLearningMemory()

# ============================================================
# 🎯 المسارات الرئيسية
#============================================================

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template error: {e}")
        return f"❌ خطأ في تحميل الصفحة: {e}", 500

@app.route('/api/ask', methods=['POST'])
@rate_limit
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "❌ السؤال فارغ"}), 400
        
        # 1. البحث في الذاكرة أولاً
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        # البحث في الذاكرة الذاتية
        memory_result = memory.get_from_memory(question)
        if memory_result:
            return jsonify({
                "success": True,
                "answer": memory_result["answer"],
                "steps": memory_result.get("steps"),
                "confidence": memory_result["confidence"],
                "source": "memory",
                "cached": True
            })
        
        # البحث في الكاش العادي
        cached = cache_db.get(q_hash)
        if cached:
            return jsonify({
                "success": True,
                "answer": cached["answer"],
                "confidence": cached["confidence"],
                "source": "cache",
                "cached": True
            })
        
        # 2. محاولة الحل المحلي مع الخطوات
        result, confidence, intent, metadata = math_engine.process(question)
        
        if result and confidence >= 0.7:
            # الحصول على الخطوات التفصيلية
            expr_str = metadata.get("expression", "")
            steps_result = math_engine.step_solver.solve_with_steps(question, intent, expr_str)
            
            # حفظ في الذاكرة
            memory.add_to_memory(
                question=question,
                answer=result,
                steps="\n".join(steps_result.get("steps", [])),
                solved_by="local",
                confidence=confidence,
                category=intent
            )
            
            return jsonify({
                "success": True,
                "answer": result,
                "steps": steps_result.get("steps", []),
                "confidence": confidence,
                "intent": intent,
                "source": "local",
                "cached": False
            })
        
        # 3. إذا فشل الحل المحلي، سجل في قائمة unanswered
        memory.add_unanswered(question)
        
        # 4. حاول استخدام Gemini
        gemini_answer = ask_gemini(question)
        if gemini_answer:
            # حفظ في الذاكرة
            memory.add_to_memory(
                question=question,
                answer=gemini_answer,
                steps=None,
                solved_by="gemini",
                confidence=0.8,
                category="llm_solved"
            )
            
            return jsonify({
                "success": True,
                "answer": gemini_answer,
                "steps": ["تم الحل باستخدام الذكاء الاصطناعي"],
                "confidence": 0.8,
                "source": "gemini",
                "fallback": True
            })
        
        # 5. فشل كل شيء
        return jsonify({
            "success": False,
            "error": "❌ لم أتمكن من حل السؤال حالياً، تم تسجيله للتعلم",
            "question_id": q_hash[:8]
        }), 400
        
    except Exception as e:
        logger.exception(f"Error: {e}")
        return jsonify({"success": False, "error": "❌ حدث خطأ داخلي"}), 500

@app.route('/api/learn', methods=['POST'])
def learn():
    """نقطة نهاية لتعليم المحرك إجابات جديدة"""
    try:
        data = request.get_json()
        question = data.get('question')
        answer = data.get('answer')
        steps = data.get('steps')
        category = data.get('category', 'manual')
        
        if not question or not answer:
            return jsonify({"success": False, "error": "بيانات غير كاملة"}), 400
        
        memory.add_to_memory(
            question=question,
            answer=answer,
            steps=steps,
            solved_by="manual",
            confidence=1.0,
            category=category
        )
        
        return jsonify({"success": True, "message": "تم التعلم بنجاح"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify({
        "success": True,
        "memory": memory.get_stats(),
        "cache": cache_db.get_stats(),
        "engine": "MathIntentEngine v29.0"
    })

@app.route('/api/pending', methods=['GET'])
def pending():
    """عرض الأسئلة المعلقة (للمسؤول)"""
    pending_questions = memory.get_next_for_llm(10)
    return jsonify({
        "success": True,
        "pending": pending_questions
    })

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔥 MECHATRONICS ASSISTANT v29.0")
    print("="*80)
    print("✅ الميزات الجديدة:")
    print("   • خطوات حل مفصلة لكل مسألة")
    print("   • ذاكرة ذاتية تتعلم من الأسئلة")
    print("   • تسجيل الأسئلة غير المحلولة")
    print("   • دعم قاعدة لوبيتال للنهايات")
    print("   • شرح تفصيلي للمشتقات والتكاملات")
    print("="*80)
    print(f"📊 إحصائيات الذاكرة: {memory.get_stats()}")
    print("="*80)
    print(f"🌐 http://{config.HOST}:{config.PORT}")
    print("="*80 + "\n")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.ENVIRONMENT == 'development'
    )

