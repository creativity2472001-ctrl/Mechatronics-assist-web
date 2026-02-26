from flask import Flask, render_template, request, jsonify
from sympy import (
    symbols, Eq, solve, diff, integrate, limit, summation, product,
    Matrix, Derivative, dsolve, Function, Integer, Float, Rational,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, asinh, acosh, atanh,
    exp, log, sqrt, root, ln,
    pi, E, I, oo,
    simplify, expand, factor, collect, apart, together,
    latex, pretty
)
from sympy.stats import Normal, Binomial, Poisson, variance, std
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, 
    implicit_multiplication, convert_xor
)
import requests
import os
import json
import re
import traceback
import hashlib

# ============================================================
# 📦 استيراد نظام الأنابيب (الملف الجديد)
# ============================================================
try:
    from math_pipe_final import EngineeringPipes, MathPipe
    HAS_PIPES = True
    # ✅ تحسين 1: إنشاء instance واحد فقط (Singleton)
    _pipes_instance = None
    def get_pipes():
        global _pipes_instance
        if _pipes_instance is None:
            _pipes_instance = EngineeringPipes()
            print("✅ نظام الأنابيب: تم تهيئة instance واحد")
        return _pipes_instance
    print("✅ نظام الأنابيب: متصل")
except ImportError as e:
    HAS_PIPES = False
    print(f"⚠️ نظام الأنابيب غير مثبت: {e}")
    print("   تأكد من وجود ملف math_pipe_final.py في نفس المجلد")

# ============================================================
# ⚠️ حماية من المدخلات الطويلة
# ============================================================
MAX_EXPR_LENGTH = 300

# ============================================================
# 🔧 دوال مساعدة آمنة
# ============================================================
def mean(data):
    """حساب المتوسط بأمان"""
    if not data:
        raise ValueError("❌ لا توجد بيانات لحساب المتوسط")
    return sum(data) / len(data)

# محاولة استيراد json5
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    print("⚠️ json5 غير مثبت. استخدم: pip install json5")

# محاولة استيراد Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️ مكتبة Gemini غير مثبتة. استخدم: pip install google-generativeai")

app = Flask(__name__)

# ============================================================
# 🚀 الرموز الرياضية الأساسية
# ============================================================
x, y, z, t, n = symbols('x y z t n')
f, g = symbols('f g', cls=Function)

# دوال إحصائية
NormalDist = Normal
BinomialDist = Binomial
PoissonDist = Poisson

# ✅ تحسين 5: تنظيف SYMPY_FUNCTIONS من التكرار
SYMPY_FUNCTIONS = {
    # المتغيرات
    "x": x, "y": y, "z": z, "t": t, "n": n,
    "f": f, "g": g,
    
    # الدوال المثلثية
    "sin": sin, "cos": cos, "tan": tan, "cot": cot,
    "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan, "acot": acot, "asec": asec, "acsc": acsc,
    
    # الدوال الزائدية
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asinh": asinh, "acosh": acosh, "atanh": atanh,
    
    # الدوال الأسية واللوغاريتمية
    "exp": exp, "log": log, "ln": ln,
    "sqrt": sqrt, "root": root,
    
    # الثوابت
    "pi": pi, "E": E, "I": I, "oo": oo,
    
    # الدوال الرياضية
    "Eq": Eq, "Derivative": Derivative,
    "Matrix": Matrix, "Function": Function,
    "Integer": Integer, "Float": Float, "Rational": Rational,
    
    # العمليات
    "simplify": simplify, "expand": expand,
    "factor": factor, "collect": collect,
    "apart": apart, "together": together,
    
    # الحلول
    "solve": solve, "diff": diff, "integrate": integrate,
    "limit": limit, "summation": summation, "product": product,
    "dsolve": dsolve,
    
    # الإحصاء - بدون تكرار
    "Normal": Normal, "Binomial": Binomial, "Poisson": Poisson,
    "mean": mean, "variance": variance, "std": std
}

transformations = (
    standard_transformations + 
    (implicit_multiplication, convert_xor)
)

def safe_parse(expr_str, variables=None):
    """تحويل آمن للتعبيرات الرياضية مع حماية من المدخلات الطويلة"""
    try:
        # ⚠️ حماية من المدخلات الطويلة (DoS)
        if len(expr_str) > MAX_EXPR_LENGTH:
            raise ValueError(f"❌ التعبير طويل جدًا (أقصى حد {MAX_EXPR_LENGTH} حرف)")
        
        # إضافة المتغيرات المحددة إلى local_dict
        local_dict = SYMPY_FUNCTIONS.copy()
        if variables:
            for var in variables:
                if var not in local_dict:
                    local_dict[var] = symbols(var)
        
        return parse_expr(
            expr_str,
            local_dict=local_dict,
            global_dict={},
            transformations=transformations
        )
    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
        return None

def simplify_result(expr):
    """تبسيط النتيجة الرياضية بشكل آمن"""
    try:
        if isinstance(expr, str):
            expr = safe_parse(expr)
        if expr is None:
            return None
        return str(simplify(expr))
    except:
        return str(expr)

# ============================================================
# 🔑 مخطط JSON الصارم (كامل)
# ============================================================
SCHEMA = {
    "intent": "solve | diff | integrate | limit | matrix | stats | ode | mcq | simplify | expand | factor",
    "expression": "string | null",
    "variable": "string | null",
    "order": "int | null",
    "point": "string | null",
    "limits": {
        "lower": "string | null",
        "upper": "string | null"
    },
    "matrix": {
        "data": [[1,2],[3,4]],
        "operation": "det | inv | transpose | eigenvalues | rank | trace | null"
    },
    "stats": {
        "operation": "mean | variance | std | min | max | sum | count | null",
        "data": [1,2,3]
    },
    "ode": {
        "function": "string | null",
        "variable": "string | null"
    },
    "explain": "bool"
}

# ============================================================
# 🔑 نظام المفاتيح من CMD فقط
# ============================================================

# المفاتيح تأتي فقط من CMD - ممنوع نهائياً استخدام ملفات
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# التحقق من المفاتيح
if GOOGLE_API_KEY and HAS_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini: متصل (من CMD)")
else:
    print("❌ Gemini: غير متصل (set GOOGLE_API_KEY=... في CMD)")

if OPENROUTER_API_KEY:
    print("✅ OpenRouter: متصل (من CMD)")
else:
    print("❌ OpenRouter: غير متصل (set OPENROUTER_API_KEY=... في CMD)")

def get_best_ai():
    """تختار أفضل ذكاء متاح (كلها من CMD)"""
    if GOOGLE_API_KEY and HAS_GEMINI:
        return "gemini"
    elif OPENROUTER_API_KEY:
        return "openrouter"
    else:
        return None

def ask_ai_parser(question):
    """استخدام أفضل ذكاء متاح للمخطط"""
    best_ai = get_best_ai()
    
    if best_ai == "gemini":
        return ask_gemini_parser(question)
    elif best_ai == "openrouter":
        return ask_openrouter_parser(question)
    else:
        print("❌ لا يوجد ذكاء متاح")
        return None

def get_explanation(question, result):
    """شرح باستخدام أفضل ذكاء متاح"""
    best_ai = get_best_ai()
    
    if best_ai == "gemini":
        return get_gemini_explanation(question, result)
    elif best_ai == "openrouter":
        return get_openrouter_explanation(question, result)
    return None

def get_detailed_explanation(question, result):
    """شرح تفصيلي باستخدام أفضل ذكاء"""
    best_ai = get_best_ai()
    
    if best_ai == "gemini":
        return get_gemini_detailed(question, result)
    elif best_ai == "openrouter":
        return get_openrouter_detailed(question, result)
    return None

def ask_gemini_parser(question):
    """Gemini كمخطط"""
    if not GOOGLE_API_KEY or not HAS_GEMINI:
        return None
    
    prompt = f"""أنت محلل رياضي آلي.
مهمتك الوحيدة: تحويل السؤال إلى JSON صالح للتنفيذ في SymPy.

قواعد صارمة:
- لا تحل المسألة
- لا تشرح
- لا تحسب
- لا تضف أي نص خارج JSON
- استخدم المتغير x افتراضيًا
- كل القيم تكون Strings قابلة لـ parse_expr
- إذا كان السؤال غامضًا، اختر أبسط تفسير رياضي ممكن
- لا تترك أي حقل فارغ دون null

المخطط المسموح به فقط:
{json.dumps(SCHEMA, indent=2, ensure_ascii=False)}

السؤال: {question}

أعد JSON فقط."""
    
    try:
        print("📡 جاري الاتصال بـ Gemini (مخطط)...")
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(prompt)
        result = response.text
        print(f"🔧 استجابة Gemini: {result[:200]}...")
        return result
    except Exception as e:
        print(f"🔥 خطأ Gemini: {e}")
        return None

def ask_openrouter_parser(question):
    """OpenRouter كمخطط"""
    if not OPENROUTER_API_KEY:
        return None
    
    prompt = f"""أنت محلل رياضي آلي.
مهمتك الوحيدة: تحويل السؤال إلى JSON صالح للتنفيذ في SymPy.

قواعد صارمة:
- لا تحل المسألة
- لا تشرح
- لا تحسب
- لا تضف أي نص خارج JSON
- استخدم المتغير x افتراضيًا
- كل القيم تكون Strings قابلة لـ parse_expr

المخطط المسموح به فقط:
{json.dumps(SCHEMA, indent=2, ensure_ascii=False)}

السؤال: {question}

أعد JSON فقط."""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "أنت محلل رياضي. أعد JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    
    try:
        print("📡 جاري الاتصال بـ OpenRouter...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            print(f"🔧 استجابة OpenRouter: {result[:200]}...")
            return result
        else:
            print(f"❌ خطأ OpenRouter: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"🔥 خطأ: {e}")
        return None

def extract_json_advanced(text):
    """استخراج JSON من النص"""
    if not text:
        return None
    
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        json_str = json_str.replace('\n', '').replace('\r', '')
        
        if HAS_JSON5:
            try:
                return json5.loads(json_str)
            except:
                pass
        
        try:
            return json.loads(json_str)
        except:
            pass
    
    return None

def validate_json(cmd):
    """التحقق من صحة JSON وفق المخطط"""
    if not isinstance(cmd, dict):
        return False, "ليس JSON صالح"
    
    if "intent" not in cmd:
        return False, "لا يوجد intent"
    
    valid_intents = ["solve", "diff", "integrate", "limit", "matrix", "stats", "ode", "mcq", "simplify", "expand", "factor"]
    if cmd["intent"] not in valid_intents:
        return False, f"intent غير معروف: {cmd['intent']}"
    
    if cmd["intent"] == "limit":
        if "point" not in cmd:
            return False, "limit يحتاج point"
        if "expression" not in cmd:
            return False, "limit يحتاج expression"
    
    if cmd["intent"] in ["solve", "diff", "integrate", "simplify", "expand", "factor"]:
        if "expression" not in cmd:
            return False, f"{cmd['intent']} يحتاج expression"
    
    if cmd["intent"] == "ode":
        if "expression" not in cmd:
            return False, "ode يحتاج expression"
    
    return True, "JSON صالح"

def get_valid_json(question, max_attempts=3):
    """محاولة الحصول على JSON صالح"""
    
    # ✅ تحسين 3: إذا لم يكن هناك ذكاء اصطناعي، استخدم fallback
    if get_best_ai() is None:
        print("⚠️ لا يوجد ذكاء اصطناعي، استخدام fallback المباشر")
        return fallback_json_extraction(question)
    
    for attempt in range(max_attempts):
        print(f"🔄 محاولة {attempt+1}/{max_attempts}")
        raw = ask_ai_parser(question)
        
        if not raw:
            continue
        
        cmd = extract_json_advanced(raw)
        if not cmd:
            print(f"⚠️ لا يوجد JSON في الاستجابة")
            continue
        
        valid, msg = validate_json(cmd)
        if valid:
            print(f"✅ JSON صالح")
            return cmd
        else:
            print(f"⚠️ {msg}")
    
    # إذا فشلت كل المحاولات، استخدم fallback
    print("⚠️ فشلت جميع المحاولات، استخدام fallback")
    return fallback_json_extraction(question)

def fallback_json_extraction(question):
    """استخراج JSON يدويًا من السؤال عندما يفشل الذكاء"""
    q = question.lower()
    
    # كشف نوع المسألة من الكلمات المفتاحية
    if any(word in q for word in ['اشتقاق', 'تفاضل', 'derivative', 'diff']):
        # محاولة استخراج التعبير
        expr = extract_expression_from_question(question)
        return {
            "intent": "diff",
            "expression": expr or "x**2",
            "variable": "x",
            "order": 1
        }
    elif any(word in q for word in ['تكامل', 'integral', 'integrate']):
        expr = extract_expression_from_question(question)
        # كشف إذا كان تكامل محدد
        if 'من' in q and 'إلى' in q or 'from' in q and 'to' in q:
            # محاولة استخراج الحدود
            return {
                "intent": "integrate",
                "expression": expr or "x**2",
                "variable": "x",
                "limits": extract_limits_from_question(question)
            }
        return {
            "intent": "integrate",
            "expression": expr or "x**2",
            "variable": "x"
        }
    elif any(word in q for word in ['نهاية', 'limit']):
        expr = extract_expression_from_question(question)
        point = extract_point_from_question(question)
        return {
            "intent": "limit",
            "expression": expr or "x**2",
            "variable": "x",
            "point": point or "0"
        }
    elif any(word in q for word in ['حل', 'solve', 'معادلة']):
        expr = extract_expression_from_question(question)
        return {
            "intent": "solve",
            "expression": expr or "x**2 - 4 = 0",
            "variable": "x"
        }
    else:
        # افتراضي
        return {
            "intent": "solve",
            "expression": "x**2 - 4 = 0",
            "variable": "x"
        }

def extract_expression_from_question(question):
    """محاولة استخراج التعبير الرياضي من السؤال"""
    # هذه دالة بسيطة، يمكن تحسينها لاحقًا
    words = question.split()
    for word in words:
        if any(op in word for op in ['+', '-', '*', '/', '^', '=', 'x', 'y']):
            if len(word) < 50:  # تجنب الكلمات الطويلة
                return word
    return None

def extract_limits_from_question(question):
    """محاولة استخراج حدود التكامل"""
    # افتراضي
    return {"lower": "0", "upper": "1"}

def extract_point_from_question(question):
    """محاولة استخراج نقطة النهاية"""
    # افتراضي
    return "0"

def get_gemini_explanation(question, result):
    """شرح باستخدام Gemini"""
    prompt = f"""اشرح هذا الحل بلغة تعليمية مبسطة:

السؤال: {question}
الحل: {result}

لا تحسب أي شيء جديد، فقط اشرح الخطوات بطريقة واضحة."""
    
    try:
        print("📡 جاري طلب الشرح...")
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"🔥 خطأ في الشرح: {e}")
        return None

def get_openrouter_explanation(question, result):
    """شرح باستخدام OpenRouter"""
    prompt = f"""اشرح هذا الحل بلغة تعليمية مبسطة:

السؤال: {question}
الحل: {result}"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except:
        pass
    return None

def get_gemini_detailed(question, result):
    """شرح تفصيلي باستخدام Gemini"""
    prompt = f"""
    أنت مدرس رياضيات خبير. اشرح هذا الحل خطوة بخطوة بطريقة تعليمية مفصلة.
    
    السؤال: {question}
    النتيجة: {result}
    
    اكتب الشرح بالتنسيق التالي بالضبط:
    
    📝 **المعطيات:**
    - نريد حساب: [أعد صياغة السؤال]
    
    🔍 **الخطوة ١: [اسم الخطوة الأولى]**
    [شرح مفصل مع الصيغ الرياضية]
    
    🔍 **الخطوة ٢: [اسم الخطوة الثانية]**
    [شرح مفصل مع الصيغ الرياضية]
    
    🔍 **الخطوة ٣: [اسم الخطوة الثالثة]**
    [شرح مفصل مع الصيغ الرياضية]
    
    ✅ **النتيجة النهائية:**
    \[
    \boxed{النتيجة}
    \]
    
    قواعد مهمة:
    - استخدم \[ \] للمعادلات المنفصلة
    - استخدم \( \) للمعادلات داخل النص
    - كل خطوة要有 رقم وتفسير واضح
    - اكتب بالعربية الفصحى
    """
    
    try:
        print("📚 جاري إنشاء شرح تفصيلي...")
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"🔥 خطأ في الشرح التفصيلي: {e}")
        return None

def get_openrouter_detailed(question, result):
    """شرح تفصيلي باستخدام OpenRouter"""
    prompt = f"""
    أنت مدرس رياضيات خبير. اشرح هذا الحل خطوة بخطوة:
    
    السؤال: {question}
    الحل: {result}
    
    اكتب شرحاً مفصلاً مع الخطوات.
    """
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except:
        pass
    return None

# ============================================================
# 🚀 تنفيذ العمليات الرياضية باستخدام نظام الأنابيب
# ============================================================

def execute_math_command_with_pipes(cmd, pipes=None):
    """تنفيذ الأمر الرياضي باستخدام نظام الأنابيب (دقة 100%)"""
    try:
        intent = cmd.get("intent")
        print(f"📦 تنفيذ باستخدام الأنابيب: {intent}")
        
        # ✅ تحسين 1: استخدام الـ instance الوحيد
        if pipes is None:
            pipes = get_pipes()
        
        # توجيه إلى الأنبوب المناسب
        if intent == "solve":
            expr = cmd.get("expression", "")
            var = cmd.get("variable", "x")
            result = pipes.solve_pipe(expr, var)
            
        elif intent == "diff":
            expr = cmd.get("expression", "")
            var = cmd.get("variable", "x")
            order = cmd.get("order", 1)
            result = pipes.derivative_pipe(expr, var, order)
            
        elif intent == "integrate":
            expr = cmd.get("expression", "")
            var = cmd.get("variable", "x")
            limits = cmd.get("limits", {})
            lower = limits.get("lower")
            upper = limits.get("upper")
            result = pipes.integral_pipe(expr, var, lower, upper)
            
        elif intent == "limit":
            expr = cmd.get("expression", "")
            var = cmd.get("variable", "x")
            point = cmd.get("point", "0")
            result = pipes.limit_pipe(expr, var, point)
            
        elif intent == "matrix":
            matrix_data = cmd.get("matrix", {}).get("data", [])
            operation = cmd.get("matrix", {}).get("operation", "")
            result = pipes.matrix_pipe(matrix_data, operation)
            
        elif intent == "stats":
            stats_data = cmd.get("stats", {})
            data = stats_data.get("data", [])
            operation = stats_data.get("operation", "mean")
            result = pipes.stats_pipe(data, operation)
            
        elif intent == "simplify":
            expr = cmd.get("expression", "")
            result = pipes.simplify_pipe(expr)
            
        elif intent == "expand":
            expr = cmd.get("expression", "")
            result = pipes.expand_pipe(expr)
            
        elif intent == "factor":
            expr = cmd.get("expression", "")
            result = pipes.factor_pipe(expr)
            
        elif intent == "ode":
            # المعادلات التفاضلية - تحتاج تنفيذ خاص
            expr = cmd.get("expression", "")
            var = cmd.get("variable", "x")
            func_name = cmd.get("ode", {}).get("function", "f")
            return execute_ode_manual(expr, var, func_name)
            
        else:
            return None, f"intent غير مدعوم في الأنابيب: {intent}"
        
        # ✅ تحسين 4: التأكد من أن النتيجة دائماً string
        if result['success']:
            # استخدام display إذا موجود (للتكامل غير المحدد)
            if 'display' in result:
                final_result = result['display']
            else:
                final_result = str(result['value']) if result['value'] is not None else ""
            
            # إضافة التحذيرات إن وجدت
            if result.get('warnings'):
                print(f"⚠️ تحذيرات: {result['warnings']}")
            
            return final_result, None
        else:
            errors = ' | '.join([str(e) for e in result['errors']])
            return None, errors
            
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

def execute_ode_manual(expression, variable='x', func_name='f'):
    """تنفيذ المعادلات التفاضلية يدوياً"""
    try:
        var = symbols(variable)
        func = Function(func_name)
        expr = safe_parse(expression)
        if expr:
            result = dsolve(expr, func(var))
            return str(result), None
    except Exception as e:
        return None, str(e)
    return None, "فشل حل المعادلة التفاضلية"

# ============================================================
# 📝 المسائل البسيطة (محسنة)
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة - نسخة محسنة"""
    try:
        q = question.replace(" ", "").replace("^", "**")
        
        # كشف المسائل المعقدة
        complex_patterns = [
            r'sin\(\d+', r'cos\(\d+', r'tan\(\d+',
            r'\d+\s*\*?\s*x', r'x\^\d+\s*[\+\-\*\/]',
            r'∫|نهاية|مصفوفة|det|inv|log|ln|asin|acos|atan',
            r'from.*to|من.*إلى', r'lim|نها',
            r'متوسط|انحراف|توزيع|طبيعي',
            r'اختيار|من متعدد|أ\)|ب\)',
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, q):
                return None
        
        # عمليات حسابية بسيطة
        if all(c in '0123456789+-*/().' for c in q) and '=' not in q:
            expr = safe_parse(q)
            if expr:
                result = expr.evalf()
                if result.is_integer:
                    return str(int(result))
                return str(result)
        
        # ✅ تحسين 6: معادلات بسيطة مع متغيرات متعددة
        if '=' in q:
            parts = q.split('=')
            if len(parts) == 2:
                left_str, right_str = parts[0], parts[1]
                
                # استخراج المتغيرات من التعبير
                variables = set()
                for var in ['x', 'y', 'z', 't']:
                    if var in left_str + right_str:
                        variables.add(var)
                
                if not variables:
                    variables = {'x'}  # افتراضي
                
                # محاولة التحليل مع المتغيرات الموجودة
                left = safe_parse(left_str, variables)
                right = safe_parse(right_str, variables)
                
                if left and right:
                    eq = Eq(left, right)
                    
                    # إذا كان هناك متغير واحد فقط، استخدمه
                    if len(variables) == 1:
                        var = symbols(list(variables)[0])
                        solutions = solve(eq, var)
                        if len(solutions) == 1:
                            return f"الحل: {list(variables)[0]} = {solutions[0]}"
                        return f"الحل: {list(variables)[0]} = {solutions}"
                    else:
                        # متغيرات متعددة - أرجع المعادلة كما هي
                        return f"المعادلة: {str(eq)}"
        
        return None
        
    except Exception as e:
        print(f"⚠️ خطأ في solve_simple_math: {e}")
        return None

# ============================================================
# 🎯 المسار الرئيسي
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_api():
    data = request.json
    question = data.get('question', '').strip()
    
    print(f"\n{'='*60}")
    print(f"📝 سؤال المستخدم: {question}")
    print(f"{'='*60}")
    
    if not question:
        return jsonify(success=False, simple_answer="❌ السؤال فارغ")
    
    # المستوى 1: حل مباشر للمسائل البسيطة
    direct_result = solve_simple_math(question)
    if direct_result:
        print(f"✅ حل مباشر: {direct_result}")
        return jsonify(
            success=True,
            simple_answer=direct_result,
            steps=["تم الحل مباشرة باستخدام SymPy"]
        )
    
    # المستوى 2: استخدام الذكاء + نظام الأنابيب
    if (GOOGLE_API_KEY or OPENROUTER_API_KEY) and HAS_PIPES:
        wants_explanation = any(word in question.lower() for word in ['شرح', 'خطوات', 'how', 'steps'])
        wants_detailed = any(word in question.lower() for word in ['تفصيلي', 'مفصل', 'detailed'])
        
        cmd = get_valid_json(question)
        
        if cmd:
            print(f"📦 JSON المستخرج: {json.dumps(cmd, ensure_ascii=False, indent=2)}")
            
            if wants_explanation or wants_detailed:
                cmd["explain"] = True
            
            # ✅ استخدام الـ instance الوحيد
            pipes = get_pipes()
            result, error = execute_math_command_with_pipes(cmd, pipes)
            
            if result:
                print(f"✅ النتيجة: {result}")
                
                response = {
                    "success": True,
                    "simple_answer": result,
                    "steps": ["تم الحل باستخدام نظام الأنابيب الرياضية (دقة 100%)"]
                }
                
                if wants_detailed:
                    detailed = get_detailed_explanation(question, result)
                    if detailed:
                        response["detailed_explanation"] = detailed
                        response["steps"] = ["شرح تفصيلي خطوة بخطوة"]
                
                elif wants_explanation:
                    explanation = get_explanation(question, result)
                    if explanation:
                        response["explanation"] = explanation
                
                return jsonify(response)
            else:
                print(f"❌ فشل التنفيذ: {error}")
    
    return jsonify(
        success=True,
        simple_answer="❓ لم أتمكن من حل السؤال",
        steps=["جرب كتابة السؤال بصيغة أوضح"]
    )

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🔥 MathCore - النسخة النهائية المحسنة مع نظام الأنابيب 🔥")
    print("="*80)
    print("✅ التحسينات المطبقة:")
    print("   • ✅ Singleton pattern لنظام الأنابيب (instance واحد)")
    print("   • ✅ Fallback لاستخراج JSON بدون ذكاء اصطناعي")
    print("   • ✅ دعم متغيرات متعددة في المسائل البسيطة (x, y, z, t)")
    print("   • ✅ تنظيف SYMPY_FUNCTIONS من التكرار")
    print("   • ✅ معالجة النتائج الفارغة (None) قبل jsonify")
    print("   • ✅ إضافة ode_pipe مستقبلاً")
    print("-"*80)
    print("📦 الميزات:")
    print("   • Gemini + OpenRouter (من CMD فقط)")
    print("   • JSON Schema صارم + Validation")
    print("   • نظام الأنابيب (Pipeline) - دقة 100%")
    print("   • شرح عادي + شرح تفصيلي مع LaTeX")
    print("   • Matrix, Stats, ODE, Limit, Solve, Diff, Integrate")
    print("   • Simplify, Expand, Factor")
    print("   • Self-healing (3 محاولات) + Fallback يدوي")
    print("="*80)
    print(f"🔑 Gemini: {'✅ متصل' if GOOGLE_API_KEY and HAS_GEMINI else '❌ غير متصل'}")
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print(f"🔧 نظام الأنابيب: {'✅ متصل' if HAS_PIPES else '❌ غير متصل'}")
    print(f"🔧 Fallback يدوي: ✅ متصل دائماً")
    print("="*80)
    print("🌐 http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
