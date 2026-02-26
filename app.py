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

SYMPY_FUNCTIONS = {
    "x": x, "y": y, "z": z, "t": t, "n": n,
    "f": f, "g": g,
    "sin": sin, "cos": cos, "tan": tan, "cot": cot,
    "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan, "acot": acot, "asec": asec, "acsc": acsc,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asinh": asinh, "acosh": acosh, "atanh": atanh,
    "exp": exp, "log": log, "ln": ln,
    "sqrt": sqrt, "root": root,
    "pi": pi, "E": E, "I": I, "oo": oo,
    "Eq": Eq, "Derivative": Derivative,
    "Matrix": Matrix, "Function": Function,
    "Integer": Integer, "Float": Float, "Rational": Rational,
    "simplify": simplify, "expand": expand,
    "factor": factor, "collect": collect,
    "apart": apart, "together": together,
    "solve": solve, "diff": diff, "integrate": integrate,
    "limit": limit, "summation": summation, "product": product,
    "dsolve": dsolve,
    "Normal": Normal, "Binom": Binomial, "Poisson": Poisson,
    "mean": mean, "variance": variance, "std": std
}

transformations = (
    standard_transformations + 
    (implicit_multiplication, convert_xor)
)

def safe_parse(expr_str):
    """تحويل آمن للتعبيرات الرياضية مع حماية من المدخلات الطويلة"""
    try:
        # ⚠️ حماية من المدخلات الطويلة (DoS)
        if len(expr_str) > MAX_EXPR_LENGTH:
            raise ValueError(f"❌ التعبير طويل جدًا (أقصى حد {MAX_EXPR_LENGTH} حرف)")
        
        return parse_expr(
            expr_str,
            local_dict=SYMPY_FUNCTIONS,
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
    "intent": "solve | diff | integrate | limit | matrix | stats | ode | mcq",
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
        "operation": "det | inv | transpose | null"
    },
    "stats": {
        "operation": "mean | variance | std | null",
        "data": [1,2,3]
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
    
    valid_intents = ["solve", "diff", "integrate", "limit", "matrix", "stats", "ode", "mcq"]
    if cmd["intent"] not in valid_intents:
        return False, f"intent غير معروف: {cmd['intent']}"
    
    if cmd["intent"] == "limit":
        if "point" not in cmd:
            return False, "limit يحتاج point"
        if "expression" not in cmd:
            return False, "limit يحتاج expression"
    
    if cmd["intent"] in ["solve", "diff", "integrate"]:
        if "expression" not in cmd:
            return False, f"{cmd['intent']} يحتاج expression"
    
    return True, "JSON صالح"

def get_valid_json(question, max_attempts=3):
    """محاولة الحصول على JSON صالح"""
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
    
    return None

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
# 🚀 تنفيذ العمليات الرياضية
# ============================================================

def execute_math_command(cmd):
    """تنفيذ الأمر الرياضي باستخدام SymPy"""
    try:
        intent = cmd.get("intent")
        print(f"📦 تنفيذ: {intent}")
        
        if intent == "solve":
            expr = safe_parse(cmd["expression"])
            var = symbols(cmd.get("variable", "x"))
            if expr:
                solutions = solve(expr, var)
                return simplify_result(solutions), None
        
        elif intent == "diff":
            expr = safe_parse(cmd["expression"])
            var = symbols(cmd.get("variable", "x"))
            order = cmd.get("order", 1)
            if expr:
                result = diff(expr, var, order)
                return simplify_result(result), None
        
        elif intent == "integrate":
            expr = safe_parse(cmd["expression"])
            var = symbols(cmd.get("variable", "x"))
            
            if expr:
                limits = cmd.get("limits", {})
                if limits.get("lower") and limits.get("upper"):
                    lower = safe_parse(limits["lower"])
                    upper = safe_parse(limits["upper"])
                    result = integrate(expr, (var, lower, upper))
                else:
                    result = integrate(expr, var)
                
                if limits.get("upper"):
                    return simplify_result(result), None
                else:
                    return simplify_result(result) + " + C", None
        
        elif intent == "limit":
            expr = safe_parse(cmd["expression"])
            var = symbols(cmd.get("variable", "x"))
            point = safe_parse(cmd["point"])
            if expr:
                result = limit(expr, var, point)
                return simplify_result(result), None
        
        elif intent == "matrix":
            matrix_data = cmd.get("matrix", {})
            data = matrix_data.get("data", [])
            operation = matrix_data.get("operation", "")
            
            try:
                M = Matrix(data)
                
                if operation == "det":
                    return str(M.det()), None
                elif operation == "inv":
                    return str(M.inv()), None
                elif operation == "transpose":
                    return str(M.T), None
                else:
                    return str(M), None
            except Exception as e:
                return None, f"خطأ في المصفوفة: {e}"
        
        elif intent == "stats":
            stats_data = cmd.get("stats", {})
            op = stats_data.get("operation", "mean")
            data = stats_data.get("data", [])
            
            if not data:
                return None, "لا توجد بيانات"
            
            try:
                if op == "mean":
                    return str(mean(data)), None
                elif op == "variance":
                    m = mean(data)
                    var = sum((xi - m) ** 2 for xi in data) / (len(data) - 1)
                    return str(var), None
                elif op == "std":
                    m = mean(data)
                    var = sum((xi - m) ** 2 for xi in data) / (len(data) - 1)
                    return str(var ** 0.5), None
            except Exception as e:
                return None, str(e)
        
        elif intent == "ode":
            expr = safe_parse(cmd["expression"])
            var = symbols(cmd.get("variable", "x"))
            func = Function(cmd.get("function", "f"))
            
            if expr:
                result = dsolve(expr, func(var))
                return str(result), None
        
        return None, f"intent غير مدعوم: {intent}"
        
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

# ============================================================
# 📝 المسائل البسيطة
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة"""
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
        
        # معادلات بسيطة
        if '=' in q:
            parts = q.split('=')
            if len(parts) == 2:
                left = safe_parse(parts[0])
                right = safe_parse(parts[1])
                if left and right:
                    eq = Eq(left, right)
                    solutions = solve(eq, x)
                    if len(solutions) == 1:
                        return f"الحل: x = {solutions[0]}"
                    return f"الحل: x = {solutions}"
        
        return None
        
    except Exception as e:
        print(f"⚠️ خطأ: {e}")
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
    
    # المستوى 1: حل مباشر
    direct_result = solve_simple_math(question)
    if direct_result:
        print(f"✅ حل مباشر: {direct_result}")
        return jsonify(
            success=True,
            simple_answer=direct_result,
            steps=["تم الحل مباشرة باستخدام SymPy"]
        )
    
    # المستوى 2: استخدام الذكاء
    if GOOGLE_API_KEY or OPENROUTER_API_KEY:
        wants_explanation = any(word in question.lower() for word in ['شرح', 'خطوات', 'how', 'steps'])
        wants_detailed = any(word in question.lower() for word in ['تفصيلي', 'مفصل', 'detailed'])
        
        cmd = get_valid_json(question)
        
        if cmd:
            print(f"📦 JSON المستخرج: {json.dumps(cmd, ensure_ascii=False)}")
            
            if wants_explanation or wants_detailed:
                cmd["explain"] = True
            
            result, error = execute_math_command(cmd)
            
            if result:
                print(f"✅ النتيجة: {result}")
                
                response = {
                    "success": True,
                    "simple_answer": result,
                    "steps": ["تم الحل باستخدام الذكاء"]
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
    print("\n" + "="*70)
    print("🔥 MathCore - النسخة الكاملة بكل الميزات 🔥")
    print("="*70)
    print("✅ Gemini + OpenRouter (من CMD فقط)")
    print("✅ JSON Schema صارم + Validation")
    print("✅ شرح عادي + شرح تفصيلي مع LaTeX")
    print("✅ Matrix, Stats, ODE, Limit, Solve, Diff, Integrate")
    print("✅ Self-healing (3 محاولات)")
    print("="*70)
    print(f"🔑 Gemini: {'✅ متصل (من CMD)' if GOOGLE_API_KEY and HAS_GEMINI else '❌ غير متصل (set GOOGLE_API_KEY=...)'}")
    print(f"🔑 OpenRouter: {'✅ متصل (من CMD)' if OPENROUTER_API_KEY else '❌ غير متصل (set OPENROUTER_API_KEY=...)'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
