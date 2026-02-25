from flask import Flask, render_template, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, limit, Function, Integer, sin, cos, tan, log, exp, sqrt, pi, oo, I
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
import requests
import os
import json
import re
import traceback
from dotenv import load_dotenv

# محاولة استيراد json5
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    print("⚠️ json5 غير مثبت. استخدم: pip install json5")

load_dotenv()

app = Flask(__name__)

# ============================================================
# الرموز الأساسية
# ============================================================

x, y, z, t = symbols('x y z t')
f = Function('f')

SAFE_MATH = {
    "x": x, "y": y, "z": z, "t": t,
    "sin": sin, "cos": cos, "tan": tan,
    "log": log, "exp": exp, "sqrt": sqrt,
    "pi": pi, "oo": oo, "I": I,
    "Eq": Eq, "Function": Function,
    "Integer": Integer
}

transformations = standard_transformations + (implicit_multiplication,)

def safe_parse(expr_str):
    try:
        return parse_expr(expr_str, local_dict=SAFE_MATH, global_dict={}, transformations=transformations)
    except Exception as e:
        print(f"❌ خطأ في تحليل التعبير: {e}")
        return None

# ============================================================
# مفتاح OpenRouter
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("⚠️ مفتاح OpenRouter غير موجود. المسائل المعقدة لن تعمل.")

# ============================================================
# مسائل بسيطة مباشرة
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة بـ SymPy"""
    try:
        question = question.replace("^", "**").replace(" ", "")
        
        # حسابات بسيطة
        if re.fullmatch(r'[\d\+\-\*/\.\(\)]+', question):
            expr = safe_parse(question)
            if expr is not None:
                return str(expr.evalf())
        
        # معادلات بسيطة
        if '=' in question:
            parts = question.split('=')
            if len(parts) == 2:
                left = safe_parse(parts[0])
                right = safe_parse(parts[1])
                if left is not None and right is not None:
                    eq = Eq(left, right)
                    solutions = solve(eq, x)
                    return f"الحل: x = {solutions}"
        
        return None
    except Exception as e:
        print(f"⚠️ خطأ في الحل المباشر: {e}")
        return None

# ============================================================
# الاتصال بـ OpenRouter
# ============================================================

def clean_json_text(text):
    if not text:
        return None
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def extract_json_advanced(text):
    if not text:
        return None
    cleaned = clean_json_text(text)
    if not cleaned:
        return None
    if HAS_JSON5:
        try:
            data = json5.loads(cleaned)
            if isinstance(data, dict):
                return data
        except: pass
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except: pass
    return None

def ask_openrouter(question):
    if not OPENROUTER_API_KEY:
        return None
    prompt = f"""أنت محلل رياضي دقيق. أعد JSON صالح فقط.
السؤال: {question}"""
    
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "system", "content": "أنت محلل رياضي. أعد JSON فقط."},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000
    }
    
    try:
        print("📡 جاري الاتصال بـ OpenRouter...")
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            print(f"🔧 استجابة OpenRouter: {result[:200]}...")
            return result
        else:
            print(f"❌ خطأ من OpenRouter: {response.status_code}")
            return None
    except Exception as e:
        print(f"🔥 خطأ في الاتصال: {e}")
        return None

# ============================================================
# تنفيذ العمليات الرياضية
# ============================================================

def execute_math_command(command_json):
    try:
        cmd_type = command_json.get("type", "")
        if cmd_type == "solve":
            expr = safe_parse(command_json.get("expression", ""))
            var = symbols(command_json.get("variable", "x"))
            if expr: return str(solve(expr, var)), None
        elif cmd_type == "diff":
            expr = safe_parse(command_json.get("expression", ""))
            var = symbols(command_json.get("variable", "x"))
            order = command_json.get("order", 1)
            if expr: return str(diff(expr, var, order)), None
        elif cmd_type == "integrate":
            expr = safe_parse(command_json.get("expression", ""))
            var = symbols(command_json.get("variable", "x"))
            if expr:
                if "lower" in command_json and "upper" in command_json:
                    lower = safe_parse(str(command_json["lower"]))
                    upper = safe_parse(str(command_json["upper"]))
                    return str(integrate(expr, (var, lower, upper))), None
                return str(integrate(expr, var)) + " + C", None
        elif cmd_type == "limit":
            expr = safe_parse(command_json.get("expression", ""))
            var = symbols(command_json.get("variable", "x"))
            point = command_json.get("point", 0)
            if expr: return str(limit(expr, var, point)), None
        elif cmd_type == "calculate":
            expr = safe_parse(command_json.get("expression", ""))
            if expr: return str(expr.evalf()), None
        return None, f"نوع العملية '{cmd_type}' غير مدعوم"
    except Exception as e:
        print(f"❌ خطأ في التنفيذ: {e}")
        traceback.print_exc()
        return None, str(e)

# ============================================================
# Flask Routes
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_route():
    data = request.json
    question = data.get('question', '').strip()
    print(f"\n📝 سؤال المستخدم: {question}")
    
    if not question:
        return jsonify({"success": False, "simple_answer": "❌ السؤال فارغ"})
    
    # المستوى 1: SymPy مباشر
    simple_result = solve_simple_math(question)
    if simple_result:
        print("✅ تم الحل مباشرة بـ SymPy")
        return jsonify({"success": True, "simple_answer": simple_result, "domain": "رياضيات", "confidence": 100})
    
    # المستوى 2: OpenRouter
    if OPENROUTER_API_KEY:
        analysis = ask_openrouter(question)
        if analysis:
            command_json = extract_json_advanced(analysis)
            if command_json:
                result, error = execute_math_command(command_json)
                if result:
                    return jsonify({"success": True, "simple_answer": result, "domain": "رياضيات", "confidence": 95})
    
    # اقتراح صيغة
    examples = ["x^2 + 5x + 6 = 0", "مشتقة sin(2x)", "تكامل x^2 من 0 إلى 2", "1+1", "2*3"]
    import random
    example = random.choice(examples)
    
    return jsonify({"success": True, "simple_answer": "❓ لم أتمكن من حل السؤال",
                    "suggestion": f"جرب صيغة واضحة مثل: {example}", "domain": "رياضيات", "confidence": 0})

# ============================================================
# تشغيل التطبيق
# ============================================================

if __name__ == '__main__':
    print("\n🚀 MathCore - SymPy + OpenRouter فقط")
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
