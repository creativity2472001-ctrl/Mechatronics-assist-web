from flask import Flask, render_template, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, limit, Function, Integer, pretty
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
import requests, os, json, traceback, re
from dotenv import load_dotenv

# محاولة استيراد json5
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False
    print("⚠️ json5 غير مثبت. استخدم: pip install json5")

# تحميل متغيرات البيئة
load_dotenv()

app = Flask(__name__)

# ==================== الرموز الرياضية ====================
x, y, z, t = symbols('x y z t')
f = Function('f')

SAFE_MATH = {
    "x": x, "y": y, "z": z, "t": t,
    "sin": __import__('sympy').sin,
    "cos": __import__('sympy').cos,
    "tan": __import__('sympy').tan,
    "log": __import__('sympy').log,
    "exp": __import__('sympy').exp,
    "sqrt": __import__('sympy').sqrt,
    "pi": __import__('sympy').pi,
    "oo": __import__('sympy').oo,
    "I": __import__('sympy').I,
    "Eq": Eq,
    "Derivative": __import__('sympy').Derivative,
    "Matrix": __import__('sympy').Matrix,
    "Function": Function,
    "f": f,
    "Integer": Integer
}

transformations = standard_transformations + (implicit_multiplication,)

def safe_parse(expr_str):
    try:
        return parse_expr(
            expr_str, 
            local_dict=SAFE_MATH, 
            global_dict={}, 
            transformations=transformations
        )
    except Exception as e:
        print(f"❌ خطأ في parse: {e}")
        return None

# ==================== إعداد مفتاح OpenRouter ====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==================== وظائف OpenRouter ====================
def clean_json_text(text):
    if not text: return None
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def extract_json_advanced(text):
    cleaned = clean_json_text(text)
    if not cleaned: return None
    if HAS_JSON5:
        try:
            return json5.loads(cleaned)
        except:
            pass
    try:
        return json.loads(cleaned)
    except:
        pass
    return None

def ask_openrouter(question):
    if not OPENROUTER_API_KEY: 
        return None
        
    prompt = f"""أنت محلل رياضي. حوّل أي سؤال كلامي أو غامض إلى JSON لصيغة SymPy. أعد JSON فقط.

السؤال: {question}

أمثلة JSON:
{{"type": "solve", "expression": "x**2 + 5*x + 6", "variable": "x"}}
{{"type": "diff", "expression": "sin(2*x)", "variable": "x", "order": 1}}
{{"type": "integrate", "expression": "x**2", "variable": "x", "lower": 0, "upper": 2"}}
{{"type": "calculate", "expression": "2+2"}}
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "أنت محلل رياضي دقيق. أعد JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    try:
        print("📡 جاري الاتصال بـ OpenRouter...")
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if r.status_code == 200:
            result = r.json()['choices'][0]['message']['content']
            print(f"🔧 استجابة: {result[:100]}...")
            return result
        else:
            print(f"❌ خطأ OpenRouter: {r.status_code}")
    except Exception as e:
        print(f"🔥 خطأ: {e}")
    return None

# ==================== تنفيذ SymPy ====================
def execute_math_command(cmd):
    try:
        t = cmd.get("type")
        
        if t == "solve":
            expr = safe_parse(cmd.get("expression", ""))
            var = symbols(cmd.get("variable", "x"))
            if expr:
                solutions = solve(expr, var)
                return solutions, None
                
        elif t == "diff":
            expr = safe_parse(cmd.get("expression", ""))
            var = symbols(cmd.get("variable", "x"))
            order = cmd.get("order", 1)
            if expr:
                return diff(expr, var, order), None
                
        elif t == "integrate":
            expr = safe_parse(cmd.get("expression", ""))
            var = symbols(cmd.get("variable", "x"))
            if expr:
                if "lower" in cmd and "upper" in cmd:
                    lower = safe_parse(str(cmd["lower"]))
                    upper = safe_parse(str(cmd["upper"]))
                    return integrate(expr, (var, lower, upper)), None
                else:
                    return integrate(expr, var) + " + C", None
                
        elif t == "limit":
            expr = safe_parse(cmd.get("expression", ""))
            var = symbols(cmd.get("variable", "x"))
            point = safe_parse(str(cmd.get("point", 0)))
            if expr:
                return limit(expr, var, point), None
                
        elif t == "calculate":
            expr = safe_parse(cmd.get("expression", ""))
            if expr:
                return expr.evalf(), None
                
        return None, f"نوع العملية {t} غير مدعوم"
        
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

# ==================== الحل المباشر لـ SymPy ====================
def solve_simple_math(question):
    try:
        q = question.replace(" ", "").replace("^", "**")
        print(f"🔍 معالجة: {q}")
        
        # ===== الحسابات البسيطة (أرقام فقط) =====
        if all(c in '0123456789+-*/().' for c in q):
            try:
                result = eval(q)
                print(f"📊 eval: {q} = {result}")
                return f"الحل المباشر: {result}"
            except:
                expr = safe_parse(q)
                if expr:
                    result = expr.evalf()
                    print(f"📊 SymPy: {q} = {result}")
                    return f"الحل المباشر: {result}"
        
        # ===== المعادلات =====
        if '=' in q:
            print("✅ تم التعرف على معادلة")
            parts = q.split('=')
            if len(parts) == 2:
                left = safe_parse(parts[0])
                right = safe_parse(parts[1])
                if left and right:
                    eq = Eq(left, right)
                    vars_in_eq = list(left.free_symbols.union(right.free_symbols))
                    if not vars_in_eq:
                        return str(eq)
                    solutions = solve(eq, vars_in_eq)
                    # عرض طريقة الحل باستخدام pretty
                    solution_str = ", ".join([f"{pretty(var)} = {pretty(val)}" for var, val in zip(vars_in_eq, solutions)]) if solutions else "لا يوجد حل"
                    return f"الحل: {solution_str}"
                else:
                    print("⚠️ فشل parsing للمعادلة")
        return None
        
    except Exception as e:
        print(f"⚠️ خطأ في الحل المباشر: {e}")
        return None

# ==================== مسارات API ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_api():
    data = request.json
    q = data.get('question', '').strip()
    
    print(f"\n{'='*50}")
    print(f"📝 سؤال المستخدم: {q}")
    print(f"{'='*50}")
    
    if not q:
        return jsonify(success=False, simple_answer="❌ السؤال فارغ")

    # المستوى 1: حل مباشر
    simple_result = solve_simple_math(q)
    if simple_result:
        print(f"✅ حل مباشر: {simple_result}")
        return jsonify(
            success=True, 
            simple_answer=simple_result, 
            domain="رياضيات", 
            confidence=100
        )

    # المستوى 2: OpenRouter لفهم السؤال الكلامي
    if OPENROUTER_API_KEY:
        print("🔄 استخدام OpenRouter...")
        analysis = ask_openrouter(q)
        if analysis:
            cmd_json = extract_json_advanced(analysis)
            if cmd_json:
                print(f"📦 JSON: {cmd_json}")
                result, error = execute_math_command(cmd_json)
                if result:
                    return jsonify(
                        success=True, 
                        simple_answer=f"الحل عبر OpenRouter: {result}", 
                        domain="رياضيات", 
                        confidence=95
                    )
                else:
                    print(f"❌ فشل التنفيذ: {error}")

    # فشل كل شيء
    return jsonify(
        success=True, 
        simple_answer="❓ لم أتمكن من حل السؤال. جرب كتابته بصيغة واضحة مثل:\n• 1+1\n• x+5=10\n• مشتقة sin(x)\n• تكامل x^2", 
        domain="رياضيات", 
        confidence=0
    )

# ==================== تشغيل التطبيق ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MathCore - SymPy + OpenRouter")
    print("="*60)
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
