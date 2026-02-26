from flask import Flask, render_template, request, jsonify
from sympy import (
    symbols, Eq, solve, diff, integrate, limit, summation, product,
    Matrix, Derivative, dsolve, Function, Integer, Float, Rational,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, asinh, acosh, atanh,
    exp, log, sqrt, root,
    pi, E, I, oo,
    simplify, expand, factor, collect, apart, together,
    latex, pretty
)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, 
    implicit_multiplication, convert_xor
)
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
# 🚀 الرموز الرياضية الأساسية
# ============================================================
x, y, z, t, n = symbols('x y z t n')
f, g = symbols('f g', cls=Function)

SYMPY_FUNCTIONS = {
    "x": x, "y": y, "z": z, "t": t, "n": n,
    "f": f, "g": g,
    "sin": sin, "cos": cos, "tan": tan, "cot": cot,
    "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "acot": acot, "asec": asec, "acsc": acsc,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asinh": asinh, "acosh": acosh, "atanh": atanh,
    "exp": exp, "log": log, "ln": log,
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
    "dsolve": dsolve
}

transformations = (
    standard_transformations + 
    (implicit_multiplication, convert_xor)
)

def safe_parse(expr_str):
    try:
        return parse_expr(
            expr_str,
            local_dict=SYMPY_FUNCTIONS,
            global_dict={},
            transformations=transformations
        )
    except Exception as e:
        print(f"❌ خطأ في التحليل: {e}")
        return None

# ============================================================
# 🔑 OpenRouter
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def ask_openrouter(question):
    if not OPENROUTER_API_KEY:
        return None
    
    prompt = f"""أنت محلل رياضي خبير. مهمتك تحويل أي سؤال رياضي إلى JSON دقيق.

السؤال: {question}

أنواع العمليات:
1. solve - حل المعادلات: {{"type": "solve", "expression": "...", "variable": "x"}}
2. diff - تفاضل: {{"type": "diff", "expression": "...", "variable": "x", "order": 1}}
3. integrate - تكامل: {{"type": "integrate", "expression": "...", "variable": "x"}}
4. limit - نهايات: {{"type": "limit", "expression": "...", "variable": "x", "point": 0}}
5. matrix - مصفوفات: {{"type": "matrix", "expression": "[[1,2],[3,4]]", "operation": "det"}}

أعد JSON فقط."""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "أعد JSON فقط."},
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
            print(f"🔧 استجابة: {result[:200]}...")
            return result
        else:
            print(f"❌ خطأ OpenRouter: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"🔥 خطأ: {e}")
        return None

def extract_json_advanced(text):
    if not text:
        return None
    
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        
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

# ============================================================
# 🚀 تنفيذ العمليات الرياضية
# ============================================================

def execute_math_command(cmd):
    try:
        cmd_type = cmd.get("type", "")
        print(f"📦 تنفيذ: {cmd_type}")
        
        if cmd_type == "solve":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            if expr:
                solutions = solve(expr, var)
                return str(solutions), None
        
        elif cmd_type == "diff":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            order = cmd.get("order", 1)
            if expr:
                result = diff(expr, var, order)
                return str(result), None
        
        elif cmd_type == "integrate":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            
            if expr:
                if "lower" in cmd and "upper" in cmd:
                    lower = safe_parse(str(cmd["lower"]))
                    upper = safe_parse(str(cmd["upper"]))
                    result = integrate(expr, (var, lower, upper))
                    return str(result), None
                else:
                    result = integrate(expr, var)
                    return str(result) + " + C", None
        
        elif cmd_type == "limit":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            point = safe_parse(str(cmd.get("point", 0)))
            if expr:
                result = limit(expr, var, point)
                return str(result), None
        
        elif cmd_type == "matrix":
            expr_str = cmd.get("expression", "")
            operation = cmd.get("operation", "")
            
            try:
                matrix_data = json.loads(expr_str) if isinstance(expr_str, str) else expr_str
                M = Matrix(matrix_data)
                
                if operation == "det":
                    return str(M.det()), None
                elif operation == "inv":
                    return str(M.inv()), None
                else:
                    return str(M), None
            except:
                return None, "خطأ في المصفوفة"
        
        return None, f"نوع العملية {cmd_type} غير مدعوم"
        
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

# ============================================================
# 📝 المسائل البسيطة - مع كشف المسائل المعقدة
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة - المعقدة تذهب لـ OpenRouter"""
    try:
        q = question.replace(" ", "").replace("^", "**")
        original_q = question
        
        # ===== كشف المسائل المعقدة =====
        complex_patterns = [
            r'sin\(\d+',      # sin(60...)
            r'cos\(\d+',      # cos(5...)
            r'tan\(\d+',      # tan(2...)
            r'\d+\s*\*?\s*x', # 2x, 5x
            r'x\^\d+\s*[\+\-\*\/]', # x^2 +, x^3 -
            r'∫|نهاية|مصفوفة|det|inv', # كلمات مفتاحية
            r'from.*to|من.*إلى', # تكامل محدد
            r'lim|نها', # نهايات
        ]
        
        for pattern in complex_patterns:
            if re.search(pattern, q):
                print(f"🔄 مسألة معقدة: تذهب لـ OpenRouter")
                return None
        
        # ===== 1. حالة = في النهاية =====
        if q.endswith('='):
            q = q[:-1]
        
        # ===== 2. العمليات الحسابية =====
        if all(c in '0123456789+-*/().' for c in q) and '=' not in q:
            try:
                result = eval(q)
                if isinstance(result, float) and result.is_integer():
                    return str(int(result))
                return str(result)
            except:
                expr = safe_parse(q)
                if expr:
                    result = expr.evalf()
                    if result.is_integer:
                        return str(int(result))
                    return str(result)
        
        # ===== 3. المعادلات =====
        if '=' in q:
            parts = q.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                if right == '':
                    return None
                try:
                    left_expr = safe_parse(left)
                    right_expr = safe_parse(right)
                    if left_expr and right_expr:
                        eq = Eq(left_expr, right_expr)
                        solutions = solve(eq, x)
                        if len(solutions) == 1:
                            return f"الحل: x = {solutions[0]}"
                        else:
                            return f"الحل: x = {solutions}"
                except:
                    pass
        
        # ===== 4. التفاضل البسيط =====
        diff_patterns = [
            (r'مشتقة.*sin', diff(sin(x), x)),
            (r'مشتقة.*cos', diff(cos(x), x)),
            (r'مشتقة.*x\*\*2', diff(x**2, x)),
        ]
        
        for pattern, result in diff_patterns:
            if re.search(pattern, original_q):
                return str(result)
        
        # ===== 5. التكامل البسيط =====
        if 'تكامل' in original_q:
            if 'sin' in original_q:
                return str(integrate(sin(x), x)) + ' + C'
            elif 'cos' in original_q:
                return str(integrate(cos(x), x)) + ' + C'
            elif 'x**2' in original_q:
                return str(integrate(x**2, x)) + ' + C'
        
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
    
    # المستوى 1: حل مباشر (للمسائل البسيطة فقط)
    direct_result = solve_simple_math(question)
    if direct_result:
        print(f"✅ حل مباشر: {direct_result}")
        return jsonify(
            success=True,
            simple_answer=direct_result,
            domain="رياضيات",
            confidence=100
        )
    
    # المستوى 2: OpenRouter للمسائل المعقدة
    if OPENROUTER_API_KEY:
        print("🔄 استخدام OpenRouter...")
        analysis = ask_openrouter(question)
        
        if analysis:
            cmd_json = extract_json_advanced(analysis)
            
            if cmd_json:
                print(f"📦 JSON: {cmd_json}")
                result, error = execute_math_command(cmd_json)
                
                if result:
                    print(f"✅ النتيجة: {result}")
                    
                    # شرح إذا طلب
                    explanation = None
                    if 'شرح' in question.lower():
                        exp = ask_openrouter(f"اشرح حل: {question}\nالنتيجة: {result}")
                        explanation = exp
                    
                    return jsonify(
                        success=True,
                        simple_answer=result,
                        explanation=explanation,
                        domain="رياضيات",
                        confidence=95
                    )
    
    # رسالة مساعدة
    return jsonify(
        success=True,
        simple_answer="❓ لم أتمكن من حل السؤال",
        domain="رياضيات",
        confidence=0
    )

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 MathCore - النسخة النهائية 🔥")
    print("="*70)
    print("✅ المسائل البسيطة: حل مباشر")
    print("✅ المسائل المعقدة: OpenRouter → SymPy → شرح")
    print("="*70)
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
