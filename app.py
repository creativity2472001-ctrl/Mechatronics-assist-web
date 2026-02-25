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
# 🚀 الرموز الرياضية الأساسية (موسعة)
# ============================================================
x, y, z, t, n = symbols('x y z t n')
f, g = symbols('f g', cls=Function)

# قاموس آمن يحتوي على كل الدوال الرياضية
SYMPY_FUNCTIONS = {
    # الرموز الأساسية
    "x": x, "y": y, "z": z, "t": t, "n": n,
    "f": f, "g": g,
    
    # الدوال المثلثية
    "sin": sin, "cos": cos, "tan": tan, "cot": cot,
    "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "acot": acot, "asec": asec, "acsc": acsc,
    
    # الدوال الزائدية
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asinh": asinh, "acosh": acosh, "atanh": atanh,
    
    # الدوال الأسية واللوغاريتمية
    "exp": exp, "log": log, "ln": log,
    "sqrt": sqrt, "root": root,
    
    # الثوابت
    "pi": pi, "E": E, "I": I, "oo": oo,
    
    # الدوال الرياضية
    "Eq": Eq, "Derivative": Derivative,
    "Matrix": Matrix, "Function": Function,
    "Integer": Integer, "Float": Float, "Rational": Rational,
    
    # عمليات الجبر
    "simplify": simplify, "expand": expand,
    "factor": factor, "collect": collect,
    "apart": apart, "together": together,
    
    # دوال الحل
    "solve": solve, "diff": diff, "integrate": integrate,
    "limit": limit, "summation": summation, "product": product,
    "dsolve": dsolve
}

# تحويلات متقدمة
transformations = (
    standard_transformations + 
    (implicit_multiplication, convert_xor)
)

def safe_parse(expr_str):
    """تحويل آمن للتعبيرات الرياضية مع دعم كل الدوال"""
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
    """إرسال السؤال لـ OpenRouter لفهمه وتحويله لـ JSON"""
    if not OPENROUTER_API_KEY:
        return None
    
    # برومبت شامل جداً يغطي كل العمليات
    prompt = f"""أنت محلل رياضي خبير. مهمتك تحويل أي سؤال رياضي إلى JSON دقيق.

السؤال: {question}

أنواع العمليات المدعومة مع أمثلة:

1. solve - حل المعادلات
   {{"type": "solve", "expression": "x**2 + 5*x + 6", "variable": "x"}}

2. diff - تفاضل عادي أو جزئي
   {{"type": "diff", "expression": "sin(2*x)", "variable": "x", "order": 1}}
   {{"type": "diff", "expression": "x**2*y**3", "variables": ["x","y"], "orders": [1,1]}}

3. integrate - تكامل محدد أو غير محدد
   {{"type": "integrate", "expression": "x**2", "variable": "x"}}
   {{"type": "integrate", "expression": "x**2", "variable": "x", "lower": 0, "upper": 2}}

4. limit - نهايات
   {{"type": "limit", "expression": "sin(x)/x", "variable": "x", "point": 0}}

5. sum - مجموع
   {{"type": "sum", "expression": "1/n**2", "variable": "n", "from": 1, "to": "oo"}}

6. matrix - عمليات مصفوفات
   {{"type": "matrix", "expression": "[[1,2],[3,4]]", "operation": "det"}}
   {{"type": "matrix", "expression": "[[1,2],[3,4]]", "operation": "inv"}}

7. simplify - تبسيط
   {{"type": "simplify", "expression": "sin(x)**2 + cos(x)**2"}}

8. expand - توسيع
   {{"type": "expand", "expression": "(x+1)**2"}}

9. factor - تحليل
   {{"type": "factor", "expression": "x**2 - 4"}}

10. dsolve - معادلات تفاضلية
    {{"type": "dsolve", "equation": "f(x).diff(x,x) + f(x)", "function": "f", "variable": "x"}}

القواعد:
1. أعد JSON فقط، لا تكتب أي كلمات أخرى
2. استخدم ** للأس
3. variable الافتراضي هو "x"
4. lower/upper يمكن أن تكون أرقاماً أو "oo" (لانهاية)

الآن حلل السؤال وأعد JSON فقط:"""
    
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
        print(f"🔥 خطأ في الاتصال: {e}")
        return None

def extract_json_advanced(text):
    """استخراج JSON من النص"""
    if not text:
        return None
    
    # البحث عن JSON
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        
        # محاولة json5
        if HAS_JSON5:
            try:
                return json5.loads(json_str)
            except:
                pass
        
        # محاولة json عادي
        try:
            return json.loads(json_str)
        except:
            pass
    
    return None

# ============================================================
# 🚀 تنفيذ العمليات الرياضية (القوة العظمى)
# ============================================================

def execute_math_command(cmd):
    """تنفيذ الأمر الرياضي باستخدام SymPy"""
    try:
        cmd_type = cmd.get("type", "")
        print(f"📦 تنفيذ: {cmd_type}")
        
        # ===== حل المعادلات =====
        if cmd_type == "solve":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            if expr:
                solutions = solve(expr, var)
                return str(solutions), None
        
        # ===== التفاضل =====
        elif cmd_type == "diff":
            expr = safe_parse(cmd.get("expression", ""))
            if "variables" in cmd:  # تفاضل جزئي
                vars_list = [symbols(v) for v in cmd["variables"]]
                orders = cmd.get("orders", [1] * len(vars_list))
                result = expr
                for var, order in zip(vars_list, orders):
                    result = diff(result, var, order)
                return str(result), None
            else:  # تفاضل عادي
                var_name = cmd.get("variable", "x")
                var = symbols(var_name)
                order = cmd.get("order", 1)
                result = diff(expr, var, order)
                return str(result), None
        
        # ===== التكامل =====
        elif cmd_type == "integrate":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            
            if "lower" in cmd and "upper" in cmd:  # تكامل محدد
                lower = safe_parse(str(cmd["lower"]))
                upper = safe_parse(str(cmd["upper"]))
                result = integrate(expr, (var, lower, upper))
                return str(result), None
            else:  # تكامل غير محدد
                result = integrate(expr, var)
                return str(result) + " + C", None
        
        # ===== النهايات =====
        elif cmd_type == "limit":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            point = safe_parse(str(cmd.get("point", 0)))
            result = limit(expr, var, point)
            return str(result), None
        
        # ===== المجاميع =====
        elif cmd_type == "sum":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "n")
            var = symbols(var_name)
            from_val = cmd.get("from", 1)
            to_val = cmd.get("to", "oo")
            to_expr = safe_parse(str(to_val)) if isinstance(to_val, str) else to_val
            result = summation(expr, (var, from_val, to_expr))
            return str(result), None
        
        # ===== المصفوفات =====
        elif cmd_type == "matrix":
            expr_str = cmd.get("expression", "")
            operation = cmd.get("operation", "")
            
            # تحويل النص إلى مصفوفة
            try:
                matrix_data = json.loads(expr_str) if isinstance(expr_str, str) else expr_str
                M = Matrix(matrix_data)
                
                if operation == "det":
                    return str(M.det()), None
                elif operation == "inv":
                    return str(M.inv()), None
                elif operation == "transpose":
                    return str(M.T), None
                else:
                    return str(M), None
            except:
                return None, "خطأ في تحويل المصفوفة"
        
        # ===== تبسيط =====
        elif cmd_type == "simplify":
            expr = safe_parse(cmd.get("expression", ""))
            if expr:
                return str(simplify(expr)), None
        
        # ===== توسيع =====
        elif cmd_type == "expand":
            expr = safe_parse(cmd.get("expression", ""))
            if expr:
                return str(expand(expr)), None
        
        # ===== تحليل =====
        elif cmd_type == "factor":
            expr = safe_parse(cmd.get("expression", ""))
            if expr:
                return str(factor(expr)), None
        
        # ===== معادلات تفاضلية =====
        elif cmd_type == "dsolve":
            eq_str = cmd.get("equation", "")
            func_name = cmd.get("function", "f")
            var_name = cmd.get("variable", "x")
            
            var = symbols(var_name)
            f_func = Function(func_name)
            
            # تحويل المعادلة
            eq = safe_parse(eq_str.replace(func_name, func_name))
            if eq:
                result = dsolve(eq, f_func(var))
                return str(result), None
        
        # ===== حساب مباشر =====
        elif cmd_type == "calculate":
            expr = safe_parse(cmd.get("expression", ""))
            if expr:
                return str(expr.evalf()), None
        
        return None, f"نوع العملية {cmd_type} غير مدعوم"
        
    except Exception as e:
        traceback.print_exc()
        return None, str(e)

# ============================================================
# 📝 المسائل البسيطة (بدون API)
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة"""
    try:
        q = question.replace(" ", "").replace("^", "**")
        
        # حساب عددي
        if all(c in '0123456789+-*/().' for c in q):
            try:
                # eval آمن للأرقام فقط
                result = eval(q)
                return str(result)
            except:
                expr = safe_parse(q)
                if expr:
                    return str(expr.evalf())
        
        # معادلة بسيطة
        if '=' in q:
            parts = q.split('=')
            if len(parts) == 2:
                left = safe_parse(parts[0])
                right = safe_parse(parts[1])
                if left and right:
                    eq = Eq(left, right)
                    solutions = solve(eq, x)
                    return f"الحل: x = {solutions}"
        
        # كشف الأنماط العربية
        patterns = [
            (r'مشتقة.*sin', diff(sin(x), x)),
            (r'مشتقة.*cos', diff(cos(x), x)),
            (r'مشتقة.*tan', diff(tan(x), x)),
            (r'تكامل.*sin', integrate(sin(x), x)),
            (r'تكامل.*cos', integrate(cos(x), x)),
        ]
        
        for pattern, result in patterns:
            if re.search(pattern, question):
                return str(result) + (" + C" if "تكامل" in pattern else "")
        
        return None
    except Exception as e:
        print(f"⚠️ خطأ في الحل المباشر: {e}")
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
    
    # المستوى 1: حل مباشر (بدون API)
    direct_result = solve_simple_math(question)
    if direct_result:
        print(f"✅ حل مباشر: {direct_result}")
        return jsonify(
            success=True,
            simple_answer=direct_result,
            domain="رياضيات",
            confidence=100
        )
    
    # المستوى 2: استخدام OpenRouter للأسئلة المعقدة
    if OPENROUTER_API_KEY:
        print("🔄 استخدام OpenRouter...")
        analysis = ask_openrouter(question)
        
        if analysis:
            cmd_json = extract_json_advanced(analysis)
            
            if cmd_json:
                print(f"📦 JSON المستخرج: {cmd_json}")
                result, error = execute_math_command(cmd_json)
                
                if result:
                    print(f"✅ النتيجة: {result}")
                    return jsonify(
                        success=True,
                        simple_answer=result,
                        domain="رياضيات",
                        confidence=95
                    )
                else:
                    print(f"❌ فشل التنفيذ: {error}")
    
    # رسالة مساعدة للمستخدم
    examples = [
        "x^2 + 5x + 6 = 0",
        "مشتقة sin(2x)",
        "تكامل x^2 من 0 إلى 2",
        "نهاية sin(x)/x عندما x -> 0",
        "مصفوفة [[1,2],[3,4]] محدد",
        "1+1",
        "2*x - 4 = 0"
    ]
    
    import random
    example = random.choice(examples)
    
    return jsonify(
        success=True,
        simple_answer="❓ لم أتمكن من حل السؤال",
        suggestion=f"جرب صيغة واضحة مثل:\n• {example}",
        domain="رياضيات",
        confidence=0
    )

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 MathCore - النسخة القوية جداً 🔥")
    print("="*70)
    print("✅ SymPy: 50+ دالة رياضية")
    print("✅ العمليات: solve, diff, integrate, limit, sum, matrix, simplify, expand, factor, dsolve")
    print("✅ الدوال: مثلثية، زائدية، أسية، لوغاريتمية")
    print("✅ التعامل مع: معادلات، مصفوفات، تفاضل جزئي")
    print("="*70)
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
