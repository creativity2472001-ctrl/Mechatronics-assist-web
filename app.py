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
    latex, pretty, solve_poly_system
)
from sympy.stats import Normal, Binomial, Poisson, mean, variance, std
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, 
    implicit_multiplication, convert_xor
)
import requests
import os
import json
import re
import traceback
import random
from dotenv import load_dotenv

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

load_dotenv()
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
    # إحصائيات
    "Normal": Normal, "Binom": Binomial, "Poisson": Poisson,
    "mean": mean, "variance": variance, "std": std
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

def simplify_result(expr):
    """تبسيط النتيجة الرياضية"""
    try:
        if isinstance(expr, str):
            expr_obj = safe_parse(expr)
            if expr_obj:
                return str(simplify(expr_obj))
        return str(simplify(expr))
    except:
        return str(expr)

# ============================================================
# 🔑 Gemini
# ============================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY and HAS_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ Gemini: متصل")
else:
    print("❌ Gemini: غير متصل")

def ask_gemini(question):
    if not GOOGLE_API_KEY or not HAS_GEMINI:
        return None
    
    prompt = f"""أنت محلل رياضي خبير. مهمتك تحويل أي سؤال رياضي إلى JSON دقيق.

السؤال: {question}

أنواع العمليات:
1. solve - حل المعادلات
2. diff - تفاضل
3. integrate - تكامل
4. limit - نهايات
5. matrix - مصفوفات
6. stats - إحصاء (متوسط، انحراف، توزيع طبيعي)
7. log - لوغاريتمات متقدمة
8. trig_inv - دوال مثلثية عكسية
9. ode - معادلات تفاضلية
10. mcq - اختيار من متعدد

أعد JSON فقط."""
    
    try:
        print("📡 جاري الاتصال بـ Gemini...")
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(prompt)
        result = response.text
        print(f"🔧 استجابة: {result[:200]}...")
        return result
    except Exception as e:
        print(f"🔥 خطأ Gemini: {e}")
        return None

def ask_gemini_with_steps(question):
    """إرسال السؤال لـ Gemini مع طلب الخطوات والشرح"""
    if not GOOGLE_API_KEY or not HAS_GEMINI:
        return None
    
    prompt = f"""أنت مدرس رياضيات خبير. مهمتك:
1. حل السؤال خطوة بخطوة
2. اكتب كل خطوة بوضوح
3. قدم النتيجة النهائية مبسطة

السؤال: {question}

أعد الإجابة بصيغة JSON:
{{
  "steps": ["خطوة 1: ...", "خطوة 2: ..."],
  "result": "النتيجة النهائية المبسطة",
  "explanation": "شرح عام للحل"
}}

أعد JSON فقط."""
    
    try:
        print("📡 جاري الاتصال بـ Gemini...")
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(prompt)
        result = response.text
        
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json_match.group()
        return None
    except Exception as e:
        print(f"🔥 خطأ Gemini: {e}")
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
# 🚀 تنفيذ العمليات الرياضية الموسعة
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
                return simplify_result(solutions), None
        
        elif cmd_type == "diff":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            order = cmd.get("order", 1)
            if expr:
                result = diff(expr, var, order)
                return simplify_result(result), None
        
        elif cmd_type == "integrate":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            
            if expr:
                if "lower" in cmd and "upper" in cmd:
                    lower = safe_parse(str(cmd["lower"]))
                    upper = safe_parse(str(cmd["upper"]))
                    result = integrate(expr, (var, lower, upper))
                else:
                    result = integrate(expr, var)
                return simplify_result(result) + (" + C" if "upper" not in cmd else ""), None
        
        elif cmd_type == "limit":
            expr = safe_parse(cmd.get("expression", ""))
            var_name = cmd.get("variable", "x")
            var = symbols(var_name)
            point = safe_parse(str(cmd.get("point", 0)))
            if expr:
                result = limit(expr, var, point)
                return simplify_result(result), None
        
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
                elif operation == "transpose":
                    return str(M.T), None
                else:
                    return str(M), None
            except:
                return None, "خطأ في المصفوفة"
        
        # ===== دوال مثلثية عكسية =====
        elif cmd_type == "trig_inv":
            expr = safe_parse(cmd.get("expression", ""))
            func = cmd.get("function", "asin")
            if expr:
                if func == "asin":
                    return str(asin(expr)), None
                elif func == "acos":
                    return str(acos(expr)), None
                elif func == "atan":
                    return str(atan(expr)), None
                return str(expr), None
        
        # ===== لوغاريتمات =====
        elif cmd_type == "log":
            expr = safe_parse(cmd.get("expression", ""))
            base = cmd.get("base", E)
            if base == E:
                return str(ln(expr)), None
            else:
                base_expr = safe_parse(str(base))
                return str(log(expr, base_expr)), None
        
        # ===== معادلات تفاضلية =====
        elif cmd_type == "ode":
            eq_str = cmd.get("equation", "")
            func_name = cmd.get("function", "f")
            var_name = cmd.get("variable", "x")
            
            var = symbols(var_name)
            f_func = Function(func_name)
            
            eq = safe_parse(eq_str.replace(func_name, func_name))
            if eq:
                result = dsolve(eq, f_func(var))
                return str(result), None
        
        # ===== إحصاء =====
        elif cmd_type == "stats":
            op = cmd.get("operation", "mean")
            data = cmd.get("data", [])
            
            if op == "mean":
                return str(sum(data) / len(data)), None
            elif op == "variance":
                m = sum(data) / len(data)
                var = sum((xi - m) ** 2 for xi in data) / (len(data) - 1)
                return str(var), None
            elif op == "std":
                m = sum(data) / len(data)
                var = sum((xi - m) ** 2 for xi in data) / (len(data) - 1)
                return str(var ** 0.5), None
        
        # ===== اختيار من متعدد =====
        elif cmd_type == "mcq":
            options = cmd.get("options", [])
            correct = cmd.get("correct", 0)
            
            # تحليل الخيارات
            result = f"الإجابة الصحيحة: {options[correct]}"
            if len(options) > 1:
                result += f"\nالخيارات: {', '.join(options)}"
            return result, None
        
        return None, f"نوع العملية {cmd_type} غير مدعوم"
        
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
        original_q = question
        
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
                print(f"🔄 مسألة معقدة: تذهب للذكاء")
                return None
        
        # عمليات حسابية بسيطة
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
        
        # معادلات بسيطة
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
            steps=["تم الحل مباشرة باستخدام SymPy"],
            domain="رياضيات",
            confidence=100
        )
    
    # المستوى 2: Gemini مع خطوات
    if GOOGLE_API_KEY and HAS_GEMINI:
        wants_explanation = any(word in question.lower() for word in ['شرح', 'خطوات', 'how', 'steps'])
        
        if wants_explanation:
            print("🔄 استخدام Gemini مع الخطوات...")
            response_json = ask_gemini_with_steps(question)
            if response_json:
                try:
                    data = json.loads(response_json)
                    steps = data.get('steps', [])
                    result = data.get('result', '')
                    explanation = data.get('explanation', '')
                    
                    # تبسيط النتيجة
                    simplified = simplify_result(result)
                    
                    return jsonify(
                        success=True,
                        simple_answer=simplified,
                        steps=steps,
                        explanation=explanation,
                        domain="رياضيات",
                        confidence=95
                    )
                except:
                    pass
        
        # الطريقة العادية
        print("🔄 استخدام Gemini...")
        analysis = ask_gemini(question)
        if analysis:
            cmd_json = extract_json_advanced(analysis)
            if cmd_json:
                print(f"📦 JSON: {cmd_json}")
                result, error = execute_math_command(cmd_json)
                
                if result:
                    simplified = simplify_result(result)
                    
                    return jsonify(
                        success=True,
                        simple_answer=simplified,
                        steps=["تم الحل باستخدام الذكاء الاصطناعي"],
                        domain="رياضيات",
                        confidence=95
                    )
    
    # رسالة مساعدة
    return jsonify(
        success=True,
        simple_answer="❓ لم أتمكن من حل السؤال",
        steps=["جرب كتابة السؤال بصيغة أوضح أو أضف 'شرح' للحصول على خطوات"],
        domain="رياضيات",
        confidence=0
    )

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 MathCore - النسخة النهائية للمنهاج الفلسطيني 🔥")
    print("="*70)
    print("✅ التفاضل والتكامل + تبسيط + خطوات + شرح")
    print("✅ إحصاء واحتمالات (متوسط، انحراف، توزيع طبيعي)")
    print("✅ لوغاريتمات متقدمة + تغيير الأساس")
    print("✅ دوال مثلثية عكسية (arcsin, arccos, arctan)")
    print("✅ معادلات تفاضلية (ODE)")
    print("✅ اختيار من متعدد (تحليل وتفسير)")
    print("="*70)
    print(f"🔑 Gemini: {'✅ متصل' if GOOGLE_API_KEY and HAS_GEMINI else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
