from flask import Flask, render_template, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, limit, summation, Matrix, Derivative, dsolve, Function, Integer
from sympy import sin, cos, tan, log, exp, sqrt, pi, oo, I
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
# الإعدادات الأساسية
# ============================================================

# الرموز الرياضية الأساسية
x, y, z, t = symbols('x y z t')
f = Function('f')

# مفتاح OpenRouter فقط (لا DeepSeek)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("⚠️ تحذير: مفتاح OpenRouter غير موجود في ملف .env")
    print("⚠️ التطبيق سيعمل فقط على المسائل البسيطة")

# ============================================================
# الأمان: قاموس آمن للدوال الرياضية
# ============================================================

SAFE_MATH = {
    "x": x, "y": y, "z": z, "t": t,
    "sin": sin, "cos": cos, "tan": tan,
    "log": log, "exp": exp, "sqrt": sqrt,
    "pi": pi, "oo": oo, "I": I,
    "Eq": Eq, "Derivative": Derivative,
    "Matrix": Matrix, "Function": Function,
    "f": f, "Integer": Integer
}

transformations = standard_transformations + (implicit_multiplication,)

def safe_parse(expr_str):
    """تحويل آمن للتعبيرات الرياضية"""
    try:
        return parse_expr(
            expr_str, 
            local_dict=SAFE_MATH,
            global_dict={},
            transformations=transformations
        )
    except Exception as e:
        print(f"❌ خطأ في تحليل التعبير: {e}")
        return None

# ============================================================
# استخراج JSON من OpenRouter
# ============================================================

def clean_json_text(text):
    """تنظيف النص من أي كلمات قبل أو بعد JSON"""
    if not text:
        return None
    
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return None

def extract_json_advanced(text):
    """استخراج JSON باستخدام json5 إن وجد"""
    if not text:
        return None
    
    cleaned = clean_json_text(text)
    if not cleaned:
        return None
    
    # json5
    if HAS_JSON5:
        try:
            data = json5.loads(cleaned)
            if isinstance(data, dict):
                return data
        except:
            pass
    
    # json عادي
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except:
        pass
    
    return None

# ============================================================
# الاتصال بـ OpenRouter فقط
# ============================================================

def ask_openrouter(question):
    """إرسال استفسار إلى OpenRouter"""
    if not OPENROUTER_API_KEY:
        return None
    
    # برومبت محسن
    prompt = f"""أنت محلل رياضي دقيق. مهمتك إرجاع JSON صالح فقط.

السؤال: {question}

أنواع العمليات:
1. solve - حل المعادلات
2. diff - تفاضل عادي أو جزئي
3. integrate - تكامل محدد أو غير محدد
4. limit - إيجاد النهايات
5. sum - حساب المجاميع
6. matrix - عمليات المصفوفات
7. simplify - تبسيط التعبيرات
8. dsolve - حل المعادلات التفاضلية

قواعد:
- أعد JSON فقط، لا تكتب أي كلمات أخرى
- استخدم "**" للأس (مثال: x**2)
- variable افتراضي هو "x"

أمثلة:
1. معادلة: {{"type": "solve", "expression": "x**2 + 5*x + 6", "variable": "x"}}
2. تفاضل: {{"type": "diff", "expression": "sin(2*x)", "variable": "x", "order": 1}}
3. تكامل: {{"type": "integrate", "expression": "x**2", "variable": "x", "lower": 0, "upper": 2}}
4. نهاية: {{"type": "limit", "expression": "sin(x)/x", "variable": "x", "point": 0}}

الآن حلل السؤال وأعد JSON فقط:"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",  # استخدام DeepSeek عبر OpenRouter
        "messages": [
            {"role": "system", "content": "أنت محلل رياضي. أعد JSON فقط بدون أي نصوص أخرى."},
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
            print(f"❌ خطأ من OpenRouter: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"🔥 خطأ في الاتصال: {e}")
        return None

# ============================================================
# تنفيذ العمليات الرياضية
# ============================================================

def execute_math_command(command_json):
    """تنفيذ الأمر الرياضي"""
    try:
        cmd_type = command_json.get("type", "")
        
        # solve - حل المعادلات
        if cmd_type == "solve":
            expr_str = command_json.get("expression", "")
            expr = safe_parse(expr_str)
            var_name = command_json.get("variable", "x")
            var = symbols(var_name)
            
            if expr is not None:
                result = solve(expr, var)
                return str(result), None
        
        # diff - تفاضل
        elif cmd_type == "diff":
            expr_str = command_json.get("expression", "")
            expr = safe_parse(expr_str)
            var_name = command_json.get("variable", "x")
            var = symbols(var_name)
            order = command_json.get("order", 1)
            
            if expr is not None:
                result = diff(expr, var, order)
                return str(result), None
        
        # integrate - تكامل
        elif cmd_type == "integrate":
            expr_str = command_json.get("expression", "")
            expr = safe_parse(expr_str)
            var_name = command_json.get("variable", "x")
            var = symbols(var_name)
            
            if expr is not None:
                if "lower" in command_json and "upper" in command_json:
                    lower = safe_parse(str(command_json["lower"]))
                    upper = safe_parse(str(command_json["upper"]))
                    result = integrate(expr, (var, lower, upper))
                else:
                    result = integrate(expr, var)
                return str(result) + (" + C" if "upper" not in command_json else ""), None
        
        # limit - نهايات
        elif cmd_type == "limit":
            expr_str = command_json.get("expression", "")
            expr = safe_parse(expr_str)
            var_name = command_json.get("variable", "x")
            var = symbols(var_name)
            point = command_json.get("point", 0)
            
            if expr is not None:
                point_expr = safe_parse(str(point)) if isinstance(point, str) else point
                result = limit(expr, var, point_expr)
                return str(result), None
        
        # مسائل بسيطة - حساب مباشر
        elif cmd_type == "calculate":
            expr_str = command_json.get("expression", "")
            expr = safe_parse(expr_str)
            if expr is not None:
                return str(expr.evalf()), None
        
        else:
            return None, f"نوع العملية '{cmd_type}' غير مدعوم"
            
    except Exception as e:
        print(f"❌ خطأ في التنفيذ: {e}")
        traceback.print_exc()
        return None, str(e)

# ============================================================
# المسائل البسيطة (بدون OpenRouter)
# ============================================================

def solve_simple_math(question):
    """حل المسائل البسيطة مباشرة بـ SymPy"""
    try:
        question = question.replace(" ", "")
        
        # حسابات بسيطة
        if question.isdigit() or '+' in question or '-' in question or '*' in question or '/' in question:
            try:
                expr = safe_parse(question)
                if expr is not None:
                    return str(expr.evalf())
            except:
                pass
        
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
        
        # تفاضل بسيط
        patterns = [
            (r'مشتقة.*sin\(x\)', diff(sin(x), x)),
            (r'مشتقة.*cos\(x\)', diff(cos(x), x)),
            (r'مشتقة.*x\*\*2', diff(x**2, x)),
            (r'diff.*sin\(x\)', diff(sin(x), x)),
        ]
        
        for pattern, result in patterns:
            if re.search(pattern, question):
                return str(result)
        
        # تكامل بسيط
        if 'تكامل' in question or 'integral' in question:
            if 'sin' in question:
                return str(integrate(sin(x), x)) + ' + C'
            elif 'cos' in question:
                return str(integrate(cos(x), x)) + ' + C'
            elif 'x**2' in question or 'x^2' in question:
                return str(integrate(x**2, x)) + ' + C'
        
        return None
    except Exception as e:
        print(f"⚠️ خطأ في الحل المباشر: {e}")
        return None

# ============================================================
# المسار الرئيسي
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    question = data.get('question', '').strip()
    
    print(f"\n{'='*60}")
    print(f"📝 سؤال المستخدم: {question}")
    print(f"{'='*60}")
    
    if not question:
        return jsonify({
            "success": False,
            "simple_answer": "❌ السؤال فارغ"
        })
    
    # المستوى 1: مسائل بسيطة
    simple_result = solve_simple_math(question)
    if simple_result:
        print("✅ تم الحل مباشرة بـ SymPy")
        return jsonify({
            "success": True,
            "simple_answer": simple_result,
            "domain": "رياضيات",
            "confidence": 100
        })
    
    # المستوى 2: OpenRouter
    if OPENROUTER_API_KEY:
        analysis = ask_openrouter(question)
        
        if analysis:
            command_json = extract_json_advanced(analysis)
            
            if command_json:
                print(f"📦 JSON: {command_json}")
                result, error = execute_math_command(command_json)
                
                if result:
                    return jsonify({
                        "success": True,
                        "simple_answer": result,
                        "domain": "رياضيات",
                        "confidence": 95
                    })
                else:
                    print(f"❌ فشل التنفيذ: {error}")
    
    # رسالة للمستخدم
    examples = [
        "x^2 + 5x + 6 = 0",
        "مشتقة sin(2x)",
        "تكامل x^2 من 0 إلى 2",
        "1+1",
        "2*3"
    ]
    
    import random
    example = random.choice(examples)
    
    return jsonify({
        "success": True,
        "simple_answer": "❓ لم أتمكن من حل السؤال",
        "suggestion": f"جرب صيغة واضحة مثل: {example}",
        "domain": "رياضيات",
        "confidence": 0
    })

# ============================================================
# التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 MathCore - OpenRouter + SymPy فقط")
    print("="*70)
    print("✅ مسائل بسيطة: 100% (بدون API)")
    print("✅ مسائل معقدة: عبر OpenRouter")
    print("✅ لا حاجة لـ DeepSeek المباشر")
    print("="*70)
    print(f"🔑 OpenRouter: {'✅ متصل' if OPENROUTER_API_KEY else '❌ غير متصل'}")
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
