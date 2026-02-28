from flask import Flask, render_template, request, jsonify
import sympy as sp
import google.generativeai as genai
import re
import os
import math

app = Flask(__name__)

# ============================================================
# 🧮 المستوى الأول: الآلة الحاسبة الذكية (محلية)
# ============================================================
class SmartCalculator:
    def __init__(self):
        self.x = sp.symbols('x')
        
    def solve_simple(self, expression):
        """حل المسائل البسيطة مع خطوات"""
        try:
            steps = []
            expr = expression.strip()
            
            # 1️⃣ معادلات بسيطة (x+5=10)
            if '=' in expr and 'x' in expr:
                return self._solve_equation(expr)
            
            # 2️⃣ دوال مثلثية (sin30, cos60)
            trig_match = re.search(r'(sin|cos|tan)(\d+)', expr)
            if trig_match:
                func, angle = trig_match.groups()
                return self._solve_trig(func, float(angle))
            
            # 3️⃣ جذور (√16, ∛27)
            if '√' in expr or 'sqrt' in expr:
                num = re.search(r'(\d+)', expr)
                if num:
                    return self._solve_sqrt(float(num.group(1)))
            
            # 4️⃣ عمليات حسابية بسيطة
            return self._solve_arithmetic(expr)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _solve_equation(self, expr):
        """حل معادلة مثل x+5=10"""
        steps = []
        left, right = expr.split('=')
        
        steps.append(f"**المعادلة:** {left} = {right}")
        steps.append(f"**الخطوة 1:** ننقل الحدود")
        
        # حل المعادلة
        eq = sp.Eq(sp.sympify(left), sp.sympify(right))
        solution = sp.solve(eq, self.x)
        
        steps.append(f"**الخطوة 2:** {self.x} = {solution[0]}")
        steps.append(f"**التحقق:** {left.replace('x', str(solution[0]))} = {right}")
        
        return {
            "success": True,
            "result": f"x = {solution[0]}",
            "steps": steps
        }
    
    def _solve_trig(self, func, angle):
        """حل دوال مثلثية مع خطوات"""
        steps = []
        rad = math.radians(angle)
        
        steps.append(f"**المطلوب:** حساب {func}({angle}°)")
        steps.append(f"**الخطوة 1:** تحويل الزاوية إلى راديان")
        steps.append(f"{angle}° = {rad:.4f} راديان")
        
        if func == 'sin':
            result = math.sin(rad)
            steps.append(f"**الخطوة 2:** sin({rad:.4f}) = {result:.4f}")
        elif func == 'cos':
            result = math.cos(rad)
            steps.append(f"**الخطوة 2:** cos({rad:.4f}) = {result:.4f}")
        elif func == 'tan':
            result = math.tan(rad)
            steps.append(f"**الخطوة 2:** tan({rad:.4f}) = {result:.4f}")
        
        return {
            "success": True,
            "result": f"{func}({angle}°) = {result:.4f}",
            "steps": steps
        }
    
    def _solve_sqrt(self, num):
        """حل الجذور مع خطوات"""
        steps = []
        steps.append(f"**المطلوب:** حساب √{num}")
        
        # تحليل العدد
        factors = []
        n = num
        i = 2
        while i * i <= n:
            while n % i == 0:
                factors.append(i)
                n //= i
            i += 1
        if n > 1:
            factors.append(n)
        
        if factors:
            steps.append(f"**الخطوة 1:** تحليل {int(num)}: {' × '.join(map(str, factors))}")
        
        result = math.sqrt(num)
        steps.append(f"**الخطوة 2:** √{int(num)} = {result:.4f}")
        
        return {
            "success": True,
            "result": f"√{int(num)} = {result:.4f}",
            "steps": steps
        }
    
    def _solve_arithmetic(self, expr):
        """عمليات حسابية بسيطة"""
        steps = []
        steps.append(f"**المطلوب:** حساب {expr}")
        
        # حساب باستخدام SymPy
        result = sp.sympify(expr).evalf()
        
        if '+' in expr:
            a, b = expr.split('+')
            steps.append(f"**الخطوة 1:** نجمع {a} + {b}")
        elif '-' in expr:
            a, b = expr.split('-')
            steps.append(f"**الخطوة 1:** نطرح {b} من {a}")
        elif '*' in expr or '×' in expr:
            a, b = expr.replace('×', '*').split('*')
            steps.append(f"**الخطوة 1:** نضرب {a} × {b}")
        elif '/' in expr or '÷' in expr:
            a, b = expr.replace('÷', '/').split('/')
            steps.append(f"**الخطوة 1:** نقسم {a} ÷ {b}")
        
        steps.append(f"**النتيجة:** {result}")
        
        return {
            "success": True,
            "result": str(result),
            "steps": steps
        }


# ============================================================
# 🤖 المستوى الثاني: Gemini + SymPy للمسائل المعقدة
# ============================================================
class AdvancedSolver:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        self.x, self.y = sp.symbols('x y')
    
    def solve_complex(self, question):
        """حل المسائل المعقدة باستخدام Gemini + SymPy"""
        
        # Gemini يحول السؤال لكود SymPy
        prompt = f"""
        Convert this math problem to Python code using sympy.
        Show complete step-by-step solution.
        
        Problem: {question}
        
        Rules:
        1. Use sympy for calculations
        2. Print each step with explanation
        3. Show final answer
        4. Use Arabic for explanations
        
        Example for "integrate x^2 from 0 to 1":
        ```python
        import sympy as sp
        x = sp.symbols('x')
        
        print("**المطلوب:** حساب ∫ x² dx من 0 إلى 1")
        
        f = x**2
        print("**الخطوة 1:** الدالة f(x) = x²")
        
        integral = sp.integrate(f, (x, 0, 1))
        print("**الخطوة 2:** ∫ x² dx = [x³/3] من 0 إلى 1")
        
        result = integral.evalf()
        print(f"**النتيجة:** {{result}}")
        ```
        
        Return only the code.
        """
        
        response = self.model.generate_content(prompt)
        code = self._extract_code(response.text)
        
        # تنفيذ الكود
        output = self._execute_code(code)
        
        return {
            "success": True,
            "result": output,
            "code": code
        }
    
    def _extract_code(self, text):
        code_pattern = r'```python\n(.*?)```'
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def _execute_code(self, code):
        import sys
        import io
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            exec(code, {'sp': sp, 'x': self.x, 'y': self.y})
            return new_stdout.getvalue()
        finally:
            sys.stdout = old_stdout


# ============================================================
# 🚀 تهيئة المحركات
# ============================================================
calculator = SmartCalculator()
api_key = os.environ.get('GEMINI_API_KEY')
advanced = AdvancedSolver(api_key) if api_key else None


# ============================================================
# 🎯 المسارات
# ============================================================
@app.route('/')
def home():
    return render_template('calculator.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        expression = data.get('expression', '').strip()
        
        if not expression:
            return jsonify({"success": False, "error": "التعبير فارغ"})
        
        # محاولة الحل بالآلة الحاسبة أولاً
        result = calculator.solve_simple(expression)
        
        if result and result.get('success'):
            return jsonify({
                "success": True,
                "result": result['result'],
                "steps": result['steps'],
                "level": "simple"
            })
        
        # إذا فشلت، استخدم Gemini للمسائل المعقدة
        if advanced:
            complex_result = advanced.solve_complex(expression)
            return jsonify({
                "success": True,
                "result": complex_result['result'],
                "steps": complex_result['result'].split('\n'),
                "level": "advanced"
            })
        
        return jsonify({
            "success": False,
            "error": "لم أتمكن من حل المسألة"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧮 الآلة الحاسبة الذكية + Gemini للمسائل المعقدة")
    print("="*70)
    print("✅ المستوى الأول: آلة حاسبة ذكية")
    print("   • عمليات حسابية: 2+2, 5×3, 10÷2")
    print("   • دوال مثلثية: sin30°, cos60°, tan45°")
    print("   • جذور: √16, ∛27")
    print("   • معادلات بسيطة: x+5=10")
    print("✅ المستوى الثاني: Gemini + SymPy للم
