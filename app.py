from flask import Flask, render_template, request, jsonify
import sympy as sp
import google.generativeai as genai
import math
import re
import os
import sys
import io
import json
import hashlib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ============================================================
# 💾 نظام الذاكرة الذاتية
# ============================================================
class MemorySystem:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS solutions (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    level TEXT NOT NULL,
                    code TEXT,
                    uses INTEGER DEFAULT 1,
                    created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_question ON solutions(question)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON solutions(last_used)")
    
    def get(self, question):
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT answer, steps, level, code, uses
                FROM solutions WHERE id = ?
            """, (q_hash,))
            row = cursor.fetchone()
            if row:
                conn.execute("UPDATE solutions SET uses = uses + 1, last_used = CURRENT_TIMESTAMP WHERE id = ?", (q_hash,))
                conn.commit()
                return {
                    "answer": row["answer"],
                    "steps": json.loads(row["steps"]),
                    "level": row["level"],
                    "code": row["code"],
                    "uses": row["uses"]
                }
        return None
    
    def save(self, question, answer, steps, level, code=None):
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        steps_json = json.dumps(steps)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO solutions 
                (id, question, answer, steps, level, code)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (q_hash, question[:200], answer, steps_json, level, code))
            conn.commit()
    
    def stats(self):
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM solutions").fetchone()[0]
            total_uses = conn.execute("SELECT SUM(uses) FROM solutions").fetchone()[0] or 0
            return {"total": total, "total_uses": total_uses}


# ============================================================
# 🧮 المستوى الأول: الآلة الحاسبة الذكية (محلية)
# ============================================================
class SmartCalculator:
    def __init__(self):
        self.x = sp.symbols('x')
        
    def solve(self, expression):
        """حل المسائل البسيطة مع خطوات تفصيلية"""
        try:
            # تنظيف التعبير أولاً
            original = expression
            expression = expression.replace('^', '**').replace('×', '*').replace('÷', '/')
            
            # 1️⃣ عمليات حسابية بسيطة
            if self._is_arithmetic(expression):
                result = self._solve_arithmetic(expression)
                if result and result.get('success'):
                    return result
            
            # 2️⃣ دوال مثلثية
            if self._is_trig(expression):
                return self._solve_trig(expression)
            
            # 3️⃣ جذور
            if self._is_root(expression):
                return self._solve_root(expression)
            
            # 4️⃣ لوغاريتمات
            if self._is_log(expression):
                return self._solve_log(expression)
            
            # 5️⃣ معادلات بسيطة
            if '=' in expression and 'x' in expression:
                return self._solve_equation(expression)
            
            return {"success": False, "error": "تعبير غير مدعوم أو غير صالح"}
            
        except Exception as e:
            return {"success": False, "error": f"خطأ في المعالجة: {str(e)}"}
    
    def _is_arithmetic(self, expr):
        return any(op in expr for op in ['+', '-', '*', '/'])
    
    def _is_trig(self, expr):
        return any(t in expr.lower() for t in ['sin', 'cos', 'tan', 'جتا', 'جا', 'ظا'])
    
    def _is_root(self, expr):
        return '√' in expr or 'sqrt' in expr or 'جذر' in expr
    
    def _is_log(self, expr):
        return 'log' in expr.lower() or 'ln' in expr.lower() or 'لوغ' in expr
    
    def _extract_numbers(self, expr):
        return [float(n) for n in re.findall(r'-?\d+\.?\d*', expr)]
    
    def _extract_angle(self, expr):
        match = re.search(r'(\d+)', expr)
        return float(match.group(1)) if match else None
    
    def _solve_arithmetic(self, expr):
        """حل العمليات الحسابية مع خطوات"""
        steps = []
        steps.append(f"📝 **المطلوب:** حساب {expr}")
        
        try:
            # حساب يدوي للعمليات البسيطة
            if '+' in expr:
                a, b = expr.split('+')
                a, b = float(a), float(b)
                steps.append(f"**الخطوة 1:** نجمع {a} + {b}")
                result = a + b
            elif '-' in expr:
                a, b = expr.split('-')
                a, b = float(a), float(b)
                steps.append(f"**الخطوة 1:** نطرح {b} من {a}")
                result = a - b
            elif '*' in expr:
                a, b = expr.split('*')
                a, b = float(a), float(b)
                steps.append(f"**الخطوة 1:** نضرب {a} × {b}")
                result = a * b
            elif '/' in expr:
                a, b = expr.split('/')
                a, b = float(a), float(b)
                if b == 0:
                    return {"success": False, "error": "لا يمكن القسمة على صفر"}
                steps.append(f"**الخطوة 1:** نقسم {a} ÷ {b}")
                result = a / b
            else:
                # استخدام SymPy للتعبيرات المعقدة
                result = sp.sympify(expr).evalf()
                steps.append(f"**النتيجة:** {result}")
            
            if result.is_integer():
                result = int(result)
            
            steps.append(f"✅ **النتيجة:** {result}")
            
            return {
                "success": True,
                "answer": str(result),
                "steps": steps,
                "level": "simple"
            }
        except Exception as e:
            return {"success": False, "error": f"تعبير حسابي غير صالح: {str(e)}"}
    
    def _solve_trig(self, expr):
        """حل الدوال المثلثية مع خطوات"""
        steps = []
        steps.append(f"📝 **المطلوب:** حساب {expr}")
        
        try:
            angle = self._extract_angle(expr)
            if angle is None:
                return {"success": False, "error": "لم يتم العثور على زاوية"}
            
            rad = math.radians(angle)
            
            if 'sin' in expr.lower() or 'جا' in expr:
                steps.append(f"**الخطوة 1:** sin(θ) = المقابل / الوتر")
                steps.append(f"**الخطوة 2:** تحويل {angle}° إلى راديان: {rad:.4f} rad")
                result = math.sin(rad)
            elif 'cos' in expr.lower() or 'جتا' in expr:
                steps.append(f"**الخطوة 1:** cos(θ) = المجاور / الوتر")
                steps.append(f"**الخطوة 2:** تحويل {angle}° إلى راديان: {rad:.4f} rad")
                result = math.cos(rad)
            elif 'tan' in expr.lower() or 'ظا' in expr:
                steps.append(f"**الخطوة 1:** tan(θ) = المقابل / المجاور")
                steps.append(f"**الخطوة 2:** تحويل {angle}° إلى راديان: {rad:.4f} rad")
                result = math.tan(rad)
            else:
                return {"success": False, "error": "دالة مثلثية غير معروفة"}
            
            if abs(result) < 1e-10:
                result = 0.0
            
            steps.append(f"✅ **النتيجة:** {result:.6f}")
            
            return {
                "success": True,
                "answer": f"{result:.6f}",
                "steps": steps,
                "level": "simple"
            }
        except Exception as e:
            return {"success": False, "error": f"خطأ في الدالة المثلثية: {str(e)}"}
    
    def _solve_root(self, expr):
        """حل الجذور مع خطوات"""
        steps = []
        steps.append(f"📝 **المطلوب:** حساب {expr}")
        
        try:
            numbers = self._extract_numbers(expr)
            if not numbers:
                return {"success": False, "error": "لم يتم العثور على رقم"}
            
            num = numbers[0]
            
            if '∛' in expr or 'cbrt' in expr.lower():
                steps.append(f"**الخطوة 1:** جذر تكعيبي")
                result = num ** (1/3)
            else:
                steps.append(f"**الخطوة 1:** جذر تربيعي")
                result = math.sqrt(num)
            
            steps.append(f"**الخطوة 2:** الحساب = {result:.6f}")
            steps.append(f"✅ **النتيجة:** {result:.6f}")
            
            return {
                "success": True,
                "answer": f"{result:.6f}",
                "steps": steps,
                "level": "simple"
            }
        except Exception as e:
            return {"success": False, "error": f"تعبير جذر غير صالح: {str(e)}"}
    
    def _solve_log(self, expr):
        """حل اللوغاريتمات مع خطوات"""
        steps = []
        
        try:
            numbers = self._extract_numbers(expr)
            if not numbers:
                return {"success": False, "error": "لم يتم العثور على رقم"}
            
            num = numbers[0]
            
            if 'ln' in expr.lower():
                steps.append(f"📝 **المطلوب:** حساب ln({num})")
                steps.append(f"**الخطوة 1:** ln({num}) = logₑ({num})")
                result = math.log(num)
            else:
                steps.append(f"📝 **المطلوب:** حساب log({num})")
                steps.append(f"**الخطوة 1:** log({num}) = log₁₀({num})")
                result = math.log10(num)
            
            steps.append(f"✅ **النتيجة:** {result:.6f}")
            
            return {
                "success": True,
                "answer": f"{result:.6f}",
                "steps": steps,
                "level": "simple"
            }
        except Exception as e:
            return {"success": False, "error": f"تعبير لوغاريتمي غير صالح: {str(e)}"}
    
    def _solve_equation(self, expr):
        """حل معادلات بسيطة مع خطوات"""
        steps = []
        
        try:
            left, right = expr.split('=')
            steps.append(f"📝 **المعادلة:** {left} = {right}")
            
            # حل باستخدام SymPy
            eq = sp.Eq(sp.sympify(left), sp.sympify(right))
            solution = sp.solve(eq, self.x)
            
            if not solution:
                return {"success": False, "error": "لا يوجد حل"}
            
            steps.append(f"**الخطوة 1:** ننقل الحدود")
            steps.append(f"**الخطوة 2:** نبسط المعادلة")
            steps.append(f"**الخطوة 3:** {self.x} = {solution[0]}")
            
            return {
                "success": True,
                "answer": f"x = {solution[0]}",
                "steps": steps,
                "level": "simple"
            }
        except Exception as e:
            return {"success": False, "error": f"معادلة غير صالحة: {str(e)}"}


# ============================================================
# 🤖 المستوى الثاني: Gemini + Code Execution
# ============================================================
class GeminiSolver:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-001')
                print("🤖 Gemini متصل")
            except Exception as e:
                print(f"⚠️ خطأ في اتصال Gemini: {e}")
    
    def solve(self, question):
        if not self.model:
            return {
                "success": False,
                "error": "Gemini غير متاح"
            }
        
        try:
            # 1️⃣ Gemini يكتب الكود
            code = self._generate_code(question)
            
            # 2️⃣ تنفيذ الكود
            output = self._execute_code(code)
            
            # 3️⃣ تنقية النتائج
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            steps = lines[:-1] if len(lines) > 1 else lines
            answer = lines[-1] if lines else output
            
            return {
                "success": True,
                "answer": answer,
                "steps": steps,
                "code": code,
                "level": "advanced"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"خطأ في Gemini: {str(e)}"
            }
    
    def _generate_code(self, question):
        prompt = f"""
        Write Python code to solve this math problem step by step.
        Use sympy library.
        Show each step with print statements in Arabic.
        
        Problem: {question}
        
        Return only the code, no explanations.
        """
        
        response = self.model.generate_content(prompt)
        return self._extract_code(response.text)
    
    def _extract_code(self, text):
        code_pattern = r'```python\n(.*?)```'
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def _execute_code(self, code):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            # بيئة آمنة للتنفيذ
            x, y, z = sp.symbols('x y z')
            safe_globals = {
                'sp': sp,
                'math': math,
                'x': x,
                'y': y,
                'z': z,
                '__builtins__': {
                    'print': print,
                    'range': range,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'abs': abs,
                    'round': round
                }
            }
            exec(code, safe_globals)
            return new_stdout.getvalue()
        except Exception as e:
            return f"خطأ في تنفيذ الكود: {e}"
        finally:
            sys.stdout = old_stdout


# ============================================================
# 🚀 تهيئة المحركات
# ============================================================
calculator = SmartCalculator()
memory = MemorySystem()

api_key = os.environ.get('GEMINI_API_KEY')
gemini = GeminiSolver(api_key)


# ============================================================
# 🎯 المسارات الرئيسية
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "السؤال فارغ"})
        
        print(f"\n🔍 سؤال: {question}")
        
        # 1️⃣ البحث في الذاكرة أولاً
        memory_result = memory.get(question)
        if memory_result:
            print(f"💾 من الذاكرة (استخدام {memory_result['uses']})")
            return jsonify({
                "success": True,
                "answer": memory_result["answer"],
                "steps": memory_result["steps"],
                "level": "memory",
                "from_memory": True,
                "uses": memory_result["uses"]
            })
        
        # 2️⃣ المستوى الأول: آلة حاسبة
        simple_result = calculator.solve(question)
        if simple_result and simple_result.get('success'):
            print(f"✅ حل محلي: {simple_result['answer']}")
            memory.save(
                question=question,
                answer=simple_result['answer'],
                steps=simple_result['steps'],
                level='simple'
            )
            return jsonify({
                "success": True,
                "answer": simple_result['answer'],
                "steps": simple_result['steps'],
                "level": "simple"
            })
        
        # 3️⃣ المستوى الثاني: Gemini
        if gemini and gemini.model:
            print(f"🤢 إرسال إلى Gemini...")
            advanced_result = gemini.solve(question)
            if advanced_result.get('success'):
                print(f"✅ حل من Gemini")
                memory.save(
                    question=question,
                    answer=advanced_result['answer'],
                    steps=advanced_result['steps'],
                    level='advanced',
                    code=advanced_result.get('code')
                )
                return jsonify({
                    "success": True,
                    "answer": advanced_result['answer'],
                    "steps": advanced_result['steps'],
                    "code": advanced_result.get('code'),
                    "level": "advanced"
                })
        
        return jsonify({
            "success": False,
            "error": "لم أتمكن من حل المسألة"
        })
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/memory/stats', methods=['GET'])
def memory_stats():
    return jsonify({
        "success": True,
        "stats": memory.stats()
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧮 الآلة الحاسبة الذكية + الذاكرة + Gemini")
    print("="*70)
    print("✅ المستوى الأول: آلة حاسبة محلية")
    print("   • 1+1, 5×3, 10÷2")
    print("   • sin30, cos60, tan45")
    print("   • √16, log100, ln(e)")
    print("   • x+5=10, 2x=8")
    print()
    print("✅ المستوى الثاني: Gemini + Code Execution")
    print("   • تكاملات، مشتقات، نهايات")
    print("   • مسائل معقدة")
    print()
    print("✅ الذاكرة الذاتية")
    print("   • تحفظ كل سؤال")
    print("   • تستخدم الحلول المخزنة")
    print("="*70)
    print(f"🤖 Gemini: {'✅ متصل' if gemini and gemini.model else '❌ غير متصل'}")
    print(f"💾 الذاكرة: {memory.stats()['total']} سؤال")
    print(f"🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True)
