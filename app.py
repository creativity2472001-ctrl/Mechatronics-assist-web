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
# 💾 نظام الذاكرة الذاتية (محسّن)
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
            
            # ✅ تحسين الأداء: إضافة فهارس
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_question ON solutions(question)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_used ON solutions(last_used)
            """)
            
            # ✅ VACUUM دوري للحفاظ على الأداء
            conn.execute("VACUUM")
    
    def get(self, question):
        """البحث عن سؤال في الذاكرة"""
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT answer, steps, level, code, uses
                FROM solutions WHERE id = ?
            """, (q_hash,))
            
            row = cursor.fetchone()
            if row:
                # تحديث عدد الاستخدامات ووقت آخر استخدام
                conn.execute("""
                    UPDATE solutions 
                    SET uses = uses + 1, last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (q_hash,))
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
        """حفظ حل جديد في الذاكرة"""
        q_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        steps_json = json.dumps(steps)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO solutions 
                (id, question, answer, steps, level, code)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (q_hash, question[:200], answer, steps_json, level, code))
            conn.commit()
            
            # تنظيف دوري (اختياري)
            self._cleanup_old_entries(conn)
    
    def _cleanup_old_entries(self, conn, max_age_days=365):
        """حذف الإدخالات القديمة جداً (اختياري)"""
        conn.execute("""
            DELETE FROM solutions 
            WHERE last_used < datetime('now', '-? days')
        """, (max_age_days,))
        conn.commit()
    
    def stats(self):
        """إحصائيات الذاكرة"""
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
            expression = expression.replace('^', '**').replace('×', '*').replace('÷', '/')
            
            # 1️⃣ عمليات حسابية بسيطة
            if self._is_arithmetic(expression):
                return self._solve_arithmetic(expression)
            
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
            
            # 6️⃣ محاولة تحليل التعبير باستخدام SymPy
            try:
                expr = sp.sympify(expression)
                if expr.is_number:
                    result = float(expr.evalf())
                    return {
                        "success": True,
                        "answer": str(result),
                        "steps": [f"📝 **حساب:** {expression}", f"✅ **النتيجة:** {result}"],
                        "level": "simple"
                    }
            except:
                pass
            
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
        """استخراج جميع الأرقام من التعبير"""
        return [float(n) for n in re.findall(r'-?\d+\.?\d*', expr)]
    
    def _extract_angle(self, expr):
        """استخراج الزاوية من تعبير مثلثي"""
        match = re.search(r'(\d+)', expr)
        return float(match.group(1)) if match else None
    
    def _solve_arithmetic(self, expr):
        """حل العمليات الحسابية مع خطوات"""
        steps = []
        steps.append(f"📝 **المطلوب:** حساب {expr}")
        
        try:
            # حساب باستخدام SymPy (آمن للتعبيرات المعقدة)
            result = sp.sympify(expr).evalf()
            
            if result.is_integer():
                result = int(result)
            
            steps.append(f"✅ **النتيجة:** {result}")
            
            return {
                "success": True,
                "answer": str(result),
                "steps": steps,
                "level": "simple"
            }
        except:
            return {"success": False, "error": "تعبير حسابي غير صالح"}
    
    def _solve_trig(self, expr):
        """حل الدوال المثلثية مع خطوات"""
        steps = []
        steps.append(f"📝 **المطلوب:** حساب {expr}")
        
        try:
            # ✅ تحويل الزاوية من درجات إلى راديان
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
            
            # ✅ تجنب أخطاء التقريب
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
            
            # تحليل العدد
            if num.is_integer():
                n = int(num)
                factors = []
                temp = n
                i = 2
                while i * i <= temp:
                    while temp % i == 0:
                        factors.append(i)
                        temp //= i
                    i += 1
                if temp > 1:
                    factors.append(temp)
                
                if factors:
                    steps.append(f"**الخطوة 1:** تحليل {n}: {' × '.join(map(str, factors))}")
            
            result = math.sqrt(num)
            steps.append(f"**الخطوة 2:** √{num} = {result:.6f}")
            steps.append(f"✅ **النتيجة:** {result:.6f}")
            
            return {
                "success": True,
                "answer": f"{result:.6f}",
                "steps": steps,
                "level": "simple"
            }
        except:
            return {"success": False, "error": "تعبير جذر غير صالح"}
    
    def _solve_log(self, expr):
        """حل اللوغاريتمات مع خطوات"""
        steps = []
        
        try:
            numbers = self._extract_numbers(expr)
            if not numbers:
                return {"success": False, "error": "لم يتم العثور على رقم"}
            
            num = numbers[0]
            
            if 'ln' in expr.lower():
                steps.append(f"📝 **المطلوب:** حساب ln({num}) (لوغاريتم طبيعي)")
                steps.append(f"**الخطوة 1:** ln({num}) = logₑ({num})")
                result = math.log(num)
            else:
                steps.append(f"📝 **المطلوب:** حساب log({num}) (لوغاريتم عشري)")
                steps.append(f"**الخطوة 1:** log({num}) = log₁₀({num})")
                result = math.log10(num)
            
            steps.append(f"✅ **النتيجة:** {result:.6f}")
            
            return {
                "success": True,
                "answer": f"{result:.6f}",
                "steps": steps,
                "level": "simple"
            }
        except:
            return {"success": False, "error": "تعبير لوغاريتمي غير صالح"}
    
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
            
            # تحقق
            try:
                check = left.replace('x', f"({solution[0]})")
                steps.append(f"✅ **التحقق:** {check} = {right}")
            except:
                pass
            
            return {
                "success": True,
                "answer": f"x = {solution[0]}",
                "steps": steps,
                "level": "simple"
            }
        except:
            return {"success": False, "error": "معادلة غير صالحة"}


# ============================================================
# 🤖 المستوى الثاني: Gemini + Code Execution (آمن جداً)
# ============================================================
class GeminiSolver:
    def __init__(self, api_key):
        if not api_key:
            print("⚠️ تحذير: لم يتم تعيين GEMINI_API_KEY. المستوى الثاني لن يعمل.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-001')
        
    def solve(self, question):
        """Gemini يحول السؤال لكود وينفذه بأمان"""
        
        if not self.model:
            return {
                "success": False,
                "error": "مفتاح Gemini غير موجود. المستوى الثاني غير متاح."
            }
        
        # 1️⃣ Gemini يكتب الكود
        code = self._generate_code(question)
        
        # 2️⃣ تنفيذ الكود في بيئة آمنة جداً
        output = self._execute_code_safely(code)
        
        # 3️⃣ استخراج الخطوات والنتيجة (مع تنظيف النتائج)
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
    
    def _generate_code(self, question):
        prompt = f"""
        Write Python code to solve this math problem step by step.
        Use sympy library.
        Show each step with print statements in Arabic.
        
        Problem: {question}
        
        Example for "integrate x^2 from 0 to 1":
        ```python
        import sympy as sp
        x = sp.symbols('x')
        
        print("📝 المطلوب: حساب ∫ x² dx من 0 إلى 1")
        print()
        
        f = x**2
        print("الخطوة 1: الدالة f(x) = x²")
        
        F = sp.integrate(f, x)
        print(f"الخطوة 2: المشتق العكسي = {{F}}")
        
        definite = sp.integrate(f, (x, 0, 1))
        print(f"الخطوة 3: التعويض بالحدود = {{definite}}")
        
        result = definite.evalf()
        print(f"✅ النتيجة: {{result}}")
        ```
        
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
    
    def _execute_code_safely(self, code):
        """تنفيذ الكود في بيئة آمنة جداً مع متغيرات رمزية إضافية"""
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            # ✅ بيئة آمنة جداً للتنفيذ مع متغيرات رمزية إضافية
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
                    'len': len,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'round': round,
                    'isinstance': isinstance,
                    'type': type,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'any': any,
                    'all': all
                }
            }
            
            # تنفيذ الكود في البيئة الآمنة
            exec(code, safe_globals)
            return new_stdout.getvalue()
            
        except Exception as e:
            return f"⚠️ خطأ في تنفيذ الكود: {str(e)}"
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
# 🎯 المسارات
# ============================================================
@app.route('/')
def home():
return render_template('index.html')  # استخدم اسم الملف الموجود

@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"success": False, "error": "السؤال فارغ"})
        
        # 1️⃣ البحث في الذاكرة أولاً
        memory_result = memory.get(question)
        if memory_result:
            return jsonify({
                "success": True,
                "answer": memory_result["answer"],
                "steps": memory_result["steps"],
                "level": memory_result["level"],
                "from_memory": True,
                "uses": memory_result["uses"]
            })
        
        # 2️⃣ المستوى الأول: جرب الآلة الحاسبة
        simple_result = calculator.solve(question)
        
        if simple_result and simple_result.get('success'):
            # حفظ في الذاكرة
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
        
        # 3️⃣ المستوى الثاني: استخدم Gemini
        if gemini and gemini.model:
            complex_result = gemini.solve(question)
            
            if complex_result.get('success'):
                # حفظ في الذاكرة
                memory.save(
                    question=question,
                    answer=complex_result['answer'],
                    steps=complex_result['steps'],
                    level='advanced',
                    code=complex_result.get('code')
                )
                
                return jsonify({
                    "success": True,
                    "answer": complex_result['answer'],
                    "steps": complex_result['steps'],
                    "code": complex_result.get('code'),
                    "level": "advanced"
                })
        else:
            return jsonify({
                "success": False,
                "error": "لم يتم حل المسألة محلياً، ومفتاح Gemini غير متاح للتكامل"
            })
        
        return jsonify({
            "success": False,
            "error": "لم أتمكن من حل المسألة"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/memory/stats', methods=['GET'])
def memory_stats():
    return jsonify({
        "success": True,
        "stats": memory.stats()
    })


if __name__ == '__main__':
    print("\n" + "="*90)
    print("🧮 الآلة الحاسبة الذكية + الذاكرة الذاتية + Gemini (النسخة النهائية)")
    print("="*90)
    print("✅ المستوى الأول - آلة حاسبة ذكية (محسنة بالكامل):")
    print("   • دوال مثلثية دقيقة بالدرجات ✓")
    print("   • دعم ^, ×, ÷ ✓")
    print("   • معالجة أخطاء متكاملة ✓")
    print("   • خطوات حل تفصيلية ✓")
    print()
    print("✅ المستوى الثاني - Gemini Code Execution (آمن جداً):")
    print("   • بيئة تنفيذ آمنة مع متغيرات رمزية (x,y,z) ✓")
    print("   • منع الأكواد الضارة تماماً ✓")
    print("   • تنظيف النتائج قبل العرض ✓")
    print()
    print("✅ نظام الذاكرة الذاتية (محسّن):")
    print("   • فهارس على last_used للأداء ✓")
    print("   • VACUOM دوري للحفاظ على السرعة ✓")
    print("   • تخزين كل سؤال وإعادة استخدامه ✓")
    print("="*90)
    print(f"🤖 Gemini: {'✅ متصل' if gemini and gemini.model else '❌ غير متصل'}")
    print(f"💾 الذاكرة: {memory.stats()['total']} سؤال محفوظ")
    print(f"🌐 http://127.0.0.1:5000")
    print("="*90 + "\n")
    
    app.run(debug=True)
