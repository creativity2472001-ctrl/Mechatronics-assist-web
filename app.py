from flask import Flask, render_template, request, jsonify
import sympy as sp
import re

app = Flask(__name__)

# آلة حاسبة بسيطة (تشتغل بدون Gemini)
def simple_calc(expr):
    try:
        expr = expr.replace(' ', '')
        if '+' in expr:
            a, b = expr.split('+')
            return float(a) + float(b)
        elif '-' in expr:
            a, b = expr.split('-')
            return float(a) - float(b)
        elif '*' in expr:
            a, b = expr.split('*')
            return float(a) * float(b)
        elif '/' in expr:
            a, b = expr.split('/')
            if float(b) == 0:
                return None
            return float(a) / float(b)
    except:
        return None
    return None

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
        
        # 1️⃣ جرب الحل بالآلة الحاسبة أولاً
        calc_result = simple_calc(question)
        if calc_result is not None:
            return jsonify({
                "success": True,
                "answer": str(calc_result),
                "steps": [f"📝 حساب {question}", f"✅ النتيجة: {calc_result}"],
                "source": "calculator"
            })
        
        # 2️⃣ جرب حل معادلات بسيطة
        if '=' in question and 'x' in question:
            try:
                left, right = question.split('=')
                x = sp.symbols('x')
                expr = sp.sympify(left) - sp.sympify(right)
                solution = sp.solve(expr, x)
                if solution:
                    return jsonify({
                        "success": True,
                        "answer": f"x = {solution[0]}",
                        "steps": [f"📝 حل {question}", f"✅ x = {solution[0]}"],
                        "source": "solver"
                    })
            except:
                pass
        
        return jsonify({
            "success": False,
            "error": "لم أتمكن من حل المسألة"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
