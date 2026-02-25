from flask import Flask, render_template, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, limit, parse_expr, sin, cos, tan, log, exp
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# الرموز الرياضية الأساسية
x, y, z, t = symbols('x y z t')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "السؤال فارغ"})
    
    try:
        result = solve_simple_math(question)
        
        return jsonify({
            "success": True,
            "question": question,
            "result": result
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

def solve_simple_math(question):
    if "=" in question:
        left, right = question.split("=")
        try:
            left_expr = parse_expr(left)
            right_expr = parse_expr(right)
            eq = Eq(left_expr, right_expr)
            solutions = solve(eq, x)
            return f"الحل: x = {solutions[0] if solutions else 'لا يوجد حل'}"
        except:
            pass
    
    if "مشتقة" in question or "diff" in question:
        try:
            if "sin" in question:
                return str(diff(sin(x), x))
            elif "x**2" in question:
                return str(diff(x**2, x))
        except:
            pass
    
    return "لم أتمكن من حل هذا السؤال بعد. سأتحسن مع إضافة DeepSeek!"

if __name__ == '__main__':
    print("🚀 التطبيق يعمل على: http://127.0.0.1:5000")
    app.run(debug=True)
