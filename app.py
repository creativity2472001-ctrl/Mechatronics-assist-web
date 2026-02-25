from flask import Flask, request, jsonify, render_template
from sympy import symbols, Eq, solve, diff, integrate, limit, sympify, Function, sin, cos, tan, log, exp, sqrt, oo, Sum, factorial
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
import traceback

app = Flask(__name__)

# ==================== الرموز المتاحة ====================
x, y, z, t, n, k = symbols('x y z t n k')
f = Function('f')
SAFE_MATH = {
    "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
    "f": f,
    "sin": sin, "cos": cos, "tan": tan, "log": log, "ln": log,
    "exp": exp, "sqrt": sqrt, "pi": 3.141592653589793, "oo": oo,
    "Eq": Eq, "Function": Function,
    "sum": Sum, "factorial": factorial
}
transformations = standard_transformations + (implicit_multiplication,)

def preprocess(expr_str):
    expr_str = expr_str.replace('^', '**').replace(' ', '')
    return expr_str

def safe_parse(expr_str):
    try:
        return parse_expr(preprocess(expr_str), local_dict=SAFE_MATH, transformations=transformations)
    except:
        return None

# ==================== حل الأسئلة ====================
def solve_expression(question):
    try:
        q = question.strip()

        # ===== الحساب المباشر =====
        if all(c in '0123456789+-*/(). ' for c in q):
            try:
                return eval(preprocess(q))
            except:
                expr = safe_parse(q)
                if expr:
                    return expr.evalf()

        # ===== المعادلات =====
        if '=' in q:
            if '&&' in q:
                # أنظمة المعادلات
                eqs = q.split('&&')
                sympy_eqs = []
                all_vars = set()
                for e in eqs:
                    parts = e.split('=')
                    if len(parts) == 2:
                        left = safe_parse(parts[0])
                        right = safe_parse(parts[1])
                        if left and right:
                            eq = Eq(left, right)
                            sympy_eqs.append(eq)
                            all_vars.update(left.free_symbols.union(right.free_symbols))
                if sympy_eqs:
                    solutions = solve(sympy_eqs, list(all_vars))
                    return solutions
            else:
                parts = q.split('=')
                if len(parts) == 2:
                    left = safe_parse(parts[0])
                    right = safe_parse(parts[1])
                    if left is not None and right is not None:
                        eq = Eq(left, right)
                        vars_in_eq = list(left.free_symbols.union(right.free_symbols))
                        if not vars_in_eq:
                            return str(eq)
                        solutions = solve(eq, vars_in_eq)
                        return {str(v): str(s) for v, s in zip(vars_in_eq, solutions)}

        # ===== مشتقات =====
        if q.startswith('diff(') and q.endswith(')'):
            expr_content = q[5:-1]
            parts = expr_content.split(',')
            if len(parts) >= 2:
                expr = safe_parse(parts[0])
                var = symbols(parts[1].strip())
                order = int(parts[2].strip()) if len(parts) == 3 else 1
                if expr:
                    return diff(expr, var, order)

        # ===== تكاملات =====
        if q.startswith('integrate(') and q.endswith(')'):
            expr_content = q[10:-1]
            parts = expr_content.split(',')
            expr = safe_parse(parts[0])
            var = symbols(parts[1].strip()) if len(parts) > 1 else x
            if expr:
                if len(parts) == 4:  # تكامل محدد
                    lower = safe_parse(parts[2])
                    upper = safe_parse(parts[3])
                    return integrate(expr, (var, lower, upper))
                else:  # تكامل غير محدد
                    return integrate(expr, var)

        # ===== النهاية (limit) =====
        if q.startswith('limit(') and q.endswith(')'):
            expr_content = q[6:-1]
            parts = expr_content.split(',')
            if len(parts) == 3:
                expr = safe_parse(parts[0])
                var = symbols(parts[1].strip())
                point = safe_parse(parts[2])
                if expr:
                    return limit(expr, var, point)

        # ===== التسلسل والمتتابعات =====
        if q.startswith('sum(') and q.endswith(')'):
            expr_content = q[4:-1]
            parts = expr_content.split(',')
            if len(parts) == 3:
                expr = safe_parse(parts[0])
                var = symbols(parts[1].strip())
                limit_val = safe_parse(parts[2])
                if expr:
                    return Sum(expr, (var, 0, limit_val)).doit()

        # ===== الحساب عبر SymPy =====
        expr = safe_parse(q)
        if expr:
            return expr.evalf()

        return "❌ لم أتمكن من حل السؤال. جرب كتابته بصيغة واضحة."

    except Exception as e:
        traceback.print_exc()
        return f"🔥 خطأ: {e}"

# ==================== API ====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify(success=False, answer="❌ السؤال فارغ")
    result = solve_expression(question)
    return jsonify(success=True, answer=result)

# ==================== تشغيل التطبيق ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MathCore - النسخة الخارقة 2026")
    print("🌐 http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True)
