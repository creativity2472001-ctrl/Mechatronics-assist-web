"""
حل المعادلات الجبرية - نسخة مبسطة للاختبار
"""
import re
from sympy import symbols, Eq, solve

x = symbols('x')

def is_equation(question):
    return '=' in question

def solve(question):
    try:
        # تنظيف بسيط
        q = question.replace(' ', '').replace('x', 'x')
        if '=' in q:
            left, right = q.split('=')
            # تحويل الطرفين إلى تعبيرات sympy
            left_expr = eval(left)
            right_expr = eval(right)
            eq = Eq(left_expr, right_expr)
            solutions = solve(eq, x)
            if len(solutions) == 1:
                return f'x = {solutions[0]}'
            elif len(solutions) > 1:
                return ', '.join([f'x = {s}' for s in solutions])
        return None
    except:
        return None
