"""
حل المعادلات الجبرية
مثل: x+5=10, 2x-4=8, x²-5x+6=0
"""
import re
from sympy import symbols, Eq, solve, sympify

x = symbols('x')

def is_equation(question):
    """تحديد إذا كان السؤال معادلة"""
    return '=' in question and 'x' in question.lower()

def solve(question):
    """حل المعادلة"""
    try:
        # تنظيف السؤال
        q = question.lower().replace(' ', '').replace('=', '=', 1)
        
        if '=' in q:
            left, right = q.split('=')
        else:
            left, right = q, '0'
        
        left_expr = sympify(left)
        right_expr = sympify(right)
        
        eq = Eq(left_expr, right_expr)
        solutions = solve(eq, x)
        
        if len(solutions) == 0:
            return 'لا يوجد حل'
        elif len(solutions) == 1:
            return f'x = {solutions[0]}'
        else:
            return ', '.join([f'x = {s}' for s in solutions])
    except Exception as e:
        return None
