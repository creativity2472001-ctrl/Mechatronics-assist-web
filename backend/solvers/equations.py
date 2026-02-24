"""
حل المعادلات الجبرية
مثل: x+5=10, 2x-4=8, x²-5x+6=0
"""
from normalizer import normalize, parse_equation
from sympy import solve

def is_equation(question):
    """تحديد إذا كان السؤال معادلة"""
    return '=' in question

def solve(question):
    """حل المعادلة"""
    try:
        q = normalize(question)
        eq = parse_equation(q)
        
        if eq is None:
            return None
        
        solutions = solve(eq)
        
        if len(solutions) == 0:
            return 'لا يوجد حل'
        elif len(solutions) == 1:
            return f'x = {solutions[0]}'
        else:
            return ', '.join([f'x = {s}' for s in solutions])
    except Exception as e:
        return None
