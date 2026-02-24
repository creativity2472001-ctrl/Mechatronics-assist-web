"""
حل المعادلات الجبرية - نسخة كاملة ومستقلة
"""
import re
from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation
)

x = symbols('x')

# دوال التحويل (مستقلة داخل الملف)
transformations = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor, function_exponentiation)
)

def normalize(question: str) -> str:
    """
    تحويل السؤال إلى صيغة تفهمها SymPy
    """
    expr = question.lower().strip()
    
    # استبدال الرموز الخاصة
    replacements = {
        '×': '*',
        '÷': '/',
        '^': '**',
        '√': 'sqrt',
        'π': 'pi',
        '∞': 'oo',
        '²': '**2',
        '³': '**3'
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    
    # 2x → 2*x
    expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
    # x2 → x**2
    expr = re.sub(r'([a-zA-Z])(\d)', r'\1**\2', expr)
    
    return expr

def parse_expression(expr_str: str):
    """
    تحويل النص إلى تعبير SymPy
    """
    try:
        return parse_expr(
            expr_str,
            transformations=transformations,
            evaluate=True
        )
    except:
        return None

def parse_equation(equation_str: str):
    """
    تحويل نص المعادلة إلى كائن Eq
    """
    if '=' in equation_str:
        left, right = equation_str.split('=', 1)
        left_expr = parse_expression(left.strip())
        right_expr = parse_expression(right.strip())
        if left_expr is not None and right_expr is not None:
            return Eq(left_expr, right_expr)
    else:
        expr = parse_expression(equation_str.strip())
        if expr is not None:
            return Eq(expr, 0)
    return None

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
        
        solutions = solve(eq, x)
        
        if len(solutions) == 0:
            return 'لا يوجد حل'
        elif len(solutions) == 1:
            return f'x = {solutions[0]}'
        else:
            return ', '.join([f'x = {s}' for s in solutions])
    except Exception as e:
        return None
