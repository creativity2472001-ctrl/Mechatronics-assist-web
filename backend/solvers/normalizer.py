"""
محول الرموز الرياضية إلى صيغة تفهمها SymPy
نسخة احترافية - مع تحسينات نهائية
"""
import re
import logging
from sympy import Eq
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation
)

# إعداد التسجيل
logger = logging.getLogger(__name__)

# التحويلات الذكية من SymPy
transformations = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor, function_exponentiation)
)

def normalize(question: str) -> str:
    """
    تحويل الرموز الخاصة فقط (ما يعجز عنه SymPy)
    """
    expr = question.lower().strip()
    
    # استبدال الرموز الخاصة
    replacements = {
        '×': '*',
        '÷': '/',
        'π': 'pi',
        '∞': 'oo',
        'θ': 'theta',
        'λ': 'lam',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        '²': '**2',
        '³': '**3'
    }
    for old, new in replacements.items():
        expr = expr.replace(old, new)
    
    # معالجة ln (فقط الكلمة الكاملة)
    expr = re.sub(r'\bln\b', 'log', expr)
    
    # معالجة الجذر: sqrtx → sqrt(x)
    expr = re.sub(r'sqrt([a-zA-Z0-9]+)', r'sqrt(\1)', expr)
    
    return expr

def parse_expression(expr_str: str):
    """
    تحويل النص إلى تعبير SymPy باستخدام التحويلات الذكية
    """
    try:
        return parse_expr(
            expr_str,
            transformations=transformations,
            evaluate=True
        )
    except Exception as e:
        logger.error(f"Parse error: {e}")
        return None

def parse_equation(equation_str: str):
    """
    تحويل معادلة إلى كائن Eq مباشرة
    مثال: "x+5=10" → Eq(x+5, 10)
    """
    if '=' in equation_str:
        # تقسيم عند أول = فقط
        left, right = equation_str.split('=', 1)
        left = left.strip()
        right = right.strip()
        
        left_expr = parse_expression(left)
        right_expr = parse_expression(right)
        
        if left_expr is not None and right_expr is not None:
            return Eq(left_expr, right_expr)
    else:
        expr = parse_expression(equation_str.strip())
        if expr is not None:
            return Eq(expr, 0)
    
    return None
