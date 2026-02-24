"""
تحليل العبارات الجبرية
مثل: factor x**2-4
"""
import re
from sympy import factor
from .normalizer import normalize, parse_expression

def is_factor(question):
    """تحديد إذا كان السؤال تحليل"""
    q = question.lower()
    keywords = ['factor', 'تحليل']
    return any(kw in q for kw in keywords)

def solve(question):
    """تحليل العبارة"""
    try:
        q = normalize(question)
        
        expr_str = q
        for word in ['factor', 'تحليل']:
            expr_str = expr_str.replace(word, '')
        
        expr = parse_expression(expr_str.strip())
        if expr is None:
            return None
        
        result = factor(expr)
        return str(result)
    except:
        return None
