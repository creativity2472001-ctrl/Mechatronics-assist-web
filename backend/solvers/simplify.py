"""
تبسيط التعبيرات الجبرية
مثل: simplify (x**2-1)/(x-1)
"""
import re
from sympy import simplify
from .normalizer import normalize, parse_expression

def is_simplify(question):
    """تحديد إذا كان السؤال تبسيط"""
    q = question.lower()
    keywords = ['simplify', 'تبسيط']
    return any(kw in q for kw in keywords)

def solve(question):
    """تبسيط التعبير"""
    try:
        q = normalize(question)
        
        expr_str = q
        for word in ['simplify', 'تبسيط']:
            expr_str = expr_str.replace(word, '')
        
        expr = parse_expression(expr_str.strip())
        if expr is None:
            return None
        
        result = simplify(expr)
        return str(result)
    except:
        return None
