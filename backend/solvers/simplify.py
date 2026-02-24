"""
تبسيط التعبيرات الجبرية
مثل: simplify (x**2-1)/(x-1)
"""
import re
from sympy import simplify, sympify

def is_simplify(question):
    """تحديد إذا كان السؤال تبسيط"""
    q = question.lower()
    keywords = ['simplify', 'تبسيط']
    return any(kw in q for kw in keywords)

def solve(question):
    """تبسيط التعبير"""
    try:
        expr = question.lower()
        for word in ['simplify', 'تبسيط']:
            expr = expr.replace(word, '')
        
        expr = sympify(expr.strip())
        result = simplify(expr)
        return str(result)
    except:
        return None
