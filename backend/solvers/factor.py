"""
تحليل العبارات الجبرية
مثل: factor x**2-4
"""
import re
from sympy import factor, sympify

def is_factor(question):
    """تحديد إذا كان السؤال تحليل"""
    q = question.lower()
    keywords = ['factor', 'تحليل']
    return any(kw in q for kw in keywords)

def solve(question):
    """تحليل العبارة"""
    try:
        expr = question.lower()
        for word in ['factor', 'تحليل']:
            expr = expr.replace(word, '')
        
        expr = sympify(expr.strip())
        result = factor(expr)
        return str(result)
    except:
        return None
