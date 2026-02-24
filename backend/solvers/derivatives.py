"""
حساب المشتقات (التفاضل)
مثل: derivative of x**3, differentiate sin(x)
"""
import re
from sympy import symbols, diff, sympify, sin, cos, tan

x = symbols('x')

def is_derivative(question):
    """تحديد إذا كان السؤال تفاضل"""
    q = question.lower()
    keywords = ['derivative', 'differentiate', 'مشتقة', 'اشتقاق']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب المشتقة"""
    try:
        # استخراج الدالة
        expr = question.lower()
        for word in ['derivative', 'differentiate', 'مشتقة', 'اشتقاق', 'of', 'لـ']:
            expr = expr.replace(word, '')
        
        expr = sympify(expr.strip())
        result = diff(expr, x)
        return str(result)
    except:
        return None
