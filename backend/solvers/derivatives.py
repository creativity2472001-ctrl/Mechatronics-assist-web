"""
حساب المشتقات (التفاضل)
مثل: derivative of x**3, differentiate sin(x)
"""
import re
from sympy import symbols, diff
from normalizer import normalize, parse_expression

x = symbols('x')

def is_derivative(question):
    """تحديد إذا كان السؤال تفاضل"""
    q = question.lower()
    keywords = ['derivative', 'differentiate', 'مشتقة', 'اشتقاق']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب المشتقة"""
    try:
        q = normalize(question)
        
        # استخراج الرتبة (مشتقة ثانية، ثالثة)
        order = 1
        if 'second' in q or 'ثانية' in q:
            order = 2
        elif 'third' in q or 'ثالثة' in q:
            order = 3
        
        # إزالة الكلمات المفتاحية
        for word in ['derivative', 'differentiate', 'مشتقة', 'اشتقاق', 'of', 'لـ', 'second', 'third', 'ثانية', 'ثالثة']:
            q = q.replace(word, '')
        
        expr = parse_expression(q.strip())
        if expr is None:
            return None
        
        result = diff(expr, x, order)
        return str(result)
    except:
        return None
