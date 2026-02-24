"""
حساب النهايات
مثل: limit of sin(x)/x as x->0
"""
import re
from sympy import symbols, limit, oo
from .normalizer import normalize, parse_expression

x = symbols('x')

def is_limit(question):
    """تحديد إذا كان السؤال نهاية"""
    q = question.lower()
    keywords = ['limit', 'lim', 'نهاية']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب النهاية"""
    try:
        q = normalize(question)
        
        # استخراج النقطة
        numbers = re.findall(r'-?\d+\.?\d*', q)
        point = float(numbers[0]) if numbers else 0
        
        # استخراج الدالة
        expr_str = q
        for word in ['limit', 'lim', 'نهاية', 'as', '->']:
            expr_str = expr_str.replace(word, '')
        
        expr = parse_expression(expr_str.strip())
        if expr is None:
            return None
        
        result = limit(expr, x, point)
        return str(result)
    except:
        return None
