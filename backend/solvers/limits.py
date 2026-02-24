"""
حساب النهايات
مثل: limit of sin(x)/x as x->0
"""
import re
from sympy import symbols, limit, sympify, oo

x = symbols('x')

def is_limit(question):
    """تحديد إذا كان السؤال نهاية"""
    q = question.lower()
    keywords = ['limit', 'lim', 'نهاية']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب النهاية"""
    try:
        # استخراج النقطة
        numbers = re.findall(r'\d+', question)
        point = float(numbers[0]) if numbers else 0
        
        # استخراج الدالة
        expr = question.lower()
        for word in ['limit', 'lim', 'نهاية', 'as', '->']:
            expr = expr.replace(word, '')
        
        expr = sympify(expr.strip())
        result = limit(expr, x, point)
        return str(result)
    except:
        return None
