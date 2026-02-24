"""
حساب التكاملات
مثل: integral of x**2, ∫ x dx
"""
import re
from sympy import symbols, integrate
from .normalizer import normalize, parse_expression

x = symbols('x')

def is_integral(question):
    """تحديد إذا كان السؤال تكامل"""
    q = question.lower()
    keywords = ['integral', '∫', 'تكامل']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب التكامل"""
    try:
        q = normalize(question)
        
        # استخراج الحدود
        numbers = re.findall(r'-?\d+\.?\d*', q)
        has_limits = 'from' in q or 'to' in q or 'من' in q or 'إلى' in q
        
        # إزالة الكلمات المفتاحية
        expr_str = q
        for word in ['integral', '∫', 'تكامل', 'of', 'لـ', 'from', 'to', 'من', 'إلى']:
            expr_str = expr_str.replace(word, '')
        
        expr = parse_expression(expr_str.strip())
        if expr is None:
            return None
        
        # تكامل محدد
        if has_limits and len(numbers) >= 2:
            lower, upper = float(numbers[0]), float(numbers[1])
            result = integrate(expr, (x, lower, upper))
            return str(result)
        
        # تكامل غير محدد
        result = integrate(expr, x)
        return f'{result} + C'
    except:
        return None
