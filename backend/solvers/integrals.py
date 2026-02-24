"""
حساب التكاملات
مثل: integral of x**2, ∫ x dx
"""
import re
from sympy import symbols, integrate, sympify

x = symbols('x')

def is_integral(question):
    """تحديد إذا كان السؤال تكامل"""
    q = question.lower()
    keywords = ['integral', '∫', 'تكامل']
    return any(kw in q for kw in keywords)

def solve(question):
    """حساب التكامل"""
    try:
        # استخراج الدالة
        expr = question.lower()
        for word in ['integral', '∫', 'تكامل', 'of', 'لـ']:
            expr = expr.replace(word, '')
        
        expr = sympify(expr.strip())
        
        # تكامل محدد؟
        if 'from' in question or 'to' in question:
            numbers = re.findall(r'\d+', question)
            if len(numbers) >= 2:
                lower, upper = float(numbers[0]), float(numbers[1])
                result = integrate(expr, (x, lower, upper))
                return str(result)
        
        # تكامل غير محدد
        result = integrate(expr, x)
        return f'{result} + C'
    except:
        return None
