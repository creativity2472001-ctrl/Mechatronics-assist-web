"""
حل العمليات الحسابية البسيطة
مثل: 1+1, 2*3, (5+3)/2
"""
from .normalizer import normalize, parse_expression

def is_arithmetic(question):
    """تحديد إذا كان السؤال عملية حسابية"""
    try:
        q = normalize(question)
        expr = parse_expression(q)
        return expr is not None and len(expr.free_symbols) == 0
    except:
        return False

def solve(question):
    """حل العملية الحسابية"""
    try:
        q = normalize(question)
        expr = parse_expression(q)
        if expr is None:
            return None
        
        result = expr.evalf()
        if abs(result - round(result)) < 1e-10:
            return str(int(result))
        return str(result)
    except:
        return None
