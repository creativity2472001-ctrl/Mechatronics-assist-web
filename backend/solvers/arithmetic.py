"""
حل العمليات الحسابية البسيطة
مثل: 1+1, 2*3, (5+3)/2
"""
import re

def is_arithmetic(question):
    """تحديد إذا كان السؤال عملية حسابية"""
    q = question.replace(' ', '').replace('=', '')
    return bool(re.match(r'^[\d+\-*/()]+$', q))

def solve(question):
    """حل العملية الحسابية"""
    q = question.replace(' ', '').replace('=', '')
    try:
        result = eval(q)
        return str(result)
    except:
        return None
