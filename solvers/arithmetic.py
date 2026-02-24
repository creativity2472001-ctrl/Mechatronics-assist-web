"""
حل العمليات الحسابية البسيطة
"""
import re

def is_arithmetic(question):
    q = question.replace(' ', '').replace('=', '')
    return bool(re.match(r'^[\d+\-*/()]+$', q))

def solve(question):
    q = question.replace(' ', '').replace('=', '')
    try:
        result = eval(q)
        return str(result)
    except:
        return None
