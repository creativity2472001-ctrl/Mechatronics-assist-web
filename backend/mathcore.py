"""
MathCore v3.5 - النسخة المبسطة التي تحل 1+1
"""
from sympy import symbols, Eq, solve, diff, integrate, limit, simplify, pi, oo
import re

class MathCore:
    def __init__(self):
        self.x = symbols('x')
        self.timeout_config = {'simple': 10, 'medium': 60, 'heavy': 90, 'complex': 120, 'default': 60}
        self.cpu_count = 1
    
    def solve(self, question: str, language: str = 'ar', user_id: str = 'default', timeout: float = None) -> dict:
        # ✅ حل فوري للأرقام
        q = question.replace(' ', '').replace('=', '')
        if re.match(r'^[\d+\-*/()]+$', q):
            try:
                result = eval(q)
                return {
                    'success': True,
                    'simple_answer': str(result),
                    'domain': 'mathematics',
                    'confidence': 100
                }
            except:
                pass
        
        return {
            'success': False,
            'simple_answer': 'لم أتمكن من حل المسألة',
            'domain': 'mathematics',
            'confidence': 0
        }
