"""
MathCore - الملف الرئيسي (نسخة مبسطة للبداية)
"""
from solvers import arithmetic, equations

class MathCore:
    def __init__(self):
        self.timeout_config = {'simple': 10, 'medium': 60, 'heavy': 90, 'complex': 120, 'default': 60}
        self.cpu_count = 1
    
    def solve(self, question: str, language: str = 'ar', user_id: str = 'default', timeout: float = None) -> dict:
        # 1. العمليات الحسابية
        if arithmetic.is_arithmetic(question):
            result = arithmetic.solve(question)
            if result:
                return self._success(result)
        
        # 2. المعادلات
        if equations.is_equation(question):
            result = equations.solve(question)
            if result:
                return self._success(result)
        
        return self._error('لم أتمكن من حل المسألة')
    
    def _success(self, answer):
        return {
            'success': True,
            'simple_answer': answer,
            'domain': 'mathematics',
            'confidence': 100
        }
    
    def _error(self, message):
        return {
            'success': False,
            'simple_answer': message,
            'domain': 'mathematics',
            'confidence': 0
        }
