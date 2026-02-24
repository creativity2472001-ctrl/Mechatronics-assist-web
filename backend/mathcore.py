"""
MathCore - الملف الرئيسي
يستدعي الحلول من مجلد solvers
"""
from solvers import arithmetic, equations, derivatives, integrals, limits, simplify, factor

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
        
        # 2. المعادلات (سنضيفها لاحقًا)
        # 3. التفاضل (سنضيفه لاحقًا)
        # 4. التكامل (سنضيفه لاحقًا)
        # 5. النهايات (سنضيفها لاحقًا)
        # 6. التبسيط (سنضيفه لاحقًا)
        # 7. التحليل (سنضيفه لاحقًا)
        
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
