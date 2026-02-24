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
        
        # 2. المعادلات
        if equations.is_equation(question):
            result = equations.solve(question)
            if result:
                return self._success(result)
        
        # 3. التفاضل
        if derivatives.is_derivative(question):
            result = derivatives.solve(question)
            if result:
                return self._success(result)
        
        # 4. التكامل
        if integrals.is_integral(question):
            result = integrals.solve(question)
            if result:
                return self._success(result)
        
        # 5. النهايات
        if limits.is_limit(question):
            result = limits.solve(question)
            if result:
                return self._success(result)
        
        # 6. التبسيط
        if simplify.is_simplify(question):
            result = simplify.solve(question)
            if result:
                return self._success(result)
        
        # 7. التحليل
        if factor.is_factor(question):
            result = factor.solve(question)
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
