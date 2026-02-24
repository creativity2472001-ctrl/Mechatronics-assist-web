"""
MathCore - النسخة النهائية
"""
from solvers import arithmetic, equations, derivatives, integrals, limits, simplify, factor

class MathCore:
    def __init__(self):
        self.timeout_config = {'simple': 10, 'medium': 60, 'heavy': 90, 'complex': 120, 'default': 60}
        self.cpu_count = 1
    
    def solve(self, question: str, language: str = 'ar', user_id: str = 'default', timeout: float = None) -> dict:
        # ترتيب الأولويات (من الأسهل للأصعب)
        solvers = [
            ('arithmetic', arithmetic.is_arithmetic, arithmetic.solve),
            ('equations', equations.is_equation, equations.solve),
            ('derivatives', derivatives.is_derivative, derivatives.solve),
            ('integrals', integrals.is_integral, integrals.solve),
            ('limits', limits.is_limit, limits.solve),
            ('simplify', simplify.is_simplify, simplify.solve),
            ('factor', factor.is_factor, factor.solve)
        ]
        
        for name, detector, solver in solvers:
            if detector(question):
                result = solver(question)
                if result:
                    return self._success(result, name)
        
        return self._error('لم أتمكن من حل المسألة')
    
    def _success(self, answer, domain):
        return {
            'success': True,
            'simple_answer': answer,
            'domain': domain,
            'confidence': 100
        }
    
    def _error(self, message):
        return {
            'success': False,
            'simple_answer': message,
            'domain': 'unknown',
            'confidence': 0
        }
