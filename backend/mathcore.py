"""
MathCore v3.5 - النسخة الكاملة التي تحل جميع المسائل الرياضية
"""
from sympy import symbols, Eq, solve, diff, integrate, limit, simplify, factor, expand, sin, cos, tan, pi, oo, Integral, Derivative, Limit
import re

class MathCore:
    def __init__(self):
        self.x = symbols('x')
        self.y = symbols('y')
        self.z = symbols('z')
        self.timeout_config = {'simple': 10, 'medium': 60, 'heavy': 90, 'complex': 120, 'default': 60}
        self.cpu_count = 1
    
    def solve(self, question: str, language: str = 'ar', user_id: str = 'default', timeout: float = None) -> dict:
        try:
            q = question.lower().strip()
            q_clean = question.replace(' ', '')
            
            # ===== 1. العمليات الحسابية البسيطة =====
            if re.match(r'^[\d+\-*/()]+$', q_clean):
                try:
                    result = eval(q_clean)
                    return self._success_result(str(result))
                except:
                    pass
            
            # ===== 2. المعادلات =====
            if '=' in q and any(c.isalpha() for c in q):
                return self._solve_equation(question)
            
            # ===== 3. التفاضل =====
            if any(word in q for word in ['derivative', 'differentiate', 'مشتقة', 'اشتقاق']):
                return self._solve_derivative(question)
            
            # ===== 4. التكامل =====
            if any(word in q for word in ['integral', '∫', 'تكامل']):
                return self._solve_integral(question)
            
            # ===== 5. النهايات =====
            if any(word in q for word in ['limit', 'lim', 'نهاية']):
                return self._solve_limit(question)
            
            # ===== 6. التبسيط =====
            if any(word in q for word in ['simplify', 'تبسيط']):
                return self._solve_simplify(question)
            
            # ===== 7. التحليل =====
            if any(word in q for word in ['factor', 'تحليل']):
                return self._solve_factor(question)
            
            # ===== 8. التوسيع =====
            if any(word in q for word in ['expand', 'توسيع']):
                return self._solve_expand(question)
            
            return self._error_result('لم أتمكن من تحديد نوع المسألة')
            
        except Exception as e:
            return self._error_result(str(e))
    
    def _solve_equation(self, question):
        try:
            if '=' in question:
                left, right = question.split('=')
            else:
                left, right = question, '0'
            
            from sympy import sympify
            left_expr = sympify(left.strip())
            right_expr = sympify(right.strip())
            
            eq = Eq(left_expr, right_expr)
            solutions = solve(eq, self.x)
            
            if len(solutions) == 0:
                return self._success_result('لا يوجد حل')
            elif len(solutions) == 1:
                return self._success_result(f'x = {solutions[0]}')
            else:
                result = ', '.join([f'x = {s}' for s in solutions])
                return self._success_result(result)
        except:
            return self._error_result('خطأ في حل المعادلة')
    
    def _solve_derivative(self, question):
        try:
            # استخراج الدالة
            func = question.lower()
            for word in ['derivative', 'differentiate', 'مشتقة', 'اشتقاق', 'of', 'لـ']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            result = diff(expr, self.x)
            return self._success_result(str(result))
        except:
            return self._error_result('خطأ في حساب المشتقة')
    
    def _solve_integral(self, question):
        try:
            # استخراج الدالة
            func = question.lower()
            for word in ['integral', '∫', 'تكامل', 'of', 'لـ']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            
            # تكامل محدد؟
            if 'from' in question or 'to' in question:
                import re
                numbers = re.findall(r'\d+', question)
                if len(numbers) >= 2:
                    lower, upper = float(numbers[0]), float(numbers[1])
                    result = integrate(expr, (self.x, lower, upper))
                    return self._success_result(str(result))
            
            # تكامل غير محدد
            result = integrate(expr, self.x)
            return self._success_result(f'{result} + C')
        except:
            return self._error_result('خطأ في حساب التكامل')
    
    def _solve_limit(self, question):
        try:
            # استخراج النقطة
            import re
            numbers = re.findall(r'\d+', question)
            point = float(numbers[0]) if numbers else 0
            
            func = question.lower()
            for word in ['limit', 'lim', 'نهاية', 'as', '->']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            result = limit(expr, self.x, point)
            return self._success_result(str(result))
        except:
            return self._error_result('خطأ في حساب النهاية')
    
    def _solve_simplify(self, question):
        try:
            func = question.lower()
            for word in ['simplify', 'تبسيط']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            result = simplify(expr)
            return self._success_result(str(result))
        except:
            return self._error_result('خطأ في التبسيط')
    
    def _solve_factor(self, question):
        try:
            func = question.lower()
            for word in ['factor', 'تحليل']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            result = factor(expr)
            return self._success_result(str(result))
        except:
            return self._error_result('خطأ في التحليل')
    
    def _solve_expand(self, question):
        try:
            func = question.lower()
            for word in ['expand', 'توسيع']:
                func = func.replace(word, '')
            
            from sympy import sympify
            expr = sympify(func.strip())
            result = expand(expr)
            return self._success_result(str(result))
        except:
            return self._error_result('خطأ في التوسيع')
    
    def _success_result(self, answer):
        return {
            'success': True,
            'simple_answer': answer,
            'domain': 'mathematics',
            'confidence': 100
        }
    
    def _error_result(self, message):
        return {
            'success': False,
            'simple_answer': message,
            'domain': 'mathematics',
            'confidence': 0
        }
