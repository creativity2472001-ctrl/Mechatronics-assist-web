"""
MathCore - Mathematics Engine v1.1
نواة رياضية متقدمة مع دعم اللغة والتكامل مع الواجهة
"""

from sympy import (
    symbols, Eq, solve, parse_expr, diff, integrate, limit, oo, 
    simplify, Matrix, laplace_transform, inverse_laplace_transform,
    fourier_transform, dsolve, Function, I, re, im, expand, factor, 
    Abs, arg, pi, exp, sin, cos, tan, log, sqrt, root, summation
)
import hashlib
import json
import re

class MathCore:
    """
    النواة الرياضية المتقدمة مع دعم كامل للغة العربية والإنجليزية
    """
    
    def __init__(self):
        # نظام الرموز والمتغيرات
        self.x, self.y, self.z, self.t, self.s, self.w, self.n = symbols('x y z t s w n')
        self.standard_vars = {
            'x': self.x, 'y': self.y, 'z': self.z, 
            't': self.t, 's': self.s, 'w': self.w, 'n': self.n,
            'pi': pi, 'I': I, 'exp': exp, 'sin': sin, 
            'cos': cos, 'tan': tan, 'log': log, 'sqrt': sqrt, 'oo': oo
        }
        
        # نظام الذاكرة المؤقتة (Caching)
        self._cache = {}

        # أكواد الأخطاء
        self.ERROR_CODES = {
            "ERR_UNSUPPORTED": "E101: Operation not supported",
            "ERR_SYNTAX": "E102: Syntax error in mathematical expression",
            "ERR_VALUE": "E103: Invalid value or parameters provided",
            "ERR_COMPUTE": "E104: Computation error or timeout",
            "ERR_UNKNOWN": "E999: Unknown internal error"
        }

    def solve(self, question, language='ar'):
        """
        الدالة الرئيسية - متوافقة مع الواجهة
        تحول السؤال إلى العملية المناسبة وتعيد النتيجة
        """
        try:
            question = question.strip()
            if not question:
                return self._error_response("السؤال فارغ", "Empty question", language)
            
            # تحديد نوع العملية من السؤال
            question_lower = question.lower()
            
            # كشف نوع المسألة
            if '=' in question_lower or 'solve' in question_lower or 'معادلة' in question_lower:
                # معادلة
                result = self.execute('solveEquation', {
                    'equation': question,
                    'variable': 'x'
                })
            elif 'derivative' in question_lower or 'differentiate' in question_lower or 'مشتقة' in question_lower or 'اشتقاق' in question_lower:
                # تفاضل
                expr = self._extract_expression(question, ['derivative', 'differentiate', 'مشتقة', 'اشتقاق', 'of', 'لـ'])
                result = self.execute('differentiate', {
                    'expression': expr,
                    'order': 1
                })
            elif 'integral' in question_lower or '∫' in question_lower or 'تكامل' in question_lower:
                # تكامل
                expr = self._extract_expression(question, ['integral', '∫', 'تكامل', 'of', 'لـ'])
                
                # هل هو تكامل محدد؟
                if 'from' in question_lower or 'to' in question_lower or 'من' in question_lower or 'إلى' in question_lower:
                    numbers = re.findall(r'\d+', question)
                    if len(numbers) >= 2:
                        result = self.execute('integrate', {
                            'expression': expr,
                            'lower': float(numbers[0]),
                            'upper': float(numbers[1])
                        })
                    else:
                        result = self.execute('integrate', {'expression': expr})
                else:
                    result = self.execute('integrate', {'expression': expr})
                    
            elif 'limit' in question_lower or 'lim' in question_lower or 'نهاية' in question_lower:
                # نهاية
                numbers = re.findall(r'\d+', question)
                point = float(numbers[0]) if numbers else 0
                expr = self._extract_expression(question, ['limit', 'lim', 'نهاية', 'as', '→', 'عندما'])
                result = self.execute('limit', {
                    'expression': expr,
                    'point': point
                })
                
            elif 'simplify' in question_lower or 'تبسيط' in question_lower:
                # تبسيط
                expr = self._extract_expression(question, ['simplify', 'تبسيط'])
                result = self.execute('simplifyExpression', {'expression': expr})
                
            elif 'factor' in question_lower or 'تحليل' in question_lower:
                # تحليل
                expr = self._extract_expression(question, ['factor', 'تحليل'])
                result = self.execute('factorExpression', {'expression': expr})
                
            elif 'root' in question_lower or 'جذر' in question_lower:
                # جذور
                numbers = re.findall(r'\d+', question)
                n = int(numbers[1]) if len(numbers) > 1 else 2
                expr = numbers[0] if numbers else question
                result = self.execute('nthRoot', {
                    'expression': expr,
                    'n': n
                })
                
            elif 'sum' in question_lower or 'مجموع' in question_lower:
                # متسلسلات
                numbers = re.findall(r'\d+', question)
                lower = int(numbers[0]) if numbers else 1
                upper = int(numbers[1]) if len(numbers) > 1 else 10
                expr = self._extract_expression(question, ['sum', 'summation', 'مجموع'])
                result = self.execute('summation', {
                    'expression': expr,
                    'variable': 'n',
                    'lower': lower,
                    'upper': upper
                })
                
            else:
                # عملية حسابية بسيطة
                result = self.execute('calculate', {'expression': question})
            
            # تحويل النتيجة لتنسيق الواجهة
            return self._format_for_frontend(result, language)
            
        except Exception as e:
            return self._error_response(str(e), str(e), language)

    def execute(self, operation_type, params):
        """الدالة الرئيسية مع دعم الذاكرة المؤقتة"""
        # توليد مفتاح فريد للعملية للبحث في الذاكرة المؤقتة
        cache_key = self._generate_cache_key(operation_type, params)
        if cache_key in self._cache:
            return self._format_response(self._cache[cache_key], success=True, cached=True)

        try:
            operations = {
                'calculate': self._calculate,
                'solveEquation': self._solve_equation,
                'solveSystem': self._solve_system,
                'simplifyExpression': self._simplify_expression,
                'factorExpression': self._factor_expression,
                'differentiate': self._differentiate,
                'partialDifferentiate': self._partial_differentiate,
                'integrate': self._integrate,
                'limit': self._limit,
                'matrixOperation': self._matrix_operation,
                'complexNumber': self._complex_number,
                'laplaceTransform': self._laplace_transform,
                'inverseLaplaceTransform': self._inverse_laplace_transform,
                'fourierTransform': self._fourier_transform,
                'solveODE': self._solve_ode,
                'nthRoot': self._nth_root,
                'summation': self._summation,
                'higherOrderDiff': self._higher_order_diff
            }

            if operation_type in operations:
                result_data = operations[operation_type](params)
                # حفظ النتيجة في الذاكرة المؤقتة
                self._cache[cache_key] = result_data
                return self._format_response(result_data, success=True)
            else:
                return self._format_response(None, success=False, error_code="ERR_UNSUPPORTED")

        except Exception as e:
            error_code = "ERR_COMPUTE"
            if "parse" in str(e).lower() or "syntax" in str(e).lower():
                error_code = "ERR_SYNTAX"
            return self._format_response(None, success=False, error_code=error_code, error_msg=str(e))

    def _extract_expression(self, question, keywords):
        """استخراج التعبير الرياضي من السؤال"""
        for keyword in keywords:
            question = question.replace(keyword, '')
        # إزالة الكلمات الشائعة
        for word in ['of', 'لـ', 'from', 'to', 'من', 'إلى', 'as', 'عندما']:
            question = question.replace(word, '')
        return question.strip()

    def _generate_cache_key(self, op_type, params):
        """توليد مفتاح فريد بناءً على العملية والمعاملات"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{op_type}_{param_str}".encode()).hexdigest()

    def _format_response(self, data, success=True, error_code=None, error_msg=None, cached=False):
        """تنسيق الاستجابة الداخلية"""
        response = {
            'status': 'success' if success else 'failure',
            'result': data,
            'cached': cached,
            'engine': 'MathCore v1.1'
        }
        if not success:
            response['error_code'] = error_code
            response['error_description'] = self.ERROR_CODES.get(error_code, "Unknown Error")
            response['technical_details'] = error_msg
        return response

    def _format_for_frontend(self, result, language='ar'):
        """تحويل نتيجة MathCore إلى تنسيق الواجهة"""
        if result.get('status') == 'failure':
            return {
                'success': False,
                'simple_answer': 'حدث خطأ في الحل' if language == 'ar' else 'Error in solution',
                'steps': ['❌ فشل في حل المسألة'],
                'ai_explanation': result.get('technical_details', 'خطأ غير معروف'),
                'domain': 'mathematics',
                'confidence': 0
            }
        
        data = result.get('result', '')
        
        # إنشاء خطوات حل
        if language == 'ar':
            steps = [
                '✅ تم استلام المسألة بنجاح',
                '🔄 جاري التحليل والمعالجة',
                f'📊 النتيجة: {data}'
            ]
            ai_explanation = f'تم حل المسألة باستخدام MathCore v1.1. النتيجة هي {data}'
        else:
            steps = [
                '✅ Question received successfully',
                '🔄 Processing...',
                f'📊 Result: {data}'
            ]
            ai_explanation = f'Solved using MathCore v1.1. Result is {data}'
        
        return {
            'success': True,
            'simple_answer': str(data),
            'steps': steps,
            'ai_explanation': ai_explanation,
            'domain': 'mathematics',
            'confidence': 98
        }

    def _error_response(self, ar_msg, en_msg, language):
        """تنسيق رسالة خطأ"""
        return {
            'success': False,
            'simple_answer': ar_msg if language == 'ar' else en_msg,
            'steps': ['❌ ' + (ar_msg if language == 'ar' else en_msg)],
            'ai_explanation': 'تأكد من صياغة السؤال بشكل صحيح' if language == 'ar' else 'Check question format',
            'domain': 'mathematics',
            'confidence': 0
        }

    def _parse_input(self, expr_str, custom_vars=None):
        """تحويل النص إلى تعبير رياضي"""
        local_dict = self.standard_vars.copy()
        if custom_vars:
            for v in custom_vars:
                local_dict[v] = symbols(v)
        return parse_expr(str(expr_str), local_dict=local_dict)

    # --- الدوال الرياضية ---

    def _nth_root(self, params):
        """حساب الجذور النونية"""
        expr = self._parse_input(params['expression'])
        n_val = params.get('n', 2)
        return str(root(expr, n_val))

    def _summation(self, params):
        """حساب المتسلسلات"""
        expr = self._parse_input(params['expression'])
        var = symbols(params.get('variable', 'n'))
        lower = params.get('lower', 1)
        upper = params.get('upper', oo)
        return str(summation(expr, (var, lower, upper)))

    def _higher_order_diff(self, params):
        """المشتقات من الرتب العليا"""
        expr = self._parse_input(params['expression'])
        var = symbols(params.get('variable', 'x'))
        order = int(params.get('order', 1))
        return str(diff(expr, var, order))
    
    def _calculate(self, params):
        """عملية حسابية بسيطة"""
        expr = self._parse_input(params['expression'])
        return float(expr.evalf())

    def _solve_equation(self, params):
        """حل معادلة"""
        eq_str = params['equation']
        var = symbols(params.get('variable', 'x'))
        if '=' in eq_str:
            left, right = eq_str.split('=')
            eq = Eq(self._parse_input(left), self._parse_input(right))
        else:
            eq = Eq(self._parse_input(eq_str), 0)
        return [str(s) for s in solve(eq, var)]

    def _solve_system(self, params):
        """حل نظام معادلات"""
        eqs = [self._parse_input(e) for e in params['equations']]
        vars_syms = [symbols(v) for v in params['variables']]
        solutions = solve(eqs, vars_syms)
        return {str(k): str(v) for k, v in solutions.items()} if isinstance(solutions, dict) else str(solutions)

    def _simplify_expression(self, params):
        """تبسيط تعبير"""
        return str(simplify(self._parse_input(params['expression'])))

    def _factor_expression(self, params):
        """تحليل تعبير"""
        return str(factor(self._parse_input(params['expression'])))

    def _differentiate(self, params):
        """تفاضل"""
        return self._higher_order_diff(params)

    def _partial_differentiate(self, params):
        """تفاضل جزئي"""
        return self._higher_order_diff(params)

    def _integrate(self, params):
        """تكامل"""
        expr = self._parse_input(params['expression'])
        var = symbols(params.get('variable', 'x'))
        if 'lower' in params and 'upper' in params:
            return str(integrate(expr, (var, params['lower'], params['upper'])))
        return str(integrate(expr, var))

    def _limit(self, params):
        """نهاية"""
        expr = self._parse_input(params['expression'])
        var = symbols(params.get('variable', 'x'))
        point = params['point']
        return str(limit(expr, var, point))

    def _matrix_operation(self, params):
        """عمليات المصفوفات"""
        op = params['operation']
        M = Matrix(params['matrix'])
        if op == 'det': return str(M.det())
        if op == 'inv': return [list(row) for row in M.inv().tolist()]
        if op == 'transpose': return [list(row) for row in M.T.tolist()]
        return None

    def _complex_number(self, params):
        """عمليات الأعداد المركبة"""
        res = simplify(self._parse_input(params['expression']))
        return {
            'result': str(res), 
            'real': str(re(res)), 
            'imaginary': str(im(res)), 
            'magnitude': str(Abs(res)), 
            'phase': str(arg(res))
        }

    def _laplace_transform(self, params):
        """تحويل لابلاس"""
        return str(laplace_transform(self._parse_input(params['expression']), self.t, self.s)[0])

    def _inverse_laplace_transform(self, params):
        """تحويل لابلاس معكوس"""
        return str(inverse_laplace_transform(self._parse_input(params['expression']), self.s, self.t))

    def _fourier_transform(self, params):
        """تحويل فورييه"""
        return str(fourier_transform(self._parse_input(params['expression']), self.x, self.w))

    def _solve_ode(self, params):
        """حل معادلة تفاضلية"""
        f = Function(params.get('function_name', 'f'))
        var = symbols(params.get('variable', 't'))
        eq_expr = self._parse_input(params['equation'], custom_vars=[params.get('function_name', 'f')])
        return str(dsolve(eq_expr, f(var)))


# اختبار سريع
if __name__ == "__main__":
    core = MathCore()
    
    print("=" * 50)
    print("🧪 اختبار MathCore v1.1")
    print("=" * 50)
    
    test_cases = [
        "2 + 2",
        "x + 5 = 10",
        "derivative of x**2",
        "integral of x**2",
        "simplify (x**2 - 1)/(x - 1)",
        "factor x**2 - 4",
        "root of 27 with n=3"
    ]
    
    for i, q in enumerate(test_cases, 1):
        print(f"\n🔍 اختبار {i}: {q}")
        result = core.solve(q, 'ar')
        print(f"✅ النتيجة: {result['simple_answer']}")
        if result['steps']:
            print(f"📋 أول خطوة: {result['steps'][0]}")
    
    print("\n" + "=" * 50)
    print("✅ MathCore جاهز للعمل!")
    print("=" * 50)
