# math_pipe_final.py
from sympy import *
from sympy.parsing.sympy_parser import (
    standard_transformations, 
    implicit_multiplication, 
    convert_xor,
    implicit_application,
    function_exponentiation,
    split_symbols
)
from typing import Any, Callable, List, Dict, Optional, Union
import traceback

# التحويلات الآمنة لـ SymPy
SAFE_TRANSFORMATIONS = (
    standard_transformations + 
    (implicit_multiplication, convert_xor, implicit_application)
)

class MathPipe:
    """
    نظام الأنابيب الرياضي - يضمن التنفيذ الصحيح خطوة بخطوة
    """
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.steps = []
        self.errors = []
        self.warnings = []
        self.metadata = {}
    
    def then(self, func: Callable, *args, **kwargs) -> 'MathPipe':
        """إضافة مرحلة للأنبوب مع التحقق من الأخطاء"""
        try:
            step_name = getattr(func, '__name__', str(func))
            
            if self.value is not None:
                self.value = func(self.value, *args, **kwargs)
            else:
                self.value = func(*args, **kwargs)
            
            self.steps.append({
                'name': step_name,
                'args': str(args) if args else '',
                'kwargs': str(kwargs) if kwargs else '',
                'success': True,
                'value_preview': str(self.value)[:100] if self.value is not None else 'None'
            })
        except Exception as e:
            self.errors.append({
                'step': step_name,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            self.steps.append({
                'name': step_name,
                'success': False,
                'error': str(e)
            })
        return self
    
    def validate(self, check_func: Callable, error_msg: str = None) -> 'MathPipe':
        """التحقق من صحة القيمة في مرحلة ما"""
        try:
            if not check_func(self.value):
                error = error_msg or f"فشل التحقق في المرحلة {len(self.steps)}"
                self.errors.append({'type': 'validation', 'error': error})
        except Exception as e:
            self.errors.append({'type': 'validation', 'error': str(e)})
        return self
    
    def if_error(self, fallback_func: Callable) -> 'MathPipe':
        """تنفيذ دالة بديلة في حالة الخطأ"""
        if self.errors:
            try:
                self.value = fallback_func(self.value)
                self.errors = []  # مسح الأخطاء إذا نجح البديل
                self.warnings.append("تم استخدام حل بديل بسبب خطأ سابق")
            except:
                pass
        return self
    
    def get_result(self) -> Dict[str, Any]:
        """الحصول على النتيجة النهائية مع كامل التفاصيل"""
        return {
            'value': self.value,
            'steps': self.steps,
            'errors': self.errors,
            'warnings': self.warnings,
            'success': len(self.errors) == 0,
            'metadata': self.metadata
        }
    
    def reset(self) -> 'MathPipe':
        """إعادة تعيين الأنبوب"""
        self.value = None
        self.steps = []
        self.errors = []
        self.warnings = []
        return self


class EngineeringPipes:
    """
    أنابيب جاهزة للمسائل الهندسية والرياضية
    النسخة النهائية مع جميع التحسينات
    """
    
    def __init__(self):
        # تعريف المتغيرات الأساسية
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.f, self.g = symbols('f g', cls=Function)
        self.C = symbols('C')  # ثابت التكامل
        
        # قاموس محلي للتحليل
        self.local_dict = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
            'f': self.f, 'g': self.g,
            'C': self.C,
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'exp': exp, 'log': log, 'ln': ln,
            'sqrt': sqrt, 'pi': pi, 'E': E
        }
    
    # ============================================================
    # 🔧 دوال مساعدة للتحليل (محسنة)
    # ============================================================
    
    def _parse_expression(self, expr_str: str):
        """تحويل النص إلى تعبير SymPy مع تصحيح الأخطاء الشائعة"""
        if not expr_str or not isinstance(expr_str, str):
            return None
        
        try:
            # تصحيح الأخطاء الشائعة
            corrected = expr_str.replace('^', '**')
            
            # تحويل = إلى == للمعادلات
            if '=' in corrected and '==' not in corrected:
                parts = corrected.split('=')
                if len(parts) == 2:
                    corrected = f"Eq({parts[0]}, {parts[1]})"
            
            # محاولة التحليل مع التحويلات الآمنة والمتغيرات المحلية
            return parse_expr(
                corrected, 
                transformations=SAFE_TRANSFORMATIONS,
                local_dict=self.local_dict
            )
        except Exception as e:
            print(f"⚠️ فشل تحليل التعبير: {e}")
            
            # محاولة أخيرة مع تجاهل الأخطاء واستخدام المتغيرات المحلية
            try:
                return parse_expr(
                    expr_str, 
                    transformations=SAFE_TRANSFORMATIONS, 
                    evaluate=False,
                    local_dict=self.local_dict
                )
            except:
                return None
    
    def _ensure_equation(self, expr):
        """التأكد أن التعبير هو معادلة"""
        if expr is None:
            return None
        if not isinstance(expr, Eq):
            # إذا لم يكن معادلة، افترض أنه يساوي 0
            return Eq(expr, 0)
        return expr
    
    def _format_solutions(self, solutions):
        """تنسيق الحلول بشكل مقروء وموحد"""
        if not solutions:
            return "لا يوجد حلول"
        
        # إذا كان الحل على شكل قاموس (للمعادلات متعددة المتغيرات)
        if isinstance(solutions, list) and all(isinstance(s, dict) for s in solutions):
            formatted = []
            for sol_dict in solutions:
                formatted.append(", ".join([f"{k} = {v}" for k, v in sol_dict.items()]))
            return formatted
        
        # إذا كان هناك حل واحد
        if isinstance(solutions, list) and len(solutions) == 1:
            return [f"x = {solutions[0]}"]
        
        # حلول متعددة
        if isinstance(solutions, list):
            return [f"x = {s}" for s in solutions]
        
        # حالة أخرى
        return str(solutions)
    
    def _safe_simplify(self, expr):
        """تبسيط آمن مع التعامل مع النصوص"""
        if expr is None:
            return None
        if isinstance(expr, str):
            return expr
        try:
            return simplify(expr)
        except:
            return expr
    
    # ============================================================
    # 📁 أنابيب الجبر والمعادلات
    # ============================================================
    
    def solve_pipe(self, expression: str, variable: str = 'x') -> Dict:
        """أنبوب حل المعادلات"""
        
        var = symbols(variable)
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'solve'
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(self._ensure_equation)
            .then(solve, var, dict=True)
            .then(self._format_solutions)
            .then(self._safe_simplify)
            .get_result())
        
        return result
    
    # ============================================================
    # 📁 أنابيب التفاضل والتكامل
    # ============================================================
    
    def derivative_pipe(self, expression: str, variable: str = 'x', order: int = 1) -> Dict:
        """أنبوب الاشتقاق"""
        
        var = symbols(variable)
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'derivative'
        pipe.metadata['order'] = order
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(diff, var, order)
            .then(self._safe_simplify)
            .get_result())
        
        return result
    
    def integral_pipe(self, expression: str, variable: str = 'x', 
                      lower: str = None, upper: str = None) -> Dict:
        """أنبوب التكامل"""
        
        var = symbols(variable)
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'integral'
        
        # تحليل التعبير أولاً
        pipe.then(self._parse_expression, expression)
        
        # تحليل الحدود إذا وجدت
        lower_expr = None
        upper_expr = None
        
        if lower:
            lower_expr = self._parse_expression(lower)
            pipe.metadata['lower'] = lower
        if upper:
            upper_expr = self._parse_expression(upper)
            pipe.metadata['upper'] = upper
        
        # تنفيذ التكامل حسب نوعه
        if lower_expr is not None and upper_expr is not None:
            # تكامل محدد
            pipe.then(integrate, (var, lower_expr, upper_expr))
            pipe.metadata['integral_type'] = 'definite'
        else:
            # تكامل غير محدد - نحتفظ بالتعبير كـ SymPy
            pipe.then(integrate, var)
            pipe.metadata['integral_type'] = 'indefinite'
        
        # تبسيط النتيجة (التي لا تزال SymPy)
        pipe.then(self._safe_simplify)
        
        result = pipe.get_result()
        
        # إضافة + C بشكل منفصل للتكامل غير المحدد (فقط للعرض)
        if result['success'] and pipe.metadata.get('integral_type') == 'indefinite':
            # نضيف خاصية display للعرض، ولكن نبقي value كـ SymPy للعمليات اللاحقة
            result['display'] = f"{result['value']} + C"
        
        return result
    
    # ============================================================
    # 📁 أنابيب النهايات
    # ============================================================
    
    def limit_pipe(self, expression: str, variable: str = 'x', point: str = '0') -> Dict:
        """أنبوب النهايات"""
        
        var = symbols(variable)
        point_expr = self._parse_expression(point)
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'limit'
        pipe.metadata['point'] = point
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(limit, var, point_expr)
            .then(self._safe_simplify)
            .get_result())
        
        return result
    
    # ============================================================
    # 📁 أنابيب المصفوفات
    # ============================================================
    
    def matrix_pipe(self, matrix_data: List[List[float]], operation: str = None) -> Dict:
        """أنبوب عمليات المصفوفات"""
        
        pipe = MathPipe()
        pipe.metadata['original_matrix'] = matrix_data
        pipe.metadata['problem_type'] = 'matrix'
        pipe.metadata['operation'] = operation or 'none'
        
        # التحقق من صحة المصفوفة
        pipe.validate(lambda d: d and len(d) > 0, "مصفوفة فارغة")
        pipe.validate(lambda d: all(len(row) == len(d[0]) for row in d), "أبعاد المصفوفة غير متسقة")
        
        # إنشاء المصفوفة
        pipe.then(lambda d: Matrix(d), matrix_data)
        pipe.metadata['matrix_shape'] = lambda: f"{pipe.value.rows}×{pipe.value.cols}" if pipe.value else "unknown"
        
        # إذا لم يتم تحديد عملية، نضع علامة خاصة
        if not operation or operation == 'none':
            pipe.metadata['note'] = "عرض المصفوفة فقط (لم يتم تحديد عملية)"
            result = pipe.get_result()
            result['value_preview'] = f"مصفوفة {pipe.metadata.get('matrix_shape', '')}"
            return result
        
        # التحقق من إمكانية تنفيذ العملية
        if operation in ['inverse', 'inv']:
            pipe.validate(lambda m: m.det() != 0, "المصفوفة غير قابلة للعكس (المحدد = 0)")
        elif operation in ['determinant', 'det']:
            pipe.validate(lambda m: m.is_square, "المحدد يحتاج مصفوفة مربعة")
        elif operation in ['eigenvalues']:
            pipe.validate(lambda m: m.is_square, "القيم الذاتية تحتاج مصفوفة مربعة")
        
        # تنفيذ العملية المطلوبة
        if operation in ['determinant', 'det']:
            pipe.then(lambda m: m.det())
        elif operation in ['inverse', 'inv']:
            pipe.then(lambda m: m.inv())
        elif operation in ['transpose', 'T']:
            pipe.then(lambda m: m.T)
        elif operation in ['eigenvalues']:
            pipe.then(lambda m: m.eigenvals())
        elif operation in ['rank']:
            pipe.then(lambda m: m.rank())
        elif operation in ['trace']:
            pipe.then(lambda m: m.trace())
        
        result = pipe.get_result()
        return result
    
    # ============================================================
    # 📁 أنابيب الإحصاء
    # ============================================================
    
    def stats_pipe(self, data: List[float], operation: str) -> Dict:
        """أنبوب العمليات الإحصائية"""
        
        pipe = MathPipe(data)
        pipe.metadata['original_data'] = data
        pipe.metadata['problem_type'] = 'statistics'
        pipe.metadata['operation'] = operation
        pipe.metadata['data_size'] = len(data)
        
        # تحقق عام
        pipe.validate(lambda d: len(d) > 0, "لا توجد بيانات")
        
        if operation in ['mean', 'متوسط']:
            result = (pipe
                .then(lambda d: sum(d) / len(d))
                .get_result())
        
        elif operation in ['variance', 'تباين']:
            result = (pipe
                .validate(lambda d: len(d) > 1, "التباين يحتاج على الأقل قيمتين")
                .then(self._calculate_variance)
                .get_result())
        
        elif operation in ['std', 'انحراف']:
            result = (pipe
                .validate(lambda d: len(d) > 1, "الانحراف المعياري يحتاج على الأقل قيمتين")
                .then(self._calculate_variance)
                .then(lambda v: v ** 0.5)
                .get_result())
        
        elif operation in ['min', 'أصغر']:
            result = pipe.then(min).get_result()
        
        elif operation in ['max', 'أكبر']:
            result = pipe.then(max).get_result()
        
        elif operation in ['sum', 'مجموع']:
            result = pipe.then(sum).get_result()
        
        elif operation in ['count', 'عدد']:
            result = pipe.then(len).get_result()
        
        else:
            result = {'success': False, 'errors': [f'عملية غير معروفة: {operation}']}
        
        return result
    
    def _calculate_variance(self, data):
        """حساب التباين (بافتراض عينة)"""
        n = len(data)
        if n <= 1:
            return 0
        mean_val = sum(data) / n
        return sum((x - mean_val) ** 2 for x in data) / (n - 1)
    
    # ============================================================
    # 📁 أنابيب إضافية
    # ============================================================
    
    def simplify_pipe(self, expression: str) -> Dict:
        """أنبوب تبسيط التعبيرات"""
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'simplify'
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(simplify)
            .get_result())
        
        return result
    
    def expand_pipe(self, expression: str) -> Dict:
        """أنبوب فك الأقواس"""
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'expand'
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(expand)
            .get_result())
        
        return result
    
    def factor_pipe(self, expression: str) -> Dict:
        """أنبوب التحليل إلى عوامل"""
        
        pipe = MathPipe()
        pipe.metadata['original_expression'] = expression
        pipe.metadata['problem_type'] = 'factor'
        
        result = (pipe
            .then(self._parse_expression, expression)
            .validate(lambda e: e is not None, "تعبير غير صالح")
            .then(factor)
            .get_result())
        
        return result
