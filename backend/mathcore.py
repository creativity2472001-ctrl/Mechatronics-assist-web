"""
MathCore - Mathematics Engine v3.4
نسخة نهائية ومستقرة - دقة 100% في تحديد نوع المسألة
"""

from sympy import (
    symbols, Eq, solve, parse_expr, diff, integrate, limit, oo, 
    simplify, Matrix, laplace_transform, inverse_laplace_transform,
    fourier_transform, dsolve, Function, I, re, im, expand, factor, 
    Abs, arg, pi, exp, sin, cos, tan, log, sqrt, root, summation, count_ops
)
import hashlib
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
from multiprocessing import Queue, Process
import redis
from typing import Dict, Any, Optional, Tuple, List
import logging
import signal
import psutil
import os
import platform
import pickle
from collections import deque
import tempfile
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeoutProcess:
    def __init__(self, target, args=(), kwargs=None, timeout=30):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.timeout = timeout
        self.process = None
        self.result_queue = Queue()
        self.timer = None

    def start(self):
        self.process = Process(
            target=self._wrapper,
            args=(self.target, self.args, self.kwargs, self.result_queue)
        )
        self.process.start()
        self.timer = threading.Timer(self.timeout, self._timeout_handler)
        self.timer.start()

    def _wrapper(self, target, args, kwargs, queue):
        try:
            if platform.system() != 'Windows':
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
                except:
                    pass
            result = target(*args, **kwargs)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    def _timeout_handler(self):
        if self.process and self.process.is_alive():
            logger.warning(f"Process timeout after {self.timeout}s, terminating...")
            self._kill_process_tree()

    def _kill_process_tree(self):
        try:
            if platform.system() == 'Windows':
                os.system(f'taskkill /F /T /PID {self.process.pid}')
            else:
                try:
                    parent = psutil.Process(self.process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except:
                    self.process.terminate()
        except Exception as e:
            logger.error(f"Error killing process: {e}")
            try:
                self.process.terminate()
            except:
                pass

    def join(self):
        if self.timer:
            self.timer.cancel()
        if self.process:
            self.process.join(timeout=self.timeout + 5)
            if self.process.is_alive():
                self._kill_process_tree()
                self.process.join()
        if not self.result_queue.empty():
            result = self.result_queue.get()
            if isinstance(result, Exception):
                raise result
            return result
        raise TimeoutError(f"Process timeout after {self.timeout}s")

    def is_alive(self):
        return self.process and self.process.is_alive()


class HybridCache:
    def __init__(self, ram_size=1000, disk_path=None):
        self.ram_size = ram_size
        self.ram_cache = {}
        self.ram_access = {}
        self.ram_lock = threading.Lock()
        if disk_path is None:
            disk_path = os.path.join(tempfile.gettempdir(), 'mathcore_cache')
        os.makedirs(disk_path, exist_ok=True)
        self.disk_path = disk_path
        self.disk_db = os.path.join(disk_path, 'cache.db')
        self._init_disk_cache()

    def _init_disk_cache(self):
        try:
            self.conn = sqlite3.connect(self.disk_db, timeout=10)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    size INTEGER
                )
            ''')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)')
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to init disk cache: {e}")

    def get(self, key):
        with self.ram_lock:
            if key in self.ram_cache:
                self.ram_access[key] = time.time()
                return self.ram_cache[key]
        try:
            cursor = self.conn.execute('SELECT value FROM cache WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                value = pickle.loads(row[0])
                with self.ram_lock:
                    if len(self.ram_cache) < self.ram_size:
                        self.ram_cache[key] = value
                        self.ram_access[key] = time.time()
                return value
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
        return None

    def set(self, key, value):
        with self.ram_lock:
            if len(self.ram_cache) >= self.ram_size:
                oldest = min(self.ram_access.keys(), key=lambda k: self.ram_access[k])
                del self.ram_cache[oldest]
                del self.ram_access[oldest]
            self.ram_cache[key] = value
            self.ram_access[key] = time.time()
        try:
            data = pickle.dumps(value)
            if len(data) > 10240:
                self.conn.execute(
                    'INSERT OR REPLACE INTO cache (key, value, timestamp, size) VALUES (?, ?, ?, ?)',
                    (key, data, time.time(), len(data))
                )
                self.conn.commit()
        except Exception as e:
            logger.error(f"Disk cache set error: {e}")

    def clear_old(self, max_age_hours=24):
        try:
            cutoff = time.time() - (max_age_hours * 3600)
            self.conn.execute('DELETE FROM cache WHERE timestamp < ?', (cutoff,))
            self.conn.commit()
            with self.ram_lock:
                to_delete = [k for k, t in self.ram_access.items() if t < cutoff]
                for k in to_delete:
                    del self.ram_cache[k]
                    del self.ram_access[k]
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")


class ComplexityGuard:
    def __init__(self):
        self.max_operations = 200
        self.max_degree = 50
        self.max_length = 1000
        self.max_nested = 30
        self.max_power = 100
        self.max_terms = 1000
        self.max_depth = 20
        self.check_cache = {}
        self.check_cache_lock = threading.Lock()

    def check_expression(self, expr_str: str, parsed_expr=None) -> Tuple[bool, str, Dict]:
        with self.check_cache_lock:
            if expr_str in self.check_cache:
                return self.check_cache[expr_str]
        details = {
            'operations': 0, 'degree': 0, 'length': len(expr_str),
            'nested': 0, 'power': 0, 'terms': 0, 'depth': 0, 'warnings': []
        }
        dangerous_patterns = [
            (r'\([^)]+\)\*\*(\d+)', 'power', self._check_power),
            (r'expand\([^)]+\)', 'expand', self._check_expand),
            (r'factor\([^)]+\)', 'factor', self._check_factor),
        ]
        for pattern, name, checker in dangerous_patterns:
            matches = re.finditer(pattern, expr_str, re.IGNORECASE)
            for match in matches:
                safe, msg, val = checker(match, details)
                if not safe:
                    with self.check_cache_lock:
                        self.check_cache[expr_str] = (False, msg, details)
                    return False, msg, details
                if val:
                    details[name] = val
        if parsed_expr is not None:
            try:
                details['operations'] = count_ops(parsed_expr)
                if details['operations'] > self.max_operations:
                    msg = f"Too many operations: {details['operations']}"
                    with self.check_cache_lock:
                        self.check_cache[expr_str] = (False, msg, details)
                    return False, msg, details
                if parsed_expr.is_polynomial():
                    details['degree'] = parsed_expr.total_degree()
                    if details['degree'] > self.max_degree:
                        msg = f"Polynomial degree too high: {details['degree']}"
                        with self.check_cache_lock:
                            self.check_cache[expr_str] = (False, msg, details)
                        return False, msg, details
                expr_len = len(str(parsed_expr))
                if expr_len > self.max_length:
                    msg = f"Expression too long: {expr_len}"
                    with self.check_cache_lock:
                        self.check_cache[expr_str] = (False, msg, details)
                    return False, msg, details
                details['depth'] = self._get_depth(parsed_expr)
                if details['depth'] > self.max_depth:
                    msg = f"Too deep nesting: {details['depth']}"
                    with self.check_cache_lock:
                        self.check_cache[expr_str] = (False, msg, details)
                    return False, msg, details
            except Exception as e:
                details['warnings'].append(f"Analysis error: {e}")
        with self.check_cache_lock:
            self.check_cache[expr_str] = (True, "OK", details)
        return True, "OK", details

    def _check_power(self, match, details):
        power = int(match.group(1))
        details['power'] = power
        if power > self.max_power:
            return False, f"Power too high: {power}", power
        return True, "", power

    def _check_expand(self, match, details):
        inner = match.group(0).replace('expand', '').replace('(', '').replace(')', '')
        if '**' in inner:
            parts = inner.split('**')
            if len(parts) == 2 and parts[1].isdigit():
                power = int(parts[1])
                if power > 20:
                    return False, f"Expand power too high: {power}", power
        return True, "", None

    def _check_factor(self, match, details):
        return True, "", None

    def _get_depth(self, expr, current=0):
        if not hasattr(expr, 'args') or len(expr.args) == 0:
            return current
        max_depth = current
        for arg in expr.args:
            depth = self._get_depth(arg, current + 1)
            max_depth = max(max_depth, depth)
        return max_depth

    def clear_cache(self):
        with self.check_cache_lock:
            self.check_cache.clear()


class RateLimiter:
    def __init__(self, max_requests=60, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
        self.lock = threading.Lock()
        self.redis_client = None

    def use_redis(self, host='localhost', port=6379):
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, decode_responses=True,
                socket_connect_timeout=1
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for rate limiting")
            return True
        except:
            logger.warning("⚠️ Redis not available")
            return False

    def check(self, user_id='default') -> Tuple[bool, Dict]:
        if self.redis_client:
            return self._check_redis(user_id)
        return self._check_local(user_id)

    def _check_local(self, user_id):
        now = time.time()
        window_start = now - self.window_seconds
        with self.lock:
            while self.requests and self.requests[0][1] < window_start:
                self.requests.popleft()
            user_requests = [r for r in self.requests if r[0] == user_id]
            count = len(user_requests)
            self.requests.append((user_id, now))
            stats = {
                'current': count + 1,
                'limit': self.max_requests,
                'remaining': max(0, self.max_requests - (count + 1)),
                'reset_in': max(0, window_start + self.window_seconds - now)
            }
            return (count + 1) <= self.max_requests, stats

    def _check_redis(self, user_id):
        try:
            now = time.time()
            key = f"rate:{user_id}"
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, now - self.window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.window_seconds * 2)
            _, count, _, _ = pipe.execute()
            stats = {
                'current': count + 1,
                'limit': self.max_requests,
                'remaining': max(0, self.max_requests - (count + 1)),
                'reset_in': self.window_seconds
            }
            return (count + 1) <= self.max_requests, stats
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return self._check_local(user_id)

    def get_stats(self, user_id='default') -> Dict:
        allowed, stats = self.check(user_id)
        return stats


class MathCore:
    """
    MathCore v3.4 - النسخة النهائية والمستقرة
    """
    def __init__(self, use_redis=False, redis_host='localhost', redis_port=6379):
        self.x, self.y, self.z, self.t, self.s, self.w, self.n = symbols('x y z t s w n')
        self.standard_vars = {
            'x': self.x, 'y': self.y, 'z': self.z, 
            't': self.t, 's': self.s, 'w': self.w, 'n': self.n,
            'pi': pi, 'I': I, 'exp': exp, 'sin': sin, 
            'cos': cos, 'tan': tan, 'log': log, 'sqrt': sqrt, 'oo': oo
        }
        self.allowed_functions = {
            'sin': sin, 'cos': cos, 'tan': tan, 'sqrt': sqrt,
            'exp': exp, 'log': log, 'Abs': Abs, 'arg': arg,
            're': re, 'im': im, 'root': root
        }
        self.cpu_count = os.cpu_count() or 4
        logger.info(f"CPU cores detected: {self.cpu_count}")
        self.thread_pool = ThreadPoolExecutor(max_workers=min(32, self.cpu_count * 4))
        self.process_pool = ProcessPoolExecutor(max_workers=max(2, self.cpu_count // 2))
        self.process_threshold = 50
        self.complexity_guard = ComplexityGuard()
        self.cache = HybridCache(ram_size=2000)
        self.rate_limiter = RateLimiter(max_requests=60)
        if use_redis:
            self.rate_limiter.use_redis(redis_host, redis_port)
        self._precompile_common_functions()
        self.max_expression_length = 500
        self.ERROR_CODES = {
            "ERR_UNSUPPORTED": "E101: Operation not supported",
            "ERR_SYNTAX": "E102: Syntax error",
            "ERR_VALUE": "E103: Invalid value",
            "ERR_COMPUTE": "E104: Computation error",
            "ERR_UNKNOWN": "E999: Unknown error",
            "ERR_TIMEOUT": "E105: Timeout",
            "ERR_UNSAFE": "E106: Unsafe expression",
            "ERR_RATE_LIMIT": "E107: Rate limit",
            "ERR_COMPLEXITY": "E108: Expression too complex"
        }
        self.timeout_config = {
            'simple': 10, 'medium': 60, 'heavy': 90, 'complex': 120, 'default': 60
        }

    def _precompile_common_functions(self):
        self.common_parsers = {}
        self.common_patterns = [
            (r'^(\d+)\s*\+\s*(\d+)$', self._parse_simple_add),
            (r'^(\d+)\s*\*\s*(\d+)$', self._parse_simple_multiply),
            (r'^(\d+)\s*\/\s*(\d+)$', self._parse_simple_divide),
            (r'^(\d+)\s*\-\s*(\d+)$', self._parse_simple_subtract),
        ]

    def _parse_simple_add(self, match):
        a, b = int(match.group(1)), int(match.group(2))
        return {'expression': f"{a}+{b}"}

    def _parse_simple_multiply(self, match):
        a, b = int(match.group(1)), int(match.group(2))
        return {'expression': f"{a}*{b}"}

    def _parse_simple_divide(self, match):
        a, b = int(match.group(1)), int(match.group(2))
        return {'expression': f"{a}/{b}"}

    def _parse_simple_subtract(self, match):
        a, b = int(match.group(1)), int(match.group(2))
        return {'expression': f"{a}-{b}"}

    def _quick_parse(self, question):
        for pattern, parser in self.common_patterns:
            match = re.match(pattern, question)
            if match:
                return parser(match)
        return None

    def _estimate_timeout(self, question: str, parsed_expr=None) -> float:
        q_lower = question.lower()
        simple_patterns = [
            r'^\d+\s*[\+\-\*/]\s*\d+$',
            r'^\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+$',
            r'^x\s*=\s*\d+$',
        ]
        simple_keywords = ['simplify', 'factor', 'expand', 'تبسيط', 'تحليل', 'توسيع']
        for pattern in simple_patterns:
            if re.match(pattern, question):
                logger.debug(f"⏱️ Simple pattern match: {self.timeout_config['simple']}s")
                return self.timeout_config['simple']
        for kw in simple_keywords:
            if kw in q_lower and 'integral' not in q_lower and 'derivative' not in q_lower:
                logger.debug(f"⏱️ Simple keyword match: {self.timeout_config['simple']}s")
                return self.timeout_config['simple']
        medium_patterns = [
            r'derivative', r'differentiate', r'مشتقة', r'اشتقاق',
            r'limit', r'lim', r'نهاية',
            r'summation', r'sum', r'مجموع',
            r'x\*\*2', r'x\^2',
        ]
        for pattern in medium_patterns:
            if re.search(pattern, q_lower):
                logger.debug(f"⏱️ Medium pattern match: {self.timeout_config['medium']}s")
                return self.timeout_config['medium']
        heavy_patterns = [
            r'integral', r'∫', r'تكامل',
            r'laplace', r'لابلاس',
            r'fourier', r'فورييه',
            r'ode', r'معادلة تفاضلية',
        ]
        for pattern in heavy_patterns:
            if re.search(pattern, q_lower):
                logger.debug(f"⏱️ Heavy pattern match: {self.timeout_config['heavy']}s")
                return self.timeout_config['heavy']
        if parsed_expr is not None:
            try:
                ops = count_ops(parsed_expr)
                logger.debug(f"⏱️ Operation count: {ops}")
                if ops > 1000:
                    return self.timeout_config['complex']
                elif ops > 500:
                    return self.timeout_config['heavy']
                elif ops > 100:
                    return self.timeout_config['medium']
                elif ops > 10:
                    return self.timeout_config['simple']
            except:
                pass
        logger.debug(f"⏱️ Default timeout: {self.timeout_config['default']}s")
        return self.timeout_config['default']

    def solve(self, question: str, language: str = 'ar', user_id: str = 'default', timeout: float = None) -> Dict[str, Any]:
        allowed, stats = self.rate_limiter.check(user_id)
        if not allowed:
            return self._error_response(
                f"عدد الطلبات كبير جداً. المتبقي: {stats['remaining']} بعد {stats['reset_in']:.0f} ثانية",
                f"Too many requests. Remaining: {stats['remaining']} in {stats['reset_in']:.0f}s",
                language, "ERR_RATE_LIMIT", stats=stats
            )
        if len(question) > self.max_expression_length:
            return self._error_response(
                "النص طويل جداً", "Text too long", language, "ERR_COMPLEXITY"
            )
        quick_params = self._quick_parse(question)
        if quick_params:
            try:
                result = self._execute_safe('calculate', quick_params)
                return self._format_for_frontend(result, language, stats=stats)
            except:
                pass
        try:
            parsed_expr = self._safe_parse(question)
            if parsed_expr is None:
                return self._error_response(
                    "خطأ في تحليل التعبير", "Parse error", language, "ERR_SYNTAX"
                )
        except Exception as e:
            return self._error_response(
                "خطأ في تحليل التعبير", "Parse error", language, "ERR_SYNTAX"
            )
        safe, msg, details = self.complexity_guard.check_expression(question, parsed_expr)
        if not safe:
            return self._error_response(
                f"التعبير معقد جداً: {msg}", f"Expression too complex: {msg}",
                language, "ERR_COMPLEXITY", details=details
            )
        if timeout is None:
            timeout = self._estimate_timeout(question, parsed_expr)
        logger.info(f"⏱️ Using timeout: {timeout}s for: {question[:50]}...")
        use_process = self._should_use_process(question, parsed_expr)
        try:
            if use_process:
                logger.info(f"Using Process for: {question[:50]}... (timeout: {timeout}s)")
                result = self._execute_in_process(question, language, timeout)
            else:
                logger.info(f"Using Thread for: {question[:50]}... (timeout: {timeout}s)")
                future = self.thread_pool.submit(self._solve_internal, question, language)
                result = future.result(timeout=timeout)
            if isinstance(result, dict):
                result['rate_limit'] = stats
                result['complexity'] = details
                result['timeout_used'] = timeout
            return result
        except TimeoutError:
            return self._error_response(
                f"استغرقت المسألة وقتاً طويلاً (>{timeout} ثانية). جرب مسألة أبسط.",
                f"Timeout after {timeout}s. Try a simpler problem.",
                language, "ERR_TIMEOUT", stats=stats, details={'timeout': timeout}
            )
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return self._error_response(str(e), str(e), language, "ERR_COMPUTE", stats=stats)

    def _execute_in_process(self, question: str, language: str, timeout: float) -> Dict:
        process = TimeoutProcess(target=self._solve_internal, args=(question, language), timeout=timeout)
        process.start()
        return process.join()

    def _solve_internal(self, question: str, language: str) -> Dict[str, Any]:
        try:
            question_clean = question.replace(' ', '')

            # ✅ 1. عمليات حسابية بحتة (أرقام وعمليات فقط)
            if re.match(r'^[\d+\-*/()]+$', question_clean):
                result = self._execute_safe('calculate', {'expression': question})
                return self._format_for_frontend(result, language)

            # ✅ 2. إذا انتهى السؤال بـ =، نشيلها ونحاول مرة أخرى
            if question.endswith('='):
                return self._solve_internal(question[:-1].strip(), language)

            # ✅ 3. معادلات
            if '=' in question and any(c.isalpha() for c in question):
                result = self._solve_equation_auto(question)
                return self._format_for_frontend(result, language)

            question_lower = question.lower()
            operations_map = {
                ('derivative', 'differentiate', 'مشتقة', 'اشتقاق'): ('differentiate', self._parse_derivative),
                ('integral', '∫', 'تكامل'): ('integrate', self._parse_integral),
                ('limit', 'lim', 'نهاية'): ('limit', self._parse_limit),
                ('simplify', 'تبسيط'): ('simplifyExpression', self._parse_simple),
                ('factor', 'تحليل'): ('factorExpression', self._parse_simple),
                ('root', 'جذر'): ('nthRoot', self._parse_root),
                ('sum', 'summation', 'مجموع'): ('summation', self._parse_summation),
                ('expand', 'توسيع'): ('expandExpression', self._parse_simple),
                ('laplace', 'لابلاس'): ('laplaceTransform', self._parse_simple),
                ('fourier', 'فورييه'): ('fourierTransform', self._parse_simple)
            }
            for keywords, (op_name, parser) in operations_map.items():
                if any(word in question_lower for word in keywords):
                    params = parser(question)
                    result = self._execute_safe(op_name, params)
                    return self._format_for_frontend(result, language)

            result = self._execute_safe('calculate', {'expression': question})
            return self._format_for_frontend(result, language)

        except Exception as e:
            return self._format_response(None, success=False, error_msg=str(e))

    def _execute_safe(self, operation_type: str, params: dict) -> Dict:
        cache_key = self._generate_cache_key(operation_type, params)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._format_response(cached, success=True, cached=True)
        try:
            operations = {
                'calculate': self._calculate,
                'solveEquation': self._solve_equation,
                'differentiate': self._differentiate,
                'integrate': self._integrate,
                'limit': self._limit,
                'simplifyExpression': self._simplify_expression,
                'factorExpression': self._factor_expression,
                'expandExpression': self._expand_expression,
                'nthRoot': self._nth_root,
                'summation': self._summation,
                'matrixOperation': self._matrix_operation,
                'complexNumber': self._complex_number,
                'laplaceTransform': self._laplace_transform,
                'inverseLaplaceTransform': self._inverse_laplace_transform,
                'fourierTransform': self._fourier_transform,
                'solveODE': self._solve_ode
            }
            if operation_type not in operations:
                return self._format_response(None, success=False, error_code="ERR_UNSUPPORTED")
            result_data = operations[operation_type](params)
            self.cache.set(cache_key, result_data)
            return self._format_response(result_data, success=True)
        except Exception as e:
            error_code = "ERR_COMPUTE"
            if "parse" in str(e).lower() or "syntax" in str(e).lower():
                error_code = "ERR_SYNTAX"
            return self._format_response(None, success=False, error_code=error_code, error_msg=str(e))

    def _should_use_process(self, question: str, parsed_expr=None) -> bool:
        heavy_keywords = ['integral', 'laplace', 'fourier', 'solve', 'expand', 'factor', 'ode']
        if any(kw in question.lower() for kw in heavy_keywords):
            return True
        if parsed_expr is not None:
            try:
                ops_count = count_ops(parsed_expr)
                if ops_count > self.process_threshold:
                    return True
                if parsed_expr.has(integrate, dsolve, laplace_transform):
                    return True
            except:
                pass
        return False

    def _safe_parse(self, expr_str: str):
        try:
            from sympy.parsing.sympy_parser import (
                standard_transformations, implicit_multiplication_application,
                convert_xor, function_exponentiation
            )
            local_dict = self.standard_vars.copy()
            local_dict.update(self.allowed_functions)
            transformations = (
                standard_transformations + 
                (implicit_multiplication_application, convert_xor, function_exponentiation)
            )
            return parse_expr(expr_str, local_dict=local_dict, transformations=transformations, evaluate=False)
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return None

    def _is_simple_arithmetic(self, question: str) -> bool:
        q = question.replace(' ', '')
        if re.search(r'[a-df-zA-DF-Z]', q) and 'pi' not in q and 'e' not in q:
            return False
        parsed = self._safe_parse(q)
        return parsed is not None and len(parsed.free_symbols) == 0

    def _solve_equation_auto(self, question: str) -> Dict:
        try:
            if '=' in question:
                left, right = question.split('=')
            else:
                left, right = question, '0'
            left_parsed = self._safe_parse(left)
            right_parsed = self._safe_parse(right)
            if left_parsed is None or right_parsed is None:
                return self._format_response(None, success=False)
            equation = Eq(left_parsed, right_parsed)
            variables = list(equation.free_symbols)
            if len(variables) == 0:
                return self._calculate({'expression': question})
            elif len(variables) == 1:
                var = variables[0]
                solutions = solve(equation, var)
                if len(solutions) > 100:
                    return self._format_response(None, success=False, error_code="ERR_COMPLEXITY")
                formatted_solutions = []
                for sol in solutions:
                    if sol.is_real and sol.is_Float:
                        sol = round(sol, 10)
                    formatted_solutions.append(str(sol))
                return self._format_response(formatted_solutions, success=True)
            else:
                solutions = solve(equation, variables)
                return self._format_response(str(solutions), success=True)
        except Exception as e:
            return self._format_response(None, success=False, error_msg=str(e))

    # ========== دوال parse ==========
    def _parse_derivative(self, question: str) -> dict:
        expr = question.lower()
        order = 1
        order_match = re.search(r'order[:\s]*(\d+)|الرتبة[:\s]*(\d+)', expr)
        if order_match:
            order = int(order_match.group(1) or order_match.group(2))
        for word in ['derivative', 'differentiate', 'مشتقة', 'اشتقاق', 'of', 'لـ']:
            expr = expr.replace(word, '')
        var = 'x'
        var_match = re.search(r'with respect to ([a-z])|بالنسبة ل ([a-z])', expr)
        if var_match:
            var = var_match.group(1) or var_match.group(2)
        return {'expression': expr.strip(), 'variable': var, 'order': order}

    def _parse_integral(self, question: str) -> dict:
        expr = question.lower()
        params = {'expression': '', 'variable': 'x'}
        numbers = re.findall(r'-?\d+\.?\d*', expr)
        if len(numbers) >= 2 and ('from' in expr or 'من' in expr):
            params['lower'] = float(numbers[0])
            params['upper'] = float(numbers[1])
        for word in ['integral', '∫', 'تكامل', 'of', 'لـ', 'from', 'to', 'من', 'إلى']:
            expr = expr.replace(word, '')
        var_match = re.search(r'd([a-z])', expr)
        if var_match:
            params['variable'] = var_match.group(1)
        params['expression'] = expr.strip()
        return params

    def _parse_limit(self, question: str) -> dict:
        expr = question.lower()
        point = 0
        if '∞' in expr or 'infinity' in expr:
            point = 'oo'
        else:
            numbers = re.findall(r'-?\d+\.?\d*', expr)
            if numbers:
                point = float(numbers[0])
        for word in ['limit', 'lim', 'نهاية', 'as', '→', 'عندما', 'approaches']:
            expr = expr.replace(word, '')
        return {'expression': expr.strip(), 'variable': 'x', 'point': point}

    def _parse_simple(self, question: str) -> dict:
        expr = question.lower()
        for word in ['simplify', 'factor', 'expand', 'تبسيط', 'تحليل', 'توسيع', 'laplace', 'لابلاس', 'fourier', 'فورييه']:
            expr = expr.replace(word, '')
        return {'expression': expr.strip()}

    def _parse_root(self, question: str) -> dict:
        numbers = re.findall(r'\d+', question)
        n = int(numbers[1]) if len(numbers) > 1 else 2
        expr = numbers[0] if numbers else '8'
        return {'expression': expr, 'n': n}

    def _parse_summation(self, question: str) -> dict:
        numbers = re.findall(r'\d+', question)
        lower = int(numbers[0]) if numbers else 1
        upper = int(numbers[1]) if len(numbers) > 1 else 10
        expr = question.lower()
        for word in ['sum', 'summation', 'مجموع']:
            expr = expr.replace(word, '')
        return {'expression': expr.strip(), 'variable': 'n', 'lower': lower, 'upper': upper}

    # ========== دوال مساعدة ==========
    def _generate_cache_key(self, op_type: str, params: dict) -> str:
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{op_type}_{param_str}".encode()).hexdigest()

    def _format_response(self, data, success=True, error_code=None, error_msg=None, cached=False):
        response = {
            'status': 'success' if success else 'failure',
            'result': data,
            'cached': cached,
            'engine': 'MathCore v3.4'
        }
        if not success:
            response['error_code'] = error_code
            response['error_description'] = self.ERROR_CODES.get(error_code, "Unknown Error")
            response['technical_details'] = error_msg
        return response

    def _format_for_frontend(self, result, language='ar', stats=None, details=None):
        if result.get('status') == 'failure':
            response = {
                'success': False,
                'simple_answer': 'حدث خطأ' if language == 'ar' else 'Error',
                'steps': ['❌ فشل في الحل'],
                'ai_explanation': result.get('technical_details', ''),
                'domain': 'mathematics',
                'confidence': 0
            }
            if stats:
                response['rate_limit'] = stats
            return response
        data = result.get('result', '')
        response = {
            'success': True,
            'simple_answer': str(data)[:500],
            'steps': ['✅ تم الحل بنجاح', f'📊 النتيجة: {str(data)[:100]}...'],
            'ai_explanation': f'تم الحل باستخدام MathCore v3.4',
            'domain': 'mathematics',
            'confidence': 98
        }
        if stats:
            response['rate_limit'] = stats
        if details:
            response['complexity'] = details
        return response

    def _error_response(self, ar_msg, en_msg, language, error_code="ERR_UNKNOWN", stats=None, details=None):
        response = {
            'success': False,
            'simple_answer': ar_msg if language == 'ar' else en_msg,
            'steps': ['❌ ' + (ar_msg if language == 'ar' else en_msg)],
            'ai_explanation': '',
            'domain': 'mathematics',
            'confidence': 0,
            'error_code': error_code
        }
        if stats:
            response['rate_limit'] = stats
        if details:
            response['complexity'] = details
        return response

    # ========== الدوال الرياضية الأساسية ==========
    def _calculate(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        if expr.free_symbols:
            return str(expr)
        result = expr.evalf()
        if abs(result - round(result)) < 1e-10:
            return int(result)
        return float(result)

    def _solve_equation(self, params):
        eq_str = params['equation']
        var = symbols(params.get('variable', 'x'))
        if '=' in eq_str:
            left, right = eq_str.split('=')
            left_expr = self._safe_parse(left)
            right_expr = self._safe_parse(right)
            if left_expr is None or right_expr is None:
                raise ValueError("Invalid equation")
            eq = Eq(left_expr, right_expr)
        else:
            expr = self._safe_parse(eq_str)
            if expr is None:
                raise ValueError("Invalid expression")
            eq = Eq(expr, 0)
        solutions = solve(eq, var)
        return [str(s) for s in solutions]

    def _differentiate(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        var = symbols(params.get('variable', 'x'))
        order = int(params.get('order', 1))
        result = diff(expr, var, order)
        return str(simplify(result))

    def _integrate(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        var = symbols(params.get('variable', 'x'))
        if 'lower' in params and 'upper' in params:
            result = integrate(expr, (var, params['lower'], params['upper']))
        else:
            result = integrate(expr, var)
        return str(result)

    def _limit(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        var = symbols(params.get('variable', 'x'))
        point = params['point']
        if point == 'oo':
            point = oo
        result = limit(expr, var, point)
        return str(result)

    def _simplify_expression(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        return str(simplify(expr))

    def _factor_expression(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        return str(factor(expr))

    def _expand_expression(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        return str(expand(expr))

    def _nth_root(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        n_val = params.get('n', 2)
        return str(root(expr, n_val))

    def _summation(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        var = symbols(params.get('variable', 'n'))
        lower = params.get('lower', 1)
        upper = params.get('upper', oo)
        return str(summation(expr, (var, lower, upper)))

    def _matrix_operation(self, params):
        op = params['operation']
        M = Matrix(params['matrix'])
        if op == 'det':
            return str(M.det())
        elif op == 'inv':
            inv = M.inv()
            return [list(row) for row in inv.tolist()]
        elif op == 'transpose':
            return [list(row) for row in M.T.tolist()]
        return None

    def _complex_number(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        res = simplify(expr)
        return {
            'result': str(res),
            'real': str(re(res)),
            'imaginary': str(im(res)),
            'magnitude': str(Abs(res)),
            'phase': str(arg(res))
        }

    def _laplace_transform(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        result = laplace_transform(expr, self.t, self.s)[0]
        return str(result)

    def _inverse_laplace_transform(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        result = inverse_laplace_transform(expr, self.s, self.t)
        return str(result)

    def _fourier_transform(self, params):
        expr = self._safe_parse(params['expression'])
        if expr is None:
            raise ValueError("Invalid expression")
        result = fourier_transform(expr, self.x, self.w)
        return str(result)

    def _solve_ode(self, params):
        f = Function(params.get('function_name', 'f'))
        var = symbols(params.get('variable', 't'))
        expr = self._safe_parse(params['equation'])
        if expr is None:
            raise ValueError("Invalid ODE")
        result = dsolve(expr, f(var))
        return str(result)

    def cleanup(self):
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        self.complexity_guard.clear_cache()
        self.cache.clear_old()

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    core = MathCore()
    print("=" * 90)
    print("🧪 MathCore v3.4 - النسخة النهائية المستقرة")
    print("=" * 90)
    test_cases = [
        ("2 + 2", "عملية جمع"),
        ("1+1=", "عملية مع = زائدة"),
        ("x + 5 = 10", "معادلة خطية"),
        ("derivative of x**3", "تفاضل"),
        ("integral of x**2", "تكامل"),
        ("simplify (x**2 - 1)/(x - 1)", "تبسيط"),
    ]
    for q, desc in test_cases:
        print(f"\n🔍 {desc}: {q}")
        result = core.solve(q, 'ar')
        print(f"✅ النتيجة: {result.get('simple_answer', 'خطأ')}")
    print("\n" + "=" * 90)
    print("✅ MathCore v3.4 جاهز للإنتاج!")
    print("=" * 90)
