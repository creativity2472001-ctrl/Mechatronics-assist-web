"""
MathCore Server - الخادم الرئيسي للمشروع (متوافق مع v3.3)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# استيراد MathCore من نفس المجلد
from mathcore import MathCore

# إنشاء تطبيق Flask - المسارات المعدلة ✅
app = Flask(__name__, 
            static_folder='templates',      # ✅ تم التعديل
            template_folder='templates')     # ✅ تم التعديل
CORS(app)  # للسماح بالتواصل مع الواجهة

# إنشاء كائن MathCore (نسخة v3.3)
math_core = MathCore()

@app.route('/')
def index():
    """عرض الصفحة الرئيسية"""
    # ✅ تم التعديل - مسار مباشر
    return send_from_directory('templates', 'index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    """حل المسائل الرياضية"""
    try:
        # استلام البيانات من الواجهة
        data = request.json
        question = data.get('question', '').strip()
        language = data.get('language', 'ar')
        user_id = data.get('user_id', 'default')
        
        logger.info(f"📩 سؤال جديد: {question[:50]}...")
        
        # التحقق من وجود سؤال
        if not question:
            return jsonify({
                'success': False,
                'simple_answer': 'الرجاء إدخال سؤال' if language == 'ar' else 'Please enter a question',
                'steps': [],
                'ai_explanation': '',
                'domain': 'mathematics',
                'confidence': 0
            })
        
        # حل المسألة باستخدام mathcore.py
        result = math_core.solve(
            question=question,
            language=language,
            user_id=user_id,
            timeout=None
        )
        
        logger.info(f"✅ تم الحل: {result.get('simple_answer', '')[:50]}...")
        
        # إرسال النتيجة للواجهة
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ خطأ: {str(e)}")
        return jsonify({
            'success': False,
            'simple_answer': 'حدث خطأ في الخادم',
            'steps': [str(e)],
            'ai_explanation': 'يرجى المحاولة مرة أخرى',
            'domain': 'mathematics',
            'confidence': 0
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """التحقق من أن الخادم يعمل"""
    return jsonify({
        'status': 'healthy',
        'engine': 'MathCore v3.3',
        'timeout_config': math_core.timeout_config,
        'message': 'Server is running'
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """إحصائيات عن الخادم"""
    return jsonify({
        'cpu_cores': math_core.cpu_count,
        'thread_pool': math_core.thread_pool._max_workers,
        'process_pool': math_core.process_pool._max_workers,
        'timeout_config': math_core.timeout_config
    })

# تشغيل الخادم
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MathCore Server v3.3 Starting...")
    print("="*60)
    print(f"📁 المجلد الحالي: {os.getcwd()}")
    print(f"📁 مجلد الواجهة: templates")
    print(f"📄 ملف الرياضيات: mathcore.py (v3.3)")
    print(f"⚙️  Timeout config: {math_core.timeout_config}")
    print(f"🖥️  CPU cores: {math_core.cpu_count}")
    print("\n🌐 رابط الواجهة: http://localhost:5000")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
