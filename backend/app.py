"""
MathCore Server - الخادم الرئيسي للمشروع
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# استيراد MathCore من نفس المجلد
from mathcore import MathCore

# إنشاء تطبيق Flask
app = Flask(__name__, 
            static_folder='../templates',
            template_folder='../templates')
CORS(app)  # للسماح بالتواصل مع الواجهة

# إنشاء كائن MathCore
math_core = MathCore()

@app.route('/')
def index():
    """عرض الصفحة الرئيسية"""
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    """حل المسائل الرياضية"""
    try:
        # استلام البيانات من الواجهة
        data = request.json
        question = data.get('question', '').strip()
        language = data.get('language', 'ar')
        
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
        result = math_core.solve(question, language)
        
        # إرسال النتيجة للواجهة
        return jsonify(result)
        
    except Exception as e:
        # في حالة حدوث خطأ
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
        'engine': 'MathCore v1.1',
        'message': 'Server is running'
    })

# تشغيل الخادم
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 MathCore Server Starting...")
    print("="*50)
    print(f"📁 المجلد الحالي: {os.getcwd()}")
    print(f"📁 مجلد الواجهة: {app.template_folder}")
    print(f"📄 ملف الرياضيات: mathcore.py")
    print("\n🌐 رابط الواجهة: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
