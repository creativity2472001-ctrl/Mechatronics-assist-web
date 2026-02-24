from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mathcore import MathCore

# ✅ المسار الصحيح (templates داخل backend)
app = Flask(__name__, 
            static_folder='templates',
            template_folder='templates')

CORS(app)
math_core = MathCore()

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        question = data.get('question', '').strip()
        language = data.get('language', 'ar')
        user_id = data.get('user_id', 'default')
        
        if not question:
            return jsonify({'success': False, 'simple_answer': 'الرجاء إدخال سؤال'})
        
        result = math_core.solve(question, language, user_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy', 'engine': 'MathCore'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
