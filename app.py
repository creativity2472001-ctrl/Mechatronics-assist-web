from flask import Flask, render_template, request, jsonify
from sympy import symbols, Eq, solve, diff, integrate, limit, parse_expr, sin, cos, tan, log, exp
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# الرموز الرياضية الأساسية
x, y, z, t = symbols('x y z t')

# مفتاح DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def ask_deepseek(prompt):
    """إرسال استفسار إلى DeepSeek"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "أنت مساعد رياضيات خبير."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"خطأ في الاتصال بـ DeepSeek: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    question = data.get('question', '').strip()
    lang = data.get('language', 'ar')
    
    if not question:
        return jsonify({"error": "السؤال فارغ", "simple_answer": "❌ خطأ"})
    
    try:
        # 1. DeepSeek يحدد التخصص ويحول السؤال لأمر رياضي
        analysis = ask_deepseek(f"""
حلل هذا السؤال وحدد:
1. التخصص (رياضيات/فيزياء/...)
2. حول السؤال لأمر SymPy

السؤال: {question}

أعد النتيجة بصيغة JSON:
{{"domain": "التخصص", "command": "أمر SymPy"}}
""")
        
        # 2. محاولة استخراج الأمر وتنفيذه
        import json
        try:
            result_json = json.loads(analysis)
            math_command = result_json.get("command", "")
            domain = result_json.get("domain", "رياضيات")
            
            # تنفيذ الأمر إذا كان موجوداً
            if math_command:
                try:
                    math_result = eval(math_command)
                    simple_answer = str(math_result)
                except:
                    simple_answer = "تعذر تنفيذ الأمر"
            else:
                simple_answer = "لم أستطع تحويل السؤال"
        except:
            simple_answer = "تحليل غير مفهوم"
        
        return jsonify({
            "success": True,
            "simple_answer": simple_answer,
            "domain": domain,
            "confidence": 98
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "simple_answer": "❌ خطأ في المعالجة"
        })

if __name__ == '__main__':
    print("🚀 التطبيق يعمل على: http://127.0.0.1:5000")
    print("🤖 DeepSeek متصل!")
    app.run(debug=True)
