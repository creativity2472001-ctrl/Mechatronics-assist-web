#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mechatronics Assistant - النسخة الاحترافية النهائية
يدعم: Gemini, DeepSeek, OpenRouter مع Code Execution
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
import traceback
import re
from typing import Optional, Dict, Any
from datetime import datetime

# تكوين التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # لدعم العربية
app.config['SECRET_KEY'] = os.urandom(24)

# ============================================================
# 🔑 نظام المفاتيح (من CMD فقط)
# ============================================================

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# ============================================================
# 📊 عرض حالة المفاتيح (للمطور فقط)
# ============================================================

print("\n" + "="*70)
print("🚀 MECHATRONICS ASSISTANT - النسخة الاحترافية النهائية")
print("="*70)

if GEMINI_API_KEY:
    print(f"✅ Gemini: متصل (مفتاح: {GEMINI_API_KEY[:8]}...)")
else:
    print("❌ Gemini: غير متصل")

if DEEPSEEK_API_KEY:
    print(f"✅ DeepSeek: متصل (مفتاح: {DEEPSEEK_API_KEY[:8]}...)")
else:
    print("❌ DeepSeek: غير متصل")

if OPENROUTER_API_KEY:
    print(f"✅ OpenRouter: متصل (مفتاح: {OPENROUTER_API_KEY[:8]}...)")
else:
    print("❌ OpenRouter: غير متصل")

print("="*70 + "\n")

# ============================================================
# 🧠 نظام كشف المجال الذكي (بدون كلمات مفتاحية)
# ============================================================

def detect_domain(question: str) -> tuple:
    """
    كشف المجال من السؤال نفسه دون الحاجة لكلمات مفتاحية
    """
    if not question:
        return False, None
    
    q = question
    q_lower = question.lower()
    
    # ============================================================
    # 📐 كشف الرياضيات
    # ============================================================
    math_patterns = [
        # رموز رياضية
        r'[∫∑∏√∞πθφωαβγ∂∇∈∉⊂⊃∩∪≈≠≤≥±∓·×÷°′″]',
        # دوال مثلثية
        r'sin|cos|tan|cot|sec|csc',
        r'arcsin|arccos|arctan',
        # لوغاريتمات
        r'log|ln|lg|e\^|exp',
        # تفاضل وتكامل
        r'diff|derivative|مشتقة',
        r'int|integral|تكامل',
        # نهايات
        r'lim|limit|نهاية',
        # معادلات
        r'x\s*[\+\-\*\/]\s*\d+',
        r'\d+\s*[\+\-\*\/]\s*x',
        r'x\^\d',
        r'[a-z]\s*\*\s*\d+',
        r'=\s*[\d\-]+',
        # مصفوفات
        r'\[\s*\[.*\]\s*\]',
        r'matrix|مصفوفة|det|محدد',
        # أعداد مركبة
        r'i\s*[\+\-\*\/]|complex|مركب',
        # متسلسلات
        r'sum|∑|product|∏',
        # إحصاء
        r'mean|average|متوسط|variance|تباين|std|انحراف',
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return True, "رياضيات"
    
    # ============================================================
    # ⚡ كشف الفيزياء
    # ============================================================
    physics_patterns = [
        # قوانين أساسية
        r'f\s*=\s*m\s*\*?\s*a',
        r'v\s*=\s*d/t',
        r'p\s*=\s*m\s*\*?\s*v',
        r'e\s*=\s*m\s*\*?\s*c\^2',
        # وحدات
        r'newton|نيوتن|n',
        r'joule|جول|j',
        r'watt|واط|w',
        r'pascal|باسكال|pa',
        # مفاهيم
        r'force|قوة',
        r'mass|كتلة',
        r'acceleration|تسارع',
        r'velocity|سرعة',
        r'gravity|جاذبية|9\.8',
        r'light|ضوء|3e8|c\s*=',
        r'energy|طاقة',
        r'work|شغل',
        r'power|قدرة',
        r'pressure|ضغط',
        r'density|كثافة',
        r'wave|موجة|frequency|تردد',
        r'sound|صوت',
        r'electric|كهرباء|charge|شحنة',
        r'magnetic|مغناطيس|field|مجال',
        r'quantum|كم',
    ]
    
    for pattern in physics_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return True, "فيزياء"
    
    # ============================================================
    # 🔧 كشف الميكانيكا
    # ============================================================
    mechanics_patterns = [
        # إجهاد وانفعال
        r'stress|إجهاد',
        r'strain|انفعال',
        r'young|يونج|modulus|معامل',
        # عناصر ميكانيكية
        r'beam|عارضة',
        r'torque|عزم',
        r'gear|ترس',
        r'spring|نابض|زنبرك',
        r'pulley|بكرة',
        r'lever|رافعة',
        # حركة
        r'vibration|اهتزاز',
        r'fatigue|كلل',
        r'fluid|مائع',
        r'pump|مضخة',
        r'turbine|عنفة',
        r'piston|مكبس',
        r'cylinder|أسطوانة',
        # ديناميكا
        r'kinematics|حركيات',
        r'dynamics|ديناميكا',
        r'statics|ستاتيكا',
        r'equilibrium|توازن',
    ]
    
    for pattern in mechanics_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return True, "ميكانيكا"
    
    # ============================================================
    # 💡 كشف الكهرباء والإلكترونيات
    # ============================================================
    electrical_patterns = [
        # قوانين أساسية
        r'v\s*=\s*i\s*\*?\s*r',
        r'p\s*=\s*v\s*\*?\s*i',
        # وحدات
        r'ohm|أوم',
        r'volt|فولت|v',
        r'amp|أمبير|a',
        r'farad|فاراد|f',
        r'henry|هنري|h',
        # عناصر
        r'resistor|مقاومة',
        r'capacitor|مكثف',
        r'inductor|ملف',
        r'diode|دايود',
        r'transistor|ترانزستور',
        r'op[- ]?amp|مكبر',
        # دوائر
        r'circuit|دائرة',
        r'arduino|raspberry',
        r'sensor|حساس|مستشعر',
        r'led|ضوء',
        r'power supply|مصدر طاقة',
        r'battery|بطارية',
        # إشارات
        r'frequency|تردد',
        r'filter|مرشح',
        r'amplifier|مضخم',
        r'digital|رقمي',
        r'analog|تناظري',
        r'signal|إشارة',
        r'pwm|تعديل',
    ]
    
    for pattern in electrical_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return True, "كهرباء وإلكترونيات"
    
    # ============================================================
    # 🤖 كشف PLC والمحركات
    # ============================================================
    plc_patterns = [
        # PLC
        r'plc',
        r'ladder|سلم',
        r'logic|منطق',
        # محركات
        r'motor|محرك',
        r'servo|سيرفو',
        r'stepper|ستبير',
        r'actuator|مشغل',
        # تحكم
        r'control|تحكم',
        r'pid',
        r'feedback|تغذية عكسية',
        # صناعة
        r'industrial|صناعي',
        r'automation|أتمتة',
        r'conveyor|ناقل',
        r'robotics|روبوت',
        r'scada',
        r'hmi',
        # حساسات
        r'encoder|مشفّر',
        r'proximity|قرب',
    ]
    
    for pattern in plc_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            return True, "PLC ومحركات"
    
    return False, None

# ============================================================
# 🤖 دوال الذكاء الاصطناعي (مع التبديل التلقائي)
# ============================================================

def ask_gemini(question: str) -> Optional[str]:
    """Gemini مع Code Execution"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        import google.generativeai as genai
        from google.generativeai.types import Tool
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        code_execution_tool = Tool(
            function_declarations=[{
                "name": "execute_python",
                "description": "Execute Python code for mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }]
        )
        
        model = genai.GenerativeModel(
            model_name='models/gemini-2.0-flash-001',
            tools=[code_execution_tool]
        )
        
        logger.info(f"Sending question to Gemini: {question[:100]}...")
        
        response = model.generate_content(
            question,
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 4096
            }
        )
        
        return response.text
        
    except ImportError:
        logger.error("google-generativeai not installed")
        return None
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        return None

def ask_deepseek(question: str) -> Optional[str]:
    """DeepSeek مع Tool Calling"""
    if not DEEPSEEK_API_KEY:
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        
        tools = [{
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute Python code for calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }
        }]
        
        logger.info(f"Sending question to DeepSeek: {question[:100]}...")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "أنت مساعد هندسي متخصص. استخدم Python للحسابات."},
                {"role": "user", "content": question}
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        logger.error("openai not installed")
        return None
    except Exception as e:
        logger.error(f"DeepSeek error: {str(e)}")
        return None

def ask_openrouter(question: str) -> Optional[str]:
    """OpenRouter"""
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        logger.info(f"Sending question to OpenRouter: {question[:100]}...")
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "أنت مساعد هندسي متخصص."},
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        logger.error("openai not installed")
        return None
    except Exception as e:
        logger.error(f"OpenRouter error: {str(e)}")
        return None

# ============================================================
# 🎯 نظام التبديل التلقائي (بدون أن يشعر المستخدم)
# ============================================================

def ask_ai_smart(question: str) -> Optional[str]:
    """
    تجربة APIs بالترتيب: Gemini → DeepSeek → OpenRouter
    بدون أن يشعر المستخدم بأي أخطاء
    """
    # قائمة APIs بالترتيب
    apis = [
        (ask_gemini, "Gemini"),
        (ask_deepseek, "DeepSeek"),
        (ask_openrouter, "OpenRouter")
    ]
    
    for api_func, api_name in apis:
        try:
            logger.info(f"Trying {api_name}...")
            result = api_func(question)
            if result and "خطأ" not in result and "⚠️" not in result:
                logger.info(f"✅ {api_name} succeeded")
                return result
        except Exception as e:
            logger.error(f"{api_name} failed: {str(e)}")
            continue
    
    return None

# ============================================================
# 🎯 المسارات الرئيسية
# ============================================================

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    """API الإجابة على الأسئلة"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "طلب غير صالح"}), 400
        
        question = data.get('question', '').strip()
        language = data.get('language', 'ar')
        
        if not question:
            return jsonify({"success": False, "error": "السؤال فارغ"}), 400
        
        # 1️⃣ كشف المجال (بدون كلمات مفتاحية)
        allowed, domain = detect_domain(question)
        
        # 2️⃣ إذا كان خارج المجال → رسالة مناسبة
        if not allowed:
            return jsonify({
                "success": False,
                "error": "❌ هذا السؤال خارج نطاق التطبيق. التطبيق متخصص في: الرياضيات، الفيزياء، الميكانيكا، الكهرباء، الإلكترونيات، المحركات، PLC",
                "domain_error": True
            })
        
        # 3️⃣ تجربة APIs بالترتيب (التبديل التلقائي)
        answer = ask_ai_smart(question)
        
        # 4️⃣ إذا فشلت كل APIs → رسالة عامة
        if not answer:
            return jsonify({
                "success": False,
                "error": "❌ عذراً، لم نتمكن من الإجابة على سؤالك حالياً. الرجاء المحاولة لاحقاً.",
                "domain": domain
            })
        
        # 5️⃣ النجاح
        return jsonify({
            "success": True,
            "answer": answer,
            "domain": domain
        })
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": "❌ حدث خطأ غير متوقع. الرجاء المحاولة لاحقاً."
        }), 500

@app.route('/api/help', methods=['GET'])
def get_help():
    """الحصول على المساعدة"""
    language = request.args.get('lang', 'ar')
    return jsonify({
        "help": "📝 طريقة الاستخدام:\nاكتب أي سؤال في الرياضيات، الفيزياء، الميكانيكا، الكهرباء، الإلكترونيات، المحركات، أو PLC وسيقوم التطبيق بالإجابة مع شرح مفصل.",
        "about": "🚀 تطبيق المساعد الهندسي v3.0 - يدعم 6 لغات و 7 مجالات هندسية."
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """الحالة (للمطور)"""
    return jsonify({
        "status": "running",
        "version": "3.0"
    })

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 MECHATRONICS ASSISTANT v3.0 - جاهز للتشغيل")
    print("="*70)
    print("📝 المستخدم يرى فقط الإجابات - لا أخطاء تقنية")
    print("🔄 التبديل بين APIs تلقائي (Gemini → DeepSeek → OpenRouter)")
    print("🧠 كشف المجال ذكي (بدون كلمات مفتاحية)")
    print("="*70)
    print("🌐 http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
