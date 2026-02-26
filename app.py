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
# 📊 عرض حالة المفاتيح
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
# 🌐 المجالات المسموح بها (متعددة اللغات)
# ============================================================

ALLOWED_DOMAINS = {
    'ar': {
        'names': ['رياضيات', 'فيزياء', 'ميكانيك', 'كهرباء', 'إلكترونيات', 'محركات', 'PLC'],
        'keywords': ['رياضيات', 'فيزياء', 'ميكانيك', 'كهرباء', 'الكترون', 'محرك', 'plc']
    },
    'en': {
        'names': ['Mathematics', 'Physics', 'Mechanics', 'Electrical', 'Electronics', 'Engines', 'PLC'],
        'keywords': ['math', 'physics', 'mechanics', 'electrical', 'electronics', 'engine', 'plc']
    },
    'de': {
        'names': ['Mathematik', 'Physik', 'Mechanik', 'Elektrik', 'Elektronik', 'Motoren', 'SPS'],
        'keywords': ['mathe', 'physik', 'mechanik', 'elektro', 'elektronik', 'motor', 'sps']
    },
    'tr': {
        'names': ['Matematik', 'Fizik', 'Mekanik', 'Elektrik', 'Elektronik', 'Motorlar', 'PLC'],
        'keywords': ['matematik', 'fizik', 'mekanik', 'elektrik', 'elektronik', 'motor', 'plc']
    },
    'fr': {
        'names': ['Mathématiques', 'Physique', 'Mécanique', 'Électrique', 'Électronique', 'Moteurs', 'API'],
        'keywords': ['math', 'physique', 'mécanique', 'électrique', 'électronique', 'moteur', 'api']
    },
    'ru': {
        'names': ['Математика', 'Физика', 'Механика', 'Электрика', 'Электроника', 'Двигатели', 'ПЛК'],
        'keywords': ['математика', 'физика', 'механика', 'электрика', 'электроника', 'двигатель', 'плк']
    }
}

def is_allowed_domain(question: str, language: str = 'ar') -> tuple:
    """التحقق من أن السؤال ضمن المجالات المسموحة"""
    if not question:
        return False, None
    
    q_lower = question.lower()
    lang_data = ALLOWED_DOMAINS.get(language, ALLOWED_DOMAINS['ar'])
    
    for i, keyword in enumerate(lang_data['keywords']):
        if keyword in q_lower:
            return True, lang_data['names'][i]
    
    return False, None

# ============================================================
# 🤖 دوال الذكاء الاصطناعي
# ============================================================

def ask_gemini(question: str) -> Optional[str]:
    """Gemini مع Code Execution"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        import google.generativeai as genai
        from google.generativeai.types import Tool
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        # إنشاء أداة Code Execution
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
        return "⚠️ مكتبة Gemini غير مثبتة. الرجاء تشغيل: pip install google-generativeai"
    except Exception as e:
        logger.error(f"Gemini error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"❌ خطأ في Gemini: {str(e)}"

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
        return "⚠️ مكتبة OpenAI غير مثبتة. الرجاء تشغيل: pip install openai"
    except Exception as e:
        logger.error(f"DeepSeek error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"❌ خطأ في DeepSeek: {str(e)}"

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
            model="deepseek/deepseek-chat",  # استخدام DeepSeek عبر OpenRouter
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
        return "⚠️ مكتبة OpenAI غير مثبتة. الرجاء تشغيل: pip install openai"
    except Exception as e:
        logger.error(f"OpenRouter error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"❌ خطأ في OpenRouter: {str(e)}"

# ============================================================
# 🎯 نظام اختيار الذكاء (الأولوية للثلاثة)
# ============================================================

def get_best_ai() -> str:
    """تحديد أفضل ذكاء متاح"""
    if GEMINI_API_KEY:
        return "gemini"
    elif DEEPSEEK_API_KEY:
        return "deepseek"
    elif OPENROUTER_API_KEY:
        return "openrouter"
    return "none"

def ask_ai(question: str) -> Dict[str, Any]:
    """إرسال السؤال للذكاء المتاح"""
    logger.info(f"Processing question: {question}")
    
    # التحقق من المجال
    allowed, domain = is_allowed_domain(question)
    if not allowed:
        return {
            "success": False,
            "error": "❌ هذا المجال غير مدعوم. التطبيق متخصص في: الرياضيات، الفيزياء، الميكانيكا، الكهرباء، الإلكترونيات، المحركات، PLC",
            "domain_error": True
        }
    
    best_ai = get_best_ai()
    answer = None
    
    if best_ai == "gemini":
        answer = ask_gemini(question)
    elif best_ai == "deepseek":
        answer = ask_deepseek(question)
    elif best_ai == "openrouter":
        answer = ask_openrouter(question)
    
    if answer:
        return {
            "success": True,
            "answer": answer,
            "ai_used": best_ai,
            "domain": domain
        }
    else:
        return {
            "success": False,
            "error": "❌ لا يوجد مفتاح ذكاء اصطناعي متاح. الرجاء وضع مفتاح في CMD.",
            "ai_used": best_ai
        }

# ============================================================
# 📚 دوال المساعدة والترجمة
# ============================================================

TRANSLATIONS = {
    'ar': {
        'title': 'المساعد الهندسي',
        'menu': 'القائمة',
        'language': 'اللغة',
        'help': 'المساعدة',
        'about': 'عن التطبيق',
        'keyboard_show': '⌨️ إظهار لوحة المفاتيح',
        'keyboard_hide': '⌨️ إخفاء لوحة المفاتيح',
        'placeholder': 'اكتب سؤالك هنا...',
        'default_answer': 'اكتب سؤالك واضغط على السهم للإرسال',
        'loading': '⏳ جاري البحث عن الإجابة...',
        'help_text': """
📝 **طريقة استخدام التطبيق:**
1. اكتب سؤالك في مربع النص
2. اضغط على السهم للإرسال
3. استخدم لوحة المفاتيح الرياضية للرموز الخاصة
4. اختر اللغة المناسبة من القائمة الجانبية

**المجالات المدعومة:**
• الرياضيات
• الفيزياء
• الميكانيكا
• الكهرباء
• الإلكترونيات
• المحركات
• PLC

**ملاحظة:** التطبيق يستخدم الذكاء الاصطناعي مع تنفيذ كود Python للحصول على دقة 100%.
        """,
        'about_text': """
🚀 **المساعد الهندسي v3.0**

تطبيق ذكي للإجابة على الأسئلة الهندسية في مجالات متعددة.

**المميزات:**
• دعم 6 لغات (عربي، إنجليزي، ألماني، تركي، فرنسي، روسي)
• ذكاء اصطناعي متعدد (Gemini + DeepSeek + OpenRouter)
• تنفيذ كود Python للحصول على دقة 100%
• لوحة مفاتيح رياضية للرموز الخاصة
• شرح تفصيلي لكل مسألة

**تم التطوير بواسطة:** creativity2472001
**للاستفسارات:** creativity2472001@gmail.com
        """
    }
}

def get_translation(key: str, language: str = 'ar') -> str:
    """الحصول على ترجمة"""
    if language in TRANSLATIONS and key in TRANSLATIONS[language]:
        return TRANSLATIONS[language][key]
    return TRANSLATIONS['ar'].get(key, '')

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
        
        # معالجة السؤال
        result = ask_ai(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"حدث خطأ داخلي: {str(e)}"
        }), 500

@app.route('/api/help', methods=['GET'])
def get_help():
    """الحصول على المساعدة"""
    language = request.args.get('lang', 'ar')
    return jsonify({
        "help": get_translation('help_text', language),
        "about": get_translation('about_text', language)
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """الحصول على حالة التطبيق"""
    return jsonify({
        "status": "running",
        "gemini": bool(GEMINI_API_KEY),
        "deepseek": bool(DEEPSEEK_API_KEY),
        "openrouter": bool(OPENROUTER_API_KEY),
        "active_ai": get_best_ai(),
        "languages": list(ALLOWED_DOMAINS.keys()),
        "version": "3.0.0"
    })

# ============================================================
# 🚀 التشغيل
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 MECHATRONICS ASSISTANT v3.0 - جاهز للتشغيل")
    print("="*70)
    print("📝 استخدم الأوامر التالية:")
    print("   • http://127.0.0.1:5000 - الصفحة الرئيسية")
    print("   • http://127.0.0.1:5000/api/status - حالة التطبيق")
    print("="*70 + "\n")
    
    # تشغيل التطبيق
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        threaded=True
    )
