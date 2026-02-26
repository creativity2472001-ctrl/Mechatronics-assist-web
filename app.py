from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# ============================================================
# 🔑 المفاتيح
# ============================================================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') or os.environ.get('OPENROUTER_API_KEY')

# تهيئة Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('models/gemini-3-flash-preview')
    print("✅ Gemini متصل")

# ============================================================
# 🎯 المجالات المسموح بها
# ============================================================
ALLOWED_DOMAINS = [
    "رياضيات", "Mathematics",
    "فيزياء", "Physics",
    "ميكانيك", "Mechanics",
    "كهرباء", "Electrical",
    "إلكترونيات", "Electronics",
    "محركات", "Engines", "Motors",
    "PLC", "Programmable Logic Controller",
    "دوائر كهربائية", "Electrical Circuits",
    "دوائر إلكترونية", "Electronic Circuits",
    "هندسة", "Engineering"
]

def is_allowed_domain(question):
    """التحقق من أن السؤال ضمن المجالات المسموحة"""
    q_lower = question.lower()
    for domain in ALLOWED_DOMAINS:
        if domain.lower() in q_lower:
            return True, domain
    return False, None

# ============================================================
# 🤖 الدوال الذكية
# ============================================================
def ask_gemini(question, language='ar'):
    """سؤال Gemini"""
    try:
        prompt = f"أجب عن هذا السؤال باللغة {language}:\n{question}"
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"خطأ في Gemini: {str(e)}"

def ask_openrouter(question, language='ar'):
    """سؤال OpenRouter"""
    if not OPENROUTER_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": f"أجب باللغة {language}"},
            {"role": "user", "content": question}
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    except:
        return None

# ============================================================
# 🎯 المسار الرئيسي
# ============================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '').strip()
    language = data.get('language', 'ar')
    
    if not question:
        return jsonify({
            "success": False,
            "error": "السؤال فارغ"
        })
    
    # التحقق من المجال
    allowed, domain = is_allowed_domain(question)
    if not allowed:
        return jsonify({
            "success": True,
            "answer": get_translation("domain_error", language),
            "domain_error": True
        })
    
    # المحاولة بـ Gemini أولاً
    answer = ask_gemini(question, language)
    
    # إذا فشل Gemini، جرب OpenRouter
    if "خطأ" in answer and OPENROUTER_API_KEY:
        answer = ask_openrouter(question, language)
    
    return jsonify({
        "success": True,
        "answer": answer,
        "domain": domain
    })

@app.route('/api/help', methods=['GET'])
def get_help():
    language = request.args.get('lang', 'ar')
    return jsonify({
        "help": get_translation("help", language),
        "about": get_translation("about", language)
    })

# ============================================================
# 🌐 الترجمة
# ============================================================
def get_translation(key, lang='ar'):
    translations = {
        "help": {
            "ar": "📝 طريقة استخدام التطبيق:\n1. اكتب سؤالك في المربع\n2. اضغط على السهم للإرسال\n3. يمكنك استخدام لوحة المفاتيح الرياضية للرموز الخاصة\n4. اختر اللغة المناسبة من القائمة الجانبية\n\nالمجالات المدعومة: رياضيات، فيزياء، ميكانيك، كهرباء، إلكترونيات، محركات، PLC",
            "en": "📝 How to use:\n1. Type your question\n2. Press the arrow to send\n3. Use the math keyboard for special symbols\n4. Choose language from sidebar\n\nSupported fields: Mathematics, Physics, Mechanics, Electrical, Electronics, Engines, PLC",
            "de": "📝 Anleitung:\n1. Frage eingeben\n2. Pfeil drücken zum Senden\n3. Tastatur für Sonderzeichen\n4. Sprache auswählen\n\nUnterstützte Bereiche: Mathematik, Physik, Mechanik, Elektrik, Elektronik, Motoren, SPS",
            "tr": "📝 Kullanım:\n1. Sorunuzu yazın\n2. Göndermek için oka basın\n3. Özel semboller için tuş takımı\n4. Dili seçin\n\nDesteklenen alanlar: Matematik, Fizik, Mekanik, Elektrik, Elektronik, Motorlar, PLC",
            "fr": "📝 Utilisation:\n1. Tapez votre question\n2. Appuyez sur la flèche\n3. Clavier mathématique\n4. Choisissez la langue\n\nDomaines: Mathématiques, Physique, Mécanique, Électrique, Électronique, Moteurs, API",
            "ru": "📝 Использование:\n1. Введите вопрос\n2. Нажмите стрелку\n3. Математическая клавиатура\n4. Выберите язык\n\nОбласти: Математика, Физика, Механика, Электрика, Электроника, Двигатели, ПЛК"
        },
        "about": {
            "ar": "🚀 تطبيق المساعد الهندسي\n\nيحل مسائل في:\n• الرياضيات\n• الفيزياء\n• هندسة الميكانيك\n• هندسة الكهرباء\n• هندسة الإلكترونيات\n• المحركات\n• PLC\n\nيدعم عدة لغات ولوحة مفاتيح رياضية",
            "en": "🚀 Engineering Assistant\n\nSolves problems in:\n• Mathematics\n• Physics\n• Mechanical Engineering\n• Electrical Engineering\n• Electronics\n• Engines\n• PLC\n\nMulti-language support with math keyboard",
            "de": "🚀 Engineering-Assistent\n\nLöst Probleme in:\n• Mathematik\n• Physik\n• Mechanik\n• Elektrotechnik\n• Elektronik\n• Motoren\n• SPS\n\nMehrsprachig mit Mathe-Tastatur",
            "tr": "🚀 Mühendislik Asistanı\n\nÇözer:\n• Matematik\n• Fizik\n• Mekanik\n• Elektrik\n• Elektronik\n• Motorlar\n• PLC\n\nÇoklu dil desteği",
            "fr": "🚀 Assistant d'ingénierie\n\nRésout:\n• Mathématiques\n• Physique\n• Mécanique\n• Électrique\n• Électronique\n• Moteurs\n• API\n\nMultilingue avec clavier math",
            "ru": "🚀 Инженерный помощник\n\nРешает:\n• Математика\n• Физика\n• Механика\n• Электрика\n• Электроника\n• Двигатели\n• ПЛК\n\nМногоязычная поддержка"
        },
        "domain_error": {
            "ar": "❌ هذا المجال غير مدعوم. التطبيق متخصص في: الرياضيات، الفيزياء، الميكانيك، الكهرباء، الإلكترونيات، المحركات، PLC فقط",
            "en": "❌ Domain not supported. This app specializes in: Mathematics, Physics, Mechanics, Electrical, Electronics, Engines, PLC only",
            "de": "❌ Bereich nicht unterstützt. Diese App spezialisiert sich auf: Mathematik, Physik, Mechanik, Elektrik, Elektronik, Motoren, SPS",
            "tr": "❌ Alan desteklenmiyor. Bu uygulama şunlarda uzmanlaşmıştır: Matematik, Fizik, Mekanik, Elektrik, Elektronik, Motorlar, PLC",
            "fr": "❌ Domaine non supporté. Cette app est spécialisée en: Mathématiques, Physique, Mécanique, Électrique, Électronique, Moteurs, API",
            "ru": "❌ Область не поддерживается. Приложение специализируется на: Математика, Физика, Механика, Электрика, Электроника, Двигатели, ПЛК"
        }
    }
    return translations.get(key, {}).get(lang, translations.get(key, {}).get('ar', ''))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
