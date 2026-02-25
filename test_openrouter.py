# test_openrouter.py
import os
import requests
from dotenv import load_dotenv

# ===== تحميل متغيرات البيئة =====
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise Exception("⚠️ لم يتم العثور على مفتاح OpenRouter في المتغيرات البيئية!")

# ===== اختبار الاتصال بـ OpenRouter =====
def test_openrouter(question="أوجد مشتقة sin(x)"):
    prompt = f"""أنت محلل رياضي دقيق. أعد JSON فقط لتحليل السؤال التالي:

السؤال: {question}
"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "أنت محلل رياضي دقيق. أعد JSON فقط."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    
    try:
        print("📡 جاري الاتصال بـ OpenRouter...")
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        if r.status_code == 200:
            result = r.json()['choices'][0]['message']['content']
            print("✅ استجابة OpenRouter:")
            print(result)
        else:
            print(f"❌ خطأ OpenRouter: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"🔥 خطأ أثناء الاتصال بـ OpenRouter: {e}")

# ===== تشغيل الاختبار =====
if __name__ == "__main__":
    test_openrouter()
