import os
import requests
import json

# تحميل مفتاح OpenRouter من متغير البيئة
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise Exception("⚠️ لم يتم العثور على مفتاح OpenRouter في المتغيرات البيئية!")

def ask_openrouter(question):
    prompt = f"""أنت محلل رياضي. حوّل أي سؤال إلى JSON لصيغة SymPy. أعد JSON فقط.

السؤال: {question}

أمثلة JSON:
{{"type": "solve", "expression": "x**2 + 5*x + 6", "variable": "x"}}
{{"type": "diff", "expression": "sin(2*x)", "variable": "x", "order": 1}}
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
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if r.status_code == 200:
        content = r.json()['choices'][0]['message']['content']
        try:
            parsed = json.loads(content)
        except:
            parsed = content
        return parsed
    else:
        return f"❌ خطأ OpenRouter: {r.status_code} - {r.text}"

if __name__ == "__main__":
    question = input("اكتب السؤال الرياضي: ")
    result = ask_openrouter(question)
    print("🔹 استجابة OpenRouter:")
    print(result)
