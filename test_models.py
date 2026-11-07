import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
c = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=20000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

for m in ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
          "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
          "gemini-2.5-flash-lite"]:
    try:
        r = c.models.generate_content(model=m, contents="Скажи привет одним словом")
        print("OK  ", m, "->", r.text.strip(), flush=True)
    except Exception as e:
        print("FAIL", m, "->", str(e)[:60], flush=True)