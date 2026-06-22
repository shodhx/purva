from purva.lid.env import load_env
load_env()
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="ओहिजा, अफगानिस्तान खातिर ई मुकाबला बहुते महत्वपूर्ण बा",
    config=types.GenerateContentConfig(
        system_instruction='Reply ONLY with compact JSON like {"label":"bhojpuri","confidence":0.9,"reason":"short"}',
        temperature=0,
        max_output_tokens=200,
        response_mime_type="application/json",
    ),
)
print("TEXT:", repr(resp.text))
print("---")
print("FINISH:", resp.candidates[0].finish_reason if resp.candidates else "no candidates")
print("---")
try:
    print("PARTS:", resp.candidates[0].content.parts)
except Exception as e:
    print("PARTS error:", e)