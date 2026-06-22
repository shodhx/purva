from __future__ import annotations

import json
import re
import time

from google import genai
from google.genai import types

from .env import require

MODEL = "gemini-2.5-flash"

SYSTEM = (
    "You are a language identifier for Eastern Indo-Aryan languages written in "
    "Devanagari. Distinguish Bhojpuri from Hindi. Bhojpuri markers: copula बा/बाड़ें/"
    "बाड़ू (Hindi uses है/हैं), participles कइल/गइल/भइल, genitive के (Hindi का/की/के), "
    "एह/ओह forms, and inline Latin s verb endings. Reply ONLY with compact JSON: "
    '{"label":"bhojpuri|hindi|other","confidence":0.0-1.0,"reason":"<=8 words"}'
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> dict:
    m = _JSON.search(text)
    if not m:
        return {"label": "other", "confidence": 0.0, "reason": "unparseable"}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"label": "other", "confidence": 0.0, "reason": "unparseable"}
    label = str(d.get("label", "other")).lower().strip()
    if label not in ("bhojpuri", "hindi", "other"):
        label = "other"
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"label": label, "confidence": max(0.0, min(1.0, conf)),
            "reason": str(d.get("reason", ""))[:80]}


class GeminiJudge:
    name = "gemini"

    def __init__(self, delay: float = 4.0, max_retries: int = 5):
        self.client = genai.Client(api_key=require("GEMINI_API_KEY"))
        self.delay = delay
        self.max_retries = max_retries
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0,
            max_output_tokens=80,
            response_mime_type="application/json",
        )

    def judge(self, sentence: str) -> dict:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=MODEL,
                    contents=sentence,
                    config=self.config,
                )
                time.sleep(self.delay)
                return _parse(resp.text or "")
            except Exception as e:
                msg = str(e).lower()
                wait = min(2 ** attempt, 30)
                if "rate" in msg or "429" in msg or "resource" in msg or "quota" in msg:
                    print(f"  [gemini] rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  [gemini] error: {e}")
                time.sleep(self.delay)
                return {"label": "other", "confidence": 0.0, "reason": f"error: {str(e)[:40]}"}
        return {"label": "other", "confidence": 0.0, "reason": "retries exhausted"}