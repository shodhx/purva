from __future__ import annotations

import re
import unicodedata

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?:(?:\+?91[\-\s]?)|0)?[6-9]\d{9}\b")
_HANDLE = re.compile(r"(?<!\w)@\w{2,}")
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENT_SPLIT = re.compile(r"[।॥?!]+|\.(?:\s)|\n+")
_WS = re.compile(r"\s+")

_WIKILINK = re.compile(r"\[\[[^\]]*\]\]")
_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/?>", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_BRACKET_MEDIA = re.compile(r"\b(?:File|Image|चित्र|फाइल):[^\]\|]*", re.IGNORECASE)

_BOILERPLATE = [
    re.compile(r"ग्रेगरियन कैलेंडर में साल के"),
    re.compile(r"साल के खतम होखे में अबहिन"),
    re.compile(r"^तिहुआर,?\s*छुट्टी"),
    re.compile(r"^इहो देखल जाय"),
    re.compile(r"बाहरी कड़ी"),
    re.compile(r"^घटना\s+जनम\s+निधन"),
    re.compile(r"^संदर्भ$"),
    re.compile(r"thumb|right|left|px\b"),
]


def strip_markup(text: str) -> str:
    text = _REF.sub(" ", text)
    text = _TEMPLATE.sub(" ", text)
    text = _WIKILINK.sub(" ", text)
    text = _BRACKET_MEDIA.sub(" ", text)
    text = _TAG.sub(" ", text)
    return text


def strip_pii(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _PHONE.sub(" ", text)
    text = _HANDLE.sub(" ", text)
    return text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _WS.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    text = strip_markup(text)
    parts = _SENT_SPLIT.split(text)
    return [normalize(p) for p in parts if normalize(p)]


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)


def is_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in _BOILERPLATE)


def clean_sentence(raw: str, min_chars: int = 15) -> str | None:
    cleaned = normalize(strip_pii(strip_markup(raw)))
    if len(cleaned) < min_chars:
        return None
    if not has_devanagari(cleaned):
        return None
    if is_boilerplate(cleaned):
        return None
    return cleaned