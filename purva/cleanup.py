from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .clean import (
    clean_sentence, split_sentences, strip_pii, strip_markup,
    has_devanagari, is_boilerplate, normalize,
)

EXCLUDE_FILES = {
    "reddit_romanised_raw.jsonl",
    "purva_pilot_candidates.jsonl",
    "purva_pilot_lid.jsonl",
    "hf_speech_transcripts.jsonl",
    "kavitakosh_geet.jsonl",
    "corpus_merged.jsonl",
    "corpus_clean.jsonl",
    "corpus_rejected.jsonl",
}

REGISTER = {
    "khabarbhojpuri": "news",
    "bhojpurinews": "news",
    "bhojpuri_wikipedia": "encyclopedic",
    "bhojpurisahityasarita": "literary",
    "jogira": "literary",
    "kavitakosh": "verse",
    "kavitakosh_geet": "verse",
    "sirijan": "literary",
    "openbooksonline": "literary",
    "bhojpuri_blogspot": "opinion",
    "parichaydas_bihari": "literary",
    "pandiji": "commentary",
}
LICENSE = {
    "bhojpuri_wikipedia": "redistributable_cc_by_sa",
}
DEFAULT_LICENSE = "pointer_only"

VERSE_CAT = re.compile(r"कविता|गीत|गजल|ग़ज़ल|kavita|geet|gazal|ghazal|poem|verse", re.I)
VERSE_SOURCES = {"kavitakosh", "kavitakosh_geet"}

JOGIRA_GENRES = [
    ("laghu-katha", "लघुकथा"), ("laghu", "लघुकथा"), ("kahani", "कहानी"),
    ("katha", "कथा"), ("kavita", "कविता"), ("gazal", "गजल"), ("ghazal", "गजल"),
    ("geet", "गीत"), ("proverb", "कहावत"), ("paheli", "पहेली"),
    ("bujhauwal", "पहेली"), ("rachna", "रचना"), ("composition", "रचना"),
    ("sahity", "साहित्य"),
]

TITLE_CHROME = re.compile(
    r"मुख्य पेज|मुख्य पन्ना|Bhojpuria Lokraag|Launda Naach|"
    r"^\s*\(?मुख्य|blog-post|Read more|Older Posts|Newer Posts"
)


def detect_script(text: str) -> str:
    dev = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    lat = sum(1 for ch in text if ch.isalpha() and "a" <= ch.lower() <= "z")
    t = dev + lat
    if t == 0:
        return "other"
    if dev / t >= 0.9:
        return "devanagari"
    if lat / t >= 0.9:
        return "latin"
    return "mixed"


def norm_key(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[\s।॥.,!?\-–—:;\"'()]+", "", t)
    return t


def jogira_genre(url: str) -> str | None:
    slug = url.rstrip("/").rsplit("/", 1)[-1].lower()
    for key, genre in JOGIRA_GENRES:
        if key in slug:
            return genre
    return None


def enrich(rec: dict) -> dict:
    src = rec.get("source_name", "")
    rec["register"] = REGISTER.get(src, "unknown")
    rec["license_class"] = LICENSE.get(src, DEFAULT_LICENSE)
    rec["script"] = detect_script(rec.get("cleaned_text", ""))
    if src == "jogira" and rec.get("category") in (None, "sitemap"):
        g = jogira_genre(rec.get("source_url", ""))
        if g:
            rec["category"] = g
    cat = rec.get("category") or ""
    rec["text_type"] = "verse" if (VERSE_CAT.search(cat) or src in VERSE_SOURCES) else "prose"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/corpus_clean.jsonl")
    ap.add_argument("--rejects", default="data/corpus_rejected.jsonl")
    ap.add_argument("--min-chars", type=int, default=15)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = [p for p in sorted(data_dir.glob("*.jsonl")) if p.name not in EXCLUDE_FILES]
    print("source files:")
    for f in files:
        n = sum(1 for _ in open(f, encoding="utf-8"))
        print(f"  {f.name}: {n}")

    rejects = open(args.rejects, "w", encoding="utf-8")
    def reject(rec, reason):
        rec["reject_reason"] = reason
        rejects.write(json.dumps(rec, ensure_ascii=False) + "\n")

    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    kept: list[dict] = []
    counts = Counter()

    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts["gross"] += 1
            text = rec.get("cleaned_text") or rec.get("raw_text") or ""

            if rec.get("id") in seen_exact:
                counts["exact_dupe"] += 1
                reject(rec, "exact_dupe")
                continue

            recl = clean_sentence(text, min_chars=args.min_chars)
            if recl is None:
                counts["filtered"] += 1
                reject(rec, "filtered_junk_or_short")
                continue
            if TITLE_CHROME.search(recl):
                counts["title_chrome"] += 1
                reject(rec, "title_chrome")
                continue

            nk = norm_key(recl)
            if nk in seen_near:
                counts["near_dupe"] += 1
                reject(rec, "near_dupe")
                continue

            seen_exact.add(rec.get("id"))
            seen_near.add(nk)
            rec["cleaned_text"] = recl
            kept.append(enrich(rec))
            counts["kept"] += 1

    rejects.close()
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== cleanup summary ===")
    print(f"gross in         : {counts['gross']}")
    print(f"exact dupes      : {counts['exact_dupe']}")
    print(f"near dupes       : {counts['near_dupe']}")
    print(f"filtered junk    : {counts['filtered']}")
    print(f"title chrome     : {counts['title_chrome']}")
    print(f"KEPT (clean)     : {counts['kept']}")

    by_source = Counter(r["source_name"] for r in kept)
    by_register = Counter(r["register"] for r in kept)
    by_script = Counter(r["script"] for r in kept)
    by_type = Counter(r["text_type"] for r in kept)
    print("\nby source:")
    for k, v in by_source.most_common():
        print(f"  {k:24s}: {v}")
    print("\nby register:")
    for k, v in by_register.most_common():
        print(f"  {k:16s}: {v}")
    print("\nby text_type:", dict(by_type))
    print("by script   :", dict(by_script))
    print(f"\nclean corpus  -> {args.out}")
    print(f"rejected pile -> {args.rejects}")


if __name__ == "__main__":
    main()
