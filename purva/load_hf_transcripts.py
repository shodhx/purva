from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii, normalize, strip_markup


def process_text(raw: str) -> list[str]:
    norm = normalize(strip_pii(strip_markup(raw)))
    out = []
    for s in split_sentences(norm):
        c = clean_sentence(s)
        if c:
            out.append(c)
    return out


def load_fleurs_transcripts() -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [fleurs] datasets library not installed; skip")
        return []
    for cfg in ("bho_in", "bho", "bhojpuri"):
        try:
            print(f"  [fleurs] trying config '{cfg}'")
            ds = load_dataset("google/fleurs", cfg)
            texts = []
            for split in ds:
                for row in ds[split]:
                    t = row.get("transcription") or row.get("raw_transcription") or ""
                    if t.strip():
                        texts.append(t)
            print(f"  [fleurs] config '{cfg}' gave {len(texts)} transcripts")
            return texts
        except Exception as e:
            print(f"  [fleurs] config '{cfg}' failed: {str(e)[:80]}")
    print("  [fleurs] no working Bhojpuri config found; skip")
    return []


def load_rural_women_transcripts() -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [rural] datasets library not installed; skip")
        return []
    try:
        ds = load_dataset("ai4bharat/Rural_Women_Bhojpuri")
    except Exception as e:
        print(f"  [rural] load failed: {str(e)[:100]}")
        return []
    texts = []
    for split in ds:
        for row in ds[split]:
            if any("syn" in str(k).lower() for k in row.keys() if row.get(k) is True):
                continue
            t = (row.get("transcription") or row.get("transcript")
                 or row.get("text") or row.get("sentence") or "")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    print(f"  [rural] {len(texts)} transcripts")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/hf_speech_transcripts.jsonl")
    ap.add_argument("--source", choices=["fleurs", "rural", "both"], default="both")
    args = ap.parse_args()

    start = time.time()
    sources = []
    if args.source in ("fleurs", "both"):
        print("loading FLEURS Bhojpuri...")
        sources.append(("fleurs", load_fleurs_transcripts()))
    if args.source in ("rural", "both"):
        print("loading Rural_Women Bhojpuri...")
        sources.append(("rural_women", load_rural_women_transcripts()))

    kept = 0
    with JsonlWriter(args.output) as writer:
        for src_name, transcripts in sources:
            for raw in transcripts:
                for sent in process_text(raw):
                    rec = Record(
                        id=content_hash(sent),
                        raw_text=sent,
                        cleaned_text=sent,
                        source_url=f"huggingface:{src_name}",
                        source_name=f"hf_{src_name}",
                        scrape_timestamp=datetime.now(timezone.utc).isoformat(),
                        category="speech_transcript",
                    )
                    if writer.add(rec):
                        kept += 1
        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- hf transcript summary ---")
    for src_name, transcripts in sources:
        print(f"{src_name}: {len(transcripts)} raw transcripts")
    print(f"sentences kept   : {total}")
    print(f"dupes skipped    : {dupes}")
    print(f"elapsed          : {elapsed:.0f}s")
    print(f"output           : {args.output}")


if __name__ == "__main__":
    main()