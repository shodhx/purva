from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from .schema import Record, JsonlWriter, content_hash
from .clean import split_sentences, clean_sentence
from .collect.news import NewsCollector

COLLECTORS = {"news": NewsCollector}


def build_collector(cfg: dict):
    src = cfg["source"]
    kind = src.pop("kind")
    if kind not in COLLECTORS:
        raise ValueError(f"unknown source kind {kind!r}; have {list(COLLECTORS)}")
    src["user_agent"] = cfg["user_agent"]
    return COLLECTORS[kind](**src)


def run(cfg: dict):
    collector = build_collector(cfg)
    out_path = cfg["output"]
    target = cfg.get("target_sentences", 500)
    min_chars = cfg.get("min_sentence_chars", 15)

    t0 = time.time()
    docs = raw_sents = kept = 0
    with JsonlWriter(out_path) as w:
        for doc in collector.iter_documents():
            docs += 1
            for raw in split_sentences(doc.text):
                raw_sents += 1
                cleaned = clean_sentence(raw, min_chars=min_chars)
                if cleaned is None:
                    continue
                rec = Record(
                    id=content_hash(cleaned),
                    raw_text=raw,
                    cleaned_text=cleaned,
                    source_url=doc.url,
                    source_name=collector.name,
                )
                if w.add(rec):
                    kept += 1
                if kept >= target:
                    break
            if kept >= target:
                break
        written, dupes = w.written, w.skipped_dupes

    elapsed = time.time() - t0
    print("\n--- scrape summary ---")
    print(f"source            : {collector.name}")
    print(f"articles fetched  : {docs}")
    print(f"raw sentences     : {raw_sents}")
    print(f"kept (new)        : {written}")
    print(f"dropped as dupes  : {dupes}")
    print(f"output            : {out_path}")
    print(f"elapsed           : {elapsed:.1f}s "
          f"({elapsed / max(written,1):.2f}s per kept sentence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg)


if __name__ == "__main__":
    main()