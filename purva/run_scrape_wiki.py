from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii, normalize
from .collect.wiki import WikiCollector


def process_text(raw: str) -> list[str]:
    norm = normalize(strip_pii(raw))
    out = []
    for s in split_sentences(norm):
        c = clean_sentence(s)
        if c:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--total", type=int, default=4476)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src = cfg["source"]

    collector = WikiCollector(
        source_name=src["source_name"],
        api_url=src["api_url"],
        user_agent=cfg["user_agent"],
        request_delay=src.get("request_delay", 0.5),
        mode=src.get("mode", "random"),
    )

    print(f"target {args.total} sentences from {src['api_url']}\n")
    start = time.time()
    kept = 0
    articles = 0

    with JsonlWriter(cfg["output"]) as writer:
        for doc in collector.iter_documents():
            if kept >= args.total:
                break
            articles += 1
            for sent in process_text(doc.text):
                if kept >= args.total:
                    break
                rec = Record(
                    id=content_hash(sent),
                    raw_text=sent,
                    cleaned_text=sent,
                    source_url=doc.url,
                    source_name=doc.meta["source"],
                    scrape_timestamp=datetime.now(timezone.utc).isoformat(),
                    category=doc.meta.get("category"),
                )
                if writer.add(rec):
                    kept += 1
            if articles % 25 == 0:
                print(f"  {articles} articles, {kept}/{args.total} sentences")

        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- wikipedia scrape summary ---")
    print(f"articles seen   : {articles}")
    print(f"sentences kept   : {total}/{args.total}")
    print(f"dupes skipped    : {dupes}")
    print(f"elapsed          : {elapsed:.0f}s")
    print(f"output           : {cfg['output']}")
    if total < args.total:
        print("\nfell short of target; wiki likely exhausted usable articles")


if __name__ == "__main__":
    main()