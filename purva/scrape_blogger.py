from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii
from .collect.blogger import BloggerCollector


def process_text(raw: str) -> list[str]:
    out = []
    for s in split_sentences(strip_pii(raw)):
        c = clean_sentence(s)
        if c:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src = cfg["source"]

    collector = BloggerCollector(
        source_name=src["source_name"],
        base_url=src["base_url"],
        user_agent=cfg["user_agent"],
        request_delay=src.get("request_delay", 1.0),
        batch=src.get("batch", 150),
    )

    print(f"walking blogger feed {src['base_url']}\n")
    start = time.time()
    kept = 0
    posts = 0

    with JsonlWriter(cfg["output"]) as writer:
        for doc in collector.iter_documents():
            posts += 1
            for sent in process_text(doc.text):
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
            if posts % 100 == 0:
                print(f"  {posts} posts, {kept} sentences")
            if args.limit and posts >= args.limit:
                break

        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- blogger scrape summary ---")
    print(f"posts processed  : {posts}")
    print(f"sentences kept    : {total}")
    print(f"dupes skipped     : {dupes}")
    print(f"elapsed           : {elapsed:.0f}s")
    print(f"output            : {cfg['output']}")


if __name__ == "__main__":
    main()