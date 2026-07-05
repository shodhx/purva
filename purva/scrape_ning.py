from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii, normalize, strip_markup
from .collect.ning import NingCollector


def process_text(raw: str) -> list[str]:
    norm = normalize(strip_pii(strip_markup(raw)))
    out = []
    for s in split_sentences(norm):
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

    collector = NingCollector(
        source_name=src["source_name"],
        group_url=src["group_url"],
        user_agent=cfg["user_agent"],
        link_selector=src.get("link_selector", ".topic h3 a"),
        content_selector=src.get("content_selector", ".xg_user_generated p"),
        request_delay=src.get("request_delay", 2.0),
        max_pages=src.get("max_pages", 100),
        respect_robots=src.get("respect_robots", True),
    )

    print(f"walking ning group {src['group_url']}\n")
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
            if posts % 20 == 0:
                print(f"  {posts} posts, {kept} sentences")
            if args.limit and posts >= args.limit:
                break

        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- ning scrape summary ---")
    print(f"posts fetched    : {posts}")
    print(f"sentences kept    : {total}")
    print(f"dupes skipped     : {dupes}")
    print(f"elapsed           : {elapsed:.0f}s")
    print(f"output            : {cfg['output']}")


if __name__ == "__main__":
    main()