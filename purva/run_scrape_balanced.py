from __future__ import annotations

import argparse
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii, normalize
from .collect.wordpress import WordpressCollector


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
    ap.add_argument("--total", type=int, default=4000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src = cfg["source"]

    collector = WordpressCollector(
        source_name=src["source_name"],
        base_url=src["base_url"],
        link_selector=src["link_selector"],
        content_selector=src["content_selector"],
        user_agent=cfg["user_agent"],
        category_paths=src.get("category_paths"),
        category_exclude=src.get("category_exclude"),
        request_delay=src.get("request_delay", 2.0),
        max_pages_per_category=src.get("max_pages_per_category", 50),
        respect_robots=src.get("respect_robots", True),
    )

    cats = collector.discover_categories()
    if not cats:
        print("no categories discovered; aborting")
        return
    n = len(cats)
    quota = args.total // n
    print(f"\n{n} categories discovered, quota = {quota} sentences each "
          f"(target {args.total})\n")
    for c in cats:
        print(f"  - {collector.category_label(c)}")
    print()

    per_cat = defaultdict(int)
    start = time.time()

    with JsonlWriter(cfg["output"]) as writer:
        for cat in cats:
            label = collector.category_label(cat)
            if per_cat[label] >= quota:
                continue
            for doc in collector.iter_category_articles(cat):
                if per_cat[label] >= quota:
                    break
                for sent in process_text(doc.text):
                    if per_cat[label] >= quota:
                        break
                    rec = Record(
                        id=content_hash(sent),
                        raw_text=sent,
                        cleaned_text=sent,
                        source_url=doc.url,
                        source_name=doc.meta["source"],
                        scrape_timestamp=datetime.now(timezone.utc).isoformat(),
                        category=label,
                    )
                    if writer.add(rec):
                        per_cat[label] += 1
            print(f"  [{label}] {per_cat[label]}/{quota}"
                  f"{' (exhausted)' if per_cat[label] < quota else ' done'}")

        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- scrape summary ---")
    print(f"categories      : {n}")
    print(f"quota each       : {quota}")
    print(f"total kept       : {total}/{args.total}")
    print(f"dupes skipped    : {dupes}")
    for label, count in sorted(per_cat.items()):
        flag = "" if count >= quota else "  <-- short"
        print(f"  {label:24s}: {count}{flag}")
    short = [l for l, c in per_cat.items() if c < quota]
    if short:
        print(f"\n{len(short)} categories under quota; redistribute if desired")
    print(f"elapsed          : {elapsed:.0f}s")
    print(f"output           : {cfg['output']}")


if __name__ == "__main__":
    main()