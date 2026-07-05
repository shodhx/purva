from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .schema import Record, content_hash, JsonlWriter
from .clean import clean_sentence, split_sentences, strip_pii, normalize, strip_markup
from .collect.wordpress import WordpressCollector


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

    collector = WordpressCollector(
        source_name=src["source_name"],
        base_url=src["base_url"],
        link_selector=src.get("link_selector", "a"),
        content_selector=src["content_selector"],
        user_agent=cfg["user_agent"],
        request_delay=src.get("request_delay", 2.0),
        respect_robots=src.get("respect_robots", True),
    )

    include = src.get("slug_include", [])
    exclude = src.get("slug_exclude", [])
    print(f"sitemap-walk of {src['base_url']}")
    print(f"  include: {include}")
    print(f"  exclude: {exclude}\n")

    start = time.time()
    kept = 0
    articles = 0

    with JsonlWriter(cfg["output"]) as writer:
        for doc in collector.iter_sitemap_articles(include, exclude):
            articles += 1
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
            if articles % 20 == 0:
                print(f"  {articles} articles, {kept} sentences")
            if args.limit and articles >= args.limit:
                break

        total = writer.written
        dupes = writer.skipped_dupes

    elapsed = time.time() - start
    print(f"\n--- sitemap scrape summary ---")
    print(f"articles fetched : {articles}")
    print(f"sentences kept    : {total}")
    print(f"dupes skipped     : {dupes}")
    print(f"elapsed           : {elapsed:.0f}s")
    print(f"output            : {cfg['output']}")


if __name__ == "__main__":
    main()