"""Join the corpus and all five committee judges' merged output into a
single analysis-ready file — one row per sentence, judges keyed by short
name, everything else passed through unmodified.

Pure join: no majority votes, consensus labels, or agreement scores are
computed here. That is Phase 4 (aggregation)'s job, and it needs the raw
per-judge votes untouched — including the null slots where a judge's
output failed to parse — not a version already collapsed by this step.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# The corpus fields carried through as-is. lid_label/lid_confidence (legacy,
# always null in data/corpus_lid.jsonl) are deliberately excluded in favor of
# the lid_model_*/lid_verdict fields that actually carry the LID result.
CORPUS_FIELDS = (
    "id", "raw_text", "cleaned_text", "source_url", "source_name",
    "scrape_timestamp", "category", "register", "text_type", "script",
    "license_class", "lid_model_label", "lid_model_confidence", "lid_verdict",
)

# The 7 fields a judge emits per sentence (see prompts/judge_prompt_v1.txt
# and run_judge.py); a parse-failed row has none of these, hence None below
# rather than a partial dict.
JUDGE_FIELDS = (
    "subjectivity", "polarity", "confidence", "domain",
    "narrative_voice", "sentiment_target", "rationale",
)

# data/committee/merged/<stem>.jsonl -> short key used in the "judges"
# object and as the Parquet column prefix (judge_<short>_<field>).
JUDGE_SHORT_NAMES = {
    "aya-expanse-8b": "aya",
    "gemma-2-9b": "gemma",
    "llama-3.1-8b": "llama",
    "mistral-nemo-12b": "mistral",
    "qwen2.5-14b": "qwen",
}

# Snapshot invariant for this corpus (RUNS.md / merge_shards.py both confirm
# full coverage at this size) — a hard assert, not a soft expectation, so a
# silently truncated or duplicated input is caught immediately.
EXPECTED_ROWS = 90_207


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_membership(chunks_dir: Path) -> dict[str, int]:
    id_to_chunk: dict[str, int] = {}
    chunk_files = sorted(chunks_dir.glob("chunk_*.jsonl"))
    if not chunk_files:
        sys.exit(f"no chunk_*.jsonl files found under {chunks_dir} — run make_chunks.py first")
    for f in chunk_files:
        chunk_n = int(f.stem.split("_")[1])
        for row in load_jsonl(f):
            id_to_chunk[row["id"]] = chunk_n
    return id_to_chunk


def load_judge_votes(path: Path) -> dict[str, dict | None]:
    """{id: {7 fields}} for a clean parse, {id: None} for parse_failed."""
    votes: dict[str, dict | None] = {}
    for row in load_jsonl(path):
        rid = row["id"]
        votes[rid] = None if row.get("parse_failed") else {k: row.get(k) for k in JUDGE_FIELDS}
    return votes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_lid.jsonl")
    ap.add_argument("--chunks-dir", default="data/chunks")
    ap.add_argument("--merged-dir", default="data/committee/merged")
    ap.add_argument("--output-jsonl", default="data/purva_master.jsonl")
    ap.add_argument("--output-parquet", default="data/purva_master.parquet")
    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_parquet = Path(args.output_parquet)
    for p in (out_jsonl, out_parquet):
        if p.exists():
            sys.exit(f"refusing to overwrite existing {p} — remove it first if you want to rebuild")

    corpus_path = Path(args.corpus)
    corpus_rows = load_jsonl(corpus_path)
    print(f"corpus: {len(corpus_rows)} rows from {corpus_path}")

    corpus_ids = [r["id"] for r in corpus_rows]
    if len(corpus_ids) != len(set(corpus_ids)):
        dupes = [i for i, c in Counter(corpus_ids).items() if c > 1]
        sys.exit(f"corpus has {len(dupes)} duplicate id(s), e.g. {dupes[:5]}")
    corpus_id_set = set(corpus_ids)

    id_to_chunk = load_chunk_membership(Path(args.chunks_dir))
    missing_chunk = [rid for rid in corpus_ids if rid not in id_to_chunk]
    if missing_chunk:
        sys.exit(f"{len(missing_chunk)} corpus id(s) have no chunk assignment, e.g. {missing_chunk[:5]}")

    merged_dir = Path(args.merged_dir)
    judge_votes: dict[str, dict[str, dict | None]] = {}
    for stem, short in JUDGE_SHORT_NAMES.items():
        path = merged_dir / f"{stem}.jsonl"
        if not path.exists():
            sys.exit(f"missing merged judge file: {path}")
        votes = load_judge_votes(path)
        extra = set(votes) - corpus_id_set
        if extra:
            sys.exit(f"judge {short} has {len(extra)} id(s) not in the corpus, e.g. {sorted(extra)[:5]}")
        judge_votes[short] = votes
        print(f"judge {short} ({stem}): {len(votes)} rows")

    master_rows: list[dict] = []
    n_judges_counter: Counter[int] = Counter()
    for row in corpus_rows:
        rid = row["id"]
        judges_obj: dict[str, dict | None] = {}
        n = 0
        for short in JUDGE_SHORT_NAMES.values():
            v = judge_votes[short].get(rid)
            judges_obj[short] = v
            if v is not None:
                n += 1
        n_judges_counter[n] += 1
        out_row = {k: row.get(k) for k in CORPUS_FIELDS}
        out_row["chunk"] = id_to_chunk[rid]
        out_row["judges"] = judges_obj
        out_row["n_judges"] = n
        master_rows.append(out_row)

    # --- verification (hard-fail on any mismatch) ---
    print("\n=== verification ===")
    out_ids = [r["id"] for r in master_rows]
    assert len(out_ids) == len(set(out_ids)), "duplicate ids in output"
    assert set(out_ids) == corpus_id_set, "output id set does not match corpus id set"
    assert len(master_rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(master_rows)}"
    print(f"output rows: {len(master_rows)} (every corpus id present exactly once, no foreign ids)")

    print("n_judges distribution:")
    for n in sorted(n_judges_counter, reverse=True):
        print(f"  {n}: {n_judges_counter[n]}")

    for short, votes in judge_votes.items():
        valid_in_file = sum(1 for v in votes.values() if v is not None)
        valid_in_master = sum(1 for r in master_rows if r["judges"][short] is not None)
        assert valid_in_file == valid_in_master, (
            f"judge {short}: merged file has {valid_in_file} valid rows but master has {valid_in_master}"
        )
        print(f"judge {short}: {valid_in_master} valid, {len(master_rows) - valid_in_master} null")

    print("all verification checks passed\n")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in master_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl} ({out_jsonl.stat().st_size / 1e6:.1f} MB)")

    flat_rows = []
    for row in master_rows:
        flat = {k: row[k] for k in CORPUS_FIELDS}
        flat["chunk"] = row["chunk"]
        flat["n_judges"] = row["n_judges"]
        for short in JUDGE_SHORT_NAMES.values():
            v = row["judges"][short]
            for field in JUDGE_FIELDS:
                flat[f"judge_{short}_{field}"] = v[field] if v is not None else None
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
