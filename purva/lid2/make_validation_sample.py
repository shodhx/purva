from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SEED = 42
TARGET_TOTAL = 500
MIN_NON_BHOJPURI = 150

FIELDS = ["id", "cleaned_text", "source_name", "lid_model_label", "lid_verdict", "human_language"]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def proportional_allocation(by_source: dict[str, list[dict]], total_n: int, target_total: int) -> dict[str, int]:
    raw = {s: len(rows) / total_n * target_total for s, rows in by_source.items()}
    alloc = {s: int(raw[s]) for s in by_source}
    remainder = target_total - sum(alloc.values())
    ranked = sorted(by_source, key=lambda s: raw[s] - alloc[s], reverse=True)
    for s in ranked[:remainder]:
        alloc[s] += 1
    return alloc


def build_sample(rows: list[dict], target_total: int = TARGET_TOTAL, min_non_bhojpuri: int = MIN_NON_BHOJPURI, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    total_n = len(rows)
    target_total = min(target_total, total_n)

    sources = sorted({r["source_name"] for r in rows})
    by_source = {s: [r for r in rows if r["source_name"] == s] for s in sources}
    alloc = proportional_allocation(by_source, total_n, target_total)

    sample_rows: list[dict] = []
    remaining_pool: dict[str, list[dict]] = {}
    for s in sources:
        pool = by_source[s][:]
        rng.shuffle(pool)
        n = alloc[s]
        sample_rows.extend(pool[:n])
        remaining_pool[s] = pool[n:]

    is_non_bho = lambda r: r["lid_verdict"] != "bhojpuri"
    total_non_bho_available = sum(1 for r in rows if is_non_bho(r))
    target_non_bho = min(min_non_bhojpuri, total_non_bho_available)

    cur_non_bho = [r for r in sample_rows if is_non_bho(r)]
    if len(cur_non_bho) < target_non_bho:
        deficit = target_non_bho - len(cur_non_bho)

        candidates = [r for s in sources for r in remaining_pool[s] if is_non_bho(r)]
        rng.shuffle(candidates)
        to_add = candidates[:deficit]

        bho_in_sample = [r for r in sample_rows if not is_non_bho(r)]
        rng.shuffle(bho_in_sample)
        remove_ids = {id(r) for r in bho_in_sample[: len(to_add)]}
        sample_rows = [r for r in sample_rows if id(r) not in remove_ids]
        sample_rows.extend(to_add)

    sample_rows.sort(key=lambda r: r["id"])
    return sample_rows


def write_outputs(sample_rows: list[dict], jsonl_path: Path, csv_path: Path):
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in sample_rows:
            out = {k: (r.get(k) if k != "human_language" else None) for k in FIELDS}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in sample_rows:
            out = {k: (r.get(k) if k != "human_language" else "") for k in FIELDS}
            writer.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus_lid.jsonl")
    ap.add_argument("--out-jsonl", default="data/lid_validation_sample.jsonl")
    ap.add_argument("--out-csv", default="data/lid_validation_sample.csv")
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    sample_rows = build_sample(rows)

    write_outputs(sample_rows, Path(args.out_jsonl), Path(args.out_csv))

    non_bho = sum(1 for r in sample_rows if r["lid_verdict"] != "bhojpuri")
    print(f"corpus rows: {len(rows)}")
    print(f"sample rows: {len(sample_rows)} (non-bhojpuri: {non_bho})")

    from collections import Counter
    by_source = Counter(r["source_name"] for r in sample_rows)
    print("\nper-source sample counts:")
    for s, n in by_source.most_common():
        print(f"  {s:25s} {n:4d}")

    print(f"\nwrote {args.out_jsonl}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
