"""Split data/corpus_lid.jsonl into N_CHUNKS proportional, stratified shards.

Why chunk-major instead of judge-major: previously each judge ran over the
whole corpus before the next judge started, so no chunk was fully labeled by
every judge until the very end — Dawid-Skene aggregation and human routing
had nothing usable until then. Chunking first means each chunk becomes a
self-contained dataset (every judge has voted on every record in it) as soon
as the slowest judge finishes that chunk, so results arrive incrementally.

Stratification is the critical requirement here, not an optimization: the
corpus file is grouped by source (see data/corpus_lid.jsonl — records from
the same source_url/source_name are contiguous), so naive sequential slicing
would make chunk_01 almost entirely one source and silently invalidate every
cross-chunk comparison. Each chunk must be a proportional miniature of the
full corpus across the joint distribution of source_name x register x
text_type.

Method: largest-remainder allocation, generalized from make_pilot_set.py's
allocate-a-subset version to allocate-across-N-buckets. Within each stratum
cell, rows are shuffled (seeded) and distributed across the 9 chunks as
evenly as integer division allows; the leftover remainder items are handed
to a seeded-random subset of chunks rather than always the first ones, so
rounding bias doesn't systematically favor chunk_01 across thousands of
cells.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
N_CHUNKS = 9
TOLERANCE_PCT = 1.0  # max allowed |chunk% - corpus%| per category value before hard-fail


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def allocate_across_chunks(n: int, k: int, rng: random.Random) -> list[int]:
    """Split n items into k integer buckets as evenly as possible. The
    remainder (n % k) items go to a seeded-random subset of buckets rather
    than always the first ones, so per-cell rounding doesn't systematically
    favor low-index chunks once summed across thousands of cells."""
    base = n // k
    alloc = [base] * k
    remainder = n - base * k
    order = list(range(k))
    rng.shuffle(order)
    for idx in order[:remainder]:
        alloc[idx] += 1
    return alloc


def build_chunks(rows: list[dict], n_chunks: int = N_CHUNKS, seed: int = SEED) -> list[list[dict]]:
    rng = random.Random(seed)

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[(r.get("source_name"), r.get("register"), r.get("text_type"))].append(r)
    cells = sorted(by_cell, key=lambda c: (str(c[0]), str(c[1]), str(c[2])))

    chunk_rows: list[list[dict]] = [[] for _ in range(n_chunks)]
    for cell in cells:
        pool = by_cell[cell][:]
        rng.shuffle(pool)
        alloc = allocate_across_chunks(len(pool), n_chunks, rng)
        offset = 0
        for i, count in enumerate(alloc):
            chunk_rows[i].extend(pool[offset : offset + count])
            offset += count

    for c in chunk_rows:
        c.sort(key=lambda r: r["id"])

    return chunk_rows


def verify_partition(rows: list[dict], chunk_rows: list[list[dict]]) -> None:
    """Hard-fail (assert) if the chunk assignment is not a clean partition of
    the input record IDs: every ID in exactly one chunk, nothing dropped,
    nothing duplicated."""
    all_ids = [r["id"] for r in rows]
    assert len(all_ids) == len(set(all_ids)), "input corpus itself has duplicate IDs — cannot build a clean partition"
    all_id_set = set(all_ids)

    chunk_id_sets = [set(r["id"] for r in c) for c in chunk_rows]
    for i, (rlist, idset) in enumerate(zip(chunk_rows, chunk_id_sets), start=1):
        assert len(rlist) == len(idset), f"chunk_{i:02d} has duplicate IDs within itself"

    for i in range(len(chunk_id_sets)):
        for j in range(i + 1, len(chunk_id_sets)):
            overlap = chunk_id_sets[i] & chunk_id_sets[j]
            assert not overlap, f"chunk_{i+1:02d} and chunk_{j+1:02d} share {len(overlap)} ID(s), e.g. {sorted(overlap)[:5]}"

    union = set().union(*chunk_id_sets)
    missing = all_id_set - union
    extra = union - all_id_set
    assert not missing, f"{len(missing)} input ID(s) missing from every chunk, e.g. {sorted(missing)[:5]}"
    assert not extra, f"{len(extra)} ID(s) in chunks that aren't in the input corpus, e.g. {sorted(extra)[:5]}"
    assert union == all_id_set, "chunk union does not exactly equal the input ID set"

    print(f"verified: {len(all_id_set)} unique input IDs, chunks pairwise disjoint, union == input set")


def print_count_table(chunk_rows: list[list[dict]]) -> None:
    total = sum(len(c) for c in chunk_rows)
    print("\nper-chunk record counts:")
    print(f"  {'chunk':10s} {'records':>8s} {'% of corpus':>12s}")
    for i, c in enumerate(chunk_rows, start=1):
        pct = len(c) / total * 100 if total else 0.0
        print(f"  chunk_{i:02d}   {len(c):8d} {pct:11.2f}%")
    print(f"  {'TOTAL':10s} {total:8d}")


def distribution_table(rows: list[dict], chunk_rows: list[list[dict]], field: str) -> bool:
    """Print a table comparing each chunk's percentage breakdown of `field`
    against the full corpus, and return whether every value is within
    TOLERANCE_PCT of the full-corpus percentage."""
    total = len(rows)
    full_counts: dict = defaultdict(int)
    for r in rows:
        full_counts[r.get(field)] += 1
    values = sorted(full_counts, key=lambda v: -full_counts[v])

    chunk_totals = [len(c) for c in chunk_rows]
    chunk_counts: list[dict] = []
    for c in chunk_rows:
        counts: dict = defaultdict(int)
        for r in c:
            counts[r.get(field)] += 1
        chunk_counts.append(counts)

    ok = True
    header = f"  {field:22s} {'corpus%':>8s} " + " ".join(f"c{i:02d}%" for i in range(1, len(chunk_rows) + 1))
    print(f"\n{field} distribution (corpus vs. each chunk, %):")
    print(header)
    for v in values:
        full_pct = full_counts[v] / total * 100
        row_cells = []
        for i, counts in enumerate(chunk_counts):
            chunk_pct = counts.get(v, 0) / chunk_totals[i] * 100 if chunk_totals[i] else 0.0
            row_cells.append(chunk_pct)
            if abs(chunk_pct - full_pct) > TOLERANCE_PCT:
                ok = False
        cells_str = " ".join(f"{p:5.1f}" for p in row_cells)
        print(f"  {str(v):22.22s} {full_pct:7.2f}% {cells_str}")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus_lid.jsonl")
    ap.add_argument("--output-dir", default="data/chunks")
    ap.add_argument("--n-chunks", type=int, default=N_CHUNKS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        sys.exit(
            f"refusing to overwrite {out_dir} — chunk assignments must stay stable for the whole "
            "project, since shards produced against them in different weeks get merged later. "
            "Delete it manually first if you really intend to redraw the chunks (this will "
            "orphan any shards already produced against the old assignment)."
        )

    rows = load_rows(Path(args.input))
    print(f"loaded {len(rows)} records from {args.input}")

    chunk_rows = build_chunks(rows, n_chunks=args.n_chunks, seed=args.seed)
    verify_partition(rows, chunk_rows)
    print_count_table(chunk_rows)

    source_ok = distribution_table(rows, chunk_rows, "source_name")
    register_ok = distribution_table(rows, chunk_rows, "register")
    text_type_ok = distribution_table(rows, chunk_rows, "text_type")

    assert source_ok, f"source_name distribution deviates by more than {TOLERANCE_PCT} points in some chunk — stratification is broken"
    assert register_ok, f"register distribution deviates by more than {TOLERANCE_PCT} points in some chunk — stratification is broken"
    assert text_type_ok, f"text_type distribution deviates by more than {TOLERANCE_PCT} points in some chunk — stratification is broken"
    print(f"\nall per-chunk distributions within {TOLERANCE_PCT} points of the full corpus")

    out_dir.mkdir(parents=True)
    for i, c in enumerate(chunk_rows, start=1):
        path = out_dir / f"chunk_{i:02d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in c:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(c)} records)")


if __name__ == "__main__":
    main()
