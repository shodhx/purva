"""Draws the Phase 5 human-annotation samples (PROTOCOL.md §6) from
stratified Dawid-Skene posteriors:

  (a) routed high-entropy set, ~2,000 items
  (b) low-entropy control, 500 items
  (c) uniform random slice, 300 items, independent of entropy
  (d) shared reliability subset, 300 items, drawn from the routed set

(a)-(c) are pairwise disjoint; (d) is deliberately a subset of (a), not a
fourth disjoint slice (the task specifies it this way — three annotators
label the same 300 already-routed items for reliability statistics).

Both (a) and (b) are allocated proportionally across (register, text_type)
strata — largest-remainder allocation, the same method make_chunks.py uses
— rather than a pure top-N/bottom-N over the whole corpus, so no single
register can dominate an entropy-based selection just because it happens
to run systematically higher- or lower-entropy than everything else.

Blinding (PROTOCOL.md §9): the annotator-facing CSVs carry only corpus
fields plus an empty human_label column — no model label, posterior,
entropy, confidence, or rationale. This is asserted programmatically
against an explicit allow-list before any CSV is written, not left to
"we didn't happen to include it."
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import build_strata, load_master

SEED = 42

N_ROUTED = 2000
N_CONTROL = 500
N_RANDOM = 300
N_RELIABILITY = 300

# The only columns an annotator-facing CSV may ever contain. Anything not
# in this list is either a model output (forbidden per PROTOCOL.md §9) or
# not needed for the labeling task.
ALLOWED_CSV_COLUMNS = [
    "id", "raw_text", "cleaned_text", "source_name", "register", "text_type", "script", "human_label",
]


def assert_blinded(df: pd.DataFrame) -> None:
    extra = set(df.columns) - set(ALLOWED_CSV_COLUMNS)
    missing = set(ALLOWED_CSV_COLUMNS) - set(df.columns)
    assert not extra, f"blinding violation: annotator CSV has disallowed column(s) {extra}"
    assert not missing, f"annotator CSV is missing required column(s) {missing}"
    # Belt-and-braces: scan the actual column names (not just the allow-list
    # match) for anything that smells like a model artifact, in case a
    # future edit renames a column into something that's coincidentally
    # still in ALLOWED_CSV_COLUMNS's shape but leaks a model field.
    forbidden_substrings = ("posterior", "entropy", "rationale", "confidence", "judge_", "dawid", "mace", "glad", "majority")
    for col in df.columns:
        low = col.lower()
        assert not any(s in low for s in forbidden_substrings), f"blinding violation: column {col!r} looks like a model artifact"


def allocate_proportional(strata: np.ndarray, n_strata: int, total: int, rng: random.Random) -> np.ndarray:
    """Largest-remainder allocation of `total` across strata proportional
    to stratum size (same method as make_chunks.py) — returns an (n_strata,)
    int array summing to min(total, len(strata))."""
    sizes = np.bincount(strata, minlength=n_strata)
    corpus_n = sizes.sum()
    total = min(total, corpus_n)
    raw = sizes * total / corpus_n
    base = np.floor(raw).astype(int)
    remainder = total - base.sum()
    fracs = raw - base
    order = sorted(range(n_strata), key=lambda s: (-fracs[s], s))
    # Deterministic tie-break by stratum index after sorting by fractional
    # part; ties within identical fractional parts are broken by a seeded
    # shuffle so the remainder isn't systematically biased toward low
    # stratum indices across repeated runs with different `total` values.
    rng.shuffle(order)
    order = sorted(order, key=lambda s: -fracs[s])
    for s in order[:remainder]:
        base[s] += 1
    return base


def top_n_per_stratum(entropy: np.ndarray, strata: np.ndarray, allocation: np.ndarray, ascending: bool, excluded: np.ndarray) -> np.ndarray:
    """Indices selected: within each stratum, the `allocation[s]` available
    (not yet excluded) items with the highest (ascending=False) or lowest
    (ascending=True) entropy."""
    selected = []
    for s, n in enumerate(allocation):
        if n == 0:
            continue
        candidates = np.where((strata == s) & ~excluded)[0]
        if len(candidates) == 0:
            continue
        order = np.argsort(entropy[candidates])
        if not ascending:
            order = order[::-1]
        selected.extend(candidates[order[: min(n, len(candidates))]].tolist())
    return np.array(selected, dtype=int)


def write_set(name: str, df: pd.DataFrame, idx: np.ndarray, entropy: np.ndarray, output_dir: Path) -> dict:
    subset = df.iloc[idx].copy()
    subset_json = subset.copy()
    subset_json["stratified_ds_entropy_norm"] = entropy[idx]

    jsonl_path = output_dir / f"routing_{name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for _, row in subset_json.iterrows():
            fh.write(json.dumps({
                "id": row["id"], "raw_text": row["raw_text"], "cleaned_text": row["cleaned_text"],
                "source_name": row["source_name"], "register": row["register"], "text_type": row["text_type"],
                "script": row["script"], "stratified_ds_entropy_norm": float(row["stratified_ds_entropy_norm"]),
            }, ensure_ascii=False) + "\n")

    csv_df = subset[["id", "raw_text", "cleaned_text", "source_name", "register", "text_type", "script"]].copy()
    csv_df["human_label"] = ""
    assert_blinded(csv_df)
    csv_path = output_dir / f"routing_{name}.csv"
    csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"{name}: {len(idx)} items -> {jsonl_path}, {csv_path}")
    return {"n": len(idx), "jsonl": str(jsonl_path), "csv": str(csv_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--n-routed", type=int, default=N_ROUTED)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--n-random", type=int, default=N_RANDOM)
    ap.add_argument("--n-reliability", type=int, default=N_RELIABILITY)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)

    print(f"loading {args.master}")
    df = load_master(args.master).reset_index(drop=True)
    strata, strata_keys = build_strata(df)
    n_strata = len(strata_keys)

    print(f"loading {args.aggregated}")
    agg_rows = {json.loads(line)["id"]: json.loads(line) for line in Path(args.aggregated).read_text(encoding="utf-8").splitlines() if line.strip()}
    entropy = np.array([agg_rows[i]["stratified_dawid_skene"]["entropy_norm"] for i in df["id"]])

    n = len(df)
    excluded = np.zeros(n, dtype=bool)
    rng = random.Random(SEED)

    print("\n=== routed high-entropy set ===")
    routed_alloc = allocate_proportional(strata, n_strata, args.n_routed, rng)
    for (r, t), c in zip(strata_keys, routed_alloc):
        print(f"  {r}/{t}: {c}")
    routed_idx = top_n_per_stratum(entropy, strata, routed_alloc, ascending=False, excluded=excluded)
    excluded[routed_idx] = True

    print("\n=== reliability subset (drawn from routed set) ===")
    reliability_rng = random.Random(SEED)
    reliability_idx = np.array(sorted(reliability_rng.sample(list(routed_idx), min(args.n_reliability, len(routed_idx)))))

    print("\n=== low-entropy control ===")
    control_alloc = allocate_proportional(strata, n_strata, args.n_control, rng)
    for (r, t), c in zip(strata_keys, control_alloc):
        print(f"  {r}/{t}: {c}")
    control_idx = top_n_per_stratum(entropy, strata, control_alloc, ascending=True, excluded=excluded)
    excluded[control_idx] = True

    print("\n=== uniform random slice ===")
    remaining = np.where(~excluded)[0]
    random_rng = random.Random(SEED)
    random_idx = np.array(sorted(random_rng.sample(list(remaining), min(args.n_random, len(remaining)))))

    # Disjointness check (reliability is the one deliberate exception —
    # it's a subset of routed, not a fourth disjoint slice).
    assert set(routed_idx).isdisjoint(control_idx), "routed and control overlap"
    assert set(routed_idx).isdisjoint(random_idx), "routed and random overlap"
    assert set(control_idx).isdisjoint(random_idx), "control and random overlap"
    assert set(reliability_idx) <= set(routed_idx), "reliability subset is not contained in the routed set"

    summary = {
        "seed": SEED,
        "high_entropy": write_set("high_entropy", df, routed_idx, entropy, output_dir),
        "low_entropy_control": write_set("low_entropy_control", df, control_idx, entropy, output_dir),
        "uniform_random": write_set("uniform_random", df, random_idx, entropy, output_dir),
        "reliability_subset": write_set("reliability_subset", df, reliability_idx, entropy, output_dir),
    }
    meta_path = output_dir / "routing_sets.meta.json"
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
