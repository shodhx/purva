"""Freezes the train/dev/test partition (PROTOCOL.md §8) before any model
training happens, so no downstream benchmarking work can retroactively
influence which items ended up in which split.

Writes:
  data/splits.json      — {"train": [ids...], "dev": [ids...], "test": [ids...]}
  data/splits.meta.json — stratification scheme, seed, per-split counts,
                           and the count of human-gold items placed in test.

Stratified on (register, text_type, source_name), largest-remainder
allocation (same method as make_chunks.py / make_routing_sets.py), roughly
80/10/10. Refuses to overwrite existing output, since a split reassignment
after any training has happened invalidates every reported number.

The hard constraint this script exists to enforce: the four Phase-5 routing
sets (data/routing_*.jsonl) are the corpus's only source of human gold
labels. Every item in any of those four sets is therefore forced into test,
regardless of what the proportional allocation would otherwise assign it —
a model must never be trained or tuned on an item that will later carry a
human label. This can push a stratum cell's test share above 10% (when a
cell happens to contain more gold items than its proportional target), but
never below the number of gold items it contains.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
FRACTIONS = {"train": 0.8, "dev": 0.1, "test": 0.1}

ROUTING_FILES = (
    "data/routing_high_entropy.jsonl",
    "data/routing_low_entropy_control.jsonl",
    "data/routing_uniform_random.jsonl",
    "data/routing_reliability_subset.jsonl",
)


def load_gold_ids(routing_files: tuple[str, ...]) -> set[str]:
    gold: set[str] = set()
    for path in routing_files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    gold.add(json.loads(line)["id"])
    return gold


def largest_remainder(total: int, fracs: list[float], rng: random.Random) -> list[int]:
    """Allocate `total` items across len(fracs) buckets proportional to
    `fracs` (which need not sum to 1). Remainder items go to a
    seeded-random subset of buckets (ordered by largest fractional part,
    ties broken by a shuffle) so rounding bias doesn't systematically
    favor one bucket once summed across many strata cells."""
    norm = [f / sum(fracs) for f in fracs]
    raw = [total * f for f in norm]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    fracs_part = [r - b for r, b in zip(raw, base)]
    order = list(range(len(fracs)))
    rng.shuffle(order)
    order.sort(key=lambda k: -fracs_part[k])
    for k in order[:remainder]:
        base[k] += 1
    return base


def allocate_cell(ids: list[str], gold_ids: set[str], rng: random.Random) -> dict[str, list[str]]:
    """Split one stratification cell's IDs into train/dev/test. Gold IDs
    are always placed in test; the proportional 80/10/10 target is applied
    to everything else, with test's target reduced by the gold count
    already guaranteed to land there (and never allowed to go negative)."""
    gold = [i for i in ids if i in gold_ids]
    nongold = [i for i in ids if i not in gold_ids]
    rng.shuffle(nongold)

    n_train, n_dev, n_test = largest_remainder(len(ids), [FRACTIONS["train"], FRACTIONS["dev"], FRACTIONS["test"]], rng)
    n_test_nongold = max(0, n_test - len(gold))

    test = gold + nongold[:n_test_nongold]
    remaining = nongold[n_test_nongold:]

    n_train_r, n_dev_r = largest_remainder(len(remaining), [FRACTIONS["train"], FRACTIONS["dev"]], rng)
    return {"train": remaining[:n_train_r], "dev": remaining[n_train_r : n_train_r + n_dev_r], "test": test}


def verify(all_ids: set[str], splits: dict[str, list[str]], gold_ids: set[str]) -> None:
    sets = {name: set(ids) for name, ids in splits.items()}
    for name, idset in sets.items():
        assert len(idset) == len(splits[name]), f"{name} contains duplicate IDs"
    assert sets["train"].isdisjoint(sets["dev"]), "train and dev overlap"
    assert sets["train"].isdisjoint(sets["test"]), "train and test overlap"
    assert sets["dev"].isdisjoint(sets["test"]), "dev and test overlap"
    union = sets["train"] | sets["dev"] | sets["test"]
    assert union == all_ids, (
        f"split union does not equal the corpus ID set: "
        f"{len(all_ids - union)} missing, {len(union - all_ids)} extra"
    )
    assert gold_ids <= sets["test"], f"{len(gold_ids - sets['test'])} gold item(s) are NOT in test — routing-set leakage"
    print(f"verified: {len(all_ids)} unique IDs, splits pairwise disjoint, union == corpus, "
          f"all {len(gold_ids)} gold items confirmed in test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--output", default="data/splits.json")
    ap.add_argument("--output-meta", default="data/splits.meta.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_path, out_meta_path = Path(args.output), Path(args.output_meta)
    if out_path.exists() or out_meta_path.exists():
        sys.exit(
            f"refusing to overwrite {out_path} / {out_meta_path} — the eval split must stay frozen "
            "once any benchmarking work has referenced it. Delete both manually first if you really "
            "intend to redraw the split (this invalidates any results already reported against it)."
        )

    print(f"loading {args.master}")
    df = pd.read_parquet(args.master, columns=["id", "register", "text_type", "source_name"])
    all_ids = set(df["id"])
    print(f"{len(df)} items")

    print(f"loading gold IDs from {len(ROUTING_FILES)} routing files")
    gold_ids = load_gold_ids(ROUTING_FILES)
    gold_ids &= all_ids
    print(f"{len(gold_ids)} unique gold items across all four routing sets")

    rng = random.Random(args.seed)
    by_cell: dict[tuple, list[str]] = defaultdict(list)
    for _id, r, t, s in df[["id", "register", "text_type", "source_name"]].itertuples(index=False, name=None):
        by_cell[(r, t, s)].append(_id)

    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for cell in sorted(by_cell, key=lambda c: (str(c[0]), str(c[1]), str(c[2]))):
        ids = sorted(by_cell[cell])  # sort first for determinism before the seeded shuffle inside allocate_cell
        cell_splits = allocate_cell(ids, gold_ids, rng)
        for name in splits:
            splits[name].extend(cell_splits[name])

    for name in splits:
        splits[name].sort()

    verify(all_ids, splits, gold_ids)

    n = len(all_ids)
    counts = {name: len(ids) for name, ids in splits.items()}
    print("\nper-split counts:")
    for name, c in counts.items():
        print(f"  {name:6s} {c:7d} ({c / n:.2%})")

    gold_by_split = {name: len(set(ids) & gold_ids) for name, ids in splits.items()}
    print("\ngold items per split:", gold_by_split)

    out_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")

    meta = {
        "seed": args.seed,
        "stratification": ["register", "text_type", "source_name"],
        "target_fractions": FRACTIONS,
        "n_items": n,
        "counts": counts,
        "actual_fractions": {name: c / n for name, c in counts.items()},
        "n_gold_items": len(gold_ids),
        "gold_items_by_split": gold_by_split,
        "routing_files": list(ROUTING_FILES),
        "method": (
            "Largest-remainder proportional allocation per (register, text_type, source_name) cell. "
            "Gold items (drawn from the four Phase-5 routing sets) are forced into test within their "
            "cell before the remaining non-gold items are allocated to fill out the 80/10/10 target; "
            "a cell's test share can exceed 10% if it contains more gold items than its proportional "
            "target, but every gold item is guaranteed to be in test."
        ),
    }
    out_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_meta_path}")


if __name__ == "__main__":
    main()
