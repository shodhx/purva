from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
TARGET_TOTAL = 1000


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def proportional_allocation(by_cell: dict[tuple, list[dict]], total_n: int, target_total: int) -> dict[tuple, int]:
    raw = {cell: len(rows) / total_n * target_total for cell, rows in by_cell.items()}
    alloc = {cell: int(raw[cell]) for cell in by_cell}
    remainder = target_total - sum(alloc.values())
    ranked = sorted(by_cell, key=lambda c: raw[c] - alloc[c], reverse=True)
    for cell in ranked[:remainder]:
        alloc[cell] += 1
    return alloc


def build_pilot_set(rows: list[dict], seed: int = SEED, target_total: int = TARGET_TOTAL) -> list[dict]:
    rng = random.Random(seed)
    total_n = len(rows)

    cells = sorted({(r.get("register"), r.get("text_type")) for r in rows})
    by_cell = {cell: [r for r in rows if (r.get("register"), r.get("text_type")) == cell] for cell in cells}
    alloc = proportional_allocation(by_cell, total_n, target_total)

    selected: list[dict] = []
    print("pilot set strata (register, text_type):")
    print(f"  {'register':15s} {'text_type':10s} {'available':>10s} {'sampled':>8s}")
    for cell in cells:
        pool = by_cell[cell][:]
        rng.shuffle(pool)
        n = alloc[cell]
        chosen = pool[:n]
        selected.extend(chosen)
        register, text_type = cell
        print(f"  {str(register):15s} {str(text_type):10s} {len(pool):10d} {len(chosen):8d}")

    selected.sort(key=lambda r: r["id"])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus_lid.jsonl")
    ap.add_argument("--output", default="data/pilot_set.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(
            f"refusing to overwrite {out_path} — the pilot set must be identical across all "
            "judges. Delete it manually first if you really intend to redraw it."
        )

    rows = load_rows(Path(args.input))
    pilot_rows = build_pilot_set(rows)

    with out_path.open("w", encoding="utf-8") as fh:
        for r in pilot_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\ntotal pilot rows: {len(pilot_rows)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
