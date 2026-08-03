from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SEED = 42
PER_STRATUM = 20
MIN_LEN = 40
MAX_LEN = 200

FIELDS = ["id", "cleaned_text", "source_name", "register", "text_type", "annotator_label", "notes"]

# Strata, in selection order. Each row is assigned to the first stratum it
# matches; already-selected ids are excluded from later strata so a row is
# never drawn twice (a row can satisfy more than one predicate, e.g.
# text_type=verse and register=news).
STRATA = [
    ("verse", lambda r: r.get("text_type") == "verse"),
    ("news", lambda r: r.get("register") == "news"),
    ("literary_commentary_opinion", lambda r: r.get("register") in ("literary", "commentary", "opinion")),
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_calibration_set(rows: list[dict], seed: int = SEED, per_stratum: int = PER_STRATUM) -> list[dict]:
    rng = random.Random(seed)
    used_ids: set[str] = set()
    selected: list[dict] = []

    for name, predicate in STRATA:
        candidates = [r for r in rows if r["id"] not in used_ids and predicate(r)]
        in_range = [r for r in candidates if MIN_LEN <= len(r["cleaned_text"]) <= MAX_LEN]
        out_range = [r for r in candidates if not (MIN_LEN <= len(r["cleaned_text"]) <= MAX_LEN)]

        rng.shuffle(in_range)
        rng.shuffle(out_range)

        chosen = in_range[:per_stratum]
        if len(chosen) < per_stratum:
            chosen += out_range[: per_stratum - len(chosen)]

        if len(chosen) < per_stratum:
            print(f"warning: stratum '{name}' only has {len(chosen)}/{per_stratum} available rows")

        used_ids.update(r["id"] for r in chosen)
        selected.append((name, chosen))

    print("calibration set stratum sizes:")
    for name, chosen in selected:
        print(f"  {name:30s} {len(chosen)}")

    return [r for _, chosen in selected for r in chosen]


def write_outputs(rows: list[dict], jsonl_path: Path, csv_path: Path):
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            out = {k: (r.get(k) if k not in ("annotator_label", "notes") else None) for k in FIELDS}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            out = {k: (r.get(k) if k not in ("annotator_label", "notes") else "") for k in FIELDS}
            writer.writerow(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus_lid.jsonl")
    ap.add_argument("--out-jsonl", default="data/calibration_set.jsonl")
    ap.add_argument("--out-csv", default="data/calibration_set.csv")
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    calibration_rows = build_calibration_set(rows)

    write_outputs(calibration_rows, Path(args.out_jsonl), Path(args.out_csv))

    print(f"\ntotal calibration rows: {len(calibration_rows)}")
    print(f"wrote {args.out_jsonl}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
