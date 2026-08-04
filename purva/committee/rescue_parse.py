from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_judge import try_parse

RESULT_KEYS = ["parse_failed", "raw_response"]


def rescue_row(row: dict) -> tuple[dict, bool]:
    """Return (possibly-rewritten row, was_recovered)."""
    if not row.get("parse_failed"):
        return row, False

    parsed = try_parse(row.get("raw_response", ""))
    if parsed is None:
        return row, False

    rescued = {k: v for k, v in row.items() if k not in RESULT_KEYS}
    rescued.update(parsed)
    return rescued, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()

    shard_path = Path(args.shard)
    rows = [json.loads(x) for x in shard_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    out_rows = []
    recovered = 0
    for row in rows:
        new_row, was_recovered = rescue_row(row)
        out_rows.append(new_row)
        if was_recovered:
            recovered += 1

    out_path = Path(str(shard_path) + ".rescued.jsonl")
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    still_failed = sum(1 for r in out_rows if r.get("parse_failed"))
    failure_rate = (still_failed / total * 100) if total else 0.0

    print(f"total rows: {total}")
    print(f"recovered: {recovered}")
    print(f"still failed: {still_failed}")
    print(f"final failure rate: {failure_rate:.2f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
