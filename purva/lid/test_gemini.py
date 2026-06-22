from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import load_env
from .gemini_judge import GeminiJudge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/purva_pilot_candidates.jsonl")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    load_env()
    judge = GeminiJudge()

    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines[: args.limit]]

    print(f"testing gemini judge on {len(rows)} sentences\n")
    for i, row in enumerate(rows, 1):
        text = row["cleaned_text"]
        out = judge.judge(text)
        print(f"[{i}] {out['label']:8s} conf={out['confidence']:.2f}  {out['reason']}")
        print(f"    {text[:70]}")


if __name__ == "__main__":
    main()