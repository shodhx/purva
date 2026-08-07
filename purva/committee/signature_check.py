"""Cross-chunk signature stability check, per judge.

Chunks are stratified miniatures of the same corpus (see make_chunks.py), so
a given judge's subjectivity rate, polarity distribution, and mean
confidence should stay close across chunks. A jump larger than the
tolerance is a signal of config drift (a different prompt, quantization, or
revision silently sneaking in) rather than genuine corpus variation — the
kind of thing merge_shards.py's config-consistency check can't catch on its
own, since a judge can drift while still reporting an internally consistent
config for the run that drifted.

Hard-fails (non-zero exit) if any judge's subjectivity rate shifts by more
than SUBJECTIVITY_TOLERANCE_PTS between the two chunks compared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import REGISTRY

SUBJECTIVITY_TOLERANCE_PTS = 5.0
POLARITIES = ("positive", "negative", "neutral", "mixed")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def compute_stats(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r.get("parse_failed")]
    n = len(valid)
    n_subjective = sum(1 for r in valid if r.get("subjectivity") == "subjective")
    subjectivity_rate = n_subjective / n * 100 if n else float("nan")

    subjective_rows = [r for r in valid if r.get("subjectivity") == "subjective"]
    polarity_pct = {
        p: (sum(1 for r in subjective_rows if r.get("polarity") == p) / len(subjective_rows) * 100)
        if subjective_rows
        else float("nan")
        for p in POLARITIES
    }

    confidences = [r["confidence"] for r in valid if isinstance(r.get("confidence"), (int, float))]
    mean_confidence = sum(confidences) / len(confidences) if confidences else float("nan")

    return {
        "n": n,
        "subjectivity_rate": subjectivity_rate,
        "polarity_pct": polarity_pct,
        "mean_confidence": mean_confidence,
    }


def shard_path(chunk_dir: Path, judge: str) -> Path:
    return chunk_dir / f"{judge}__judge_prompt_v1.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-a", type=int, default=1)
    ap.add_argument("--chunk-b", type=int, required=True)
    ap.add_argument("--committee-dir", default="data/committee")
    args = ap.parse_args()

    dir_a = Path(args.committee_dir) / f"chunk_{args.chunk_a:02d}"
    dir_b = Path(args.committee_dir) / f"chunk_{args.chunk_b:02d}"

    rows_out = []
    flagged = []
    skipped = []

    for judge in sorted(REGISTRY):
        pa, pb = shard_path(dir_a, judge), shard_path(dir_b, judge)
        if not (pa.exists() and pb.exists()):
            skipped.append(judge)
            continue

        sa = compute_stats(load_rows(pa))
        sb = compute_stats(load_rows(pb))
        delta = sb["subjectivity_rate"] - sa["subjectivity_rate"]
        if abs(delta) > SUBJECTIVITY_TOLERANCE_PTS:
            flagged.append(judge)
        rows_out.append((judge, sa, sb, delta))

    if skipped:
        print(f"skipped (missing shard in chunk_{args.chunk_a:02d} or chunk_{args.chunk_b:02d}): {skipped}")

    label_a, label_b = f"c{args.chunk_a:02d}", f"c{args.chunk_b:02d}"
    print(
        f"\n{'judge':18s} {label_a + ' subj%':>9s} {label_b + ' subj%':>9s} {'delta':>7s}  "
        f"{label_a + ' pos/neg/neu/mix (subj%)':>34s}  {label_b + ' pos/neg/neu/mix (subj%)':>34s}  "
        f"{label_a + ' conf':>9s} {label_b + ' conf':>9s}"
    )
    for judge, sa, sb, delta in rows_out:
        pa_str = "/".join(f"{sa['polarity_pct'][p]:.1f}" for p in POLARITIES)
        pb_str = "/".join(f"{sb['polarity_pct'][p]:.1f}" for p in POLARITIES)
        flag = "  <== FLAG (>{:.0f}pt shift)".format(SUBJECTIVITY_TOLERANCE_PTS) if abs(delta) > SUBJECTIVITY_TOLERANCE_PTS else ""
        print(
            f"{judge:18s} {sa['subjectivity_rate']:9.2f} {sb['subjectivity_rate']:9.2f} {delta:7.2f}  "
            f"{pa_str:>34s}  {pb_str:>34s}  {sa['mean_confidence']:9.4f} {sb['mean_confidence']:9.4f}{flag}"
        )

    print(f"\nflagged judges (>{SUBJECTIVITY_TOLERANCE_PTS:.0f}pt subjectivity shift): {flagged or 'none'}")

    if flagged:
        sys.exit(
            f"signature stability check FAILED: {flagged} shifted subjectivity rate by more than "
            f"{SUBJECTIVITY_TOLERANCE_PTS} points between chunk_{args.chunk_a:02d} and chunk_{args.chunk_b:02d} "
            "-- possible config drift, not just corpus variation. Investigate before running more chunks."
        )


if __name__ == "__main__":
    main()
