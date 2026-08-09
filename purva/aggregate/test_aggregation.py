"""Permanent invariant tests for aggregation output.

Importable — run_aggregation.py calls check_unanimity() and
check_class_ratio() directly to decide between the five-class (with
identifiability priors) and four-class-fallback label space, before any
output is written.

Also runnable standalone as a regression check against whatever aggregation
currently sits on disk:

    python -m purva.aggregate.test_aggregation

Two invariants, both directly motivated by the identifiability failure
documented in data/aggregation_report.md:

  1. Unanimity: if every judge who voted on an item agrees, every
     aggregation method's consensus label for that item must be that same
     label. A method that overrides unanimous agreement is definitionally
     broken, regardless of how it justifies doing so internally.
  2. Class ratio: no method's consensus share of a class may exceed a
     stated multiple of that class's raw-vote share. This is the direct,
     checkable form of "a class no judge produces in quantity should not
     dominate the corpus."
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from ._common import JUDGES, LABELS, build_vote_matrix, load_aggregated, load_master, method_labels_from_aggregated


def unanimous_mask_and_label(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_unanimous (N,) bool, label_idx (N,) int — meaningful only where
    is_unanimous is True). Unanimity requires >=2 judges voted (a single
    vote isn't "agreement", it's just one opinion)."""
    n = votes.shape[0]
    unanimous = np.zeros(n, dtype=bool)
    label_idx = np.full(n, -1, dtype=np.int8)
    for i in range(n):
        row = votes[i]
        observed = row[row != -1]
        if observed.size >= 2 and (observed == observed[0]).all():
            unanimous[i] = True
            label_idx[i] = observed[0]
    return unanimous, label_idx


def check_unanimity(votes: np.ndarray, method_labels: dict[str, list[str]], labels: tuple[str, ...] = LABELS) -> dict:
    unanimous, label_idx = unanimous_mask_and_label(votes)
    n_unanimous = int(unanimous.sum())
    out = {"n_unanimous_items": n_unanimous, "methods": {}, "passed": True}
    unanimous_positions = np.where(unanimous)[0]
    expected = [labels[label_idx[i]] for i in unanimous_positions]
    for method, all_labels in method_labels.items():
        actual = [all_labels[i] for i in unanimous_positions]
        mismatches = [(int(i), e, a) for i, e, a in zip(unanimous_positions, expected, actual) if e != a]
        method_passed = len(mismatches) == 0
        out["methods"][method] = {
            "n_violations": len(mismatches),
            "violation_rate": len(mismatches) / n_unanimous if n_unanimous else 0.0,
            "examples": mismatches[:10],
            "passed": method_passed,
        }
        out["passed"] = out["passed"] and method_passed
    return out


def check_class_ratio(
    votes: np.ndarray, method_labels: dict[str, list[str]], labels: tuple[str, ...] = LABELS, max_ratio: float = 3.0,
) -> dict:
    total_votes = int((votes != -1).sum())
    raw_freq = {lbl: float((votes == i).sum()) / total_votes for i, lbl in enumerate(labels)}
    n = len(next(iter(method_labels.values())))
    out = {"max_ratio": max_ratio, "raw_vote_frequency": raw_freq, "methods": {}, "passed": True}
    for method, all_labels in method_labels.items():
        counts = Counter(all_labels)
        per_label = {}
        method_passed = True
        for lbl in labels:
            share = counts.get(lbl, 0) / n
            rf = raw_freq[lbl]
            if rf > 0:
                ratio = share / rf
            else:
                ratio = float("inf") if share > 0 else 1.0
            exceeds = ratio > max_ratio
            method_passed = method_passed and not exceeds
            per_label[lbl] = {"consensus_share": share, "raw_vote_share": rf, "ratio": ratio, "exceeds": exceeds}
        out["methods"][method] = {"per_label": per_label, "passed": method_passed}
        out["passed"] = out["passed"] and method_passed
    return out


def run_invariant_checks(
    votes: np.ndarray, method_labels: dict[str, list[str]], labels: tuple[str, ...] = LABELS, max_ratio: float = 3.0,
) -> tuple[bool, dict]:
    unanimity = check_unanimity(votes, method_labels, labels)
    ratio = check_class_ratio(votes, method_labels, labels, max_ratio)
    report = {"unanimity": unanimity, "class_ratio": ratio}
    return (unanimity["passed"] and ratio["passed"]), report


def _print_report(report: dict) -> None:
    u = report["unanimity"]
    print(f"unanimity: {u['n_unanimous_items']} fully-agreeing items")
    for method, v in u["methods"].items():
        status = "OK" if v["passed"] else "FAIL"
        print(f"  [{status}] {method}: {v['n_violations']} violation(s) ({v['violation_rate']:.4%})")
        for i, expected, actual in v["examples"]:
            print(f"      item_index={i} unanimous={expected!r} method_says={actual!r}")

    r = report["class_ratio"]
    print(f"\nclass ratio (max allowed = {r['max_ratio']}x raw-vote share):")
    for method, v in r["methods"].items():
        status = "OK" if v["passed"] else "FAIL"
        print(f"  [{status}] {method}:")
        for lbl, stats in v["per_label"].items():
            flag = " <-- EXCEEDS" if stats["exceeds"] else ""
            print(f"      {lbl}: consensus={stats['consensus_share']:.4f} raw={stats['raw_vote_share']:.4f} ratio={stats['ratio']:.2f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--max-ratio", type=float, default=3.0)
    args = ap.parse_args()

    df = load_master(args.master)
    votes = build_vote_matrix(df, JUDGES)
    aggregated = load_aggregated(args.aggregated)
    method_labels = method_labels_from_aggregated(df, aggregated)

    passed, report = run_invariant_checks(votes, method_labels, LABELS, args.max_ratio)
    _print_report(report)

    if not passed:
        sys.exit("\nFAILED: one or more invariant checks did not pass — see above")
    print("\nall invariant checks passed")


if __name__ == "__main__":
    main()
