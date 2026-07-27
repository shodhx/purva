#!/usr/bin/env python3
"""
PURVA L2: Inter-Annotator Agreement Diagnostic & Evaluation Tool
================================================================
Computes Global Fleiss' Kappa across multi-model ensembles, pairwise Cohen's Kappa,
percentage agreement, and per-model affective label distributions.

Target Publication Standard: ACL / EMNLP / ARR Resources & Findings
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Set

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURVA L2: Inter-Annotator Agreement & Kappa Evaluation Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=Path("data/purva_ensemble.db"),
        help="Path to SQLite master consensus database.",
    )
    return parser.parse_args()


def compute_fleiss_kappa(
    model_labels_list: List[Dict[str, str]], categories: List[str]
) -> Tuple[float, int, float]:
    """Computes Global Fleiss' Kappa (k) across all active committee annotators."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    N = len(model_labels_list)
    k = len(categories)
    ratings = np.zeros((N, k))

    for i, labels in enumerate(model_labels_list):
        for label in labels.values():
            if label in cat_idx:
                ratings[i, cat_idx[label]] += 1

    n_raters_per_row = np.sum(ratings, axis=1)
    mask = n_raters_per_row >= 2
    valid_ratings = ratings[mask]

    if valid_ratings.shape[0] == 0:
        return 0.0, 0, 0.0

    N_valid = valid_ratings.shape[0]
    n_avg = np.mean(n_raters_per_row[mask])
    if n_avg <= 1.0:
        return 0.0, N_valid, n_avg

    p_j = np.sum(valid_ratings, axis=0) / (N_valid * n_avg)
    P_i = (np.sum(valid_ratings**2, axis=1) - n_avg) / (n_avg * (n_avg - 1.0))

    P_bar = np.mean(P_i)
    P_e = np.sum(p_j**2)

    if P_e >= 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1.0 - P_e)

    return float(kappa), int(N_valid), float(n_avg)


def compute_cohens_kappa(
    labels1: List[str], labels2: List[str], categories: List[str]
) -> Tuple[float, float]:
    """Computes pairwise Cohen's Kappa (k) and raw agreement percentage between two judges."""
    n = len(labels1)
    if n == 0:
        return 0.0, 0.0

    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    confusion = np.zeros((k, k))

    for l1, l2 in zip(labels1, labels2):
        if l1 in cat_idx and l2 in cat_idx:
            confusion[cat_idx[l1], cat_idx[l2]] += 1

    total = np.sum(confusion)
    if total == 0:
        return 0.0, 0.0

    po = np.trace(confusion) / total
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (total**2)

    if pe >= 1.0:
        return 1.0, float(po)
    return float((po - pe) / (1.0 - pe)), float(po)


def main():
    args = parse_arguments()
    if not args.db.exists():
        print(f"[ERROR] Database file not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    c = conn.cursor()
    rows = c.execute("SELECT model_outputs_json FROM ensemble_decisions").fetchall()
    conn.close()

    print(f"\nLoaded {len(rows):,} total evaluated records from {args.db.resolve()}")

    categories_4 = ["positive", "negative", "neutral", "objective"]
    categories_3 = ["positive", "negative", "neutral_factual"]

    model_labels_4c: List[Dict[str, str]] = []
    model_labels_3c: List[Dict[str, str]] = []
    all_models: Set[str] = set()

    for r in rows:
        outputs = json.loads(r[0])
        lbl4 = {}
        lbl3 = {}
        for model_name, out in outputs.items():
            label_val = str(out.get("label", "error")).lower().strip()
            if label_val in categories_4:
                lbl4[model_name] = label_val
                all_models.add(model_name)
            if label_val in ["neutral", "objective"]:
                mapped = "neutral_factual"
            elif label_val in ["positive", "negative"]:
                mapped = label_val
            else:
                mapped = "error"
            if mapped != "error":
                lbl3[model_name] = mapped
        if lbl4:
            model_labels_4c.append(lbl4)
            model_labels_3c.append(lbl3)

    models_sorted = sorted(list(all_models))
    print(f"Active Committee Models: {models_sorted}")

    k4, N4, n4 = compute_fleiss_kappa(model_labels_4c, categories_4)
    k3, N3, n3 = compute_fleiss_kappa(model_labels_3c, categories_3)

    print("\n" + "=" * 70)
    print("=== 1. GLOBAL MULTI-RATER FLEISS' KAPPA ===")
    print("=" * 70)
    print(
        f"  • 4-Class Taxonomy (Raw)           : k = {k4:+.4f} (n={N4:,}, avg raters={n4:.1f})"
    )
    print(
        f"  • 3-Class Taxonomy (Neutral/Factual): k = {k3:+.4f} (n={N3:,}, avg raters={n3:.1f})"
    )

    print("\n" + "=" * 70)
    print("=== 2. PAIRWISE COHEN'S KAPPA & RAW AGREEMENT (3-Class Taxonomy) ===")
    print("=" * 70)
    for i, m1 in enumerate(models_sorted):
        for m2 in models_sorted[i + 1 :]:
            l1, l2 = [], []
            for lbls in model_labels_3c:
                if m1 in lbls and m2 in lbls:
                    l1.append(lbls[m1])
                    l2.append(lbls[m2])
            if l1:
                ck, po = compute_cohens_kappa(l1, l2, categories_3)
                print(
                    f"  • {m1} vs. {m2}: k = {ck:+.4f} | Raw Agreement: {po * 100:.1f}% (n={len(l1):,})"
                )

    print("\n" + "=" * 70)
    print("=== 3. PER-MODEL LABEL DISTRIBUTIONS (3-Class Taxonomy) ===")
    print("=" * 70)
    for model in models_sorted:
        counts = {c: 0 for c in categories_3}
        tot = 0
        for lbls in model_labels_3c:
            if model in lbls:
                label_val = lbls[model]
                if label_val in counts:
                    counts[label_val] += 1
                    tot += 1
        print(f"\n  [{model}] (Total evaluated: {tot:,})")
        for cat in categories_3:
            pct = (counts[cat] / tot * 100) if tot > 0 else 0.0
            print(f"     - {cat:16s}: {counts[cat]:7,} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("=== 4. MULTI-MODEL CONSENSUS RATES (3-Class Taxonomy) ===")
    print("=" * 70)
    unanimous = sum(
        1 for lbls in model_labels_3c if len(lbls) >= 2 and len(set(lbls.values())) == 1
    )
    majority = sum(
        1
        for lbls in model_labels_3c
        if len(lbls) >= 2 and Counter(lbls.values()).most_common(1)[0][1] >= 2
    )
    total_multi = sum(1 for lbls in model_labels_3c if len(lbls) >= 2)

    print(
        f"  • Unanimous Agreement (3/3 agree): {unanimous:,} / {total_multi:,} ({unanimous / max(1, total_multi) * 100:.1f}%)"
    )
    print(
        f"  • Clean Majority Consensus (>= 2/3): {majority:,} / {total_multi:,} ({majority / max(1, total_multi) * 100:.1f}%)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
