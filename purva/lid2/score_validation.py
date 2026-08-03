from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["bhojpuri", "hindi", "maithili", "other"]


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def check_complete(rows: list[dict]) -> None:
    missing = [r["id"] for r in rows if not (r.get("human_language") or "").strip()]
    if missing:
        sys.exit(
            f"score_validation: {len(missing)} row(s) have an empty human_language cell.\n"
            f"Fill in every row before scoring. Missing ids:\n  "
            + "\n  ".join(missing)
        )


def confusion_matrix(rows: list[dict]) -> dict[str, Counter]:
    matrix = {true: Counter() for true in CLASSES}
    for r in rows:
        true = r["human_language"].strip()
        pred = r["lid_verdict"].strip()
        if true not in matrix:
            matrix[true] = Counter()
        matrix[true][pred] += 1
    return matrix


def per_class_metrics(matrix: dict[str, Counter], all_true_classes: list[str]) -> dict[str, dict]:
    metrics = {}
    for c in all_true_classes:
        tp = matrix.get(c, Counter()).get(c, 0)
        fn = sum(v for pred, v in matrix.get(c, Counter()).items() if pred != c)
        fp = sum(matrix.get(other, Counter()).get(c, 0) for other in all_true_classes if other != c)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = tp + fn

        metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def print_confusion_matrix(matrix: dict[str, Counter], all_true_classes: list[str], all_pred_classes: list[str]):
    label = "true \\ pred"
    header = f"{label:15s}" + "".join(f"{p:>10s}" for p in all_pred_classes)
    print(header)
    for true in all_true_classes:
        row = matrix.get(true, Counter())
        print(f"{true:15s}" + "".join(f"{row.get(p, 0):10d}" for p in all_pred_classes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/lid_validation_sample.csv")
    ap.add_argument("--output", default="data/lid_validation_report.json")
    args = ap.parse_args()

    csv_path = Path(args.input)
    rows = load_rows(csv_path)
    check_complete(rows)

    matrix = confusion_matrix(rows)
    all_true_classes = sorted(set(CLASSES) | set(matrix.keys()))
    all_pred_classes = sorted(set(CLASSES) | {r["lid_verdict"].strip() for r in rows})

    metrics = per_class_metrics(matrix, all_true_classes)

    total = len(rows)
    correct = sum(1 for r in rows if r["human_language"].strip() == r["lid_verdict"].strip())
    accuracy = correct / total if total else 0.0

    print(f"n = {total}, accuracy = {accuracy:.4f} ({correct}/{total})\n")
    print("confusion matrix (rows = human_language, cols = lid_verdict):")
    print_confusion_matrix(matrix, all_true_classes, all_pred_classes)

    print("\nper-class metrics:")
    print(f"{'class':10s}{'precision':>12s}{'recall':>10s}{'f1':>10s}{'support':>10s}")
    for c in all_true_classes:
        m = metrics[c]
        print(f"{c:10s}{m['precision']:12.4f}{m['recall']:10.4f}{m['f1']:10.4f}{m['support']:10d}")

    report = {
        "n": total,
        "accuracy": round(accuracy, 4),
        "confusion_matrix": {true: dict(matrix.get(true, Counter())) for true in all_true_classes},
        "per_class_metrics": metrics,
        "classes": CLASSES,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
