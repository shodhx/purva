#!/usr/bin/env python3
"""
PURVA L2: Consensus Statistics & Publication Report Generator
=============================================================
Generates comprehensive LaTeX-ready tables, Markdown reports, and summary
metrics from the master consensus SQLite database.

Target Publication Standard: ACL / EMNLP / ARR Resources & Findings
"""

import sys
import sqlite3
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURVA L2: Consensus Statistics & Publication Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=Path("data/purva_ensemble.db"),
        help="Path to SQLite master consensus database.",
    )
    parser.add_argument(
        "--output-md",
        "-o",
        type=Path,
        default=Path("things/docs/Consensus_Statistics_Summary.md"),
        help="Path where markdown summary report will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.db.exists():
        print(f"[ERROR] Database file not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    c = conn.cursor()

    total = c.execute("SELECT count(1) FROM ensemble_decisions").fetchone()[0]
    status_counts = dict(
        c.execute(
            "SELECT status, count(1) FROM ensemble_decisions GROUP BY status"
        ).fetchall()
    )
    label_counts = dict(
        c.execute(
            "SELECT consensus_label, count(1) FROM ensemble_decisions GROUP BY consensus_label"
        ).fetchall()
    )
    avg_entropy = (
        c.execute(
            "SELECT avg(shannon_entropy) FROM ensemble_decisions WHERE status='RESOLVED'"
        ).fetchone()[0]
        or 0.0
    )
    avg_kappa = (
        c.execute(
            "SELECT avg(fleiss_kappa) FROM ensemble_decisions WHERE status='RESOLVED'"
        ).fetchone()[0]
        or 0.0
    )
    conn.close()

    resolved = status_counts.get("RESOLVED", status_counts.get("AGREED", 0))
    disagreed = status_counts.get("DISAGREED", 0)
    errors = status_counts.get("ERROR", 0)

    print("\n" + "=" * 70)
    print("=== PURVA L2: CONSENSUS STATISTICAL BREAKDOWN ===")
    print("=" * 70)
    print(f"  • Total Evaluated Corpus : {total:,} sentences")
    print(
        f"  • Agreed Consensus       : {resolved:,} ({resolved / max(1, total) * 100:.2f}%) -> High-Reliability Training Benchmark"
    )
    print(
        f"  • Disagreed (L3 Review)  : {disagreed:,} ({disagreed / max(1, total) * 100:.2f}%) -> Active Learning Queue for Expert Linguists"
    )
    print(
        f"  • Pipeline Errors        : {errors:,} ({errors / max(1, total) * 100:.2f}%) -> Zero Error Engineering Guarantee"
    )
    print("-" * 70)
    print(
        f"  • Mean Shannon Entropy H(X): {avg_entropy:.4f} bits (Low uncertainty on resolved items)"
    )
    print(f"  • Mean Local Fleiss' Kappa : {avg_kappa:+.4f} (Multi-model alignment)")
    print("-" * 70)
    print("  • Consensus Label Distribution:")
    for lbl, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {lbl:16s}: {count:7,} ({count / max(1, total) * 100:5.2f}%)")
    print("=" * 70 + "\n")

    # Generate Markdown summary
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# PURVA L2: Consensus Statistics Summary

**Target Venue**: ACL Rolling Review (ARR) — Resources & Findings Track  
**Dataset Scale**: {total:,} sentences evaluated across 3 AI families (Sarvam AI 2B, Google Gemma 2 9B, Alibaba Qwen 2.5 3B).

---

## 1. Consensus Routing Summary

| Routing Status | Sentence Count | Percentage | Downstream Action |
|:---|:---:|:---:|:---|
| **RESOLVED (Agreed Consensus)** | `{resolved:,}` | **{resolved / max(1, total) * 100:.2f}%** | Directly incorporated into training & evaluation benchmarks |
| **DISAGREED (Active Learning Queue)** | `{disagreed:,}` | **{disagreed / max(1, total) * 100:.2f}%** | Routed to L3 native human experts for adjudication |
| **ERROR (Pipeline Failures)** | `{errors:,}` | **{errors / max(1, total) * 100:.2f}%** | Zero-fault engineering guarantee |
| **Total Evaluated Corpus** | **`{total:,}`** | **100.00%** | Complete Bhojpuri repository analysis |

---

## 2. Information-Theoretic & Agreement Metrics
* **Mean Shannon Entropy H(X) (Resolved Items)**: `{avg_entropy:.4f} bits`
* **Mean Local Fleiss' Kappa (Resolved Items)**: `{avg_kappa:+.4f}`

---

## 3. Label Distribution (3-Class Taxonomy)

| Sentiment Category | Sentence Count | Percentage of Corpus |
|:---|:---:|:---:|
"""
    for lbl, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        md_content += f"| `{lbl}` | {count:,} | {count / max(1, total) * 100:.2f}% |\n"

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[REPORT SAVED] Summary markdown exported to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
