#!/usr/bin/env python3
"""
PURVA L2: 3-Class Taxonomy Dataset Synchronizer & Exporter
==========================================================
Synchronizes SQLite database labels with the 3-Class Taxonomy (Fix 1) and exports
all deliverable files (Agreed JSONL/CSV, Disagreed JSONL/CSV, 1% Audit Sample).

Target Publication Standard: ACL / EMNLP / ARR Resources & Findings
"""

import sys
import csv
import json
import random
import sqlite3
import argparse
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURVA L2: 3-Class Taxonomy Dataset Synchronizer & Exporter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=Path("data/purva_ensemble.db"),
        help="Path to master SQLite consensus database.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data"),
        help="Directory where updated JSONL and CSV files will be written.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for generating the reproducible 1%% control audit sample.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.db.exists():
        print(f"[ERROR] Database file not found: {args.db}")
        sys.exit(1)

    print(f"[1/4] Connecting to master state database: {args.db.resolve()}...")
    conn = sqlite3.connect(str(args.db))
    c = conn.cursor()

    rows = c.execute(
        "SELECT id, raw_text, cleaned_text, source_name, consensus_label, shannon_entropy, fleiss_kappa, status, model_outputs_json, timestamp FROM ensemble_decisions ORDER BY id ASC"
    ).fetchall()
    print(f"Loaded {len(rows):,} total evaluated records.")

    agreed_rows, disagreed_rows, error_rows = [], [], []
    db_updates = []

    for r in rows:
        (
            rec_id,
            raw_text,
            cleaned_text,
            source,
            old_label,
            entropy,
            kappa,
            old_status,
            outputs_json,
            ts,
        ) = r
        outputs = json.loads(outputs_json)

        mapped_vals = []
        for model_name, out in outputs.items():
            lbl = str(out.get("label", "error")).lower().strip()
            if lbl in ["neutral", "objective"]:
                mapped = "neutral_factual"
            elif lbl in ["positive", "negative"]:
                mapped = lbl
            else:
                mapped = "error"
            if mapped != "error":
                mapped_vals.append(mapped)

        if len(mapped_vals) >= 2:
            counts = Counter(mapped_vals)
            most_common_lbl, count = counts.most_common(1)[0]
            if count >= 2:
                new_status = "RESOLVED"
                new_label = most_common_lbl
            else:
                new_status = "DISAGREED"
                new_label = "disagreed"
        else:
            new_status = "ERROR"
            new_label = "error"

        record = {
            "id": rec_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "source": source,
            "consensus_label": new_label,
            "shannon_entropy": entropy,
            "fleiss_kappa": kappa,
            "status": new_status,
            "model_outputs": outputs,
            "timestamp": ts,
        }

        db_updates.append((new_label, new_status, rec_id))
        if new_status == "RESOLVED":
            agreed_rows.append(record)
        elif new_status == "DISAGREED":
            disagreed_rows.append(record)
        else:
            error_rows.append(record)

    print("[2/4] Synchronizing database table ensemble_decisions...")
    with conn:
        conn.executemany(
            "UPDATE ensemble_decisions SET consensus_label = ?, status = ? WHERE id = ?;",
            db_updates,
        )
    conn.close()
    print("Database sync complete!")

    print(f"[3/4] Exporting JSONL corpora to {args.output_dir.resolve()}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agreed_jsonl = args.output_dir / "purva_l2_agreed.jsonl"
    disagreed_jsonl = args.output_dir / "purva_l2_disagreed.jsonl"
    error_jsonl = args.output_dir / "purva_l2_errors.jsonl"
    audit_sample_csv = args.output_dir / "purva_l2_human_audit_sample.csv"

    with open(agreed_jsonl, "w", encoding="utf-8") as f:
        for rec in agreed_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(disagreed_jsonl, "w", encoding="utf-8") as f:
        for rec in disagreed_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(error_jsonl, "w", encoding="utf-8") as f:
        for rec in error_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[4/4] Exporting CSV spreadsheets and 1% control sample...")
    headers = [
        "id",
        "raw_text",
        "consensus_label",
        "shannon_entropy",
        "fleiss_kappa",
        "model_outputs",
    ]

    with open(
        args.output_dir / "purva_l2_agreed.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in agreed_rows:
            writer.writerow(
                [
                    rec["id"],
                    rec["raw_text"],
                    rec["consensus_label"],
                    rec["shannon_entropy"],
                    rec["fleiss_kappa"],
                    json.dumps(rec["model_outputs"], ensure_ascii=False),
                ]
            )

    with open(
        args.output_dir / "purva_l2_disagreed.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in disagreed_rows:
            writer.writerow(
                [
                    rec["id"],
                    rec["raw_text"],
                    rec["consensus_label"],
                    rec["shannon_entropy"],
                    rec["fleiss_kappa"],
                    json.dumps(rec["model_outputs"], ensure_ascii=False),
                ]
            )

    sample_size = max(50, int(len(agreed_rows) * 0.01))
    if agreed_rows:
        random.seed(args.seed)
        audit_sample = random.sample(agreed_rows, min(sample_size, len(agreed_rows)))
        with open(audit_sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for rec in audit_sample:
                writer.writerow(
                    [
                        rec["id"],
                        rec["raw_text"],
                        rec["consensus_label"],
                        rec["shannon_entropy"],
                        rec["fleiss_kappa"],
                        json.dumps(rec["model_outputs"], ensure_ascii=False),
                    ]
                )

    print("\n" + "=" * 70)
    print("=== EXPORT SUCCESSFUL (3-CLASS RULES) ===")
    print("=" * 70)
    print(
        f"  • Agreed Consensus (RESOLVED) : {len(agreed_rows):,} rows -> {agreed_jsonl.name} & .csv"
    )
    print(
        f"  • Disagreed Items  (DISAGREED): {len(disagreed_rows):,} rows -> {disagreed_jsonl.name} & .csv"
    )
    print(
        f"  • Human Audit Sample (1%)     : {len(audit_sample if agreed_rows else []):,} rows -> {audit_sample_csv.name}"
    )
    print(
        f"  • Error Log                   : {len(error_rows):,} rows -> {error_jsonl.name}"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
