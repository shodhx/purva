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
import math
import random
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# Import Dawid-Skene EM and Shannon Entropy from the consensus engine
sys.path.insert(0, str(Path(__file__).resolve().parent))
from consensus import dawid_skene_aggregation, calculate_shannon_entropy

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURVA L2: 3-Class Taxonomy Dataset Synchronizer & Exporter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--db", "-d",
        type=Path,
        default=Path("data/purva_ensemble.db"),
        help="Path to master SQLite consensus database."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data"),
        help="Directory where updated JSONL and CSV files will be written."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for generating the reproducible 1%% control audit sample."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if not args.db.exists():
        print(f"[ERROR] Database file not found: {args.db}")
        sys.exit(1)

    print(f"[1/5] Connecting to master state database: {args.db.resolve()}...")
    conn = sqlite3.connect(str(args.db))
    c = conn.cursor()

    rows = c.execute("SELECT id, raw_text, cleaned_text, source_name, consensus_label, shannon_entropy, fleiss_kappa, status, model_outputs_json, timestamp FROM ensemble_decisions ORDER BY id ASC").fetchall()
    print(f"Loaded {len(rows):,} total evaluated records.")

    # ── Pass 1: Map all model outputs to 3-class taxonomy and collect batch ──
    print("[2/5] Mapping 4-class labels to 3-class taxonomy (neutral/objective → neutral_factual)...")
    categories = ["positive", "negative", "neutral_factual"]
    batch_preds: List[Dict[str, str]] = []
    row_metadata: List[Dict[str, Any]] = []

    for r in rows:
        rec_id, raw_text, cleaned_text, source, old_label, entropy, kappa, old_status, outputs_json, ts = r
        outputs = json.loads(outputs_json)

        mapped_preds: Dict[str, str] = {}
        for model_name, out in outputs.items():
            lbl = str(out.get("label", "error")).lower().strip()
            if lbl in ["neutral", "objective"]:
                mapped_preds[model_name] = "neutral_factual"
            elif lbl in ["positive", "negative"]:
                mapped_preds[model_name] = lbl
            # Labels that don't map (e.g., "error") are excluded from this model's vote

        batch_preds.append(mapped_preds)
        row_metadata.append({
            "id": rec_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "source": source,
            "kappa": kappa,
            "outputs": outputs,
            "timestamp": ts,
            "n_valid_votes": len(mapped_preds),
        })

    # ── Pass 2: Run Dawid-Skene EM over the entire corpus batch ──
    print(f"[3/5] Running Dawid-Skene EM aggregation across {len(batch_preds):,} items...")
    ds_results = dawid_skene_aggregation(batch_preds, categories, max_iter=25)

    # ── Pass 3: Route items using DS posterior confidence + Shannon entropy ──
    print("[4/5] Routing items via Dawid-Skene confidence and Shannon entropy thresholds...")
    agreed_rows, disagreed_rows, error_rows = [], [], []
    db_updates = []

    for i, meta in enumerate(row_metadata):
        if meta["n_valid_votes"] < 2:
            # Fewer than 2 models returned a valid label — pipeline error
            new_status = "ERROR"
            new_label = "error"
            new_entropy = 0.0
        else:
            consensus_label, ds_conf, prob_dist = ds_results[i]
            new_entropy = calculate_shannon_entropy(prob_dist)

            if new_entropy <= 1.0 and ds_conf >= 0.60:
                new_status = "RESOLVED"
                new_label = consensus_label
            else:
                new_status = "DISAGREED"
                new_label = "disagreed"

        record = {
            "id": meta["id"],
            "raw_text": meta["raw_text"],
            "cleaned_text": meta["cleaned_text"],
            "source": meta["source"],
            "consensus_label": new_label,
            "shannon_entropy": round(new_entropy, 4),
            "fleiss_kappa": meta["kappa"],
            "status": new_status,
            "model_outputs": meta["outputs"],
            "timestamp": meta["timestamp"],
        }

        db_updates.append((new_label, new_status, round(new_entropy, 4), meta["id"]))
        if new_status == "RESOLVED":
            agreed_rows.append(record)
        elif new_status == "DISAGREED":
            disagreed_rows.append(record)
        else:
            error_rows.append(record)

    print("[5/5] Synchronizing database table ensemble_decisions...")
    with conn:
        conn.executemany("UPDATE ensemble_decisions SET consensus_label = ?, status = ?, shannon_entropy = ? WHERE id = ?;", db_updates)
    conn.close()
    print("Database sync complete!")
    
    print(f"\n[6/7] Exporting JSONL corpora to {args.output_dir.resolve()}...")
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
            
    print(f"[7/7] Exporting CSV spreadsheets and 1% control sample...")
    headers = ["id", "raw_text", "consensus_label", "shannon_entropy", "fleiss_kappa", "model_outputs"]
    
    with open(args.output_dir / "purva_l2_agreed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in agreed_rows:
            writer.writerow([rec["id"], rec["raw_text"], rec["consensus_label"], rec["shannon_entropy"], rec["fleiss_kappa"], json.dumps(rec["model_outputs"], ensure_ascii=False)])
            
    with open(args.output_dir / "purva_l2_disagreed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in disagreed_rows:
            writer.writerow([rec["id"], rec["raw_text"], rec["consensus_label"], rec["shannon_entropy"], rec["fleiss_kappa"], json.dumps(rec["model_outputs"], ensure_ascii=False)])
            
    sample_size = max(50, int(len(agreed_rows) * 0.01))
    if agreed_rows:
        random.seed(args.seed)
        audit_sample = random.sample(agreed_rows, min(sample_size, len(agreed_rows)))
        with open(audit_sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for rec in audit_sample:
                writer.writerow([rec["id"], rec["raw_text"], rec["consensus_label"], rec["shannon_entropy"], rec["fleiss_kappa"], json.dumps(rec["model_outputs"], ensure_ascii=False)])
                
    print("\n" + "="*70)
    print("=== EXPORT SUCCESSFUL (3-CLASS RULES) ===")
    print("="*70)
    print(f"  • Agreed Consensus (RESOLVED) : {len(agreed_rows):,} rows -> {agreed_jsonl.name} & .csv")
    print(f"  • Disagreed Items  (DISAGREED): {len(disagreed_rows):,} rows -> {disagreed_jsonl.name} & .csv")
    print(f"  • Human Audit Sample (1%)     : {len(audit_sample if agreed_rows else []):,} rows -> {audit_sample_csv.name}")
    print(f"  • Error Log                   : {len(error_rows):,} rows -> {error_jsonl.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
