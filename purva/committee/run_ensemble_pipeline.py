#!/usr/bin/env python3
"""
PURVA Layer 2: Master Ensemble Execution Pipeline
=================================================
A* Conference Standard CLI launcher for running multi-model consensus,
Expectation-Maximization Dawid-Skene aggregation, and corpus export.

Usage:
    python run_ensemble_pipeline.py --input data/corpus_clean.jsonl --db data/purva_ensemble.db
    python run_ensemble_pipeline.py --export-only --db data/purva_ensemble.db --out data/
"""

import sys
import argparse
from pathlib import Path

# Automatically resolve module paths relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from consensus import CommitteeOrchestrator, EnsembleStateManager
except ImportError:
    # If running inside repo root
    from purva.committee.consensus import CommitteeOrchestrator, EnsembleStateManager


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURVA L2: Multi-LLM Committee Consensus Pipeline (ACL/ARR Release)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus_clean.jsonl",
        help="Path to input scraped Bhojpuri corpus in JSONL format.",
    )
    parser.add_argument(
        "--db",
        "-d",
        type=Path,
        default=PROJECT_ROOT / "data" / "purva_ensemble.db",
        help="Path to SQLite master state database (WAL checkpointing enabled).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory where final consensus CSV and JSONL files will be exported.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=25,
        help="Number of sentences per evaluation batch.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent execution workers.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip LLM evaluation and re-export CSV/JSONL files from existing database.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n" + "=" * 70)
    print("=== PURVA L2: MULTI-LLM COMMITTEE CONSENSUS PIPELINE ===")
    print("=" * 70)
    print(f"  • Input Corpus      : {args.input.resolve()}")
    print(f"  • Master State DB   : {args.db.resolve()}")
    print(f"  • Export Directory  : {args.output_dir.resolve()}")
    print(f"  • Batch Size        : {args.batch_size}")
    print(f"  • Workers           : {args.workers}")
    print("=" * 70 + "\n")

    if args.export_only:
        print(
            "[MODE] Export-only flag detected. Exporting datasets from SQLite store..."
        )
        if not args.db.exists():
            print(f"[ERROR] Database file not found: {args.db}")
            sys.exit(1)
        state_mgr = EnsembleStateManager(args.db)
        state_mgr.export_corpora(output_dir=args.output_dir)
        state_mgr.close()
        print("[SUCCESS] Export completed.")
        return

    orchestrator = CommitteeOrchestrator(db_path=args.db)

    if not args.input.exists():
        print(f"[ERROR] Input corpus file not found: {args.input}")
        sys.exit(1)

    print("[MODE] Starting continuous batch evaluation pipeline...")
    orchestrator.run_pipeline(
        jsonl_path=args.input, batch_size=args.batch_size, max_workers=args.workers
    )


if __name__ == "__main__":
    main()
