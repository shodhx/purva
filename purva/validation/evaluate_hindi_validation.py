"""Dawid–Skene aggregation + gold evaluation for the Hindi validation run.

Reads the five judge shards for the Hindi validation set (written by
purva/committee/run_judge.py into data/committee/hindi_validation/), builds
the (N, J) vote matrix with the SAME derivation rule the main pipeline uses
(purva/aggregate/_common.build_vote_matrix: a judge's derived label is
"objective" when its subjectivity vote is objective, otherwise its polarity
vote; a judge with no parsed output is -1 and is marginalised, never
imputed), and runs the validated primary aggregation exactly as configured
for the main corpus:

  standard Dawid–Skene EM, four-class label space (objective, positive,
  negative, neutral), with the identifiability priors diag_prior=5.0,
  off_diag_prior=0.5, class_prior_strength=500.0, class prior anchored to
  raw-vote frequency — purva/aggregate/dawid_skene.py's DSConfig defaults,
  which are the values recorded in data/aggregation_report.md section 0.
  Stratified DS is NOT run: it is not the validated primary method (see
  aggregation_report.md §0 / PROTOCOL.md CHANGELOG v1.7).

Majority vote (same tie-break rule as the main pipeline: first class in
canonical label order wins) is computed for comparison.

Then, against the Hindi gold labels mapped into our four-class space:
  - per-judge accuracy
  - Dawid–Skene consensus accuracy
  - majority-vote accuracy
  - accuracy on low- vs high-entropy items (entropy from the DS posterior,
    split at the same 0.3 normalised threshold used for the main corpus's
    routing entropy distribution, data/aggregation_report.md §6)
  - per-judge subjectivity rates and their ordering

Outputs data/hindi_validation_report.json (all numbers) and a companion
data/.hindi_validation_eval.jsonl with per-item posteriors/labels for
inspection. The markdown report is written by
purva/validation/write_hindi_validation_report.py.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from purva.aggregate import dawid_skene as ds
from purva.validation._hindi_common import (
    ENTROPY_SPLIT,
    GOLD_LABELS,
    JUDGE_SHORT_NAMES,
    LABELS,
    LABEL_TO_IDX,
    load_validation_rows,
)

DEFAULT_SET = Path("data/hindi_validation_set.jsonl")
DEFAULT_SHARD_DIR = Path("data/committee/hindi_validation")
DEFAULT_OUT_JSON = Path("data/hindi_validation_report.json")
DEFAULT_OUT_JSONL = Path("data/.hindi_validation_eval.jsonl")


def load_shards(rows: list[dict], shard_dir: Path) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """{judge: {id: shard_row}} plus {judge: sidecar meta}, with duplicate
    ids hard-failed."""
    votes_by_judge: dict[str, dict[str, dict]] = {}
    meta_by_judge: dict[str, dict] = {}
    for stem, short in JUDGE_SHORT_NAMES.items():
        shard_path = shard_dir / f"{stem}__judge_prompt_v1.jsonl"
        if not shard_path.exists():
            raise SystemExit(f"missing judge shard: {shard_path}")
        by_id: dict[str, dict] = {}
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in by_id:
                raise SystemExit(f"judge {short}: duplicate id {r['id']}")
            by_id[r["id"]] = r
        meta_path = shard_path.with_suffix(".meta.json")
        meta_by_judge[short] = json.loads(meta_path.read_text(encoding="utf-8"))
        votes_by_judge[short] = by_id
    return votes_by_judge, meta_by_judge


def build_vote_matrix(rows: list[dict], votes_by_judge: dict[str, dict[str, dict]]) -> np.ndarray:
    """(N, J) int8 class-index matrix in canonical judge order; -1 = no vote."""
    judges = list(JUDGE_SHORT_NAMES.values())
    n = len(rows)
    votes = np.full((n, len(judges)), -1, dtype=np.int8)
    for i, row in enumerate(rows):
        rid = row["id"]
        for j, judge in enumerate(judges):
            r = votes_by_judge[judge].get(rid)
            if r is None or r.get("parse_failed"):
                continue
            subj = r.get("subjectivity")
            if subj not in ("objective", "subjective"):
                continue
            if subj == "objective":
                derived = "objective"
            else:
                pol = r.get("polarity")
                if pol not in ("positive", "negative", "neutral", "mixed"):
                    continue
                derived = pol
            votes[i, j] = LABEL_TO_IDX[derived]
    return votes


def check_config_consistency(meta_by_judge: dict[str, dict]) -> None:
    """The five judges' sidecars must record identical decoding conditions
    (repo identity and revision are per-judge by nature; everything that
    must be shared is asserted shared). Mirrors merge_shards.py's
    check_config_consistency for the main-corpus chunks."""
    shared_keys = ("prompt_file", "prompt_file_sha256", "seed", "max_model_len",
                   "max_num_seqs", "guided_decoding", "quantization")
    for key in shared_keys:
        values = {m.get(key) for m in meta_by_judge.values()}
        if len(values) != 1:
            raise SystemExit(f"judge sidecars differ on {key!r}: {values} — conditions were not identical")


def accuracy(pred: list[str], gold: list[str]) -> float:
    assert len(pred) == len(gold) and len(gold) > 0
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def confusion(pred: list[str], gold: list[str]) -> dict:
    """counts[gold][pred] — rows are gold, columns are predicted."""
    counts = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        counts[g][p] += 1
    per_class_recall = {}
    for g in LABELS:
        n_g = sum(counts[g].values())
        per_class_recall[g] = counts[g][g] / n_g if n_g else None
    return {"counts": counts, "per_gold_recall": per_class_recall}


def norm_entropy(posteriors: np.ndarray) -> np.ndarray:
    p = np.clip(posteriors, 1e-300, 1.0)
    return (-(p * np.log(p)).sum(axis=1)) / np.log(posteriors.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-set", default=str(DEFAULT_SET))
    ap.add_argument("--shard-dir", default=str(DEFAULT_SHARD_DIR))
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-jsonl", default=str(DEFAULT_OUT_JSONL))
    args = ap.parse_args()

    set_path = Path(args.validation_set)
    shard_dir = Path(args.shard_dir)
    out_json = Path(args.output_json)
    out_jsonl = Path(args.output_jsonl)

    rows, set_meta = load_validation_rows(set_path)
    n = len(rows)
    gold = [r["gold_label"] for r in rows]
    print(f"loaded {n} validation rows; gold distribution: {dict(Counter(gold))}")

    votes_by_judge, meta_by_judge = load_shards(rows, shard_dir)
    check_config_consistency(meta_by_judge)
    judges = list(JUDGE_SHORT_NAMES.values())
    missing = {j: sum(1 for r in rows if r["id"] not in votes_by_judge[j]) for j in judges}
    parse_failed = {
        j: sum(1 for r in rows if (v := votes_by_judge[j].get(r["id"])) is not None and v.get("parse_failed"))
        for j in judges
    }
    print(f"cross-judge shared run config: identical (checked sidecars)")
    print(f"rows missing per judge: {missing}; parse-failed per judge: {parse_failed}")

    votes = build_vote_matrix(rows, votes_by_judge)
    n_votes = int((votes != -1).sum())
    raw_freq = {LABELS[i]: int((votes == i).sum()) for i in range(len(LABELS))}
    print(f"vote matrix {votes.shape}; raw vote distribution (4-class): {raw_freq}")

    # --- validated primary method: standard DS, four-class, identifiability priors ---
    anchor = np.array([raw_freq[l] / max(1, n_votes) for l in LABELS])
    config = ds.DSConfig()  # defaults ARE the validated config; asserted below
    assert (config.diag_prior, config.off_diag_prior, config.class_prior_strength) == (5.0, 0.5, 500.0), (
        "DSConfig defaults are no longer the validated prior configuration — update this script deliberately"
    )
    print("\n=== standard Dawid-Skene (four-class, identifiability priors) ===")
    t0 = time.time()
    result = ds.run_dawid_skene(votes, config, n_classes=len(LABELS), class_prior_anchor=anchor)
    ds_runtime = time.time() - t0
    print(f"iters={result.n_iter} converged={result.converged} runtime={ds_runtime:.1f}s "
          f"final_ll={result.log_likelihood[-1]:.2f}")

    ds_post = result.posteriors
    ds_labels = [LABELS[i] for i in np.argmax(ds_post, axis=1)]
    ent_norm = norm_entropy(ds_post)

    # --- majority vote, same tie-break rule as the main pipeline ---
    counts = np.zeros((n, len(LABELS)))
    for k in range(len(LABELS)):
        counts[:, k] = (votes == k).sum(axis=1)
    mv_labels = [LABELS[i] for i in np.argmax(counts, axis=1)]
    max_count = counts.max(axis=1)
    tie_count = int(((counts == max_count[:, None]).sum(axis=1) > 1).sum())

    # --- per-judge labels and accuracy vs gold ---
    per_judge_labels: dict[str, list[str | None]] = {}
    for j, judge in enumerate(judges):
        col = votes[:, j]
        per_judge_labels[judge] = [None if col[i] == -1 else LABELS[col[i]] for i in range(n)]

    eval_block: dict = {}
    for judge in judges:
        valid = [(l, g) for l, g in zip(per_judge_labels[judge], gold) if l is not None]
        eval_block[judge] = {
            "n_votes": len(valid),
            "accuracy": accuracy([l for l, _ in valid], [g for _, g in valid]),
        }

    ds_acc = accuracy(ds_labels, gold)
    mv_acc = accuracy(mv_labels, gold)
    print(f"\nDS accuracy vs gold:     {ds_acc:.4f}")
    print(f"MV accuracy vs gold:     {mv_acc:.4f}")
    for judge in judges:
        print(f"{judge:8s} accuracy vs gold: {eval_block[judge]['accuracy']:.4f} "
              f"(n={eval_block[judge]['n_votes']})")

    # --- entropy-split evaluation ---
    low_mask = ent_norm <= ENTROPY_SPLIT
    high_mask = ~low_mask
    n_low, n_high = int(low_mask.sum()), int(high_mask.sum())

    def acc_on(mask: np.ndarray, labels: list[str]) -> float | None:
        sel = [(labels[i], gold[i]) for i in range(n) if mask[i]]
        return accuracy([l for l, _ in sel], [g for _, g in sel]) if sel else None

    entropy_block = {
        "threshold_entropy_norm": ENTROPY_SPLIT,
        "n_low": n_low,
        "n_high": n_high,
        "ds_accuracy_low": acc_on(low_mask, ds_labels),
        "ds_accuracy_high": acc_on(high_mask, ds_labels),
        "mv_accuracy_low": acc_on(low_mask, mv_labels),
        "mv_accuracy_high": acc_on(high_mask, mv_labels),
        "per_judge_accuracy_low": {},
        "per_judge_accuracy_high": {},
    }
    for judge in judges:
        entropy_block["per_judge_accuracy_low"][judge] = acc_on(low_mask, per_judge_labels[judge])
        entropy_block["per_judge_accuracy_high"][judge] = acc_on(high_mask, per_judge_labels[judge])
    print(f"\nentropy split (threshold {ENTROPY_SPLIT} normalised): low={n_low} high={n_high}")
    print(f"DS acc low={entropy_block['ds_accuracy_low']:.4f} high={entropy_block['ds_accuracy_high']:.4f}")
    print(f"MV acc low={entropy_block['mv_accuracy_low']:.4f} high={entropy_block['mv_accuracy_high']:.4f}")

    # --- subjectivity rates (raw shard votes, not the 4-class derivation) ---
    subj_rates: dict = {}
    for judge in judges:
        subj_counter = Counter()
        for row in rows:
            r = votes_by_judge[judge].get(row["id"])
            if r is None or r.get("parse_failed"):
                continue
            s = r.get("subjectivity")
            if s in ("objective", "subjective"):
                subj_counter[s] += 1
        total = sum(subj_counter.values())
        subj_rates[judge] = {
            "n": total,
            "objective": subj_counter["objective"],
            "subjective": subj_counter["subjective"],
            "subjective_rate": subj_counter["subjective"] / total if total else None,
        }
        print(f"{judge:8s} subjective rate: {subj_rates[judge]['subjective_rate']:.4f} (n={total})")
    subj_order = sorted(subj_rates, key=lambda k: -(subj_rates[k]["subjective_rate"] or 0))
    print(f"subjectivity ordering (highest subjective rate first): {subj_order}")

    report = {
        "what_this_validates": (
            "The judge-committee + Dawid-Skene aggregation PIPELINE, validated against existing "
            "third-party Hindi gold labels. This does NOT validate the Bhojpuri labels, and the "
            "label-scheme mapping (3-class polarity gold onto our 4-class space, with no subjectivity "
            "stage in the gold) introduces error that bounds every accuracy estimate from below."
        ),
        "dataset": set_meta,
        "run": {
            "judges": judges,
            "prompt_file": meta_by_judge[judges[0]].get("prompt_file"),
            "prompt_sha256": meta_by_judge[judges[0]].get("prompt_file_sha256"),
            "per_judge_sidecars": meta_by_judge,
            "rows_missing_per_judge": missing,
            "parse_failed_per_judge": parse_failed,
        },
        "label_space": {
            "labels": list(LABELS),
            "path": "four-class (objective/positive/negative/neutral), matching the main corpus's validated label space",
            "gold_label_space": list(GOLD_LABELS),
            "mapping": {"negative": "negative", "neutral": "neutral", "positive": "positive"},
            "mapping_loss": (
                "gold has no objective class and no subjectivity stage; every gold item is a polarity "
                "judgment, so 'objective' can never be a correct answer under the gold — any judge or "
                "consensus 'objective' vote is scored wrong even when it may be defensible. See the report."
            ),
        },
        "ds_config": {"diag_prior": config.diag_prior, "off_diag_prior": config.off_diag_prior,
                      "class_prior_strength": config.class_prior_strength, "alpha": config.alpha,
                      "max_iter": config.max_iter, "tol": config.tol},
        "ds_result": {"n_iter": result.n_iter, "converged": result.converged,
                      "final_log_likelihood": result.log_likelihood[-1],
                      "runtime_seconds": round(ds_runtime, 1),
                      "class_prior": {LABELS[i]: float(result.class_prior[i]) for i in range(len(LABELS))}},
        "raw_vote_distribution": raw_freq,
        "gold_distribution": dict(Counter(gold)),
        "per_judge_accuracy": eval_block,
        "ds_accuracy": ds_acc,
        "majority_vote_accuracy": mv_acc,
        "majority_vote_ties": {"count": tie_count, "rate": tie_count / n,
                               "tie_break_rule": "first class in canonical order wins"},
        "ds_vs_mv_label_disagreement": float(np.mean([a != b for a, b in zip(ds_labels, mv_labels)])),
        "entropy_split": entropy_block,
        "subjectivity": {"rates": subj_rates, "ordering_by_subjective_rate_desc": subj_order},
        "ds_confusion_vs_gold": confusion(ds_labels, gold),
        "mv_confusion_vs_gold": confusion(mv_labels, gold),
    }

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_json}")

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            fh.write(json.dumps({
                "id": row["id"],
                "gold_label": gold[i],
                "ds_label": ds_labels[i],
                "ds_posterior": {LABELS[k]: float(ds_post[i, k]) for k in range(len(LABELS))},
                "entropy_norm": float(ent_norm[i]),
                "mv_label": mv_labels[i],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl}")


if __name__ == "__main__":
    main()
