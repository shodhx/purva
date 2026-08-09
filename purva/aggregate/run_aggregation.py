"""Driver for Phase 4 aggregation (PROTOCOL.md §5). Runs every aggregator
over the full corpus and writes:

  data/purva_aggregated.jsonl — per sentence: id, the five raw votes
    (unchanged), and each aggregator's posterior/argmax/entropy.
  data/purva_aggregated.meta.json — aggregator configs, iteration counts,
    convergence status, runtime, which label-space path was taken, and
    which method is the validated primary consensus.

Label-space AND primary-method decision (see data/aggregation_report.md's
identifiability section for the full incident): five-class Dawid-Skene,
regularised with a diagonal-favouring Dirichlet prior on every confusion
matrix and a class prior shrunk toward the observed raw-vote frequency, is
attempted first for both the standard and covariate-stratified variants.
Each is checked separately against test_aggregation.py's permanent
invariants (unanimous items must keep their label; no class may exceed 3x
its raw-vote share) *before* anything is written.

What the evidence actually showed on this corpus: at five classes, BOTH
variants fail (the "mixed" catch-all). Dropping "mixed" (four-class
fallback, PROTOCOL.md CHANGELOG v1.7) fixes standard DS cleanly, but
stratified DS — PROTOCOL.md §5's designated *primary* method — still fails,
with the same pathology now landing on "neutral" instead (raw-vote share
4.3%, stratified-DS consensus share ~24%, ratio ~5.7-6.1x, robust across
every shrinkage_k0/diag_prior/class_prior_strength combination tried). This
is a structural over-parameterisation problem (a confusion matrix per
judge per stratum, for a corpus whose class balance is already thin in
several strata), not something these priors alone can patch for the
stratified variant. So: standard DS becomes the validated primary
consensus; stratified DS is still run and shipped (its per-stratum
confusion matrices remain legitimate for the reliability-hypothesis
analysis in analysis.py), but is explicitly marked non-primary and
excluded from routing/consensus decisions. This is reported, not hidden —
see data/aggregation_report.md section 0.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import baselines as bl
from . import dawid_skene as ds
from ._common import JUDGES, LABELS, argmax_labels, build_strata, build_vote_matrix, load_master, posterior_entropy, raw_vote_frequency
from .test_aggregation import run_invariant_checks

DEFAULT_OUTPUT_JSONL = "data/purva_aggregated.jsonl"
DEFAULT_OUTPUT_META = "data/purva_aggregated.meta.json"

MAX_CLASS_RATIO = 3.0
# Preference order when more than one method passes the invariant checks at
# a given label space — stratified DS is PROTOCOL.md §5's designated
# primary method, so it wins if (and only if) it's actually validated.
METHOD_PREFERENCE = ("stratified_dawid_skene", "dawid_skene")


def _raw_votes_per_item(df, i: int) -> dict:
    votes = {}
    for judge in JUDGES:
        subj = df.at[df.index[i], f"judge_{judge}_subjectivity"]
        if subj is None or (isinstance(subj, float) and np.isnan(subj)):
            votes[judge] = None
            continue
        votes[judge] = {
            "subjectivity": subj,
            "polarity": df.at[df.index[i], f"judge_{judge}_polarity"],
            "confidence": df.at[df.index[i], f"judge_{judge}_confidence"],
            "domain": df.at[df.index[i], f"judge_{judge}_domain"],
            "narrative_voice": df.at[df.index[i], f"judge_{judge}_narrative_voice"],
            "sentiment_target": df.at[df.index[i], f"judge_{judge}_sentiment_target"],
            "rationale": df.at[df.index[i], f"judge_{judge}_rationale"],
        }
    return votes


def _method_block(posteriors: np.ndarray, labels: tuple[str, ...]) -> list[dict]:
    lbls = argmax_labels(posteriors, labels)
    ent_nat, ent_norm = posterior_entropy(posteriors)
    return [
        {
            "posterior": {labels[k]: float(posteriors[i, k]) for k in range(len(labels))},
            "label": lbls[i],
            "entropy_nat": float(ent_nat[i]),
            "entropy_norm": float(ent_norm[i]),
        }
        for i in range(posteriors.shape[0])
    ]


def _run_ds_pair(votes, strata, strata_keys, labels, ds_config, sds_config):
    anchor = raw_vote_frequency(votes, labels)
    n_classes = len(labels)

    print(f"--- standard DS (n_classes={n_classes}) ---")
    t0 = time.time()
    ds_result = ds.run_dawid_skene(votes, ds_config, n_classes=n_classes, class_prior_anchor=anchor)
    ds_runtime = time.time() - t0
    print(f"iters={ds_result.n_iter} converged={ds_result.converged} runtime={ds_runtime:.1f}s "
          f"final_ll={ds_result.log_likelihood[-1]:.2f}")

    print(f"--- stratified DS (n_classes={n_classes}) ---")
    t0 = time.time()
    sds_result = ds.run_stratified_dawid_skene(votes, strata, strata_keys, sds_config, n_classes=n_classes, class_prior_anchor=anchor)
    sds_runtime = time.time() - t0
    print(f"iters={sds_result.n_iter} converged={sds_result.converged} runtime={sds_runtime:.1f}s "
          f"final_ll={sds_result.log_likelihood[-1]:.2f}")

    return ds_result, ds_runtime, sds_result, sds_runtime


def _check_both(votes, labels, ds_labels, sds_labels, max_ratio):
    _, ds_report = run_invariant_checks(votes, {"dawid_skene": ds_labels}, labels, max_ratio)
    _, sds_report = run_invariant_checks(votes, {"stratified_dawid_skene": sds_labels}, labels, max_ratio)
    ds_passed = ds_report["unanimity"]["passed"] and ds_report["class_ratio"]["passed"]
    sds_passed = sds_report["unanimity"]["passed"] and sds_report["class_ratio"]["passed"]
    return ds_passed, sds_passed, ds_report, sds_report


def _pick_primary(ds_passed: bool, sds_passed: bool) -> str | None:
    passed = {"dawid_skene": ds_passed, "stratified_dawid_skene": sds_passed}
    for name in METHOD_PREFERENCE:
        if passed[name]:
            return name
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    ap.add_argument("--output-meta", default=DEFAULT_OUTPUT_META)
    ap.add_argument("--mace-restarts", type=int, default=3)
    ap.add_argument("--diag-prior", type=float, default=5.0)
    ap.add_argument("--off-diag-prior", type=float, default=0.5)
    ap.add_argument("--class-prior-strength", type=float, default=500.0)
    ap.add_argument("--max-ratio", type=float, default=MAX_CLASS_RATIO)
    ap.add_argument("--skip-mace", action="store_true")
    ap.add_argument("--skip-glad", action="store_true")
    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_meta = Path(args.output_meta)

    print(f"loading {args.master}")
    df = load_master(args.master)
    votes5 = build_vote_matrix(df)
    strata, strata_keys = build_strata(df)
    item_ids = df["id"].to_numpy()
    print(f"{len(df)} items, {votes5.shape[1]} judges, {len(strata_keys)} strata")

    ds_config = ds.DSConfig(diag_prior=args.diag_prior, off_diag_prior=args.off_diag_prior, class_prior_strength=args.class_prior_strength)
    sds_config = ds.StratifiedDSConfig(diag_prior=args.diag_prior, off_diag_prior=args.off_diag_prior, class_prior_strength=args.class_prior_strength)

    print("\n=== attempt 1: five-class DS with identifiability priors ===")
    ds5, ds5_runtime, sds5, sds5_runtime = _run_ds_pair(votes5, strata, strata_keys, LABELS, ds_config, sds_config)
    ds5_labels = argmax_labels(ds5.posteriors, LABELS)
    sds5_labels = argmax_labels(sds5.posteriors, LABELS)
    ds5_passed, sds5_passed, ds5_report, sds5_report = _check_both(votes5, LABELS, ds5_labels, sds5_labels, args.max_ratio)
    print(f"five-class invariants: dawid_skene={'PASS' if ds5_passed else 'FAIL'} "
          f"stratified_dawid_skene={'PASS' if sds5_passed else 'FAIL'}")

    primary = _pick_primary(ds5_passed, sds5_passed)

    if primary is not None:
        path = "five_class_with_priors"
        labels, final_votes = LABELS, votes5
        final_ds, final_ds_runtime, final_sds, final_sds_runtime = ds5, ds5_runtime, sds5, sds5_runtime
        method_validity = {"dawid_skene": ds5_passed, "stratified_dawid_skene": sds5_passed}
        invariant_reports = {"dawid_skene": ds5_report, "stratified_dawid_skene": sds5_report}
    else:
        print("\nneither DS variant passes the invariant checks at five classes — "
              "falling back to four-class DS; 'mixed' is dropped from the label space "
              "and reported at the raw-vote level only (PROTOCOL.md CHANGELOG v1.7).")
        path = "four_class_fallback"
        labels = LABELS[:4]
        mixed_idx = LABELS.index("mixed")
        final_votes = votes5.copy()
        final_votes[final_votes == mixed_idx] = -1  # a judge's "mixed" vote becomes an abstention under this label space

        final_ds, final_ds_runtime, final_sds, final_sds_runtime = _run_ds_pair(final_votes, strata, strata_keys, labels, ds_config, sds_config)
        ds4_labels = argmax_labels(final_ds.posteriors, labels)
        sds4_labels = argmax_labels(final_sds.posteriors, labels)
        ds4_passed, sds4_passed, ds4_report, sds4_report = _check_both(final_votes, labels, ds4_labels, sds4_labels, args.max_ratio)
        print(f"four-class invariants: dawid_skene={'PASS' if ds4_passed else 'FAIL'} "
              f"stratified_dawid_skene={'PASS' if sds4_passed else 'FAIL'}")
        method_validity = {"dawid_skene": ds4_passed, "stratified_dawid_skene": sds4_passed}
        invariant_reports = {"dawid_skene": ds4_report, "stratified_dawid_skene": sds4_report}

        primary = _pick_primary(ds4_passed, sds4_passed)
        if primary is None:
            raise SystemExit(
                "four-class DS still fails the invariant checks for EVERY variant — refusing to write a "
                "broken aggregation. See the printed reports above; this needs a person, not another "
                "silent fallback."
            )
        if primary != "stratified_dawid_skene":
            print(
                "\nNOTE: stratified DS (PROTOCOL.md §5's designated primary method) does not pass the "
                f"invariant checks even at four classes — see data/aggregation_report.md section 0. "
                f"'{primary}' is used as the validated primary consensus instead; stratified DS is still "
                "shipped in the output for comparison/ablation, and its per-stratum confusion matrices "
                "remain valid for the reliability-hypothesis analysis, but its overall consensus labels "
                "are NOT used for routing or reported as ground truth."
            )

    print(f"\n=== label-space decision: {path} ({len(labels)} classes); primary consensus = {primary} ===")

    meta: dict = {
        "n_items": len(df), "judges": list(JUDGES), "label_space_path": path,
        "labels": list(labels), "raw_labels": list(LABELS), "max_class_ratio": args.max_ratio,
        "primary_consensus_method": primary,
        "method_passed_invariants": method_validity,
        "invariant_reports": invariant_reports,
        "methods": {},
    }

    methods: dict[str, list[dict]] = {
        "dawid_skene": _method_block(final_ds.posteriors, labels),
        "stratified_dawid_skene": _method_block(final_sds.posteriors, labels),
    }
    meta["methods"]["dawid_skene"] = {
        "config": {"diag_prior": ds_config.diag_prior, "off_diag_prior": ds_config.off_diag_prior,
                   "class_prior_strength": ds_config.class_prior_strength, "alpha": ds_config.alpha,
                   "max_iter": ds_config.max_iter, "tol": ds_config.tol},
        "n_iter": final_ds.n_iter, "converged": final_ds.converged,
        "final_log_likelihood": final_ds.log_likelihood[-1], "log_likelihood_trajectory": final_ds.log_likelihood,
        "runtime_seconds": round(final_ds_runtime, 1),
        "passed_invariants": method_validity["dawid_skene"],
    }
    meta["methods"]["stratified_dawid_skene"] = {
        "config": {"diag_prior": sds_config.diag_prior, "off_diag_prior": sds_config.off_diag_prior,
                   "class_prior_strength": sds_config.class_prior_strength, "alpha": sds_config.alpha,
                   "shrinkage_k0": sds_config.shrinkage_k0, "max_iter": sds_config.max_iter, "tol": sds_config.tol},
        "strata": [f"{r}/{t}" for r, t in strata_keys],
        "strata_sizes": {f"{r}/{t}": int(n) for (r, t), n in zip(strata_keys, final_sds.strata_sizes)},
        "n_iter": final_sds.n_iter, "converged": final_sds.converged,
        "final_log_likelihood": final_sds.log_likelihood[-1], "log_likelihood_trajectory": final_sds.log_likelihood,
        "runtime_seconds": round(final_sds_runtime, 1),
        "passed_invariants": method_validity["stratified_dawid_skene"],
    }

    print("\n=== majority vote ===")
    t0 = time.time()
    mv_result = bl.majority_vote(final_votes, labels)
    runtime = time.time() - t0
    print(f"tie_count={mv_result.extra['tie_count']} tie_rate={mv_result.extra['tie_rate']:.4f}")
    methods["majority_vote"] = _method_block(mv_result.posteriors, labels)
    meta["methods"]["majority_vote"] = {**mv_result.extra, "runtime_seconds": round(runtime, 1)}

    if args.skip_mace:
        meta["methods"]["mace"] = {"available": False, "unavailable_reason": "skipped via --skip-mace"}
    else:
        print("\n=== MACE ===")
        t0 = time.time()
        mace_result = bl.run_mace(final_votes, item_ids, labels, n_restarts=args.mace_restarts)
        runtime = time.time() - t0
        print(f"available={mace_result.available} runtime={runtime:.1f}s")
        if mace_result.available:
            methods["mace"] = _method_block(mace_result.posteriors, labels)
            meta["methods"]["mace"] = {**mace_result.extra, "runtime_seconds": round(runtime, 1), "available": True}
        else:
            print(f"MACE unavailable: {mace_result.unavailable_reason}")
            meta["methods"]["mace"] = {"available": False, "unavailable_reason": mace_result.unavailable_reason}

    if args.skip_glad:
        meta["methods"]["glad"] = {"available": False, "unavailable_reason": "skipped via --skip-glad"}
    else:
        print("\n=== GLAD ===")
        t0 = time.time()
        glad_result = bl.run_glad(final_votes, item_ids, labels)
        runtime = time.time() - t0
        print(f"available={glad_result.available} runtime={runtime:.1f}s")
        if glad_result.available:
            methods["glad"] = _method_block(glad_result.posteriors, labels)
            meta["methods"]["glad"] = {**glad_result.extra, "runtime_seconds": round(runtime, 1), "available": True}
        else:
            print(f"GLAD unavailable: {glad_result.unavailable_reason}")
            meta["methods"]["glad"] = {"available": False, "unavailable_reason": glad_result.unavailable_reason}

    # Sanity re-check on majority_vote/MACE — they can't structurally
    # violate unanimity, but this is the "permanent" test, so it runs
    # unconditionally rather than trusting that. Only the PRIMARY method
    # failing here is fatal; a non-primary method (e.g. stratified DS, by
    # design in the branch above) failing is expected and already reported.
    final_method_labels = {name: [row["label"] for row in blocks] for name, blocks in methods.items()}
    final_passed, final_report = run_invariant_checks(final_votes, final_method_labels, labels, args.max_ratio)
    primary_passed = final_report["unanimity"]["methods"][primary]["passed"] and final_report["class_ratio"]["methods"][primary]["passed"]
    print(f"\nfinal invariant check across all methods: {'PASSED' if final_passed else 'FAILED'} "
          f"(primary method '{primary}': {'PASSED' if primary_passed else 'FAILED'})")
    meta["final_invariant_report"] = final_report
    if not primary_passed:
        raise SystemExit(f"the PRIMARY consensus method '{primary}' fails the final invariant check — refusing to write output.")

    print(f"\nwriting {out_jsonl}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i in range(len(df)):
            row = {"id": item_ids[i], "votes": _raw_votes_per_item(df, i)}
            for method_name, blocks in methods.items():
                row[method_name] = blocks[i]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl} ({out_jsonl.stat().st_size / 1e6:.1f} MB)")

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_meta}")


if __name__ == "__main__":
    main()
