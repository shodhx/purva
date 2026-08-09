"""Reporting layer for Phase 4 aggregation (PROTOCOL.md §5). Reads the
master corpus and run_aggregation.py's output and writes:

  data/aggregation_report.md   — the written report.
  data/aggregation_report.json — the same numbers, machine-readable.

Covers: inter-judge agreement (Fleiss' kappa, Krippendorff's alpha,
pairwise matrix), per-judge/per-stratum confusion analysis (the
register/text_type reliability hypothesis), judge calibration (ECE/Brier
against a Dawid-Skene consensus proxy — NOT human gold, which doesn't
exist yet), leave-one-judge-out sensitivity, aggregator comparison, and
the entropy distribution that determines the Phase 5 routing budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dawid_skene as ds
from ._common import JUDGES, LABELS, N_CLASSES, argmax_labels, build_strata, build_vote_matrix, load_master

N_JUDGES = len(JUDGES)


# --------------------------------------------------------------------------
# Agreement statistics
# --------------------------------------------------------------------------

def fleiss_kappa(votes_col_per_judge: list[np.ndarray], n_categories: int) -> float:
    """Generalised Fleiss' kappa allowing a variable number of raters per
    item (Fleiss, Levin & Paik) — needed because 33 items have 4 judges,
    not 5. votes_col_per_judge[j] is that judge's category index per item,
    or -1 for no vote. P_i only involves item i's own n_i, so the standard
    per-item formula already generalises; nothing needs to be imputed."""
    n_items = len(votes_col_per_judge[0])
    counts = np.zeros((n_items, n_categories))
    for col in votes_col_per_judge:
        for k in range(n_categories):
            counts[:, k] += (col == k)
    n_i = counts.sum(axis=1)
    valid = n_i >= 2  # kappa is undefined for an item with <2 raters
    counts, n_i = counts[valid], n_i[valid]
    p_i = ((counts ** 2).sum(axis=1) - n_i) / (n_i * (n_i - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / n_i.sum()
    p_e = (p_j ** 2).sum()
    if p_e >= 1.0:
        return float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def krippendorff_alpha_nominal(votes: np.ndarray) -> float:
    import krippendorff

    reliability_data = votes.T.astype(float)
    reliability_data[reliability_data < 0] = np.nan
    return float(krippendorff.alpha(reliability_data=reliability_data, level_of_measurement="nominal"))


def pairwise_agreement(votes: np.ndarray, judges: tuple[str, ...] = JUDGES) -> dict:
    matrix = {}
    for i, ji in enumerate(judges):
        for j, jj in enumerate(judges):
            if j <= i:
                continue
            a, b = votes[:, i], votes[:, j]
            both = (a != -1) & (b != -1)
            n = int(both.sum())
            agree = float((a[both] == b[both]).mean()) if n else float("nan")
            matrix[f"{ji}-{jj}"] = {"agreement": agree, "n_compared": n}
    return matrix


def agreement_section(votes: np.ndarray, binary_votes: np.ndarray) -> dict:
    judge_cols = [votes[:, j] for j in range(votes.shape[1])]
    binary_cols = [binary_votes[:, j] for j in range(binary_votes.shape[1])]
    return {
        "fleiss_kappa_5class": fleiss_kappa(judge_cols, N_CLASSES),
        "fleiss_kappa_binary_subjectivity": fleiss_kappa(binary_cols, 2),
        "krippendorff_alpha_5class": krippendorff_alpha_nominal(votes),
        "krippendorff_alpha_binary_subjectivity": krippendorff_alpha_nominal(binary_votes),
        "pairwise_agreement_5class": pairwise_agreement(votes),
        "pairwise_agreement_binary_subjectivity": pairwise_agreement(binary_votes),
    }


# --------------------------------------------------------------------------
# Confusion matrix analysis
# --------------------------------------------------------------------------

def diagonal_strength(confusion_jkk: np.ndarray) -> float:
    return float(np.diag(confusion_jkk).mean())


def mixed_class_caveat(votes: np.ndarray, method_labels: dict[str, list[str]]) -> dict:
    """"mixed" is rare (~1% of raw votes) and, per the confusion matrices,
    essentially never correctly re-identified by any judge (near-zero
    diagonal on the mixed row for all five) — a textbook Dawid-Skene
    identifiability problem for a class no annotator reliably recognises.
    The practical symptom: DS inflates the "mixed" prior far past its raw
    vote share, and stratification inflates it further still. Surfaced
    explicitly here so a reader doesn't take the "mixed" consensus counts
    at face value without this warning."""
    mixed_idx = LABELS.index("mixed")
    n = votes.shape[0]
    raw_mixed_vote_items = int(((votes == mixed_idx).any(axis=1)).sum())
    raw_mixed_votes_total = int((votes == mixed_idx).sum())
    total_votes = int((votes != -1).sum())
    counts = {name: int(sum(lbl == "mixed" for lbl in labels)) for name, labels in method_labels.items()}
    return {
        "raw_mixed_votes_total": raw_mixed_votes_total,
        "raw_mixed_votes_share_of_all_votes": raw_mixed_votes_total / total_votes,
        "items_with_at_least_one_raw_mixed_vote": raw_mixed_vote_items,
        "items_with_at_least_one_raw_mixed_vote_share": raw_mixed_vote_items / n,
        "mixed_label_count_by_method": counts,
        "warning": (
            "\"mixed\" is voted by any judge on only "
            f"{raw_mixed_vote_items} items ({raw_mixed_vote_items / n:.1%} of the corpus) and makes up "
            f"{raw_mixed_votes_total / total_votes:.1%} of all votes cast, yet Dawid-Skene assigns the "
            f"consensus label \"mixed\" to {counts.get('dawid_skene', 0)} items (standard) and "
            f"{counts.get('stratified_dawid_skene', 0)} items (stratified) — vs. only "
            f"{counts.get('majority_vote', 0)} under majority vote. Every judge's confusion matrix shows a "
            "near-zero diagonal entry for the true-mixed row (see section 2), meaning no judge reliably "
            "re-produces \"mixed\" even when DS believes it's the true label — a known DS failure mode for "
            "a weakly-identified class, not evidence that ~30% of the corpus is actually mixed-sentiment. "
            "Treat consensus \"mixed\" labels as low-confidence pending human adjudication."
        ),
    }


def confusion_section(votes: np.ndarray, strata: np.ndarray, strata_keys: list[tuple[str, str]],
                       ds_posteriors: np.ndarray, sds_posteriors: np.ndarray, alpha: float) -> dict:
    global_confusion = ds.estimate_confusion(votes, ds_posteriors, alpha)
    strat_confusion, _global_from_stratified, strata_sizes = ds.estimate_confusion_stratified(
        votes, sds_posteriors, strata, len(strata_keys), alpha, ds.StratifiedDSConfig().shrinkage_k0,
    )

    per_judge_global = {j: diagonal_strength(global_confusion[ji]) for ji, j in enumerate(JUDGES)}
    per_judge_per_stratum = {}
    for ji, j in enumerate(JUDGES):
        per_judge_per_stratum[j] = {
            f"{r}/{t}": diagonal_strength(strat_confusion[si, ji]) for si, (r, t) in enumerate(strata_keys)
        }

    verse_strata = [si for si, (r, t) in enumerate(strata_keys) if r == "verse" or t == "verse"]
    news_prose_idx = next((si for si, (r, t) in enumerate(strata_keys) if r == "news" and t == "prose"), None)

    hypothesis: dict = {"news_prose_stratum": "news/prose" if news_prose_idx is not None else None, "per_judge": {}}
    for ji, j in enumerate(JUDGES):
        news_val = diagonal_strength(strat_confusion[news_prose_idx, ji]) if news_prose_idx is not None else None
        verse_vals = [diagonal_strength(strat_confusion[si, ji]) for si in verse_strata]
        verse_mean = float(np.mean(verse_vals)) if verse_vals else None
        hypothesis["per_judge"][j] = {
            "news_prose_diagonal_strength": news_val,
            "verse_mean_diagonal_strength": verse_mean,
            "verse_degrades_vs_news_prose": (verse_mean < news_val) if (news_val is not None and verse_mean is not None) else None,
            "delta": (verse_mean - news_val) if (news_val is not None and verse_mean is not None) else None,
        }

    any_degrade = [v["verse_degrades_vs_news_prose"] for v in hypothesis["per_judge"].values() if v["verse_degrades_vs_news_prose"] is not None]
    hypothesis["conclusion"] = (
        f"{sum(any_degrade)}/{len(any_degrade)} judges show weaker diagonal strength (worse reliability) "
        "on verse-associated strata than on news/prose" if any_degrade else "insufficient data"
    )

    return {
        "global_confusion": {j: global_confusion[ji].tolist() for ji, j in enumerate(JUDGES)},
        "global_diagonal_strength": per_judge_global,
        "per_stratum_diagonal_strength": per_judge_per_stratum,
        "strata_sizes": {f"{r}/{t}": int(n) for (r, t), n in zip(strata_keys, strata_sizes)},
        "register_text_type_hypothesis": hypothesis,
    }


# --------------------------------------------------------------------------
# Judge calibration
# --------------------------------------------------------------------------

def ece_and_brier(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    bin_report = []
    n = len(confidence)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_report.append({
            "bin": f"[{bins[b]:.1f}, {bins[b+1]:.1f}]", "n": int(mask.sum()),
            "mean_confidence": bin_conf, "empirical_accuracy": bin_acc,
        })
    brier = float(np.mean((confidence - correct) ** 2))
    return {"ece": ece, "brier_score": brier, "bins": bin_report}


def calibration_section(df: pd.DataFrame, votes: np.ndarray, consensus_labels: list[str]) -> dict:
    """Judge self-reported confidence, calibrated against the stratified-DS
    consensus argmax as a proxy for correctness. This is explicitly NOT
    accuracy against human gold — no gold exists yet (Phase 5 hasn't run) —
    it measures agreement-with-consensus-weighted-by-self-confidence, a
    standard fallback when gold is unavailable. Treat "ECE"/"Brier" here as
    calibration-against-consensus, not calibration-against-truth."""
    consensus_idx = np.array([LABELS.index(x) for x in consensus_labels])
    out = {}
    for ji, j in enumerate(JUDGES):
        voted = votes[:, ji] != -1
        confidence = df[f"judge_{j}_confidence"].to_numpy()[voted]
        correct = (votes[voted, ji] == consensus_idx[voted]).astype(float)
        stats = ece_and_brier(confidence, correct)
        stats["n_votes"] = int(voted.sum())
        stats["mean_confidence"] = float(confidence.mean())
        stats["pct_confidence_ge_0_9"] = float((confidence >= 0.9).mean())
        stats["pct_confidence_ge_0_95"] = float((confidence >= 0.95).mean())
        stats["empirical_agreement_with_consensus"] = float(correct.mean())
        out[j] = stats
    return out


# --------------------------------------------------------------------------
# Leave-one-judge-out
# --------------------------------------------------------------------------

def leave_one_out_section(votes: np.ndarray, full_ds_labels: list[str], config: ds.DSConfig) -> dict:
    out = {}
    for ji, j in enumerate(JUDGES):
        remaining = [k for k in range(votes.shape[1]) if k != ji]
        sub_votes = votes[:, remaining]
        t0 = time.time()
        result = ds.run_dawid_skene(sub_votes, config)
        runtime = time.time() - t0
        sub_labels = argmax_labels(result.posteriors)
        changed = sum(a != b for a, b in zip(full_ds_labels, sub_labels))
        out[j] = {
            "pct_labels_changed_when_excluded": changed / len(full_ds_labels),
            "n_changed": changed,
            "n_iter": result.n_iter,
            "converged": result.converged,
            "runtime_seconds": round(runtime, 1),
        }
    return out


# --------------------------------------------------------------------------
# Aggregator comparison
# --------------------------------------------------------------------------

def aggregator_comparison_section(method_labels: dict[str, list[str]]) -> dict:
    names = list(method_labels)
    n = len(next(iter(method_labels.values())))
    pairwise = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            agree = sum(x == y for x, y in zip(method_labels[a], method_labels[b])) / n
            pairwise[f"{a}-{b}"] = agree
    mv_vs_sds = None
    if "majority_vote" in method_labels and "stratified_dawid_skene" in method_labels:
        changed = sum(
            a != b for a, b in zip(method_labels["majority_vote"], method_labels["stratified_dawid_skene"])
        )
        mv_vs_sds = {"n_changed": changed, "pct_changed": changed / n}
    return {
        "pairwise_label_agreement": pairwise,
        "majority_vote_vs_stratified_ds": mv_vs_sds,
        "note": "Accuracy against human gold labels cannot be computed yet — Phase 5 human annotation has not run. Not estimated or proxied here.",
    }


# --------------------------------------------------------------------------
# Entropy distribution
# --------------------------------------------------------------------------

def _describe(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)), "std": float(np.std(x)),
        "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75)),
        "pct_below_0_3": float((x < 0.3).mean()), "pct_above_0_7": float((x > 0.7).mean()),
    }


def entropy_section(df: pd.DataFrame, entropy_norm: np.ndarray) -> dict:
    """df must have a plain 0..N-1 RangeIndex aligned with entropy_norm's
    positional order (true of a freshly-loaded parquet, asserted by the
    caller) — groupby(...).indices gives positional arrays directly, so no
    label/position translation is needed."""
    out = {"overall": _describe(entropy_norm)}
    for col in ("register", "text_type", "source_name", "script"):
        groups = {str(val): _describe(entropy_norm[pos]) for val, pos in df.groupby(col).indices.items()}
        out[col] = dict(sorted(groups.items(), key=lambda kv: -kv[1]["mean"]))
    return out


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def render_markdown(report: dict) -> str:
    lines = ["# Aggregation report", ""]
    lines.append("Generated by `purva/aggregate/analysis.py`. Machine-readable form: `data/aggregation_report.json`.")
    lines.append("")

    lines.append("## 1. Inter-judge agreement")
    a = report["agreement"]
    lines.append(f"- Fleiss' kappa (5-class): **{a['fleiss_kappa_5class']:.4f}**")
    lines.append(f"- Fleiss' kappa (binary subjectivity): **{a['fleiss_kappa_binary_subjectivity']:.4f}**")
    lines.append(f"- Krippendorff's alpha (5-class, nominal): **{a['krippendorff_alpha_5class']:.4f}**")
    lines.append(f"- Krippendorff's alpha (binary subjectivity, nominal): **{a['krippendorff_alpha_binary_subjectivity']:.4f}**")
    lines.append("")
    lines.append("Pairwise 5-class agreement (fraction agreeing, over items both judges voted on):")
    lines.append("")
    lines.append("| Pair | Agreement | N compared |")
    lines.append("|---|---|---|")
    for pair, v in a["pairwise_agreement_5class"].items():
        lines.append(f"| {pair} | {v['agreement']:.4f} | {v['n_compared']} |")
    lines.append("")

    lines.append("## 2. Confusion matrix analysis")
    c = report["confusion"]
    lines.append("Global diagonal strength (mean of confusion-matrix diagonal, higher = more reliable):")
    lines.append("")
    lines.append("| Judge | Global diagonal strength |")
    lines.append("|---|---|")
    for j, v in c["global_diagonal_strength"].items():
        lines.append(f"| {j} | {v:.4f} |")
    lines.append("")
    lines.append("### Register × text_type reliability hypothesis")
    lines.append("")
    lines.append(f"**{c['register_text_type_hypothesis']['conclusion']}**")
    lines.append("")
    lines.append("| Judge | news/prose diag. strength | verse-strata mean diag. strength | delta | degrades on verse? |")
    lines.append("|---|---|---|---|---|")
    for j, v in c["register_text_type_hypothesis"]["per_judge"].items():
        lines.append(
            f"| {j} | {v['news_prose_diagonal_strength']:.4f} | {v['verse_mean_diagonal_strength']:.4f} | "
            f"{v['delta']:+.4f} | {v['verse_degrades_vs_news_prose']} |"
        )
    lines.append("")

    lines.append("## 3. Judge calibration")
    lines.append(
        "Calibrated against the stratified-DS consensus label as a proxy for correctness "
        "(no human gold exists yet — see PROTOCOL.md §6). Expect severe overconfidence."
    )
    lines.append("")
    lines.append("| Judge | Mean confidence | % conf ≥0.9 | Agreement w/ consensus | ECE | Brier |")
    lines.append("|---|---|---|---|---|---|")
    for j, v in report["calibration"].items():
        lines.append(
            f"| {j} | {v['mean_confidence']:.3f} | {v['pct_confidence_ge_0_9']:.1%} | "
            f"{v['empirical_agreement_with_consensus']:.3f} | {v['ece']:.4f} | {v['brier_score']:.4f} |"
        )
    lines.append("")

    lines.append("## 4. Leave-one-judge-out")
    lines.append("Standard (unstratified) DS, re-run with each judge held out; % of items whose consensus label changes.")
    lines.append("")
    lines.append("| Judge held out | % labels changed | N changed |")
    lines.append("|---|---|---|")
    for j, v in report["leave_one_out"].items():
        lines.append(f"| {j} | {v['pct_labels_changed_when_excluded']:.2%} | {v['n_changed']} |")
    lines.append("")

    lines.append("## 5. Aggregator comparison")
    ac = report["aggregator_comparison"]
    lines.append("| Method pair | Label agreement |")
    lines.append("|---|---|")
    for pair, agree in ac["pairwise_label_agreement"].items():
        lines.append(f"| {pair} | {agree:.4f} |")
    if ac["majority_vote_vs_stratified_ds"]:
        mv = ac["majority_vote_vs_stratified_ds"]
        lines.append("")
        lines.append(f"Majority vote vs. stratified DS: **{mv['n_changed']}** items ({mv['pct_changed']:.2%}) change label.")
    lines.append("")
    lines.append(f"> {ac['note']}")
    lines.append("")

    lines.append("### ⚠ \"mixed\" class identifiability caveat")
    lines.append("")
    lines.append(report["mixed_class_caveat"]["warning"])
    lines.append("")

    lines.append("## 6. Entropy distribution (stratified DS, normalised to [0,1])")
    e = report["entropy"]["overall"]
    lines.append(
        f"Overall: mean={e['mean']:.4f}, median={e['median']:.4f}, "
        f"{e['pct_below_0_3']:.1%} below 0.3, {e['pct_above_0_7']:.1%} above 0.7."
    )
    for col in ("register", "text_type", "source_name", "script"):
        lines.append("")
        lines.append(f"### By `{col}` (sorted by mean entropy, descending)")
        lines.append("")
        lines.append(f"| {col} | mean | median | % > 0.7 |")
        lines.append("|---|---|---|---|")
        for val, stats in report["entropy"][col].items():
            lines.append(f"| {val} | {stats['mean']:.4f} | {stats['median']:.4f} | {stats['pct_above_0_7']:.1%} |")
    lines.append("")

    if report.get("unavailable_baselines"):
        lines.append("## Unavailable baselines")
        for name, reason in report["unavailable_baselines"].items():
            lines.append(f"- **{name}**: {reason}")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--report-md", default="data/aggregation_report.md")
    ap.add_argument("--report-json", default="data/aggregation_report.json")
    ap.add_argument("--alpha", type=float, default=1.0, help="DS smoothing alpha, must match run_aggregation.py")
    args = ap.parse_args()

    print(f"loading {args.master}")
    df = load_master(args.master).reset_index(drop=True)
    votes = build_vote_matrix(df)
    strata, strata_keys = build_strata(df)

    binary_votes = np.where(votes == -1, -1, (votes != 0).astype(np.int8))  # 0=objective, 1=subjective

    print(f"loading {args.aggregated}")
    agg_rows = [json.loads(line) for line in Path(args.aggregated).read_text(encoding="utf-8").splitlines() if line.strip()]
    agg_by_id = {r["id"]: r for r in agg_rows}
    ordered = [agg_by_id[i] for i in df["id"]]

    def posteriors_for(method: str) -> np.ndarray | None:
        if method not in ordered[0]:
            return None
        return np.array([[r[method]["posterior"][lbl] for lbl in LABELS] for r in ordered])

    def labels_for(method: str) -> list[str] | None:
        if method not in ordered[0]:
            return None
        return [r[method]["label"] for r in ordered]

    ds_posteriors = posteriors_for("dawid_skene")
    sds_posteriors = posteriors_for("stratified_dawid_skene")
    sds_labels = labels_for("stratified_dawid_skene")
    ds_labels = labels_for("dawid_skene")
    entropy_norm = np.array([r["stratified_dawid_skene"]["entropy_norm"] for r in ordered])

    method_labels = {}
    unavailable = {}
    for method in ("majority_vote", "dawid_skene", "stratified_dawid_skene", "mace", "glad"):
        labels = labels_for(method)
        if labels is not None:
            method_labels[method] = labels
        else:
            unavailable[method] = "not present in purva_aggregated.jsonl (see its .meta.json for why)"

    print("agreement statistics")
    agreement = agreement_section(votes, binary_votes)

    print("confusion matrix analysis")
    confusion = confusion_section(votes, strata, strata_keys, ds_posteriors, sds_posteriors, args.alpha)

    print("judge calibration")
    calibration = calibration_section(df, votes, sds_labels)

    print("leave-one-judge-out (5 standard-DS re-runs)")
    leave_one_out = leave_one_out_section(votes, ds_labels, ds.DSConfig())

    print("aggregator comparison")
    aggregator_comparison = aggregator_comparison_section(method_labels)

    print("mixed-class identifiability caveat")
    mixed_caveat = mixed_class_caveat(votes, method_labels)

    print("entropy distribution")
    entropy = entropy_section(df, entropy_norm)

    report = {
        "agreement": agreement,
        "confusion": confusion,
        "calibration": calibration,
        "leave_one_out": leave_one_out,
        "aggregator_comparison": aggregator_comparison,
        "mixed_class_caveat": mixed_caveat,
        "entropy": entropy,
        "unavailable_baselines": unavailable,
    }

    Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.report_json}")

    Path(args.report_md).write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
