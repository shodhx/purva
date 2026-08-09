"""Reporting layer for Phase 4 aggregation (PROTOCOL.md §5). Reads the
master corpus and run_aggregation.py's output and writes:

  data/aggregation_report.md   — the written report.
  data/aggregation_report.json — the same numbers, machine-readable.

Covers: the identifiability failure/fix (if relevant — see section 0 of
the rendered report), inter-judge agreement (Fleiss' kappa, Krippendorff's
alpha, pairwise matrix), per-judge/per-stratum confusion analysis (the
register/text_type reliability hypothesis), judge calibration against
*both* stratified-DS consensus and majority vote (ECE/Brier — NOT human
gold, which doesn't exist yet), leave-one-judge-out sensitivity, aggregator
comparison, and the entropy distribution that determines the Phase 5
routing budget.

The label space (five-class-with-priors, or the four-class fallback) is
read from purva_aggregated.meta.json rather than assumed — this module
never hardcodes LABELS for anything that depends on which path was taken.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dawid_skene as ds
from ._common import JUDGES, LABELS, build_strata, build_vote_matrix, load_aggregated, load_master, method_labels_from_aggregated, raw_vote_frequency


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
    n_classes = int(votes.max()) + 1
    return {
        "fleiss_kappa_5class": fleiss_kappa(judge_cols, n_classes),
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


def confusion_section(votes, strata, strata_keys, ds_posteriors, sds_posteriors, diag_prior, off_diag_prior, shrinkage_k0) -> dict:
    global_confusion = ds.estimate_confusion(votes, ds_posteriors, diag_prior, off_diag_prior)
    strat_confusion, _global_from_stratified, strata_sizes = ds.estimate_confusion_stratified(
        votes, sds_posteriors, strata, len(strata_keys), diag_prior, off_diag_prior, shrinkage_k0,
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
# Judge calibration — against BOTH stratified-DS consensus and majority
# vote, since the divergence between the two was itself diagnostic of the
# identifiability failure (see report section 0).
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


def calibration_against(df: pd.DataFrame, votes: np.ndarray, consensus_labels: list[str], labels: tuple[str, ...]) -> dict:
    consensus_idx = np.array([labels.index(x) for x in consensus_labels])
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


def calibration_section(df: pd.DataFrame, votes: np.ndarray, sds_labels: list[str], mv_labels: list[str], labels: tuple[str, ...]) -> dict:
    """Explicitly NOT accuracy against human gold — no gold exists yet
    (Phase 5 hasn't run). Both proxies are reported side by side because
    the SDS-vs-MV divergence in these numbers (0.375-0.468 vs 0.661-0.865
    in the run that had the identifiability bug) was itself the first
    concrete symptom of stratified DS's degenerate "mixed" catch-all."""
    return {
        "vs_stratified_dawid_skene": calibration_against(df, votes, sds_labels, labels),
        "vs_majority_vote": calibration_against(df, votes, mv_labels, labels),
    }


# --------------------------------------------------------------------------
# Leave-one-judge-out
# --------------------------------------------------------------------------

def leave_one_out_section(votes: np.ndarray, full_ds_labels: list[str], labels: tuple[str, ...], config: ds.DSConfig) -> dict:
    out = {}
    for ji, j in enumerate(JUDGES):
        remaining = [k for k in range(votes.shape[1]) if k != ji]
        sub_votes = votes[:, remaining]
        anchor = raw_vote_frequency(sub_votes, labels)
        t0 = time.time()
        result = ds.run_dawid_skene(sub_votes, config, n_classes=len(labels), class_prior_anchor=anchor)
        runtime = time.time() - t0
        sub_labels = [labels[i] for i in np.argmax(result.posteriors, axis=1)]
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

def aggregator_comparison_section(method_labels: dict[str, list[str]], primary_method: str) -> dict:
    names = list(method_labels)
    n = len(next(iter(method_labels.values())))
    pairwise = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            agree = sum(x == y for x, y in zip(method_labels[a], method_labels[b])) / n
            pairwise[f"{a}-{b}"] = agree
    mv_vs_primary = None
    if "majority_vote" in method_labels and primary_method in method_labels:
        changed = sum(
            a != b for a, b in zip(method_labels["majority_vote"], method_labels[primary_method])
        )
        mv_vs_primary = {"n_changed": changed, "pct_changed": changed / n}
    return {
        "primary_consensus_method": primary_method,
        "pairwise_label_agreement": pairwise,
        "majority_vote_vs_primary": mv_vs_primary,
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


def entropy_section(df: pd.DataFrame, entropy_norm: np.ndarray, method: str) -> dict:
    """df must have a plain 0..N-1 RangeIndex aligned with entropy_norm's
    positional order (true of a freshly-loaded parquet, asserted by the
    caller) — groupby(...).indices gives positional arrays directly, so no
    label/position translation is needed."""
    out = {"method": method, "overall": _describe(entropy_norm)}
    for col in ("register", "text_type", "source_name", "script"):
        groups = {str(val): _describe(entropy_norm[pos]) for val, pos in df.groupby(col).indices.items()}
        out[col] = dict(sorted(groups.items(), key=lambda kv: -kv[1]["mean"]))
    return out


# --------------------------------------------------------------------------
# Identifiability failure / fix section
# --------------------------------------------------------------------------

def identifiability_section(agg_meta: dict, votes5: np.ndarray) -> dict:
    path = agg_meta["label_space_path"]
    primary = agg_meta["primary_consensus_method"]
    validity = agg_meta["method_passed_invariants"]
    ds_cfg = agg_meta["methods"]["dawid_skene"]["config"]
    sds_cfg = agg_meta["methods"]["stratified_dawid_skene"]["config"]
    raw_freq5 = raw_vote_frequency(votes5, LABELS)
    mixed_idx = LABELS.index("mixed")

    description = (
        "An earlier run of this pipeline (pre-priors) found stratified DS assigning \"mixed\" to 29.6% "
        "of the corpus against a 1.0% raw-vote rate, including 6,932 unanimous-positive and 4,521 "
        "unanimous-objective items relabelled \"mixed\" — a Dawid-Skene identifiability failure: no "
        "judge produces \"mixed\" in enough quantity to constrain its confusion-matrix row, so "
        "unconstrained EM shaped that row into a catch-all for residual variance, and the higher "
        "log-likelihood this produced reflected EM exploiting a degenerate direction, not better "
        "labels. Fixed with two priors applied at every M-step (not just initialisation): a "
        "diagonal-favouring Dirichlet prior on every confusion-matrix row (encoding \"annotators beat "
        "chance\"), and class priors shrunk toward the observed raw-vote frequency. Both are asserted "
        "against permanent invariants in test_aggregation.py before any output is written: every "
        "unanimous item must keep its unanimous label, and no class's consensus share may exceed 3x "
        "its raw-vote share."
    )

    if not validity.get("stratified_dawid_skene", True):
        sds_class_ratio = agg_meta.get("invariant_reports", {}).get("stratified_dawid_skene", {}).get("class_ratio", {})
        worst_label, worst_ratio = None, 0.0
        for lbl, v in sds_class_ratio.get("per_label", {}).items():
            if v.get("exceeds") and v["ratio"] > worst_ratio:
                worst_label, worst_ratio = lbl, v["ratio"]
        description += (
            " A further finding, not anticipated by the original priors fix: even after dropping "
            f"\"mixed\" (path `{path}`), stratified DS continued to fail the invariant checks — the same "
            f"catch-all pathology reappeared on the next-sparsest class"
            + (f" (\"{worst_label}\", consensus/raw-vote ratio {worst_ratio:.2f}x)" if worst_label else "")
            + ", reproducibly across every diag_prior/off_diag_prior/class_prior_strength/shrinkage_k0 "
            "combination tried. This points to structural over-parameterisation in the stratified "
            "variant (one confusion matrix per judge per stratum, versus one per judge for standard DS) "
            f"rather than something these two priors alone can fix. Standard (unstratified) DS passes "
            f"cleanly at `{path}` and is used as the validated primary consensus method (`{primary}`); "
            "stratified DS is still run and shipped — its per-stratum confusion matrices remain valid "
            "input to the register/text_type reliability hypothesis test (section 2) — but its overall "
            "consensus labels are excluded from calibration-as-ground-truth and from routing (section 6)."
        )

    return {
        "path_taken": path,
        "primary_consensus_method": primary,
        "method_passed_invariants": validity,
        "priors_used": {
            "diag_prior": ds_cfg["diag_prior"], "off_diag_prior": ds_cfg["off_diag_prior"],
            "class_prior_strength": ds_cfg["class_prior_strength"], "shrinkage_k0": sds_cfg["shrinkage_k0"],
        },
        "raw_mixed_vote_share": float(raw_freq5[mixed_idx]),
        "final_invariant_summary": {
            "unanimity_passed": agg_meta["final_invariant_report"]["unanimity"]["passed"],
            "class_ratio_passed": agg_meta["final_invariant_report"]["class_ratio"]["passed"],
        },
        "description": description,
    }


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

def render_markdown(report: dict, labels: tuple[str, ...], path: str) -> str:
    lines = ["# Aggregation report", ""]
    lines.append("Generated by `purva/aggregate/analysis.py`. Machine-readable form: `data/aggregation_report.json`.")
    lines.append(f"Label space in use: **{list(labels)}** (path: `{path}`).")
    lines.append("")

    idf = report["identifiability"]
    lines.append("## 0. Identifiability failure and fix")
    lines.append("")
    lines.append(idf["description"])
    lines.append("")
    lines.append(f"**Path taken: `{idf['path_taken']}`. Validated primary consensus method: `{idf['primary_consensus_method']}`.**")
    lines.append(
        f"Priors used: diag_prior={idf['priors_used']['diag_prior']}, "
        f"off_diag_prior={idf['priors_used']['off_diag_prior']}, "
        f"class_prior_strength={idf['priors_used']['class_prior_strength']}, "
        f"shrinkage_k0={idf['priors_used']['shrinkage_k0']} (stratified variant)."
    )
    lines.append(
        "Per-method invariant validity: " + ", ".join(
            f"`{m}` {'PASSED' if ok else 'FAILED'}" for m, ok in idf["method_passed_invariants"].items()
        ) + "."
    )
    lines.append(
        f"Final invariant check on the shipped primary consensus: unanimity "
        f"{'PASSED' if idf['final_invariant_summary']['unanimity_passed'] else 'FAILED'}, "
        f"class-ratio {'PASSED' if idf['final_invariant_summary']['class_ratio_passed'] else 'FAILED'}."
    )
    lines.append(f"Raw \"mixed\" vote share (of all votes cast, 5-class): {idf['raw_mixed_vote_share']:.4f}.")
    lines.append("")

    lines.append("## 1. Inter-judge agreement")
    a = report["agreement"]
    lines.append(f"- Fleiss' kappa ({len(labels)}-class): **{a['fleiss_kappa_5class']:.4f}**")
    lines.append(f"- Fleiss' kappa (binary subjectivity): **{a['fleiss_kappa_binary_subjectivity']:.4f}**")
    lines.append(f"- Krippendorff's alpha ({len(labels)}-class, nominal): **{a['krippendorff_alpha_5class']:.4f}**")
    lines.append(f"- Krippendorff's alpha (binary subjectivity, nominal): **{a['krippendorff_alpha_binary_subjectivity']:.4f}**")
    lines.append("")
    lines.append(f"Pairwise {len(labels)}-class agreement (fraction agreeing, over items both judges voted on):")
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
        "Reported against both consensus methods, since their divergence was itself diagnostic "
        "(see section 0). No human gold exists yet (PROTOCOL.md §6). Expect severe overconfidence."
    )
    for key, title in (("vs_stratified_dawid_skene", "vs. stratified-DS consensus"), ("vs_majority_vote", "vs. majority vote")):
        lines.append("")
        lines.append(f"### Calibration {title}")
        lines.append("")
        lines.append("| Judge | Mean confidence | % conf ≥0.9 | Agreement w/ consensus | ECE | Brier |")
        lines.append("|---|---|---|---|---|---|")
        for j, v in report["calibration"][key].items():
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
    if ac["majority_vote_vs_primary"]:
        mv = ac["majority_vote_vs_primary"]
        lines.append("")
        lines.append(f"Majority vote vs. primary consensus (`{ac['primary_consensus_method']}`): **{mv['n_changed']}** items ({mv['pct_changed']:.2%}) change label.")
    lines.append("")
    lines.append(f"> {ac['note']}")
    lines.append("")

    lines.append(f"## 6. Entropy distribution ({report['entropy']['method']}, normalised to [0,1])")
    lines.append(
        "Sourced from the validated primary consensus method's posterior (not necessarily stratified "
        "DS — see section 0). This is also what the Phase 5 routing sets are drawn from."
    )
    lines.append("")
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
    ap.add_argument("--aggregated-meta", default="data/purva_aggregated.meta.json")
    ap.add_argument("--report-md", default="data/aggregation_report.md")
    ap.add_argument("--report-json", default="data/aggregation_report.json")
    args = ap.parse_args()

    print(f"loading {args.aggregated_meta}")
    agg_meta = json.loads(Path(args.aggregated_meta).read_text(encoding="utf-8"))
    labels = tuple(agg_meta["labels"])
    ds_cfg = agg_meta["methods"]["dawid_skene"]["config"]
    sds_cfg = agg_meta["methods"]["stratified_dawid_skene"]["config"]

    print(f"loading {args.master}")
    df = load_master(args.master).reset_index(drop=True)
    votes5 = build_vote_matrix(df)  # always the full 5-class matrix, for the identifiability section
    votes = votes5.copy()
    if len(labels) < 5:
        mixed_idx = LABELS.index("mixed")
        votes[votes == mixed_idx] = -1
    strata, strata_keys = build_strata(df)

    binary_votes = np.where(votes == -1, -1, (votes != 0).astype(np.int8))  # 0=objective, 1=subjective

    print(f"loading {args.aggregated}")
    agg_rows = load_aggregated(args.aggregated)
    method_labels = method_labels_from_aggregated(df, agg_rows)
    by_id = {r["id"]: r for r in agg_rows}
    ordered = [by_id[i] for i in df["id"]]

    def posteriors_for(method: str) -> np.ndarray:
        return np.array([[r[method]["posterior"][lbl] for lbl in labels] for r in ordered])

    primary = agg_meta["primary_consensus_method"]
    ds_posteriors = posteriors_for("dawid_skene")
    sds_posteriors = posteriors_for("stratified_dawid_skene")
    sds_labels = method_labels["stratified_dawid_skene"]
    mv_labels = method_labels["majority_vote"]
    ds_labels = method_labels["dawid_skene"]
    entropy_norm = np.array([r[primary]["entropy_norm"] for r in ordered])

    unavailable = {
        m: "not present in purva_aggregated.jsonl (see its .meta.json for why)"
        for m in ("mace", "glad") if m not in method_labels
    }

    print("identifiability section")
    identifiability = identifiability_section(agg_meta, votes5)

    print("agreement statistics")
    agreement = agreement_section(votes, binary_votes)

    print("confusion matrix analysis")
    confusion = confusion_section(votes, strata, strata_keys, ds_posteriors, sds_posteriors,
                                   ds_cfg["diag_prior"], ds_cfg["off_diag_prior"], sds_cfg["shrinkage_k0"])

    print("judge calibration (vs stratified DS and vs majority vote)")
    calibration = calibration_section(df, votes, sds_labels, mv_labels, labels)

    print("leave-one-judge-out (5 standard-DS re-runs)")
    leave_one_out = leave_one_out_section(votes, ds_labels, labels, ds.DSConfig(
        diag_prior=ds_cfg["diag_prior"], off_diag_prior=ds_cfg["off_diag_prior"],
        class_prior_strength=ds_cfg["class_prior_strength"],
    ))

    print("aggregator comparison")
    aggregator_comparison = aggregator_comparison_section(method_labels, primary)

    print("entropy distribution")
    entropy = entropy_section(df, entropy_norm, primary)

    report = {
        "label_space": list(labels),
        "label_space_path": agg_meta["label_space_path"],
        "identifiability": identifiability,
        "agreement": agreement,
        "confusion": confusion,
        "calibration": calibration,
        "leave_one_out": leave_one_out,
        "aggregator_comparison": aggregator_comparison,
        "entropy": entropy,
        "unavailable_baselines": unavailable,
    }

    Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.report_json}")

    Path(args.report_md).write_text(render_markdown(report, labels, agg_meta["label_space_path"]), encoding="utf-8")
    print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
