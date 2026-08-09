"""Threshold-ordering analysis — the paper's primary empirical finding:
that the five judges behave as if applying a shared latent subjectivity
scale at different thresholds, so that the stricter judge's "subjective"
set is (approximately) nested inside the more permissive judge's set,
rather than the judges disagreeing along independent, unrelated axes.

Writes data/threshold_ordering.md and data/threshold_ordering.json, with:

  1. Per-judge subjectivity rate, full corpus and per chunk, with the
     chunk-level standard deviation (stability check).
  2. The full pairwise nesting matrix: for ordered pair (A, B), the
     fraction of A's subjective set (among items both judges voted on)
     that B also calls subjective.
  3. The pairwise binary-subjectivity agreement matrix, and whether
     agreement decays monotonically with rank-distance along the
     rate-ordered chain of judges (evidence for a shared 1-D scale rather
     than idiosyncratic pairwise disagreement).
  4. A formal test of nesting against the null of independent labeling at
     each judge's observed marginal rate: chi-square test of independence
     per pair, plus the excess of observed nesting over the independence
     expectation (P(B=subj) unconditionally).
  5. Whether the same rate ordering holds within every register and
     text_type stratum, or only when strata are pooled.
  6. Whether an analogous ordering exists on the polarity axis
     (positive-lean vs. negative-lean per judge among its subjective
     calls), and whether the two axes correlate across judges.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from purva.aggregate._common import JUDGES, load_master

N_CHUNKS = 9


def subjectivity_rate(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> dict:
    out = {}
    for j in judges:
        s = df[f"judge_{j}_subjectivity"].dropna()
        out[j] = {"rate": float((s == "subjective").mean()), "n": int(len(s))}
    return out


def per_chunk_rates(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES, n_chunks: int = N_CHUNKS) -> dict:
    out = {}
    for j in judges:
        rates = []
        for c in range(1, n_chunks + 1):
            s = df.loc[df["chunk"] == c, f"judge_{j}_subjectivity"].dropna()
            rates.append(float((s == "subjective").mean()))
        arr = np.array(rates)
        out[j] = {
            "rates_by_chunk": rates, "mean": float(arr.mean()), "std": float(arr.std(ddof=1)),
            "min": float(arr.min()), "max": float(arr.max()),
        }
    return out


def shared_mask(df: pd.DataFrame, a: str, b: str) -> tuple[pd.Series, pd.Series]:
    sa = df[f"judge_{a}_subjectivity"]
    sb = df[f"judge_{b}_subjectivity"]
    shared = sa.notna() & sb.notna()
    return sa[shared], sb[shared]


def nesting_matrix(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> dict:
    out = {a: {} for a in judges}
    for a in judges:
        for b in judges:
            if a == b:
                out[a][b] = None
                continue
            sa, sb = shared_mask(df, a, b)
            subj_a = sa == "subjective"
            n_subj_a = int(subj_a.sum())
            out[a][b] = float((subj_a & (sb == "subjective")).sum() / n_subj_a) if n_subj_a else None
    return out


def agreement_matrix(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> dict:
    out = {a: {} for a in judges}
    for a in judges:
        for b in judges:
            if a == b:
                out[a][b] = 1.0
                continue
            sa, sb = shared_mask(df, a, b)
            out[a][b] = float((sa == sb).mean()) if len(sa) else None
    return out


def chain_distance_decay(agreement: dict, rate_order: list[str]) -> dict:
    rank = {j: i for i, j in enumerate(rate_order)}
    by_distance: dict[int, list[float]] = {}
    for a, b in combinations(rate_order, 2):
        d = abs(rank[a] - rank[b])
        by_distance.setdefault(d, []).append(agreement[a][b])
    means = {d: float(np.mean(vals)) for d, vals in sorted(by_distance.items())}
    distances = sorted(means)
    monotonic = all(means[distances[i]] >= means[distances[i + 1]] for i in range(len(distances) - 1))
    return {"mean_agreement_by_distance": means, "decays_monotonically": monotonic}


def independence_test(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> dict:
    out = {a: {} for a in judges}
    for a, b in combinations(judges, 2):
        sa, sb = shared_mask(df, a, b)
        n = len(sa)
        n11 = int(((sa == "subjective") & (sb == "subjective")).sum())
        n10 = int(((sa == "subjective") & (sb == "objective")).sum())
        n01 = int(((sa == "objective") & (sb == "subjective")).sum())
        n00 = int(((sa == "objective") & (sb == "objective")).sum())
        table = [[n11, n10], [n01, n00]]
        chi2, p, dof, _ = chi2_contingency(table)

        p_b_marginal = (n11 + n01) / n
        p_b_given_a_subj = n11 / (n11 + n10) if (n11 + n10) else None
        p_a_marginal = (n11 + n10) / n
        p_a_given_b_subj = n11 / (n11 + n01) if (n11 + n01) else None

        result = {
            "n_shared": n, "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
            "contingency": {"both_subjective": n11, f"{a}_subj_{b}_obj": n10, f"{a}_obj_{b}_subj": n01, "both_objective": n00},
        }
        out[a][b] = {
            **result,
            "observed_nesting": p_b_given_a_subj,
            "expected_under_independence": p_b_marginal,
            "excess": (p_b_given_a_subj - p_b_marginal) if p_b_given_a_subj is not None else None,
        }
        out[b][a] = {
            **result,
            "observed_nesting": p_a_given_b_subj,
            "expected_under_independence": p_a_marginal,
            "excess": (p_a_given_b_subj - p_a_marginal) if p_a_given_b_subj is not None else None,
        }
    return out


def stratum_ordering(df: pd.DataFrame, column: str, overall_order: list[str], judges: tuple[str, ...] = JUDGES) -> dict:
    overall_rank = {j: i for i, j in enumerate(overall_order)}
    pairs = list(combinations(judges, 2))
    out = {}
    for level, sub in df.groupby(column, dropna=False):
        rates = subjectivity_rate(sub, judges)
        level_order = sorted(judges, key=lambda j: rates[j]["rate"])
        level_rank = {j: i for i, j in enumerate(level_order)}
        n_inversions = sum(1 for a, b in pairs if (overall_rank[a] - overall_rank[b]) * (level_rank[a] - level_rank[b]) < 0)
        rho, p = spearmanr([overall_rank[j] for j in judges], [level_rank[j] for j in judges])
        out[str(level)] = {
            "n": int(len(sub)), "rates": {j: rates[j]["rate"] for j in judges},
            "order": level_order, "exact_match_to_overall_order": level_order == overall_order,
            "n_inversions": n_inversions, "n_pairs": len(pairs),
            "spearman_rho_vs_overall": float(rho), "spearman_p": float(p),
        }
    n_strata = len(out)
    n_exact = sum(1 for v in out.values() if v["exact_match_to_overall_order"])
    mean_rho = float(np.mean([v["spearman_rho_vs_overall"] for v in out.values()]))
    return {"by_level": out, "summary": {"n_strata": n_strata, "n_exact_match": n_exact, "mean_spearman_rho": mean_rho}}


def polarity_axis(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> dict:
    per_judge = {}
    for j in judges:
        subj = df[f"judge_{j}_subjectivity"] == "subjective"
        pol = df[f"judge_{j}_polarity"]
        valenced = subj & pol.isin(["positive", "negative"])
        n = int(valenced.sum())
        lean = float((pol[valenced] == "positive").mean()) if n else None
        per_judge[j] = {"positive_lean": lean, "n_valenced_votes": n}

    order = sorted(judges, key=lambda j: per_judge[j]["positive_lean"])
    return {"per_judge": per_judge, "order": order}


def render_markdown(report: dict) -> str:
    lines = ["# Threshold-ordering analysis", ""]
    lines.append("Generated by `purva/analysis/threshold_ordering.py`. Machine-readable form: `data/threshold_ordering.json`.")
    lines.append("")

    lines.append("## 1. Per-judge subjectivity rate (stability across chunks)")
    lines.append("")
    lines.append("| Judge | Overall rate | N | Chunk mean | Chunk std | Chunk min | Chunk max |")
    lines.append("|---|---|---|---|---|---|---|")
    for j in report["rate_order"]:
        o = report["per_judge_rate_overall"][j]
        c = report["per_judge_rate_by_chunk"][j]
        lines.append(f"| {j} | {o['rate']:.4f} | {o['n']} | {c['mean']:.4f} | {c['std']:.4f} | {c['min']:.4f} | {c['max']:.4f} |")
    lines.append("")
    lines.append(f"Rate-ordered chain (ascending): **{' < '.join(report['rate_order'])}**.")
    lines.append("")

    lines.append("## 2. Pairwise nesting matrix")
    lines.append("")
    lines.append("Row = A, column = B; cell = fraction of A's subjective set that B also calls subjective, over items both voted on.")
    lines.append("")
    header = "| A \\ B | " + " | ".join(report["rate_order"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(report["rate_order"]) + 1))
    for a in report["rate_order"]:
        cells = []
        for b in report["rate_order"]:
            v = report["nesting_matrix"][a][b]
            cells.append("—" if v is None else f"{v:.4f}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 3. Pairwise agreement matrix and chain-distance decay")
    lines.append("")
    header = "| | " + " | ".join(report["rate_order"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(report["rate_order"]) + 1))
    for a in report["rate_order"]:
        cells = [f"{report['agreement_matrix'][a][b]:.4f}" for b in report["rate_order"]]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean agreement by rank-distance along the rate-ordered chain:")
    lines.append("")
    lines.append("| Distance | Mean agreement |")
    lines.append("|---|---|")
    for d, v in report["chain_distance_decay"]["mean_agreement_by_distance"].items():
        lines.append(f"| {d} | {v:.4f} |")
    lines.append("")
    lines.append(f"**Decays monotonically with distance: {report['chain_distance_decay']['decays_monotonically']}.**")
    lines.append("")

    lines.append("## 4. Nesting vs. independence null")
    lines.append("")
    lines.append("For each ordered pair (A, B): observed P(B=subjective | A=subjective) vs. the independence "
                  "expectation P(B=subjective) [unconditional], and the excess. Chi-square test is on the "
                  "underlying 2x2 contingency table (symmetric per unordered pair).")
    lines.append("")
    lines.append("| A | B | Observed nesting | Independence expectation | Excess | chi2 | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for a in report["rate_order"]:
        for b in report["rate_order"]:
            if a == b:
                continue
            v = report["independence_test"][a][b]
            lines.append(f"| {a} | {b} | {v['observed_nesting']:.4f} | {v['expected_under_independence']:.4f} | "
                          f"{v['excess']:+.4f} | {v['chi2']:.1f} | {v['p_value']:.2e} |")
    lines.append("")

    lines.append("## 5. Ordering within strata")
    lines.append("")
    for col in ("register", "text_type"):
        s = report["stratum_ordering"][col]
        lines.append(f"### By `{col}`")
        lines.append("")
        lines.append(f"{s['summary']['n_exact_match']}/{s['summary']['n_strata']} strata reproduce the exact overall "
                      f"order; mean Spearman rho vs. overall order = {s['summary']['mean_spearman_rho']:.4f}.")
        lines.append("")
        lines.append("| Level | N | Order | Exact match | Inversions | Spearman rho |")
        lines.append("|---|---|---|---|---|---|")
        for level, v in s["by_level"].items():
            lines.append(f"| {level} | {v['n']} | {' < '.join(v['order'])} | {v['exact_match_to_overall_order']} | "
                          f"{v['n_inversions']}/{v['n_pairs']} | {v['spearman_rho_vs_overall']:.4f} |")
        lines.append("")

    lines.append("## 6. Polarity axis and cross-axis correlation")
    lines.append("")
    lines.append("Positive-lean = P(polarity = positive | subjective and polarity in {positive, negative}), i.e. "
                  "restricted to the judge's clearly-valenced subjective calls.")
    lines.append("")
    lines.append("| Judge | Positive-lean | N valenced votes |")
    lines.append("|---|---|---|")
    for j in report["polarity_axis"]["order"]:
        v = report["polarity_axis"]["per_judge"][j]
        lines.append(f"| {j} | {v['positive_lean']:.4f} | {v['n_valenced_votes']} |")
    lines.append("")
    lines.append(f"Positive-lean-ordered chain (ascending): **{' < '.join(report['polarity_axis']['order'])}**.")
    corr = report["polarity_axis"]["correlation_with_subjectivity_rate"]
    lines.append("")
    lines.append(f"Correlation between subjectivity-rate order and positive-lean order across the {corr['n']} judges: "
                  f"Pearson r = {corr['pearson_r']:+.4f} (p={corr['pearson_p']:.3f}), "
                  f"Spearman rho = {corr['spearman_rho']:+.4f} (p={corr['spearman_p']:.3f}).")
    lines.append("")
    lines.append(f"> {corr['caveat']}")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--report-md", default="data/threshold_ordering.md")
    ap.add_argument("--report-json", default="data/threshold_ordering.json")
    args = ap.parse_args()

    print(f"loading {args.master}")
    df = load_master(args.master)

    print("per-judge subjectivity rate")
    rate_overall = subjectivity_rate(df)
    rate_by_chunk = per_chunk_rates(df)
    rate_order = sorted(JUDGES, key=lambda j: rate_overall[j]["rate"])
    print("rate order:", rate_order)

    print("nesting matrix")
    nesting = nesting_matrix(df)

    print("agreement matrix + chain-distance decay")
    agreement = agreement_matrix(df)
    decay = chain_distance_decay(agreement, rate_order)

    print("independence test")
    indep = independence_test(df)

    print("stratum ordering (register, text_type)")
    stratum = {
        "register": stratum_ordering(df, "register", rate_order),
        "text_type": stratum_ordering(df, "text_type", rate_order),
    }

    print("polarity axis")
    pol = polarity_axis(df)
    sub_rates = [rate_overall[j]["rate"] for j in JUDGES]
    pol_rates = [pol["per_judge"][j]["positive_lean"] for j in JUDGES]
    pear_r, pear_p = pearsonr(sub_rates, pol_rates)
    spear_r, spear_p = spearmanr(sub_rates, pol_rates)
    pol["correlation_with_subjectivity_rate"] = {
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "spearman_rho": float(spear_r), "spearman_p": float(spear_p), "n": len(JUDGES),
        "caveat": "n=5 judges; a correlation across so few points is weak evidence regardless of the p-value and should not be over-interpreted.",
    }

    report = {
        "judges": list(JUDGES), "n_chunks": N_CHUNKS,
        "rate_order": rate_order,
        "per_judge_rate_overall": rate_overall,
        "per_judge_rate_by_chunk": rate_by_chunk,
        "nesting_matrix": nesting,
        "agreement_matrix": agreement,
        "chain_distance_decay": decay,
        "independence_test": indep,
        "stratum_ordering": stratum,
        "polarity_axis": pol,
    }

    Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.report_json}")

    Path(args.report_md).write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
