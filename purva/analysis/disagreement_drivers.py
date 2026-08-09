"""Disagreement-driver analysis (PROTOCOL.md §7b): regresses judge
disagreement on corpus covariates and sentence length. Two outcomes are
analysed in parallel:

  1. Standard-DS posterior entropy (`dawid_skene`, the validated primary
     consensus method — see data/purva_aggregated.meta.json and
     docs/aggregation_identifiability.md), a continuous 5-class-uncertainty
     measure.
  2. Binary subjectivity dissent rate: the fraction of judges who voted on
     an item but disagreed with that item's modal subjectivity call
     (objective vs. subjective) — the dimension judges are known to differ
     on systematically (PROTOCOL.md §5 register/text_type analysis).

This is an associational analysis of an observational corpus: register,
text_type, source_name, script, and sentence length are correlated with
each other (e.g. verse is concentrated in a few sources) and with
unobserved factors (topic, translation history, orthographic convention).
Coefficients describe conditional association, not the causal effect of
changing a sentence's register or length. Nothing in this module or its
output should be read as identifying a cause of disagreement.

Writes data/disagreement_analysis.md and data/disagreement_analysis.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from purva.aggregate._common import JUDGES, load_aggregated, load_master

CAUSAL_CAVEAT = (
    "Associational, not causal. This corpus is observational: register, text_type, source_name, "
    "script, and sentence length are correlated with each other and with unmeasured factors (topic, "
    "translation history, orthographic convention). Coefficients describe conditional association "
    "within this sample, not the effect of changing a sentence's register or length. No predictor "
    "here should be described as a cause of judge disagreement. Note also that source_name is highly "
    "correlated with register in this corpus (most sources specialise in one register, e.g. the "
    "Wikipedia source is entirely encyclopedic) — the Type II partial-R2 split between them is "
    "adjusted correctly but is less stable than for independent predictors; read register and "
    "source_name's contributions together rather than as two fully separable effects."
)

CATEGORICAL_PREDICTORS = ["register", "text_type", "source_name", "script"]
CONTINUOUS_PREDICTORS = ["char_count", "token_count"]
FORMULA_RHS = " + ".join([f"C({c})" for c in CATEGORICAL_PREDICTORS] + CONTINUOUS_PREDICTORS)


def build_dissent_rate(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> pd.Series:
    """Fraction of voting judges who disagree with the item's modal binary
    subjectivity call. For a binary outcome the minority count is exactly
    the dissent count, so this needs no per-row Counter loop. NaN where
    fewer than 2 judges voted (disagreement is undefined for n<2)."""
    cols = [df[f"judge_{j}_subjectivity"] for j in judges]
    n_subj = sum((c == "subjective").astype(int) for c in cols)
    n_obj = sum((c == "objective").astype(int) for c in cols)
    n_voted = n_subj + n_obj
    with np.errstate(invalid="ignore"):
        rate = np.minimum(n_subj, n_obj) / n_voted
    return pd.Series(np.where(n_voted >= 2, rate, np.nan), index=df.index)


def fit_and_report(df: pd.DataFrame, outcome: str) -> dict:
    sub = df[[outcome] + CATEGORICAL_PREDICTORS + CONTINUOUS_PREDICTORS].dropna()
    n_dropped = len(df) - len(sub)

    model = smf.ols(f"{outcome} ~ {FORMULA_RHS}", data=sub).fit()
    ci = model.conf_int(alpha=0.05)
    coefficients = {
        term: {
            "coef": float(model.params[term]),
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
            "std_err": float(model.bse[term]),
            "p_value": float(model.pvalues[term]),
        }
        for term in model.params.index
    }

    anova = anova_lm(model, typ=2)
    ssr = float(anova.loc["Residual", "sum_sq"])
    marginal_contribution = {
        term: {
            "sum_sq": float(row["sum_sq"]),
            "partial_r2": float(row["sum_sq"] / (row["sum_sq"] + ssr)),
            "f_value": float(row["F"]) if pd.notna(row["F"]) else None,
            "p_value": float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else None,
        }
        for term, row in anova.iterrows() if term != "Residual"
    }

    return {
        "outcome": outcome,
        "n_used": len(sub),
        "n_dropped_missing": n_dropped,
        "formula": f"{outcome} ~ {FORMULA_RHS}",
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "coefficients": coefficients,
        "marginal_contribution": marginal_contribution,
    }


def level_means(df: pd.DataFrame, outcomes: list[str]) -> dict:
    out: dict = {}
    for col in CATEGORICAL_PREDICTORS:
        levels = {}
        for val, sub in df.groupby(col, dropna=False):
            entry = {"n": int(len(sub))}
            for outcome in outcomes:
                valid = sub[outcome].dropna()
                entry[f"mean_{outcome}"] = float(valid.mean()) if len(valid) else None
                entry[f"std_{outcome}"] = float(valid.std()) if len(valid) > 1 else None
            levels[str(val)] = entry
        out[col] = dict(sorted(levels.items(), key=lambda kv: -(kv[1][f"mean_{outcomes[0]}"] or 0)))
    return out


def render_markdown(report: dict) -> str:
    lines = ["# Disagreement-driver analysis", ""]
    lines.append("Generated by `purva/analysis/disagreement_drivers.py`. Machine-readable form: `data/disagreement_analysis.json`.")
    lines.append("")
    lines.append(f"> **{report['caveat']}**")
    lines.append("")

    for outcome_key, title in (("entropy", "Standard-DS posterior entropy"), ("dissent_rate", "Binary subjectivity dissent rate")):
        m = report["models"][outcome_key]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"`{m['formula']}`, fit by OLS on {m['n_used']} items ({m['n_dropped_missing']} dropped for missing values).")
        lines.append(f"R² = {m['r_squared']:.4f}, adjusted R² = {m['adj_r_squared']:.4f}.")
        lines.append("")
        lines.append("### Marginal contribution per predictor (Type II ANOVA, partial R²)")
        lines.append("")
        lines.append("| Predictor | Partial R² | F | p |")
        lines.append("|---|---|---|---|")
        for term, v in sorted(m["marginal_contribution"].items(), key=lambda kv: -kv[1]["partial_r2"]):
            lines.append(f"| {term} | {v['partial_r2']:.4f} | {v['f_value']:.2f} | {v['p_value']:.2e} |")
        lines.append("")
        lines.append("### Coefficients (95% CI)")
        lines.append("")
        lines.append("| Term | Coef | 95% CI | p |")
        lines.append("|---|---|---|---|")
        for term, v in m["coefficients"].items():
            lines.append(f"| {term} | {v['coef']:+.5f} | [{v['ci_low']:+.5f}, {v['ci_high']:+.5f}] | {v['p_value']:.2e} |")
        lines.append("")

    lines.append("## Mean outcome per predictor level")
    lines.append("")
    for col, levels in report["level_means"].items():
        lines.append(f"### `{col}`")
        lines.append("")
        lines.append("| Level | N | Mean entropy | Mean dissent rate |")
        lines.append("|---|---|---|---|")
        for level, v in levels.items():
            lines.append(f"| {level} | {v['n']} | {v['mean_entropy']:.4f} | {v['mean_dissent_rate']:.4f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--report-md", default="data/disagreement_analysis.md")
    ap.add_argument("--report-json", default="data/disagreement_analysis.json")
    args = ap.parse_args()

    print(f"loading {args.master}")
    df = load_master(args.master).reset_index(drop=True)

    print(f"loading {args.aggregated}")
    agg_rows = load_aggregated(args.aggregated)
    entropy_by_id = {r["id"]: r["dawid_skene"]["entropy_norm"] for r in agg_rows}
    df["entropy"] = df["id"].map(entropy_by_id)
    assert df["entropy"].notna().all(), "every master row must have a standard-DS entropy from purva_aggregated.jsonl"

    df["char_count"] = df["cleaned_text"].str.len()
    df["token_count"] = df["cleaned_text"].str.split().str.len()
    df["dissent_rate"] = build_dissent_rate(df)

    print("fitting entropy model")
    entropy_model = fit_and_report(df, "entropy")
    print(f"  R2={entropy_model['r_squared']:.4f} n={entropy_model['n_used']}")

    print("fitting dissent-rate model")
    dissent_model = fit_and_report(df, "dissent_rate")
    print(f"  R2={dissent_model['r_squared']:.4f} n={dissent_model['n_used']}")

    print("per-level means")
    means = level_means(df, ["entropy", "dissent_rate"])

    report = {
        "caveat": CAUSAL_CAVEAT,
        "n_items": len(df),
        "predictors": {"categorical": CATEGORICAL_PREDICTORS, "continuous": CONTINUOUS_PREDICTORS},
        "models": {"entropy": entropy_model, "dissent_rate": dissent_model},
        "level_means": means,
    }

    Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.report_json}")

    Path(args.report_md).write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
