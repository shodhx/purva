"""Render data/hindi_validation_report.md from data/hindi_validation_report.json.

Markdown is generated, never hand-edited: every number in it comes from the
JSON (written by evaluate_hindi_validation.py), so the two reports cannot
drift. Run this after evaluate_hindi_validation.py.

Report structure (corrected analysis): the subjective-only comparable subset
is the headline — the Hindi gold has no objective class while the judges cast
a large fraction of objective votes, so the full-set accuracy figures measure
mapping loss rather than pipeline accuracy. The full-set figures are retained,
clearly marked as confounded by mapping loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_JSON = Path("data/hindi_validation_report.json")
DEFAULT_MD = Path("data/hindi_validation_report.md")

BHOJPURI_SUBJECTIVE_ORDER_ASC = "qwen < gemma < llama < mistral < aya"
NEAR_TIE_PTS = 1.0  # adjacent subjective rates within this many points render as "≈"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def gap_pts(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f} pts"


def render_ordering_asc(order: list[str], rates: dict) -> str:
    """Ascending subjective-rate order; near-ties (< NEAR_TIE_PTS) joined by '≈'."""
    parts = [f"{order[0]} ({pct(rates[order[0]]['subjective_rate'])})"]
    for prev, cur in zip(order, order[1:]):
        gap = (rates[cur]["subjective_rate"] - rates[prev]["subjective_rate"]) * 100
        parts.append(f" {'≈' if gap < NEAR_TIE_PTS else '<'} {cur} ({pct(rates[cur]['subjective_rate'])})")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-json", default=str(DEFAULT_JSON))
    ap.add_argument("--output", default=str(DEFAULT_MD))
    args = ap.parse_args()

    r = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    ds_ = r["dataset"]
    acc = r["per_judge_accuracy"]
    ent = r["entropy_split"]
    subj = r["subjectivity"]["rates"]
    so = r["subjective_only_subset"]
    ent_sub = r["entropy_split_subset"]
    obj_rates = so["objective_vote_rate_full_set"]
    judges = [j for j in acc]  # canonical order
    n_all = ds_["_sampled"]

    lines: list[str] = []
    add = lines.append

    add("# Hindi cross-lingual validation of the judge committee and aggregation")
    add("")
    add("> **This run validates the PIPELINE, not the Bhojpuri labels.** The committee, the frozen")
    add("> prompt, the pinned judge revisions, the decoding config, guided decoding, and the")
    add("> Dawid–Skene aggregation are exercised end-to-end against *existing third-party gold")
    add("> labels in Hindi* (the closest high-resource language sharing our script), because")
    add("> Bhojpuri gold is limited. Nothing here measures the quality of the Bhojpuri silver")
    add("> labels or of the eventual human adjudication on the Bhojpuri corpus.")
    add(">")
    add("> **Headline figures: the subjective-only comparable subset (§3).** The Hindi gold has no")
    add("> objective class, while the judges cast objective votes on a large fraction of items")
    add("> (measured in §4). Under this gold every objective vote is auto-scored wrong by")
    add("> construction, so the full-set accuracy figures (§6) are confounded by mapping loss and")
    add("> largely measure the label-scheme mismatch rather than pipeline accuracy. The")
    add("> subjective-only subset is the comparable evaluation; the full-set figures are retained")
    add("> for completeness and clearly marked.")
    add(">")
    add("> **The label-scheme mapping introduces error that bounds every accuracy number below")
    add("> from below.** The gold dataset carries 3-class polarity labels (neg/pos/neu) with no")
    add("> subjectivity stage: there is no gold \\\"objective\\\" class, so a judge (or consensus)")
    add("> \\\"objective\\\" vote is always scored wrong under this gold even when it is defensible,")
    add("> and their \\\"neu\\\" (a fact-flavored or lukewarm polarity judgment) is mapped to our")
    add("> Stage-B \\\"neutral\\\", not to our Stage-A \\\"objective\\\". Within the subjective-only")
    add("> subset the polarity stage *is* evaluated against real gold, but the subset is selected")
    add("> by the DS consensus itself, so subset figures carry a selection effect that the")
    add("> reported objective-vote rates make visible.")
    add("")

    add("## 1. Gold dataset selected")
    add("")
    add(f"- **Repo:** `{ds_['_dataset_repo_id']}` (HuggingFace Hub), file `{ds_['_dataset_file']}`")
    add(f"- **Source:** {ds_['_source_url']}")
    add(f"- **Size:** {ds_['_source_size']} sentence-level review snippets; sampled **{ds_['_sampled']}** (seed {ds_['_seed']}), stratified proportionally on the mapped gold label")
    add(f"- **Label space (gold):** 3-class polarity — neg {ds_['_source_distribution']['negative']}, pos {ds_['_source_distribution']['positive']}, neu {ds_['_source_distribution']['neutral']} — no subjectivity distinction")
    add("- **Licence:** none declared on the dataset card. The contributor (Kusumlata Patiyal) and the content (short Amazon product / movie review snippets in Hindi) place it in the IIT Patna Hindi review-corpora lineage, which is released CC BY-NC-SA 4.0; recorded in every row's `license_class` / `license_note` as `cc_by_nc_sa_4.0_assumed_iit_patna_lineage` — research, non-commercial use only.")
    add("- **Gating:** public, ungated (single flat file, no access request, no token).")
    add("- **Label mapping onto ours (PROTOCOL.md §3):** `neg` → `negative`, `pos` → `positive`, `neu` → `neutral` (Stage B). **Mapping loss, stated explicitly:** their scheme has no Stage A; `neu` is a *polarity* judgment, not our *objective* (factual-reporting) class, and no gold item can ever be correctly labeled `objective`. Scores are computed in the four-class space `{objective, positive, negative, neutral}` — the same space as the main corpus's validated aggregation — so the mapping loss shows up *inside* every full-set number rather than being papered over by excluding class rows; the subjective-only subset (§3) is the complementary correction.")
    add("- **Selection rationale:** candidates checked on the Hub on 2026-09-17 — mteb/sentiment_analysis_hindi (2,497 rows but split 1,249+1,248, each half below the 2,000 requirement, no licence), sepidmnorozy/Hindi_sentiment (no card/licence), iam-tsr/hindi-sentiments (MIT but unverifiable 2026 re-upload provenance), Process-Venue/Movie_Review_Sentiment_Hindi (~1,000 rows). The OdiaGenAI set is the only public ungated candidate meeting the ≥2,000-item sentence-level requirement with documented annotation conventions.")
    add("")

    add("## 2. Run conditions — identical to the Bhojpuri runs except the input file")
    add("")
    run_ = r["run"]
    add(f"- Prompt: `{run_['prompt_file']}` (sha256 `{run_['prompt_sha256']}`) — the frozen v1 prompt, unmodified")
    add("- Judges: the same five-judge committee (aya, gemma, llama, mistral, qwen), same pinned AWQ revisions (`purva/committee/models.py`), seed 42, temperature 0, xgrammar guided decoding, same schema — and the same package stack as the Bhojpuri runs (`requirements-kaggle-pinned.txt`: vllm 0.6.6.post1 / torch 2.5.1 / xgrammar 0.1.13, frozen so upstream dependency bumps cannot silently change the decoding conditions)")
    add("- Aggregation: standard Dawid–Skene EM, four-class, with the identifiability priors exactly as configured for the main corpus (`diag_prior=5.0`, `off_diag_prior=0.5`, `class_prior_strength=500.0`, class prior anchored to raw-vote frequency) — the validated primary method per `data/aggregation_report.md` §0")
    add(f"- Parse failures before repair: {run_['parse_failed_per_judge']}")
    add(f"- Rows missing per judge: {run_['rows_missing_per_judge']}")
    dsr = r["ds_result"]
    add(f"- DS convergence: {dsr['n_iter']} iterations, converged={dsr['converged']}, final log-likelihood {dsr['final_log_likelihood']:.2f}")
    add(f"- Raw 4-class vote distribution (all judges pooled): {r['raw_vote_distribution']}")
    add("")

    add(f"## 3. Headline result: subjective-only comparable subset (n={so['n_items']} of {n_all})")
    add("")
    add(f"**Definition.** {so['definition']}")
    add("")
    add(f"**Scoring rule.** {so['scoring_rule']}")
    add("")
    add(f"Subset size: **{so['n_items']} items** ({so['selection_rate'] * 100:.1f}% of {n_all}) — the items where the DS consensus is a polarity class.")
    add("")
    add(f"### Dawid–Skene vs majority vote on the comparable subset")
    add("")
    add(f"**Dawid–Skene polarity accuracy on the subset: {pct(so['ds']['accuracy'])}** (n scored = {so['ds']['n_scored']}; DS makes a polarity prediction on every subset item by construction).")
    add("")
    add(f"**Majority vote polarity accuracy on the subset: {pct(so['mv']['accuracy'])}** (n scored = {so['mv']['n_scored']}; {so['mv']['n_objective_abstained']} items had an objective plurality that abstains under polarity-only scoring).")
    add("")
    add(f"**Majority vote, four-class scored on the same subset: {pct(so['mv_four_class_on_subset'])}** — the matched-coverage comparison: {so['mv_four_class_note']}")
    add("")
    add(f"The DS-minus-MV gap on the subset is {gap_pts((so['ds']['accuracy'] or 0) - (so['mv']['accuracy'] or 0))} under polarity-only scoring. Both aggregators face identical mapping loss, so this comparison — like the full-set one in §6 — is unaffected by the confound; note the two framings differ in coverage: DS scores all {so['n_items']} subset items, while polarity-only MV abstains on {so['mv']['n_objective_abstained']} items where its plurality was objective — the matched-coverage comparison is the 4-class figure above.")
    add("")
    add("### Per-judge polarity accuracy on the subset, with objective-vote rates")
    add("")
    add("The full-set accuracy column is confounded: each judge's full-set figure is depressed by its own objective-vote rate (every objective vote auto-scored wrong), so the full-set ranking largely re-measures output-style propensity. Reading subset polarity accuracy next to each judge's objective-vote rate makes the confound visible.")
    add("")
    add("| Judge | Polarity acc (subset) | n scored | Obj-vote rate (within subset) | Obj-vote rate (full set) | Full-set acc (confounded) |")
    add("|---|---|---|---|---|---|")
    for judge in judges:
        pj = so["per_judge"][judge]
        add(f"| {judge} | {pct(pj['accuracy'])} | {pj['n_scored']} | {pct(pj['within_subset_objective_vote_rate'])} | {pct(obj_rates[judge])} | {pct(acc[judge]['accuracy'])} |")
    add("")
    rk = so["ranking"]
    add(f"- **Ranking on the full set (confounded):** {' > '.join(rk['full_set_by_confounded_accuracy'])}")
    add(f"- **Ranking on the subjective-only subset:** {' > '.join(rk['subjective_only_by_polarity_accuracy'])}")
    if rk["ranking_survives"]:
        add(f"- **Does the full-set ranking survive the correction? Yes.** Aya's apparent full-set lead is not merely its low objective-vote rate: it also ranks first on subset polarity accuracy.")
    else:
        moves = "; ".join(
            f"{j} full-set #{rk['full_set_by_confounded_accuracy'].index(j) + 1} → subset #{rk['subjective_only_by_polarity_accuracy'].index(j) + 1}"
            for j in rk["full_set_by_confounded_accuracy"]
            if rk["full_set_by_confounded_accuracy"].index(j) != rk["subjective_only_by_polarity_accuracy"].index(j)
        ) or "no judge moves position"
        add(f"- **Does the full-set ranking survive the correction? No.** {moves}. Aya's apparent full-set lead coincides with the lowest objective-vote rate of the five judges, so much of it reflects output-style propensity (casting fewer auto-scored-wrong objective votes) rather than better polarity judgment; the subset figures separate the two.")
    add("")

    add("## 4. Objective-vote rate — a measured label-scheme incompatibility statistic")
    add("")
    add("Fraction of **all** items on which each judge (raw derived vote) and each consensus predict `objective`:")
    add("")
    add("| Scorer | Objective votes | Rate over all items |")
    add("|---|---|---|")
    for judge in judges:
        n_votes_j = acc[judge]["n_votes"]  # judge obj rates are over valid votes
        add(f"| {judge} (judge) | {round(obj_rates[judge] * n_votes_j)} | {pct(obj_rates[judge])} |")
    add(f"| Dawid–Skene consensus | {round(obj_rates['ds_consensus'] * n_all)} | {pct(obj_rates['ds_consensus'])} |")
    add(f"| Majority-vote consensus | {round(obj_rates['mv_consensus'] * n_all)} | {pct(obj_rates['mv_consensus'])} |")
    add("")
    add(f"**The Hindi gold cannot evaluate the subjectivity stage.** The gold carries no objective class and no subjectivity annotation, so no number anywhere in this report measures whether the judges or the consensus classify subjectivity correctly. These rates are a measured incompatibility between the judges' output distribution and the gold's 3-class polarity scheme — the quantitative counterpart of the mapping-loss caveat, and the reason the full-set accuracies are confounded.")
    add("")

    add("## 5. Entropy routing test, recomputed on the subjective-only subset")
    add("")
    add("This tests whether the entropy-based routing rule — posterior entropy of the DS consensus, the quantity that drives Phase-5 human review routing on the main corpus — identifies genuinely hard cases. The earlier full-set entropy figures were also confounded by mapping loss; this subset recomputation is the meaningful version of the test.")
    add("")
    add(f"Split at normalised entropy ≤ {ent_sub['threshold_entropy_norm']} (the main corpus's routing threshold) *within the subset*: **low n={ent_sub['n_low']}**, **high n={ent_sub['n_high']}**.")
    add("")
    add("| Method | Polarity acc (low entropy) | Polarity acc (high entropy) | Gap |")
    add("|---|---|---|---|")
    add(f"| Dawid–Skene | {pct(ent_sub['ds']['low']['accuracy'])} | {pct(ent_sub['ds']['high']['accuracy'])} | {gap_pts((ent_sub['ds']['low']['accuracy'] or 0) - (ent_sub['ds']['high']['accuracy'] or 0))} |")
    add(f"| Majority vote | {pct(ent_sub['mv']['low']['accuracy'])} | {pct(ent_sub['mv']['high']['accuracy'])} | {gap_pts((ent_sub['mv']['low']['accuracy'] or 0) - (ent_sub['mv']['high']['accuracy'] or 0))} |")
    add("")
    add("Per-judge subset entropy splits are not broken out separately; the DS/MV rows above carry the routing-rule test.")
    add("")

    add("## 6. Full-set figures — CONFOUNDED by mapping loss, retained for completeness")
    add("")
    add("**Every accuracy number in this section is dominated by mapping loss and is NOT a measure of pipeline accuracy.** The gold has no objective class; judges cast objective votes on a large fraction of items (§4), and every one of those votes is auto-scored wrong by construction. Read §3 for the comparable evaluation.")
    add("")
    add(f"**Dawid–Skene consensus accuracy (full set): {pct(r['ds_accuracy'])}**")
    add("")
    add(f"**Majority-vote accuracy (full set): {pct(r['majority_vote_accuracy'])}** (for comparison)")
    add("")
    add(f"DS beats majority vote by {(r['ds_accuracy'] - r['majority_vote_accuracy']) * 100:.2f} points on the full set — both aggregators face identical mapping loss, so this gap is unaffected by the confound; it is the cleanest result of the exercise.")
    add("")
    add("| Judge | n votes | Accuracy vs gold (confounded) |")
    add("|---|---|---|")
    for judge, jacc in acc.items():
        add(f"| {judge} | {jacc['n_votes']} | {pct(jacc['accuracy'])} |")
    add("")
    add(f"DS vs majority-vote label disagreement: {r['ds_vs_mv_label_disagreement'] * 100:.2f}% of items "
        f"(majority-vote ties: {r['majority_vote_ties']['count']}, {r['majority_vote_ties']['rate'] * 100:.2f}%)")
    add("")
    add("Full-set entropy split (threshold the same as §5; confounded like everything else in this section):")
    add("")
    add(f"low n={ent['n_low']}, high n={ent['n_high']}. DS {pct(ent['ds_accuracy_low'])} (low) vs {pct(ent['ds_accuracy_high'])} (high); "
        f"MV {pct(ent['mv_accuracy_low'])} (low) vs {pct(ent['mv_accuracy_high'])} (high).")
    add("")

    add("## 7. Subjectivity ordering — stated precisely")
    add("")
    add("On the Bhojpuri corpus the per-judge subjective-rate ordering (ascending) was " + BHOJPURI_SUBJECTIVE_ORDER_ASC + ". The question is what reproduces in Hindi on different-register content.")
    add("")
    desc = r["subjectivity"]["ordering_by_subjective_rate_desc"]
    asc = list(reversed(desc))
    hi, lo = desc[0], desc[-1]
    add(f"Hindi subjective-rate ordering (ascending, with rates): {render_ordering_asc(asc, subj)}")
    add("")
    extremes_hold = hi == "aya" and lo == "qwen"
    middle_bho = ["gemma", "llama", "mistral"]
    middle_hin = [j for j in asc if j not in ("aya", "qwen")]
    assert extremes_hold, f"extremes changed unexpectedly: {desc}"
    if middle_hin != middle_bho:
        add(f"**What held: the extremes.** Qwen is the lowest subjective-rate judge and aya the highest in both languages. This cross-language stability of the extremes strengthens the finding that the judges differ in *output-style propensity* rather than language-specific sensitivity.")
        add("")
        add(f"**What did not hold: the middle three reorder.** Bhojpuri gave gemma < llama < mistral; Hindi gives {' < '.join(middle_hin)} (llama ≈ mistral near-tied). The ordering does **not** reproduce exactly, and must not be described as doing so: the claim that survives is the stability of the extremes plus a broadly shared low-vs-high split, not a fixed five-judge order.")
    else:
        add(f"**What held: the full ordering reproduces**, extremes and middle three alike (llama ≈ mistral near-tied).")
    add("")
    add("| Judge | n | objective | subjective | subjective rate |")
    add("|---|---|---|---|---|")
    for judge, s in subj.items():
        add(f"| {judge} | {s['n']} | {s['objective']} | {s['subjective']} | {pct(s['subjective_rate'])} |")
    add("")

    add("## 8. Confusion vs gold (full set — confounded by mapping loss)")
    add("")
    for name, key in (("Dawid–Skene", "ds_confusion_vs_gold"), ("Majority vote", "mv_confusion_vs_gold")):
        add(f"### {name}")
        add("")
        add("| gold \\ pred | objective | positive | negative | neutral | recall |")
        add("|---|---|---|---|---|---|")
        cm = r[key]["counts"]
        rec = r[key]["per_gold_recall"]
        for g in ("objective", "positive", "negative", "neutral"):
            row = cm[g]
            add(f"| {g} | {row['objective']} | {row['positive']} | {row['negative']} | {row['neutral']} | {pct(rec[g])} |")
        add("")
        add("(No gold items exist with label `objective` — see the mapping-loss caveat at the top.)")
        add("")

    add("## 9. Interpretation bounds")
    add("")
    add("1. **Validates:** the committee produces parseable, schema-valid labels on a related language; the frozen prompt transfers; the four-class DS with the identifiability priors runs, converges, and yields consensus labels; the routing entropy is computable and splittable.")
    add("2. **Comparable result (§3):** on the subjective-only subset — the only items where the gold can adjudicate the polarity stage — DS polarity accuracy is " + pct(so['ds']['accuracy']) + " vs majority vote " + pct(so['mv']['accuracy']) + " under polarity-only scoring (MV abstains on " + str(so['mv']['n_objective_abstained']) + " subset items), and " + pct(so['mv_four_class_on_subset']) + " for MV at matched coverage — DS leads by " + f"{((so['ds']['accuracy'] or 0) - so['mv_four_class_on_subset']) * 100:.2f}" + " points there. Both aggregators face identical mapping loss, so the DS-over-MV comparison is the cleanest cross-language finding in every framing that keeps coverage matched.")
    add("3. **Does not validate:** the Bhojpuri silver labels, the Bhojpuri register-stratum reliability findings, the eventual Bhojpuri human-adjudication quality — and, on this gold, the subjectivity stage itself (§4): the Hindi gold has no objective class, so it cannot score subjective/objective classification at all.")
    add("4. **Lower bound:** because the gold scheme is 3-class polarity with no subjectivity stage, every full-set accuracy is a floor; the subset figures are the comparable ones but carry a selection effect (the subset is defined by the DS consensus), visible through the reported objective-vote rates.")
    add("5. **Ordering (§7):** the extremes of the subjective-rate ordering reproduce across languages (qwen lowest, aya highest); the middle three reorder. This is cross-language stability of output-style propensities at the extremes — not an exact reproduction of the ordering.")
    add("")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
