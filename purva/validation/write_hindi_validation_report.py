"""Render data/hindi_validation_report.md from data/hindi_validation_report.json.

Markdown is generated, never hand-edited: every number in it comes from the
JSON (written by evaluate_hindi_validation.py), so the two reports cannot
drift. Run this after evaluate_hindi_validation.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_JSON = Path("data/hindi_validation_report.json")
DEFAULT_MD = Path("data/hindi_validation_report.md")

ORDER_DESC = "highest subjective rate first (Bhojpuri finding was aya > mistral > llama > gemma > qwen)"


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


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
    add("> **The label-scheme mapping introduces error that bounds every accuracy number below")
    add("> from below.** The gold dataset carries 3-class polarity labels (neg/pos/neu) with no")
    add("> subjectivity stage: there is no gold \"objective\" class, so a judge (or consensus)")
    add("> \"objective\" vote is always scored wrong under this gold even when it is defensible,")
    add("> and their \"neu\" (a fact-flavored or lukewarm polarity judgment) is mapped to our")
    add("> Stage-B \"neutral\", not to our Stage-A \"objective\". Every accuracy figure in this")
    add("> report is therefore a *lower bound* on true pipeline performance against a scheme")
    add("> isomorphic to ours.")
    add("")

    add("## 1. Gold dataset selected")
    add("")
    add(f"- **Repo:** `{ds_['_dataset_repo_id']}` (HuggingFace Hub), file `{ds_['_dataset_file']}`")
    add(f"- **Source:** {ds_['_source_url']}")
    add(f"- **Size:** {ds_['_source_size']} sentence-level review snippets; sampled **{ds_['_sampled']}** (seed {ds_['_seed']}), stratified proportionally on the mapped gold label")
    add(f"- **Label space (gold):** 3-class polarity — neg {ds_['_source_distribution']['negative']}, pos {ds_['_source_distribution']['positive']}, neu {ds_['_source_distribution']['neutral']} — no subjectivity distinction")
    add("- **Licence:** none declared on the dataset card. The contributor (Kusumlata Patiyal) and the content (short Amazon product / movie review snippets in Hindi) place it in the IIT Patna Hindi review-corpora lineage, which is released CC BY-NC-SA 4.0; recorded in every row's `license_class` / `license_note` as `cc_by_nc_sa_4.0_assumed_iit_patna_lineage` — research, non-commercial use only.")
    add("- **Gating:** public, ungated (single flat file, no access request, no token).")
    add("- **Label mapping onto ours (PROTOCOL.md §3):** `neg` → `negative`, `pos` → `positive`, `neu` → `neutral` (Stage B). **Mapping loss, stated explicitly:** their scheme has no Stage A; `neu` is a *polarity* judgment, not our *objective* (factual-reporting) class, and no gold item can ever be correctly labeled `objective`. Scores are computed in the four-class space `{objective, positive, negative, neutral}` — the same space as the main corpus's validated aggregation — so the mapping loss shows up *inside* every number below rather than being papered over by excluding class rows.")
    add("- **Selection rationale:** candidates checked on the Hub on 2026-09-17 — mteb/sentiment_analysis_hindi (2,497 rows but split 1,249+1,248, each half below the 2,000 requirement, no licence), sepidmnorozy/Hindi_sentiment (no card/licence), iam-tsr/hindi-sentiments (MIT but unverifiable 2026 re-upload provenance), Process-Venue/Movie_Review_Sentiment_Hindi (~1,000 rows). The OdiaGenAI set is the only public ungated candidate meeting the ≥2,000-item sentence-level requirement with documented annotation conventions.")
    add("")

    add("## 2. Run conditions — identical to the Bhojpuri runs except the input file")
    add("")
    run_ = r["run"]
    add(f"- Prompt: `{run_['prompt_file']}` (sha256 `{run_['prompt_sha256']}`) — the frozen v1 prompt, unmodified")
    add(f"- Judges: the same five-judge committee (aya, gemma, llama, mistral, qwen), same pinned AWQ revisions (`purva/committee/models.py`), seed 42, temperature 0, xgrammar guided decoding, same schema — and the same package stack as the Bhojpuri runs (`requirements-kaggle-pinned.txt`: vllm 0.6.6.post1 / torch 2.5.1 / xgrammar 0.1.13, frozen so upstream dependency bumps cannot silently change the decoding conditions)")
    add("- Aggregation: standard Dawid–Skene EM, four-class, with the identifiability priors exactly as configured for the main corpus (`diag_prior=5.0`, `off_diag_prior=0.5`, `class_prior_strength=500.0`, class prior anchored to raw-vote frequency) — the validated primary method per `data/aggregation_report.md` §0")
    add(f"- Parse failures before repair: {run_['parse_failed_per_judge']}")
    add(f"- Rows missing per judge: {run_['rows_missing_per_judge']}")
    dsr = r["ds_result"]
    add(f"- DS convergence: {dsr['n_iter']} iterations, converged={dsr['converged']}, final log-likelihood {dsr['final_log_likelihood']:.2f}")
    add(f"- Raw 4-class vote distribution (all judges pooled): {r['raw_vote_distribution']}")
    add("")

    add("## 3. Accuracy against Hindi gold")
    add("")
    add(f"**Dawid–Skene consensus accuracy: {pct(r['ds_accuracy'])}**")
    add("")
    add(f"**Majority-vote accuracy: {pct(r['majority_vote_accuracy'])}** (for comparison)")
    add("")
    add("| Judge | n votes | Accuracy vs gold |")
    add("|---|---|---|")
    for judge, jacc in acc.items():
        add(f"| {judge} | {jacc['n_votes']} | {pct(jacc['accuracy'])} |")
    add("")
    add(f"DS vs majority-vote label disagreement: {r['ds_vs_mv_label_disagreement'] * 100:.2f}% of items "
        f"(majority-vote ties: {r['majority_vote_ties']['count']}, {r['majority_vote_ties']['rate'] * 100:.2f}%)")
    add("")

    add("## 4. Low-entropy vs high-entropy items (routing-rule test)")
    add("")
    add("This tests whether the entropy-based routing rule — posterior entropy of the DS consensus, "
        "the quantity that drives Phase-5 human review routing on the main corpus — identifies "
        "genuinely hard cases.")
    add("")
    add(f"Split at normalised entropy ≤ {ent['threshold_entropy_norm']} (the same threshold the main "
        f"corpus's routing entropy distribution is reported against): **low n={ent['n_low']}**, "
        f"**high n={ent['n_high']}**.")
    add("")
    add("| Method | Accuracy (low entropy) | Accuracy (high entropy) | Gap |")
    add("|---|---|---|---|")
    add(f"| Dawid–Skene | {pct(ent['ds_accuracy_low'])} | {pct(ent['ds_accuracy_high'])} | "
        f"{(ent['ds_accuracy_low'] - ent['ds_accuracy_high']) * 100:+.2f} pts |")
    add(f"| Majority vote | {pct(ent['mv_accuracy_low'])} | {pct(ent['mv_accuracy_high'])} | "
        f"{(ent['mv_accuracy_low'] - ent['mv_accuracy_high']) * 100:+.2f} pts |")
    add("")
    add("| Judge | Accuracy (low) | Accuracy (high) |")
    add("|---|---|---|")
    for judge in acc:
        add(f"| {judge} | {pct(ent['per_judge_accuracy_low'][judge])} | {pct(ent['per_judge_accuracy_high'][judge])} |")
    add("")

    add("## 5. Per-judge subjectivity rates — does the ordering reproduce?")
    add("")
    add("On the Bhojpuri corpus, subjective-rate ordering ran Qwen lowest through Aya highest. If the "
        "same ordering reproduces in a different language on different register content, that "
        "substantially strengthens the finding that the judges differ in *output-style propensity*, "
        "not language-specific sensitivity.")
    add("")
    add("| Judge | n | objective | subjective | subjective rate |")
    add("|---|---|---|---|---|")
    for judge, s in subj.items():
        add(f"| {judge} | {s['n']} | {s['objective']} | {s['subjective']} | {pct(s['subjective_rate'])} |")
    add("")
    order = r["subjectivity"]["ordering_by_subjective_rate_desc"]
    add(f"Observed ordering ({ORDER_DESC}): **{' > '.join(order)}**")
    add("")

    add("## 6. Confusion vs gold")
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

    add("## 7. Interpretation bounds")
    add("")
    add("1. **Validates:** the committee produces parseable, schema-valid labels on a related language; the frozen prompt transfers; the four-class DS with the identifiability priors runs, converges, and yields consensus labels; the routing entropy is computable and splittable.")
    add("2. **Does not validate:** the Bhojpuri silver labels, the Bhojpuri register-stratum reliability findings, or the eventual Bhojpuri human-adjudication quality.")
    add("3. **Lower bound:** because the gold scheme is 3-class polarity with no subjectivity stage, every accuracy above is a floor — a pipeline judged against a scheme isomorphic to ours (with a real objective class) should score at or above these numbers.")
    add("")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
