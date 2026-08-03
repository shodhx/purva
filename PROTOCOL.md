# PURVA Research Protocol — Pre-registration

Version: 1.0
Date: 2026-08-03

## 1. Study overview

PURVA is a budget-aware human+LLM consensus framework for low-resource dataset curation, demonstrated on a Bhojpuri sentiment dataset. The input corpus consists of 90,207 deduplicated, metadata-tagged sentences collected from 12 documented web sources (see SOURCES.md). Each record carries the following fields: `id`, `raw_text`, `cleaned_text`, `source_url`, `source_name`, `scrape_timestamp`, `category`, `register`, `license_class`, `script`, `text_type`.

## 2. Language verification

GlotLID v3 (cis-lmu/glotlid, Apache 2.0 license) is applied to all sentences in the corpus. GlotLID replaces the originally planned IndicLID (AI4Bharat): IndicLID covers only the 22 Eighth-Schedule Indian languages and has no Bhojpuri class, so it cannot perform the Bhojpuri-vs-Hindi separation this stage requires. GlotLID predictions are mapped to a four-way verdict: `bho_Deva` → `bhojpuri`; `hin_Deva` → `hindi`; `mai_Deva` → `maithili` (Bhojpuri's closest scheduled sibling; its confusion rate with Bhojpuri is itself a reportable finding); anything else → `other`. Sentences receiving a non-Bhojpuri verdict are quarantined to a separate file, held out from downstream annotation and modeling. Before quarantine decisions are treated as final, a 500-sentence human-verified stratified sample (proportional by source) is drawn to measure GlotLID accuracy on this corpus.

## 3. Label scheme

The label scheme is frozen after one documented calibration round. It is a two-stage scheme:

- Stage A — subjectivity: `{objective, subjective}`.
- Stage B — polarity, applied only to sentences labeled subjective: `{positive, negative, neutral, mixed}`.

The final label space is `{objective, positive, negative, neutral, mixed}`.

In addition to the stage labels, the judge committee outputs, per sentence: `normalized_domain`, `narrative_voice` (`{first_person, third_person, mixed}`), `sentiment_target` (nullable free text), `confidence` (0–1), and `rationale` (maximum one sentence).

## 4. Judge committee

The committee comprises six open-weight models from six distinct organizations, run locally on Kaggle T4x2 with quantization, sequentially, with zero paid APIs:

- Llama-3.1-8B-Instruct (Meta)
- Gemma-2-9B-It (Google)
- Qwen2.5-14B-Instruct (Alibaba)
- Mistral-Nemo-12B (Mistral)
- Aya-Expanse-8B (Cohere)
- Sarvam-1 or Airavata (Indic-specialist)

The final roster is locked by a 1,000-sentence stratified pilot. Any judge with a JSON-contract failure rate above 5% is dropped from the roster; the minimum roster size is five. All judges use greedy decoding, temperature 0, and seed 42. HuggingFace revisions are pinned in this file at pilot time (to be recorded here once the pilot completes).

A prompt-sensitivity study is run using 3 prompt paraphrases on a 10% stratified subsample. The full corpus is then labeled using one frozen prompt per judge.

## 5. Aggregation

The primary aggregation method is Dawid–Skene EM with covariate-stratified confusion matrices, stratified on `register` and `text_type`, initialized from majority vote with Laplace smoothing.

Ablations are run against vanilla Dawid–Skene, majority vote, MACE, and GLAD (crowd-kit). A leave-one-judge-out ablation is used for error-correlation analysis. Judge calibration is reported via ECE and Brier score.

Routing to human review is triggered when posterior entropy exceeds a threshold τ; τ is set according to the human annotation budget. The full cost–quality frontier is reported across values of τ.

## 6. Human annotation

Three annotators (the lead and two faculty reviewers) label independently, blinded to all model outputs, followed by adjudication. The following samples are drawn:

- (a) the routed high-entropy set (~2,000 sentences)
- (b) a low-entropy control (500 sentences)
- (c) a uniform random slice (300 sentences), reserved as an unbiased test split
- (d) a shared reliability subset (300 sentences), labeled by all three annotators, for Krippendorff's α, Gwet's AC1, and Fleiss' κ

Silver-layer agreement is reported via Fleiss' κ, Krippendorff's α, and a pairwise judge-agreement matrix. Per-annotator labels are retained in the release.

## 7. Causal analyses

- (a) Counterfactual minimal-edit pairs: generated across the corpus by models (flagged as unverified), with a human-verified stratified probe set of 500–1,000 pairs.
- (b) Disagreement-driver regression: posterior entropy regressed on `register`, `text_type`, `source`, and `script`.
- (c) Spurious-correlation audit: source↔label association analysis, plus source-held-out evaluation splits.

## 8. Benchmarks

MuRIL, IndicBERT-v2, and XLM-R are fine-tuned on silver labels and evaluated on human gold labels, with routed and random slices reported separately. Bootstrap 95% confidence intervals and McNemar significance tests are reported. Results are sliced by `register`, `text_type`, entropy stratum, and the counterfactual probe set. Judge-circularity is assessed by comparing judge-family versus non-judge-family models as evaluators.

## 9. Blinding and integrity

Human annotators never see model labels or rationales prior to adjudication. Test-split items are never used for model selection or threshold selection. Every protocol deviation is logged with its date in the CHANGELOG section of this file.

## 10. Reproducibility

Only open weights are used, with pinned revisions. Prompts are stored verbatim in a `prompts/` directory (to be created in a later phase). Seeds are fixed and decoding is greedy throughout. All code is public in this repository. Zero paid APIs are used anywhere in the pipeline. All GPU work runs on free Kaggle quota.

## CHANGELOG

- v1.0 (2026-08-03): initial pre-registration.
- v1.1 (2026-08-03): §2 amended — IndicLID replaced with GlotLID; reason: IndicLID covers only Eighth-Schedule languages and has no Bhojpuri class; GlotLID provides bho_Deva/hin_Deva/mai_Deva as distinct classes.
- v1.2 (2026-08-03): sequencing amendment — LID human validation (§2) deferred to run concurrently with Phase-5 human annotation; no quarantine executed before validation scores are reviewed; lid_verdict retained as metadata; committee pilot may run on unquarantined data.
