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

In addition to the stage labels, the judge committee outputs, per sentence: `normalized_domain`, `narrative_voice` (`{first_person, third_person, mixed}`), `sentiment_target` (nullable free text), `confidence` (0–1), and `rationale` (maximum one sentence). `rationale` is collected on a 10% stratified subsample rather than the full corpus, for compute budget; see CHANGELOG v1.5. The rest of the schema is unchanged and collected on every sentence.

## 4. Judge committee

The committee comprises six open-weight models from six distinct organizations, run locally on Kaggle T4x2 with quantization, sequentially, with zero paid APIs:

- Llama-3.1-8B-Instruct (Meta)
- Gemma-2-9B-It (Google)
- Qwen2.5-14B-Instruct (Alibaba)
- Mistral-Nemo-12B (Mistral)
- Aya-Expanse-8B (Cohere)
- Airavata (AI4Bharat, Indic-specialist) — replaces the originally planned Sarvam-1, which is a base (non-instruction-tuned) model unsuited to structured JSON output; see CHANGELOG v1.3.

The final roster is locked by a 1,000-sentence stratified pilot. Any judge with a JSON-contract failure rate above 5% is dropped from the roster; the minimum roster size is five. All judges use greedy decoding, temperature 0, and seed 42. Judge runs specified revision `main` rather than a pinned commit SHA at run time; the corresponding commit SHA for each of the five judge repos was resolved post-hoc via the HF Hub API and is recorded, per judge, in `data/purva_master.meta.json` and `RUNS.md`, alongside that repo's `lastModified` timestamp so any drift between the run and the resolution stays visible. `purva/committee/models.py` now pins these resolved SHAs for future runs, and `run_judge.py` refuses to run against an unpinned `main` revision. See CHANGELOG v1.6.

A prompt-sensitivity study is run using 3 prompt paraphrases on a 10% stratified subsample. The full corpus is then labeled using one frozen prompt per judge.

Committee labeling is scoped to a 20,000-sentence stratified subsample of the corpus (proportional across `register` × `text_type` × `source_name`, seed 42; see `purva/committee/make_label_subset.py`), not the full 90,207-sentence corpus. The full corpus remains the separately released raw resource; see CHANGELOG v1.4.

## 5. Aggregation

The primary aggregation method is Dawid–Skene EM, initialized from majority vote with Laplace smoothing. Confusion matrices carry a diagonal-favouring Dirichlet prior applied at every M-step (not only at initialization), and class priors are shrunk toward the observed raw-vote frequency rather than estimated freely; both are needed to fix an identifiability failure in which a class with near-zero real annotator support became an unconstrained EM catch-all (see CHANGELOG v1.7). A covariate-stratified variant, stratified on `register` and `text_type` with per-stratum confusion matrices shrunk toward each judge's global matrix, is also run and reported — but on this corpus it does not pass the permanent unanimity/class-ratio invariants (`purva/aggregate/test_aggregation.py`) even after the same priors: the identifiability pathology recurs on whichever class is rarest, consistent with its far larger parameter count (one confusion matrix per judge per stratum, versus one per judge). It is therefore retained as an ablation and as the basis of the register×text_type reliability analysis, but excluded from consensus, calibration ground truth, and routing; standard (unstratified) Dawid–Skene is the validated primary method in practice. See CHANGELOG v1.7.

Ablations are run against the covariate-stratified variant above, majority vote, MACE, and GLAD (crowd-kit; refused above 20,000 items on this corpus — GLAD's measured runtime scales roughly quadratically and extrapolates to many hours at full scale). A leave-one-judge-out ablation is used for error-correlation analysis. Judge calibration is reported via ECE and Brier score, against both the primary consensus method and majority vote.

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

Only open weights are used. Judge runs specified revision `main` rather than a pinned commit SHA at run time; the corresponding commit SHA for each judge repo was resolved post-hoc and is recorded, per judge, in `data/purva_master.meta.json` and `RUNS.md` alongside that repo's `lastModified` timestamp (see CHANGELOG v1.6). `purva/committee/models.py` now pins these resolved SHAs and `run_judge.py` refuses to run against an unpinned `main` revision, so future runs are pinned before the fact. Prompts are stored verbatim in a `prompts/` directory (to be created in a later phase). Seeds are fixed and decoding is greedy throughout. All code is public in this repository. Zero paid APIs are used anywhere in the pipeline. All GPU work runs on free Kaggle quota.

## CHANGELOG

- v1.0 (2026-08-03): initial pre-registration.
- v1.1 (2026-08-03): §2 amended — IndicLID replaced with GlotLID; reason: IndicLID covers only Eighth-Schedule languages and has no Bhojpuri class; GlotLID provides bho_Deva/hin_Deva/mai_Deva as distinct classes.
- v1.2 (2026-08-03): sequencing amendment — LID human validation (§2) deferred to run concurrently with Phase-5 human annotation; no quarantine executed before validation scores are reviewed; lid_verdict retained as metadata; committee pilot may run on unquarantined data.
- v1.3 (2026-08-03): §4 roster note — Sarvam-1 replaced by Airavata (instruction-tuned; Sarvam-1 is a base model unsuited to structured JSON output). Revisions to be pinned at pilot completion.
- v1.4 (2026-08-04): committee labeling scoped to a 20,000-sentence stratified subsample; full corpus released as raw resource.
- v1.5 (2026-08-05): rationale collected on a 10% stratified subsample rather than the full corpus, for compute budget; schema otherwise unchanged.
- v1.6 (2026-08-09): §4 deviation recorded — judge runs specified revision "main" rather than pinned commit SHAs; SHAs resolved post-hoc and recorded in data/purva_master.meta.json and RUNS.md with repo lastModified timestamps. Future runs must pin explicit SHAs.
- v1.7 (2026-08-09): mixed excluded from the aggregation label space owing to Dawid–Skene unidentifiability for a class no judge reliably produces (1.0% of raw votes); mixed retained and reported at the raw-vote level. Additionally, the covariate-stratified Dawid–Skene variant designated as primary in §5 fails the same permanent invariant checks even at four classes (catch-all recurs on whichever class is rarest — "neutral", consensus/raw-vote ratio ~5.7x, robust across regularisation strength); standard Dawid–Skene is used as the validated primary consensus method instead, with the stratified variant retained as an ablation and for the register/text_type reliability analysis only.
