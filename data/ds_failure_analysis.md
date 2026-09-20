# Dawid-Skene failure analysis on contested (high-entropy) items

Date: 2026-09-21. Investigation only — **no aggregator code or shipped aggregation output was changed**;
`purva/aggregate/run_aggregation.py`'s four-class fallback with standard DS as primary consensus remains as-is.
Machine-readable numbers: `data/ds_failure_analysis.json` (full grid and per-part detail); the same parts are in
`scratch/ds_failure/results.json`. All DS re-runs reuse `purva.aggregate.dawid_skene.run_dawid_skene` unchanged.

**Inputs.** Contested gold: 45 human-labelled items split out of `data/annotate_50_contested_labelled.csv`
(4 `unclear` + 1 `not_bhojpuri` rows discarded to `data/gold_contested_excluded.jsonl` — the same convention as the
91+9 unanimous set; see `scratch/ds_failure/split_contested.py`). The 1 human-`mixed` row is outside every 4-class
label space, so the headline base is **n=45 with that row counted as an error for every aggregator** (the brief's
convention); n=44 with the row excluded is reported alongside. Unanimous gold: `data/gold_unanimous_91.jsonl`
(n=90 after the same mixed exclusion). Corpus: `data/purva_aggregated.jsonl` (90,207 items; votes, all four
aggregators' labels/posteriors/entropies).

## 0. Summary of findings

1. **Reproduced.** On the contested 45: DS **22.2% [11.1, 35.6]**, stratified DS **31.1% [17.8, 44.4]**, majority vote
   **44.4% [28.9, 57.8]**, MACE **46.7% [31.1, 60.0]**. Judges: qwen 40.0%, gemma 40.0%,
   llama 35.6%, aya 22.2%, mistral 33.3%*
   (*footnote below). Every individual judge matches or beats DS; aya exactly ties it.
2. **The neutral catch-all is real and corpus-wide but rare in absolute terms — and concentrated exactly where
   routing concentrates.** DS assigns a label that received **zero votes** on 465/90,207 items (**0.52%**);
   461 of the 465 are **neutral** assignments (6.8% of everything DS calls neutral), 94.4% occur on items whose
   five votes span **3 distinct labels**, and the contested-45 exemplifies the mechanism: DS assigns neutral on
   6/14 items with vote pattern {objective×2, positive×2, negative×1} — no judge voted
   neutral on any of them — and all 6 of those neutral calls are wrong against the human label.
3. **Removing neutral does not rescue DS.** Three-class DS (objective/positive/negative, same priors) lifts
   contested accuracy only 22.7%→**27.3%** (n=44) while three-class majority vote sits at **45.5%** — the gap is
   essentially unchanged (DS−MV goes from −22.7 to −18.2 points). The catch-all is a symptom; the deeper problem is
   that DS's reliability-weighted likelihood, estimated from a corpus that is ~90% near-unanimous, systematically
   misranks split votes.
4. **Not a tuning artifact.** Across a 28-cell sweep (diag_prior 1–20 × class_prior_strength 0–5,000, off_diag 0.5),
   contested accuracy stays in **[0.227, 0.295]** — never approaching MV's 0.455 — and the zero-vote rate is flat
   (0.41–0.52%) everywhere except under 10× stronger class-prior shrinkage, where it *falls* to 0.12% while
   contested accuracy still only reaches 0.295. Weakening the priors does not blow the zero-vote rate up: the
   phenomenon is EM's equilibrium on split votes, not prior-induced.
5. **Entropy routing: right direction, weak within-pool signal.** On the contested items, DS entropy barely
   separates DS's own errors from its correct calls (AUC **0.66**; error rate by entropy tertile 73% / 73% / 86%;
   Pearson r 0.26). Majority vote shows no signal at all (AUC 0.46). High DS entropy is a good *discovery*
   mechanism (it surfaces the split items where DS errs 13/14 of its zero-vote overrides and
   77% overall) but must not be sold as a within-pool difficulty ranking — the routing claim needs this
   qualification.

## 1. Contested-set results (n=45, mixed human row counted as error)

| System | Accuracy (n=45) | n=44 (mixed excluded) | Zero-vote assignments |
|---|---|---|---|
| Dawid-Skene (primary) | 22.2% [11.1, 35.6] | 22.7% | **14** (all neutral) |
| Stratified DS (non-primary) | 31.1% [17.8, 44.4] | 31.8% | 8 |
| Majority vote | 44.4% [28.9, 57.8] | 45.5% | 0 |
| MACE | 46.7% [31.1, 60.0] | 47.7% | 0 |

Individual judges (derived vote vs human label; a missing vote counts as an error):

| Judge | n=45 | n=44 |
|---|---|---|
| qwen | 40.0% | 40.9% |
| gemma | 40.0% | 40.9% |
| llama | 35.6% | 36.4% |
| aya | 22.2% | 22.7% |
| mistral | 33.3%* | 34.1%* |

\* **Mistral footnote (the one number that differs from the brief's 35.6%).** Mistral voted `mixed` on item
`8f666959e8c3eae1`, which the human also labelled `mixed`. Counting a judge's `mixed` vote as correct against a
human `mixed` label gives 16/45 = **35.6%**; the pipeline's own derivation (`_common.build_vote_matrix`, and
PROTOCOL CHANGELOG v1.7) treats `mixed` votes as abstentions, giving 15/45 = **33.3%**. This report uses the
pipeline convention everywhere (it is the same convention the aggregators are scored under). No conclusion changes
under either convention. DS's macro-F1 on the contested set is 0.247 (vs MV 0.195, MACE 0.267) — MV wins on
accuracy while leaning almost entirely on `objective`; DS's errors are spread across all four classes.

## 2. Task 1 — zero-vote assignments (assigned label received zero votes from any judge)

**Motivating example, confirmed and counted.** 14 of the contested 45 have exactly the vote pattern
{objective×2, positive×2, negative×1} (order varies by judge). DS assigns **neutral** to 6 of them
(DS posterior neutral 0.31–0.44, always the argmax or near-argmax despite zero neutral votes) and positive to the
other 8; majority vote assigns **objective** (the canonical-order tie-break) to all 14 and is right on the
9 human-objective items among them. DS is wrong on all 6 of its neutral calls here.

**Corpus-wide (90,207 items):**

| Aggregator | Zero-vote items | Rate | Of which `neutral` |
|---|---|---|---|
| Dawid-Skene | 465 | 0.516% | 461 (99.1%) |
| Stratified DS | 12,371 | 13.714% | 12,312 (99.5%) |
| Majority vote | 4 | 0.004% | 0 |
| MACE | 4 | 0.004% | 0 |

Per assigned label (DS): objective 4/38,200 (0.01%), positive 0/25,797, negative 0/19,416,
**neutral 461/6,794 (6.79%)**. The 4 items every aggregator (including MV and MACE) labels `objective` with zero
objective votes are items where **no judge cast any vote at all** (verified: 4 such items) — an
index-0/argmax-default artifact on empty items, not a modelling failure.

By number of distinct labels among the five votes (DS):

| Distinct votes | Items | Zero-vote assignments | Rate |
|---|---|---|---|
| 1 (unanimous) | 33,198 | 0 | 0% |
| 2 | 43,169 | 22 | 0.05% |
| **3** | 12,199 | **439** | **3.60%** |
| 4 | 1,637 | 0 | 0% |

94.4% of DS's zero-vote mass sits on 3-way-split items. **The problem is confined to split votes, and within them
it is neutral-specific.** Stratified DS (excluded from consensus since CHANGELOG v1.7) shows the same pathology at
16× the scale (13.7% of the corpus; 55.6% of all its neutral assignments are zero-vote), concentrated on 2-vote
items — consistent with its documented over-parameterisation failure. On DS's 465 zero-vote items the mean
assigned-class posterior is 0.47 and the mean normalised entropy is 0.80: DS itself signals high uncertainty
exactly there — the information needed to catch this behaviour already exists in the shipped posterior.

## 3. Task 2 — three-class re-run ({objective, positive, negative})

Standard DS re-run with **neutral votes mapped to abstentions** (exactly the treatment the four-class fallback
applies to `mixed`), same priors (diag 5.0 / off-diag 0.5 / class-prior anchor 500), converged in 10 iterations.
Direct neutral-vote removal touches only 2 of the 45 contested items — the cast votes there span 30 items with
{obj, pos, neg} only, 12 with {obj, pos}, 2 with {pos, neg, neu}, and 1 with {pos, neg} — and 7 judge-votes
on this set are missing entirely (parse failures), so abstention-mapping has almost no raw material to work with.

| Gold set | DS-3class | MV-3class | 4-class DS (shipped) | 4-class MV (shipped) |
|---|---|---|---|---|
| Contested n=44 | **27.3%** [13.6, 43.2] | **45.5%** [29.5, 61.4] | 22.7% | 45.5% |
| Contested n=38 (human-neutral items excluded) | **31.6%** | **52.6%** | 21.1% | 52.6% |
| Unanimous n=90 (mixed excluded) | 87.8% | 87.8% | 91.1% | 91.1% |

**DS does not recover relative to majority vote on the contested set.** The 3-class zero-vote pathology itself
vanishes (corpus-wide zero-vote rate 0.033% — 30 items, all objective, all empty-vote items — and **0 on the
contested 45**), and DS's own accuracy does improve 22.7%→27.3%, but majority vote is untouched at 45.5%: the gap
closes by only 4.6 points. The mechanism is now visible by subtraction: with neutral out of the label space DS can
no longer park split-vote posterior mass on a zero-support class, yet its likelihood still prefers the wrong
objective/positive/negative class on the same items. Removing the catch-all removes the symptom, not the ranking
failure. On the unanimous set DS-3class equals MV-3class exactly (unanimous items, trivially identical).

**Neutral mapping decision.** No principled rule exists for mapping a human `neutral` to
{objective, positive, negative}: `objective` is a subjectivity-axis judgment and positive/negative a polarity-axis
judgment; a mild/hedged stance has no nearest neighbour on either axis. Human-neutral items (6 contested, 5
unanimous) are therefore **excluded** from the neutral-void cut (reported above) and counted as errors in the full
bases.

## 4. Task 3 — prior sensitivity sweep

28 runs, diag_prior {1, 2, 3, 5, 8, 12, 20} × class_prior_strength {0, 50, 500, 5000} (off_diag 0.5; all other
config identical to shipped). Full grid in the JSON.

| | contested acc (n=44) | zero-vote rate | neutral consensus share | unanimous acc |
|---|---|---|---|---|
| min over grid | 0.227 | 0.0012 | 0.069 | 0.911 |
| max over grid | 0.295 | 0.0052 | 0.075 | 0.911 |
| shipped (5.0, 500) | 0.227 | 0.0052 | 0.075 | 0.911 |

- Contested accuracy is **flat within [0.227, 0.295]** across the entire grid — no setting recovers majority
  vote's 0.455. The largest contested-accuracy gains come from the *class-prior* anchor at 10× its shipped
  strength (cps=5000 → 0.295), i.e. from forcibly dragging neutral's prior toward its 4.3% raw-vote share — and
  even that only removes ~76% of the zero-vote assignments without fixing the ranking of split votes.
- The zero-vote rate is essentially **invariant to the confusion-matrix prior** (0.0052 at every diag_prior for
  cps ≤ 500) and *decreases* only under extreme shrinkage. If the pathology were prior-induced, weakening priors
  would inflate it; instead it is EM's stable equilibrium: with 5 categorical votes, a likelihood built from
  corpus-wide confusion matrices genuinely prefers a low-support class on specific 3-way split patterns.
- Unanimous-set accuracy is 0.911 at every cell: nothing in the grid trades easy-set performance for
  contested-set performance, because contested items are simply not represented in what the likelihood rewards.

**Verdict: structural, not a tuning artifact.**

## 5. Task 4 — does DS posterior entropy predict DS error on contested items?

Contested items only (n=44), DS normalised posterior entropy vs DS correctness:

| Statistic | Dawid-Skene | Majority vote (vote-share entropy) |
|---|---|---|
| AUC (entropy ranks error) | **0.659** | 0.465 |
| Pearson r (entropy vs 0/1 error) | 0.262 | −0.152 |
| Spearman rho | 0.233 | −0.069 |
| Mean entropy, errors vs correct | 0.761 vs 0.727 | 0.657 vs 0.694 |
| Error rate by entropy tertile | 73% / 73% / **86%** | 59% / 50% / 100% (n=1) |

**Interpretation — the routing signal is real between pools but weak within the pool.** Within the contested set,
entropy separates DS's errors from its correct calls only marginally: the bottom two entropy tertiles are
indistinguishable (73% error both), and even the top tertile is only 86%. AUC 0.66 is directionally right but far
from a usable per-item prioritiser. For majority vote there is no signal at all. What entropy demonstrably does
is find the *pool* (corpus-wide, contested items sit at the top of the entropy distribution, and DS's zero-vote
items average 0.80 normalised entropy). The honest qualification: **routing by entropy threshold is validated as a
pool-discovery mechanism; it is not validated as a within-pool difficulty ranking, and DS's own entropy should not
be read as a per-item confidence on the items that matter most.**

## 6. Conclusions and implications (no changes made)

1. The brief's core claim is confirmed and quantified: DS's remaining failure mode after the v1.7 fix is
   **neutral-as-residual-class on 3-way split votes** — 465 corpus-wide zero-vote assignments, 94% on 3-vote
   splits, 99% labelled neutral. On the contested set, DS makes 14 zero-vote assignments (all on the n=44 base) and 13 of them
   are wrong (13/14 = 93% error on exactly the items where DS
   overrode the committee); within the 14-item {obj×2, pos×2, neg×1} pattern specifically, DS sends
   6 items to neutral (zero neutral votes cast) and all 6 of those calls are
   wrong.
2. But the failure is **not caused by neutral alone**: with neutral removed from the label space DS still trails
   majority vote by ~18 points on the contested set. The catch-all is the most *visible* symptom of a deeper
   issue — likelihood weights fit on a corpus that is ~90% near-unanimous do not transfer to the contested tail,
   where they disagree with what a human rewards. On this evidence, **majority vote (or MACE) with entropy-based
   routing is the better contested-set aggregator**; DS's contested-set behaviour (not just stratified DS's, which
   fails invariants outright) must be treated as unresolved.
3. Prior tuning cannot fix it (28-cell sweep), and the entropy routing signal needs the within-pool qualification
   in §5. Until the aggregator is changed, the 14-item zero-vote pattern is *detectable in the shipped output*
   (zero supporting votes for the argmax class; mean entropy 0.80) and could be used as an explicit
   "committee-split → human" routing rule without touching DS.
4. Options for the aggregator decision (none implemented, per task constraints): (a) make majority vote/MACE the
   primary for contested items and keep DS only where the committee is near-unanimous; (b) add a hard
   zero-vote-veto post-processor to DS (never emit a class with zero supporting votes); (c) fit DS's priors on
   adjudicated contested items once Phase-5 three-annotator labels exist. Each needs its own evaluation before
   adoption.

**Caveats.** n=45 is small (95% CIs span ±15–20 points); one human annotator, not the three-annotator adjudicated
process of PROTOCOL §6; the contested set comes from a single high-entropy routing slice drawn under the shipped
DS posterior, so "contested" is DS-relative by construction; per-judge accuracies use derived votes (mixed→
abstention) per the pipeline convention, and mistral's figure is convention-sensitive (§1 footnote); the 4
all-aggregator zero-vote `objective` items are empty-vote items (no judge cast any vote), an index-default
artifact present in every aggregator including MV and MACE.

**Files produced by this task:** `data/ds_failure_analysis.md` (this file), `data/ds_failure_analysis.json`,
`data/gold_contested_45.jsonl`, `data/gold_contested_excluded.jsonl` (gold split; 45+5 per the annotation file),
plus working scripts and intermediate results under `scratch/ds_failure/` (not part of the shipped pipeline).
