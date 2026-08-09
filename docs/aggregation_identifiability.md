# Dawid–Skene identifiability failure on the `mixed` class

Source material for a methods subsection. Covers the failure, the diagnostic
evidence, the fix attempted, its partial success, and the resulting
label-space and primary-method decisions. See `PROTOCOL.md` CHANGELOG v1.7
and `data/aggregation_report.md` section 0 for the versioned record;
`purva/aggregate/dawid_skene.py` and `purva/aggregate/test_aggregation.py`
for the implementation.

## 1. Failure, as first observed

Five-class Dawid–Skene EM (label space `{objective, positive, negative,
neutral, mixed}`), with confusion matrices and class priors estimated by
unregularised MLE:

| | value |
|---|---|
| Raw `mixed` vote share (of 451,002 total votes cast) | 4,620 votes = 1.0% |
| Covariate-stratified DS consensus share of `mixed` | 26,665 items = 29.6% of the 90,207-item corpus |
| Unanimous-positive items (all 5 judges agree) relabelled `mixed` | 6,932 |
| Unanimous-objective items relabelled `mixed` | 4,521 |
| Log-likelihood | higher than the eventual fixed model |

The higher log-likelihood did not indicate a better fit to true labels: it
reflected EM exploiting a degenerate direction in an unconstrained
parameter.

## 2. Root cause

No judge produces `mixed` in the quantity needed to constrain that class's
row of any judge's confusion matrix (`P(vote = k' | true = mixed)`). With
free MLE estimation, an under-constrained confusion-matrix row is not
identified: EM is free to shape it into a catch-all that absorbs residual
variance from any item, in exchange for a small likelihood gain, regardless
of whether doing so improves the actual label. This is a standard
Dawid–Skene identifiability failure for a rare/low-support class, not a
bug in the E/M update equations.

## 3. Fix attempted

Two priors, applied identically in the standard and covariate-stratified
variants:

1. **Confusion-matrix prior.** A Dirichlet prior on every row of every
   judge's confusion matrix, applied at every M-step (not only at
   initialisation), with more pseudocount mass on the diagonal:
   `diag_prior = 5.0`, `off_diag_prior = 0.5` (encodes "annotators beat
   chance"). In the stratified variant this prior is applied per
   (judge, stratum) cell before the existing hierarchical shrinkage toward
   the judge's global matrix.
2. **Class-prior regularisation.** The class prior is shrunk toward the
   observed raw-vote frequency rather than estimated freely:
   `class_prior_strength = 500` pseudo-items of anchor weight.

Implementation: `purva/aggregate/dawid_skene.py`, `estimate_confusion()`
and `_regularised_class_prior()`.

## 4. Result of the fix, at five classes

| method | `mixed` consensus share | ratio vs. raw (1.0%) | unanimity violations |
|---|---|---|---|
| standard DS, unregularised (original failure) | ~29.6%* | ~29x | not measured directly; consistent with the 11,453 known bad relabels above |
| standard DS, with priors | 9.94% | 9.7x | 20 |
| stratified DS, with priors | 22.18% | 21.65x | 1 |

\* the 29.6% figure was measured on stratified DS, the method in production at the time; unregularised standard DS was not separately measured before the fix.

The priors reduced the effect by roughly 3x (standard DS) but did not
eliminate it. A separate sensitivity sweep pushing `class_prior_strength`
to an extreme (500 → 2,000 → 5,000 → 20,000 → 90,207, i.e. one pseudo-item
per real item) brought standard DS's ratio down to ~3.06 — still
(marginally) over the 3x threshold, and only by using an amount of
external anchoring strong enough to make the class prior nearly fixed
rather than estimated. This indicates the identifiability problem is
structural, not a matter of tuning two scalar priors further.

## 5. Permanent invariant tests

Added in `purva/aggregate/test_aggregation.py`, run automatically before
any aggregation output is written, hard-failing the pipeline if violated:

- **Unanimity invariant**: every item where all available (≥2) judges vote
  identically must receive that label under every reported method.
- **Class-ratio invariant**: no method's consensus share for a class may
  exceed 3x that class's raw-vote share.

## 6. Decision: four-class fallback

Per protocol, since `mixed` still failed both invariants after the priors
fix, it is dropped from the aggregation label space. Four-class DS is run
on `{objective, positive, negative, neutral}` (a judge's `mixed` vote is
treated as an abstention for that item, not imputed to another class).
`mixed` is retained and reported only as a raw-vote statistic (1.02% of
votes cast). See `PROTOCOL.md` CHANGELOG v1.7.

## 7. Second, unanticipated finding: the pathology moved, not disappeared

At four classes, standard DS passes both invariants cleanly. Covariate-
stratified DS — `PROTOCOL.md` §5's originally designated primary method —
does not: the same catch-all pathology reappears on `neutral`, the next
sparsest class (raw-vote share 4.3%):

| method (4-class) | `neutral` consensus share | ratio vs. raw (4.3%) | unanimity violations | invariants |
|---|---|---|---|---|
| standard DS | 7.53% | 1.75x | 0 | PASS |
| stratified DS | 24.54% | 5.71x | 5 | FAIL |

This ratio was robust to every regularisation setting tried:
`shrinkage_k0` ∈ {50, 200, 500}, `diag_prior` ∈ {5, 10}, `off_diag_prior`
= 0.5, `class_prior_strength` ∈ {500, 2000, 5000} — the ratio stayed in
5.68–6.14x across all five combinations tested, and unanimity violations
only reached 0 at `shrinkage_k0 ≥ 200` (the class-ratio violation never
cleared).

**Conclusion**: covariate-stratified Dawid–Skene is unidentifiable on this
corpus whenever a class's real annotator support is thin, independent of
which class that happens to be. Stratification makes this worse, not
better, because it multiplies the number of free confusion-matrix
parameters (one per judge per stratum, 5 judges × 8 strata × 4² = 640
entries for the 4-class case, versus 5 × 4² = 80 for standard DS) without
a proportional increase in the number of observations available to
constrain each one — several strata are small, and the sparse class's
already-thin support is spread thinner still across them.

Standard (unstratified) Dawid–Skene is therefore used as the validated
primary consensus method (`purva_aggregated.meta.json`:
`"primary_consensus_method": "dawid_skene"`). Stratified DS is still run
and shipped in the aggregation output — its per-stratum confusion matrices
remain the valid input to the register×text_type reliability analysis
(`data/aggregation_report.md` section 2), since the hypothesis there
concerns each judge's diagonal strength per stratum, not the aggregate
consensus label — but its overall consensus labels are excluded from
calibration ground truth and from the Phase-5 routing-set entropies.

## 8. Final consensus distributions (4-class, shipped)

| | raw vote share | standard DS (primary) | stratified DS (ablation, fails invariants) | majority vote | MACE |
|---|---|---|---|---|---|
| objective | 0.437 | 0.424 | 0.314 | 0.475 | 0.531 |
| positive | 0.297 | 0.286 | 0.252 | 0.306 | 0.282 |
| negative | 0.224 | 0.215 | 0.188 | 0.211 | 0.182 |
| neutral | 0.043 | 0.075 | 0.245 | 0.008 | 0.006 |

`mixed` (excluded from aggregation): 1.02% of raw votes.
