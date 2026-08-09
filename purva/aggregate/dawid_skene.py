"""Dawid-Skene EM: standard, and the covariate-stratified extension that is
this project's primary aggregation method (PROTOCOL.md §5).

Model: each judge j has a confusion matrix pi_j[k, k'] = P(judge j votes k'
| true class is k). Items have a class prior. EM alternates:
  M-step: given current per-item posteriors T, re-estimate the class prior
          and every judge's confusion matrix as posterior-weighted counts.
  E-step: given confusion matrices and prior, recompute each item's
          posterior over the classes from the judges' actual votes.

Missing votes (33 items have only 4 of 5 judges, from parse failures) are
handled by marginalisation, not imputation: a judge who didn't vote on an
item simply contributes no factor to that item's likelihood product.

IDENTIFIABILITY (see PROTOCOL.md CHANGELOG and data/aggregation_report.md
"identifiability failure" section for the full incident): an
unconstrained MLE confusion matrix is only identified for a class that at
least one judge produces in real quantity. "mixed" is ~1% of raw votes;
no judge's diagonal entry for it is meaningfully constrained by data, so a
free M-step can turn that row into a catch-all for residual variance —
observed directly as unanimous positive/objective items getting relabelled
"mixed", and as stratified DS assigning "mixed" to 29.6% of the corpus
against a 1.0% raw-vote rate. Two priors fix this:
  1. A Dirichlet prior on every confusion-matrix row, with more pseudo-mass
     on the diagonal than off it (diag_prior > off_diag_prior) — this
     encodes "annotators are better than chance" directly into the
     estimator, applied at every M-step, not just at initialisation.
  2. The class prior is shrunk toward the observed raw-vote frequency
     (class_prior_anchor) rather than estimated freely, so a class with
     near-zero raw support can't drift to dominate the prior.
Neither prior is switched off by default; DSConfig's defaults ARE the fix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ._common import N_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class DSConfig:
    alpha: float = 1.0  # add-alpha smoothing for the majority-vote INIT only (not the M-step confusion update)
    # Dirichlet pseudo-counts on each confusion-matrix row: diag_prior on
    # pi_j[k,k], off_diag_prior on every pi_j[k,k'] for k'!=k. Larger
    # diag_prior/off_diag_prior ratio = stronger "annotators beat chance"
    # assumption. 5.0/0.5 (10:1) was chosen after a sensitivity sweep (see
    # data/aggregation_report.md) — strong enough to stop "mixed" from
    # inflating past ~3x its raw-vote share, without visibly distorting the
    # well-identified classes' own diagonals.
    diag_prior: float = 5.0
    off_diag_prior: float = 0.5
    # Shrinkage strength (in pseudo-items) pulling the class prior toward
    # class_prior_anchor (typically raw_vote_frequency()). 0 disables it
    # (pure MLE, the pre-fix behaviour); the pipeline never actually runs
    # with 0 — this default exists so unit tests can isolate the confusion-
    # matrix prior's effect from the class-prior prior's effect.
    class_prior_strength: float = 500.0
    max_iter: int = 500
    # Relative, not absolute: log-likelihood here is a sum over 90,207
    # items and runs to ~1e5-1e6 in magnitude, so a fixed absolute
    # tolerance like 1e-4 would demand ~11 significant figures of
    # agreement and never trip before max_iter. Converged when
    # |ll_new - ll_old| < tol * |ll_old|.
    tol: float = 1e-6
    # Unused by the EM itself (majority-vote init is deterministic given the
    # data — there is no random restart here), kept for interface symmetry
    # with the rest of the pipeline.
    seed: int = 42


@dataclass
class StratifiedDSConfig(DSConfig):
    # Empirical-Bayes shrinkage strength: a stratum's raw confusion matrix
    # is weighted by n_stratum / (n_stratum + shrinkage_k0) toward the
    # judge's pooled (all-strata) confusion matrix, which is itself already
    # regularised by diag_prior/off_diag_prior. The two priors are
    # complementary, not redundant: shrinkage_k0 protects small STRATA
    # (e.g. the 31-item commentary/verse cell) from noisy per-stratum
    # estimates; diag_prior/off_diag_prior protects rare CLASSES (mixed)
    # from being unconstrained even in the pooled, all-9000-item case.
    shrinkage_k0: float = 50.0


@dataclass
class DSResult:
    posteriors: np.ndarray  # (N, K)
    class_prior: np.ndarray  # (K,)
    confusion: np.ndarray  # (J, K, K) standard; (S, J, K, K) stratified
    log_likelihood: list[float]
    n_iter: int
    converged: bool
    stratified: bool = False
    strata_keys: list[tuple[str, str]] | None = None  # index-aligned to confusion's S axis
    strata_sizes: np.ndarray | None = None
    global_confusion: np.ndarray | None = None  # (J, K, K), stratified's shrinkage target
    n_classes: int = N_CLASSES


def _majority_vote_init(votes: np.ndarray, alpha: float, n_classes: int) -> np.ndarray:
    n, _j = votes.shape
    counts = np.zeros((n, n_classes))
    for k in range(n_classes):
        counts[:, k] = (votes == k).sum(axis=1)
    n_voters = (votes != -1).sum(axis=1, keepdims=True)
    return (counts + alpha) / (n_voters + alpha * n_classes)


def estimate_confusion(votes: np.ndarray, T: np.ndarray, diag_prior: float = 5.0, off_diag_prior: float = 0.5) -> np.ndarray:
    """(J, K, K) confusion matrices, MAP-updated (posterior-mean form) under
    an asymmetric Dirichlet(diag_prior, off_diag_prior, ..., off_diag_prior)
    prior on each row — see module docstring. K is taken from T.shape[1],
    so this adapts automatically to a reduced label space (e.g. the
    four-class fallback)."""
    n_classes = T.shape[1]
    j_count = votes.shape[1]
    conf = np.zeros((j_count, n_classes, n_classes))
    prior_row_sum = diag_prior + off_diag_prior * (n_classes - 1)
    for j in range(j_count):
        col = votes[:, j]
        observed = col != -1
        weight_by_true = T[observed]  # (n_observed, K) posterior mass per true class
        for k in range(n_classes):
            denom = weight_by_true[:, k].sum() + prior_row_sum
            for kp in range(n_classes):
                prior = diag_prior if kp == k else off_diag_prior
                num = weight_by_true[col[observed] == kp, k].sum() + prior
                conf[j, k, kp] = num / denom
    return conf


def _regularised_class_prior(T: np.ndarray, class_prior_anchor: np.ndarray | None, class_prior_strength: float) -> np.ndarray:
    raw = T.mean(axis=0)
    if class_prior_anchor is not None and class_prior_strength > 0:
        n_items = T.shape[0]
        prior = (n_items * raw + class_prior_strength * class_prior_anchor) / (n_items + class_prior_strength)
    else:
        prior = raw
    prior = np.clip(prior, 1e-12, None)
    return prior / prior.sum()


def _e_step(votes: np.ndarray, log_class_prior: np.ndarray, log_confusion: np.ndarray) -> tuple[np.ndarray, float]:
    """log_confusion: (J, K, K). Returns (posteriors (N,K), total log-likelihood)."""
    n, j_count = votes.shape
    n_classes = log_class_prior.shape[0]
    log_post = np.broadcast_to(log_class_prior, (n, n_classes)).copy()
    for j in range(j_count):
        col = votes[:, j]
        idx = np.where(col != -1)[0]
        if idx.size == 0:
            continue
        # log_confusion[j][:, col[idx]].T has shape (len(idx), K): for each
        # observed item, the K log-probabilities of judge j's actual vote
        # under each candidate true class.
        log_post[idx, :] += log_confusion[j][:, col[idx]].T
    m = log_post.max(axis=1, keepdims=True)
    exp_shifted = np.exp(log_post - m)
    z = exp_shifted.sum(axis=1, keepdims=True)
    log_likelihood = float(np.sum(np.log(z) + m))
    posteriors = exp_shifted / z
    return posteriors, log_likelihood


def run_dawid_skene(
    votes: np.ndarray,
    config: DSConfig = DSConfig(),
    n_classes: int = N_CLASSES,
    class_prior_anchor: np.ndarray | None = None,
) -> DSResult:
    """Standard (unstratified) Dawid-Skene EM. class_prior_anchor should
    normally be raw_vote_frequency(votes, labels) — see module docstring
    for why this isn't optional in practice even though it's optional in
    the signature."""
    T = _majority_vote_init(votes, config.alpha, n_classes)
    ll_trajectory: list[float] = []
    prev_ll: float | None = None
    converged = False
    class_prior = T.mean(axis=0)
    confusion = None
    for it in range(1, config.max_iter + 1):
        class_prior = _regularised_class_prior(T, class_prior_anchor, config.class_prior_strength)
        confusion = estimate_confusion(votes, T, config.diag_prior, config.off_diag_prior)
        log_confusion = np.log(np.clip(confusion, 1e-300, None))
        log_class_prior = np.log(class_prior)
        T, ll = _e_step(votes, log_class_prior, log_confusion)
        ll_trajectory.append(ll)
        logger.info("EM iter %d: log-likelihood=%.3f", it, ll)
        if prev_ll is not None and abs(ll - prev_ll) < config.tol * abs(prev_ll):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll
    return DSResult(
        posteriors=T,
        class_prior=class_prior,
        confusion=confusion,
        log_likelihood=ll_trajectory,
        n_iter=len(ll_trajectory),
        converged=converged,
        stratified=False,
        n_classes=n_classes,
    )


def estimate_confusion_stratified(
    votes: np.ndarray, T: np.ndarray, strata: np.ndarray, n_strata: int,
    diag_prior: float, off_diag_prior: float, shrinkage_k0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (stratified_confusion (S,J,K,K), global_confusion (J,K,K),
    strata_sizes (S,)). Both the global and every per-stratum confusion
    matrix already carry the diag_prior/off_diag_prior Dirichlet prior
    (via estimate_confusion) *before* the stratum-to-global shrinkage is
    applied — the two regularisers stack rather than substitute for
    each other (see StratifiedDSConfig.shrinkage_k0's docstring)."""
    global_confusion = estimate_confusion(votes, T, diag_prior, off_diag_prior)
    n_classes = T.shape[1]
    j_count = votes.shape[1]
    strat_confusion = np.zeros((n_strata, j_count, n_classes, n_classes))
    strata_sizes = np.zeros(n_strata)
    for s in range(n_strata):
        mask = strata == s
        strata_sizes[s] = mask.sum()
        raw = estimate_confusion(votes[mask], T[mask], diag_prior, off_diag_prior) if mask.any() else global_confusion
        w = strata_sizes[s] / (strata_sizes[s] + shrinkage_k0)
        strat_confusion[s] = w * raw + (1 - w) * global_confusion
    return strat_confusion, global_confusion, strata_sizes


def run_stratified_dawid_skene(
    votes: np.ndarray,
    strata: np.ndarray,
    strata_keys: list[tuple[str, str]],
    config: StratifiedDSConfig = StratifiedDSConfig(),
    n_classes: int = N_CLASSES,
    class_prior_anchor: np.ndarray | None = None,
) -> DSResult:
    """Covariate-stratified Dawid-Skene EM — a separate confusion matrix per
    (judge, stratum) cell, hierarchically shrunk toward that judge's global
    (Dirichlet-regularised) confusion matrix."""
    n_strata = len(strata_keys)

    # _e_step assumes a single (J,K,K) confusion matrix shared by every
    # item; the stratified case needs a per-item lookup by stratum instead,
    # so this variant reimplements the E-step rather than reusing _e_step.
    T = _majority_vote_init(votes, config.alpha, n_classes)
    ll_trajectory: list[float] = []
    prev_ll: float | None = None
    converged = False
    strat_confusion = None
    global_confusion = None
    strata_sizes = None
    class_prior = T.mean(axis=0)
    for it in range(1, config.max_iter + 1):
        class_prior = _regularised_class_prior(T, class_prior_anchor, config.class_prior_strength)
        strat_confusion, global_confusion, strata_sizes = estimate_confusion_stratified(
            votes, T, strata, n_strata, config.diag_prior, config.off_diag_prior, config.shrinkage_k0,
        )
        log_strat_confusion = np.log(np.clip(strat_confusion, 1e-300, None))
        log_class_prior = np.log(class_prior)

        n, j_count = votes.shape
        log_post = np.broadcast_to(log_class_prior, (n, n_classes)).copy()
        for j in range(j_count):
            col = votes[:, j]
            idx = np.where(col != -1)[0]
            if idx.size == 0:
                continue
            cm = log_strat_confusion[strata[idx], j]  # (len(idx), K, K)
            log_post[idx, :] += cm[np.arange(idx.size), :, col[idx]]
        m = log_post.max(axis=1, keepdims=True)
        exp_shifted = np.exp(log_post - m)
        z = exp_shifted.sum(axis=1, keepdims=True)
        ll = float(np.sum(np.log(z) + m))
        T = exp_shifted / z

        ll_trajectory.append(ll)
        logger.info("stratified EM iter %d: log-likelihood=%.3f", it, ll)
        if prev_ll is not None and abs(ll - prev_ll) < config.tol * abs(prev_ll):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    return DSResult(
        posteriors=T,
        class_prior=class_prior,
        confusion=strat_confusion,
        log_likelihood=ll_trajectory,
        n_iter=len(ll_trajectory),
        converged=converged,
        stratified=True,
        strata_keys=strata_keys,
        strata_sizes=strata_sizes,
        global_confusion=global_confusion,
        n_classes=n_classes,
    )
