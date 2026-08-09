"""Dawid-Skene EM: standard, and the covariate-stratified extension that is
this project's primary aggregation method (PROTOCOL.md §5).

Model: each judge j has a confusion matrix pi_j[k, k'] = P(judge j votes k'
| true class is k). Items have a class prior. EM alternates:
  M-step: given current per-item posteriors T, re-estimate the class prior
          and every judge's confusion matrix as posterior-weighted counts.
  E-step: given confusion matrices and prior, recompute each item's
          posterior over the 5 classes from the judges' actual votes.

Missing votes (33 items have only 4 of 5 judges, from parse failures) are
handled by marginalisation, not imputation: a judge who didn't vote on an
item simply contributes no factor to that item's likelihood product. This
falls directly out of "missing at random" — the joint likelihood is a
product over *observed* votes only; there is nothing to integrate over
because a missing vote is not a censored/noisy observation of a value, it
is the absence of an observation. Imputing a value (e.g. the judge's modal
vote) would inject information the judge never actually provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ._common import N_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class DSConfig:
    alpha: float = 1.0  # Laplace/add-alpha smoothing strength
    max_iter: int = 500
    # Relative, not absolute: log-likelihood here is a sum over 90,207
    # items and runs to ~1e5-1e6 in magnitude, so a fixed absolute
    # tolerance like 1e-4 would demand ~11 significant figures of
    # agreement and never trip before max_iter. Converged when
    # |ll_new - ll_old| < tol * |ll_old|.
    tol: float = 1e-6
    # Unused by the EM itself (majority-vote init is deterministic given the
    # data — there is no random restart here), kept for interface symmetry
    # with the rest of the pipeline and so a future random-restart variant
    # doesn't need a signature change.
    seed: int = 42


@dataclass
class StratifiedDSConfig(DSConfig):
    # Empirical-Bayes shrinkage strength: a stratum's raw confusion matrix
    # is weighted by n_stratum / (n_stratum + shrinkage_k0) toward the
    # judge's pooled (all-strata) confusion matrix. Larger k0 = more
    # shrinkage for a given stratum size. 50 means a stratum needs roughly
    # 50 items before its own estimate carries as much weight as the global
    # prior — enough to keep the 31-item commentary/verse cell from
    # producing a near-empirical (and therefore wild) confusion matrix,
    # without over-smoothing the largest strata (thousands of items).
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


def _majority_vote_init(votes: np.ndarray, alpha: float) -> np.ndarray:
    n, _j = votes.shape
    counts = np.zeros((n, N_CLASSES))
    for k in range(N_CLASSES):
        counts[:, k] = (votes == k).sum(axis=1)
    n_voters = (votes != -1).sum(axis=1, keepdims=True)
    return (counts + alpha) / (n_voters + alpha * N_CLASSES)


def estimate_confusion(votes: np.ndarray, T: np.ndarray, alpha: float) -> np.ndarray:
    """(J, K, K) posterior-weighted, alpha-smoothed confusion matrices from
    the responsibilities T. conf[j, k, k'] = P(judge j votes k' | true k)."""
    n, j_count = votes.shape
    conf = np.zeros((j_count, N_CLASSES, N_CLASSES))
    for j in range(j_count):
        col = votes[:, j]
        observed = col != -1
        weight_by_true = T[observed]  # (n_observed, K) posterior mass per true class
        for k in range(N_CLASSES):
            denom = weight_by_true[:, k].sum() + alpha * N_CLASSES
            for kp in range(N_CLASSES):
                num = weight_by_true[col[observed] == kp, k].sum() + alpha
                conf[j, k, kp] = num / denom
    return conf


def _e_step(votes: np.ndarray, log_class_prior: np.ndarray, log_confusion: np.ndarray) -> tuple[np.ndarray, float]:
    """log_confusion: (J, K, K). Returns (posteriors (N,K), total log-likelihood)."""
    n, j_count = votes.shape
    log_post = np.broadcast_to(log_class_prior, (n, N_CLASSES)).copy()
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


def _run_em(
    votes: np.ndarray,
    config: DSConfig,
    confusion_step,
) -> tuple[np.ndarray, np.ndarray, object, list[float], int, bool]:
    """Shared EM loop. confusion_step(votes, T) -> (confusion_for_e_step,
    extra) where confusion_for_e_step is (J,K,K) and extra is whatever the
    caller wants carried through to the result (None for standard DS, the
    (stratified_confusion, global_confusion, strata_sizes) tuple for the
    stratified variant)."""
    T = _majority_vote_init(votes, config.alpha)
    ll_trajectory: list[float] = []
    prev_ll: float | None = None
    converged = False
    class_prior = T.mean(axis=0)
    extra = None
    for it in range(1, config.max_iter + 1):
        class_prior = np.clip(T.mean(axis=0), 1e-12, None)
        class_prior /= class_prior.sum()
        confusion_for_e_step, extra = confusion_step(votes, T)
        log_confusion = np.log(np.clip(confusion_for_e_step, 1e-300, None))
        log_class_prior = np.log(class_prior)
        T, ll = _e_step(votes, log_class_prior, log_confusion)
        ll_trajectory.append(ll)
        logger.info("EM iter %d: log-likelihood=%.3f", it, ll)
        if prev_ll is not None and abs(ll - prev_ll) < config.tol * abs(prev_ll):
            converged = True
            prev_ll = ll
            break
        prev_ll = ll
    return T, class_prior, extra, ll_trajectory, len(ll_trajectory), converged


def run_dawid_skene(votes: np.ndarray, config: DSConfig = DSConfig()) -> DSResult:
    """Standard (unstratified) Dawid-Skene EM."""
    last_confusion = {"value": None}

    def confusion_step(v, T):
        conf = estimate_confusion(v, T, config.alpha)
        last_confusion["value"] = conf
        return conf, None

    T, class_prior, _extra, ll_traj, n_iter, converged = _run_em(votes, config, confusion_step)
    return DSResult(
        posteriors=T,
        class_prior=class_prior,
        confusion=last_confusion["value"],
        log_likelihood=ll_traj,
        n_iter=n_iter,
        converged=converged,
        stratified=False,
    )


def estimate_confusion_stratified(
    votes: np.ndarray, T: np.ndarray, strata: np.ndarray, n_strata: int, alpha: float, shrinkage_k0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (shrunk_per_item_confusion placeholder unused, global_confusion,
    strata_sizes) — actually returns (stratified_confusion (S,J,K,K),
    global_confusion (J,K,K), strata_sizes (S,))."""
    global_confusion = estimate_confusion(votes, T, alpha)
    j_count = votes.shape[1]
    strat_confusion = np.zeros((n_strata, j_count, N_CLASSES, N_CLASSES))
    strata_sizes = np.zeros(n_strata)
    for s in range(n_strata):
        mask = strata == s
        strata_sizes[s] = mask.sum()
        raw = estimate_confusion(votes[mask], T[mask], alpha) if mask.any() else global_confusion
        w = strata_sizes[s] / (strata_sizes[s] + shrinkage_k0)
        strat_confusion[s] = w * raw + (1 - w) * global_confusion
    return strat_confusion, global_confusion, strata_sizes


def run_stratified_dawid_skene(
    votes: np.ndarray,
    strata: np.ndarray,
    strata_keys: list[tuple[str, str]],
    config: StratifiedDSConfig = StratifiedDSConfig(),
) -> DSResult:
    """Covariate-stratified Dawid-Skene EM — a separate confusion matrix per
    (judge, stratum) cell, hierarchically shrunk toward that judge's global
    (pooled) confusion matrix so small strata (e.g. the 31-item
    commentary/verse cell) don't produce a wild, near-empirical estimate."""
    n_strata = len(strata_keys)

    # _e_step assumes a single (J,K,K) confusion matrix shared by every
    # item; the stratified case needs a per-item lookup by stratum instead,
    # so this variant reimplements the E-step rather than reusing _e_step.
    T = _majority_vote_init(votes, config.alpha)
    ll_trajectory: list[float] = []
    prev_ll: float | None = None
    converged = False
    strat_confusion = None
    for it in range(1, config.max_iter + 1):
        class_prior = np.clip(T.mean(axis=0), 1e-12, None)
        class_prior /= class_prior.sum()
        strat_confusion, global_confusion, strata_sizes = estimate_confusion_stratified(
            votes, T, strata, n_strata, config.alpha, config.shrinkage_k0
        )
        log_strat_confusion = np.log(np.clip(strat_confusion, 1e-300, None))
        log_class_prior = np.log(class_prior)

        n, j_count = votes.shape
        log_post = np.broadcast_to(log_class_prior, (n, N_CLASSES)).copy()
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
    )
