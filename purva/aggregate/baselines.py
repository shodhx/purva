"""Baseline aggregators for comparison against Dawid-Skene (PROTOCOL.md §5):
majority vote with a documented tie-break, and MACE/GLAD via crowd-kit when
it's actually available and working. Per the brief: if crowd-kit can't be
installed or a model fails, this reports *why* rather than faking a result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._common import JUDGES, LABELS, N_CLASSES

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    posteriors: np.ndarray | None  # (N, K); None when unavailable
    available: bool = True
    unavailable_reason: str | None = None
    extra: dict = field(default_factory=dict)


def majority_vote(votes: np.ndarray) -> BaselineResult:
    """Plain vote share among judges who actually voted (denominator is
    n_voters, 4 or 5, not a fixed 5 — this is majority vote's own way of
    handling the 33 items with a missing judge, no DS-style marginalisation
    needed since there's no likelihood model here).

    Tie-break rule (documented, not hidden): ties go to whichever tied
    class comes first in the canonical order LABELS = (objective, positive,
    negative, neutral, mixed). This is exactly numpy.argmax's own
    tie-break behavior (first occurrence of the max wins), so the decision
    itself needs no extra code — only the tie *count* is worth tracking
    separately, reported in `extra`."""
    n = votes.shape[0]
    counts = np.zeros((n, N_CLASSES))
    for k in range(N_CLASSES):
        counts[:, k] = (votes == k).sum(axis=1)
    n_voters = np.clip((votes != -1).sum(axis=1, keepdims=True), 1, None)
    posteriors = counts / n_voters

    max_count = counts.max(axis=1)
    tie_count = int(((counts == max_count[:, None]).sum(axis=1) > 1).sum())
    return BaselineResult(
        name="majority_vote",
        posteriors=posteriors,
        extra={
            "tie_count": tie_count,
            "tie_rate": tie_count / n,
            "tie_break_rule": f"first class in canonical order {list(LABELS)} wins",
        },
    )


def _votes_to_long_df(votes: np.ndarray, item_ids: np.ndarray, judges: tuple[str, ...] = JUDGES) -> pd.DataFrame:
    """crowd-kit's expected long format: one row per (task, worker, label)
    — a judge with no vote on an item (-1) simply contributes no row,
    which is crowd-kit's own way of handling missing votes."""
    parts = []
    for j, name in enumerate(judges):
        col = votes[:, j]
        mask = col != -1
        parts.append(pd.DataFrame({
            "task": item_ids[mask],
            "worker": name,
            "label": [LABELS[c] for c in col[mask]],
        }))
    return pd.concat(parts, ignore_index=True)


def _proba_to_posteriors(proba: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    proba = proba.reindex(index=item_ids, columns=list(LABELS), fill_value=0.0)
    arr = proba.to_numpy(dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) == 0
    if zero_rows.any():
        # A task crowd-kit dropped output for (shouldn't happen given every
        # item has >=4 votes, but fail safe rather than divide by zero).
        arr[zero_rows] = 1.0 / N_CLASSES
        row_sums[zero_rows] = 1.0
    return arr / row_sums


def run_mace(votes: np.ndarray, item_ids: np.ndarray, n_restarts: int = 3, n_iter: int = 50, seed: int = 42) -> BaselineResult:
    """MACE (Hovy et al. 2013) via crowd-kit. n_restarts reduced from
    crowd-kit's default of 10 to 3 — a compute-budget tradeoff (each
    restart is a full EM run over ~451k (task,worker) rows), documented
    here rather than silently changed, matching this project's practice
    elsewhere (e.g. rationale collection scoped to a subsample for the same
    reason; see PROTOCOL.md CHANGELOG v1.5)."""
    try:
        from crowdkit.aggregation import MACE
    except Exception as e:  # pragma: no cover - exercised only if crowd-kit is absent/broken
        return BaselineResult("mace", None, available=False, unavailable_reason=f"crowd-kit import failed: {e!r}")
    try:
        df = _votes_to_long_df(votes, item_ids)
        model = MACE(n_restarts=n_restarts, n_iter=n_iter, random_state=seed)
        proba = model.fit_predict_proba(df)
        posteriors = _proba_to_posteriors(proba, item_ids)
        return BaselineResult("mace", posteriors, extra={"n_restarts": n_restarts, "n_iter": n_iter})
    except Exception as e:
        logger.exception("MACE fit_predict_proba failed")
        return BaselineResult("mace", None, available=False, unavailable_reason=f"crowd-kit MACE failed at runtime: {e!r}")



# Measured directly against this corpus (crowd-kit 1.4.2): GLAD's runtime
# scales roughly quadratically, not linearly — 500 items/18s, 2,000/34s,
# 5,000/207s. Extrapolating that growth rate to the full 90,207-item corpus
# lands in the range of many hours to a day for a single run, which is not
# a "crowd-kit failed" situation but is equally not something this pipeline
# can afford to wait on. Rather than silently truncate the corpus or fake a
# result, GLAD is refused outright above this size with the measurement
# that justifies it — see baselines.py's run_glad docstring/report output.
GLAD_MAX_PRACTICAL_ITEMS = 20_000


def run_glad(votes: np.ndarray, item_ids: np.ndarray, n_iter: int = 100, tol: float = 1e-5) -> BaselineResult:
    """GLAD (Whitehill et al. 2009) via crowd-kit."""
    if len(item_ids) > GLAD_MAX_PRACTICAL_ITEMS:
        return BaselineResult(
            "glad", None, available=False,
            unavailable_reason=(
                f"{len(item_ids)} items exceeds the measured-practical limit of {GLAD_MAX_PRACTICAL_ITEMS}: "
                "crowd-kit's GLAD scales roughly quadratically on this corpus (500 items/18s, 2,000/34s, "
                "5,000/207s measured directly), which extrapolates to many hours at full scale — not a "
                "crashing failure, but not a run this pipeline can complete in practical time either."
            ),
        )
    try:
        from crowdkit.aggregation import GLAD
    except Exception as e:  # pragma: no cover
        return BaselineResult("glad", None, available=False, unavailable_reason=f"crowd-kit import failed: {e!r}")
    try:
        df = _votes_to_long_df(votes, item_ids)
        model = GLAD(n_iter=n_iter, tol=tol, silent=True)
        proba = model.fit_predict_proba(df)
        posteriors = _proba_to_posteriors(proba, item_ids)
        return BaselineResult("glad", posteriors, extra={"n_iter": n_iter, "tol": tol})
    except Exception as e:
        logger.exception("GLAD fit_predict_proba failed")
        return BaselineResult("glad", None, available=False, unavailable_reason=f"crowd-kit GLAD failed at runtime: {e!r}")
