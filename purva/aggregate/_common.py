"""Shared constants and vote-matrix utilities for purva/aggregate/*.

Kept separate from dawid_skene.py so baselines.py, analysis.py, and
make_routing_sets.py can all depend on the vote-matrix/label/entropy
plumbing without depending on the EM implementation itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The final label space (PROTOCOL.md §3). Fixed order — this is the class
# index used throughout purva/aggregate/, and the tie-break priority order
# for majority vote (see baselines.py: np.argmax returns the first index
# among ties, i.e. the earliest label in this tuple wins).
LABELS = ("objective", "positive", "negative", "neutral", "mixed")
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
N_CLASSES = len(LABELS)

JUDGES = ("aya", "gemma", "llama", "mistral", "qwen")

MASTER_PARQUET = "data/purva_master.parquet"


def load_master(path: str = MASTER_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_vote_matrix(df: pd.DataFrame, judges: tuple[str, ...] = JUDGES) -> np.ndarray:
    """(N, J) int8 array of class indices; -1 marks a judge with no vote on
    that item (parse failure at run_judge.py time). Every consumer of this
    matrix must treat -1 as "no observation" — the DS likelihood
    marginalises over it by omitting that judge's factor for that item,
    never by imputing a label (see dawid_skene.py's e_step)."""
    n = len(df)
    votes = np.full((n, len(judges)), -1, dtype=np.int8)
    for j, judge in enumerate(judges):
        subj = df[f"judge_{judge}_subjectivity"]
        pol = df[f"judge_{judge}_polarity"]
        derived = np.where(subj.isna(), np.nan, np.where(subj.to_numpy() == "objective", "objective", pol))
        idx = pd.Series(derived, index=df.index).map(LABEL_TO_IDX)
        votes[:, j] = idx.fillna(-1).astype(np.int8).to_numpy()
    return votes


def build_strata(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Integer stratum id per row over (register, text_type), plus the
    ordered list of (register, text_type) keys index-aligned to that id —
    i.e. keys[strata[i]] recovers row i's stratum."""
    keys = sorted(set(zip(df["register"], df["text_type"])))
    key_to_id = {k: i for i, k in enumerate(keys)}
    ids = np.fromiter(
        (key_to_id[(r, t)] for r, t in zip(df["register"], df["text_type"])),
        dtype=np.int32, count=len(df),
    )
    return ids, keys


def posterior_entropy(posteriors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(entropy_nat, entropy_norm); entropy_norm = entropy_nat / ln(N_CLASSES)
    so it's comparable across analyses regardless of class count."""
    p = np.clip(posteriors, 1e-300, 1.0)
    entropy_nat = -(p * np.log(p)).sum(axis=1)
    return entropy_nat, entropy_nat / np.log(N_CLASSES)


def argmax_labels(posteriors: np.ndarray) -> list[str]:
    return [LABELS[i] for i in np.argmax(posteriors, axis=1)]
