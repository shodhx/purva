"""Shared constants/loaders for the Hindi cross-lingual validation run."""

from __future__ import annotations

import json
from pathlib import Path

# Judge short names and their repo stems, identical to purva/committee/build_master.py's
# JUDGE_SHORT_NAMES (the 'indic' registry entry never produced a shard and is not part
# of the five-judge committee that ran).
JUDGE_SHORT_NAMES = {
    "aya-expanse-8b": "aya",
    "gemma-2-9b": "gemma",
    "llama-3.1-8b": "llama",
    "mistral-nemo-12b": "mistral",
    "qwen2.5-14b": "qwen",
}

# Four-class aggregation label space (PROTOCOL.md CHANGELOG v1.7): canonical
# order is also the majority-vote tie-break priority order.
LABELS = ("objective", "positive", "negative", "neutral")
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

# Gold labels present in the Hindi validation set (their 3-class polarity scheme
# mapped onto ours; no "objective" gold exists — see make_hindi_validation_set.py).
GOLD_LABELS = ("negative", "neutral", "positive")

# Normalised-entropy threshold separating "low" from "high" entropy items,
# taken from the main corpus's routing entropy distribution
# (data/aggregation_report.md §6: 79.5% of items below 0.3).
ENTROPY_SPLIT = 0.3


def load_validation_rows(path: Path) -> tuple[list[dict], dict]:
    """Returns (data rows, header meta). The first line of the validation set
    is the dataset-provenance header written by make_hindi_validation_set.py;
    every subsequent line is a data row."""
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    header = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:]]
    assert all(r.get("gold_label") in GOLD_LABELS for r in rows), "unexpected gold label in validation set"
    return rows, header
