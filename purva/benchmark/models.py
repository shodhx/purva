"""Benchmark fine-tuning model registry (PROTOCOL.md §8).

Revisions resolved via HfApi().model_info(repo_id).sha on 2026-09-19 and
pinned here so every training run is reproducible from a commit SHA, not a
moving "main" pointer (same discipline as purva/committee/models.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkModelSpec:
    repo_id: str
    revision: str
    # Which text/label source this model trains on. "corpus" = the
    # Bhojpuri train split (data/splits.json) with Dawid-Skene consensus
    # labels from data/purva_aggregated.jsonl. "hindi_baseline" = the Hindi
    # cross-lingual validation set (data/hindi_validation_set.jsonl) with
    # its own third-party gold labels, and NO Bhojpuri data at all — the
    # ablation testing whether high-resource-neighbour transfer alone would
    # have sufficed.
    training_source: str


REGISTRY: dict[str, BenchmarkModelSpec] = {
    "muril": BenchmarkModelSpec(
        repo_id="google/muril-base-cased",
        revision="afd9f36c7923d54e97903922ff1b260d091d202f",
        training_source="corpus",
    ),
    "indicbert": BenchmarkModelSpec(
        repo_id="ai4bharat/IndicBERTv2-MLM-only",
        revision="8598f13fe52443bc3fc054fcd665944560145b5c",
        training_source="corpus",
    ),
    "xlmr": BenchmarkModelSpec(
        repo_id="xlm-roberta-base",
        revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        training_source="corpus",
    ),
    "muril-hindi-baseline": BenchmarkModelSpec(
        repo_id="google/muril-base-cased",
        # Same base checkpoint as "muril" above (same repo, same revision) —
        # this entry differs only in training_source, not in which weights
        # it starts from.
        revision="afd9f36c7923d54e97903922ff1b260d091d202f",
        training_source="hindi_baseline",
    ),
}
