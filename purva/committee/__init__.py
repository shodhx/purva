"""
PURVA Layer 2: Multi-Model Committee Consensus & Agreement Engine
=================================================================
A* Conference Standard package for running multi-model ensemble consensus,
Expectation-Maximization Dawid-Skene aggregation, and Fleiss' Kappa evaluations.
"""

from .consensus import (
    CommitteeOrchestrator,
    EnsembleStateManager,
    BaseJudge,
    OllamaJudge,
    VLLMJudge,
    LocalIndicJudge,
    JudgeOutput,
    dawid_skene_aggregation,
    calculate_fleiss_kappa,
    calculate_shannon_entropy,
)

__all__ = [
    "CommitteeOrchestrator",
    "EnsembleStateManager",
    "BaseJudge",
    "OllamaJudge",
    "VLLMJudge",
    "LocalIndicJudge",
    "JudgeOutput",
    "dawid_skene_aggregation",
    "calculate_fleiss_kappa",
    "calculate_shannon_entropy",
]
