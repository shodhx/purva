"""Judge committee model registry (PROTOCOL.md §4).

Revisions are pinned to "main" until the 1,000-sentence pilot completes, at
which point the resolved commit hashes are recorded here AND in PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    quantization: str
    max_model_len: int
    dtype: str


REGISTRY: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=2048,
        dtype="float16",
    ),
    "gemma-2-9b": ModelSpec(
        repo_id="google/gemma-2-9b-it",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=2048,
        dtype="float16",
    ),
    "qwen2.5-14b": ModelSpec(
        # Official AWQ variant used for T4 (16GB) fit; the unquantized
        # 14B model does not fit on a single T4.
        repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        revision="main",  # PIN AT PILOT
        quantization="awq",
        max_model_len=2048,
        dtype="float16",
    ),
    "mistral-nemo-12b": ModelSpec(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=2048,
        dtype="float16",
    ),
    "aya-expanse-8b": ModelSpec(
        repo_id="CohereForAI/aya-expanse-8b",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=2048,
        dtype="float16",
    ),
    "indic": ModelSpec(
        # PROTOCOL.md §4 lists "Sarvam-1 or Airavata" for the Indic-specialist
        # slot. Sarvam-1 is a base (non-instruction-tuned) model and is not
        # reliable at following a strict JSON output contract, so Airavata
        # (instruction-tuned on Indic languages) is used instead. See
        # PROTOCOL.md CHANGELOG v1.3.
        repo_id="ai4bharat/Airavata",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=2048,
        dtype="float16",
    ),
}
