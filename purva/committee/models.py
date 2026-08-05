"""Judge committee model registry (PROTOCOL.md §4).

Revisions are pinned to "main" until the 1,000-sentence pilot completes, at
which point the resolved commit hashes are recorded here AND in PROTOCOL.md.

max_model_len: measured against judge_prompt_v1.txt with a rendered sentence
substituted in, prompts run ~700 tokens and outputs (MAX_TOKENS in
run_judge.py) are capped at 200 tokens. 1024 gives headroom over the ~900
actually needed while avoiding the KV-cache over-reservation that caused
PreemptionMode.RECOMPUTE thrashing at max_model_len=2048 (0.18 sentences/sec
on the Gemma pilot).
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
    # vLLM reported 10,032 GPU blocks and 156x max concurrency at
    # max_model_len=1024 — 32 badly underused available KV cache. 96 is the
    # new baseline for all judges; drop to 64 for any specific model that
    # OOMs at 96 (see per-entry overrides below).
    max_num_seqs: int = 96
    # Optional pointer to a well-established AWQ build of this model, used in
    # preference to the bitsandbytes path (AWQ kernels are substantially
    # faster than bitsandbytes on Turing/T4). Left None where no repo could
    # be confidently verified to exist — see --quant in run_judge.py for the
    # fallback behavior.
    awq_repo_id: str | None = None


REGISTRY: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1024,
        dtype="float16",
        # hugging-quants is the same org (HF + community, Meta-affiliated
        # quantization partners) that publishes the Qwen-AWQ-style official
        # AWQ builds; verified to exist on the Hub.
        awq_repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    ),
    "gemma-2-9b": ModelSpec(
        repo_id="google/gemma-2-9b-it",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1024,
        dtype="float16",
        awq_repo_id="hugging-quants/gemma-2-9b-it-AWQ-INT4",
    ),
    "qwen2.5-14b": ModelSpec(
        # Official AWQ variant used for T4 (16GB) fit; the unquantized
        # 14B model does not fit on a single T4. Already AWQ, so no separate
        # awq_repo_id — there is no non-AWQ path in this registry entry for
        # --quant bnb to fall back to.
        repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        revision="main",  # PIN AT PILOT
        quantization="awq",
        max_model_len=1024,
        dtype="float16",
    ),
    "mistral-nemo-12b": ModelSpec(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1024,
        dtype="float16",
        # casperhansen is a prolific, widely-used community AWQ quantizer
        # (775k+ downloads/month on this repo at time of writing).
        awq_repo_id="casperhansen/mistral-nemo-instruct-2407-awq",
    ),
    "aya-expanse-8b": ModelSpec(
        repo_id="CohereForAI/aya-expanse-8b",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1024,
        dtype="float16",
        # Community (non-Cohere) AWQ build; confirmed to target this exact
        # base model and vLLM-compatible, with substantial (~197k/month)
        # download volume, but less established than the hugging-quants/
        # casperhansen repos above — worth a spot-check at pilot time.
        awq_repo_id="Orion-zhen/aya-expanse-8b-AWQ",
    ),
    "indic": ModelSpec(
        # PROTOCOL.md §4 lists "Sarvam-1 or Airavata" for the Indic-specialist
        # slot. Sarvam-1 is a base (non-instruction-tuned) model and is not
        # reliable at following a strict JSON output contract, so Airavata
        # (instruction-tuned on Indic languages) is used instead. See
        # PROTOCOL.md CHANGELOG v1.3. No well-established AWQ build of
        # Airavata could be confirmed to exist, so awq_repo_id is left None
        # rather than guessed.
        repo_id="ai4bharat/Airavata",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1024,
        dtype="float16",
    ),
}
