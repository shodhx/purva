"""Judge committee model registry (PROTOCOL.md §4).

Revisions are pinned to "main" until the 1,000-sentence pilot completes, at
which point the resolved commit hashes are recorded here AND in PROTOCOL.md.

max_model_len: the original ~700-token prompt estimate was wrong. Measured
directly via prompt_token_ids on a real Kaggle run (Llama's tokenizer):
judge_prompt_v1.txt with a rendered sentence substituted in actually runs
~975-1043 tokens, not ~700 — at max_model_len=1024 the prompt alone was
sometimes exceeding the budget, leaving zero-to-negative room for output
(confirmed: one failure had prompt_tokens=1043 and an empty completion,
several others had <50 tokens left and were truncated mid-object regardless
of MAX_TOKENS in run_judge.py). 1536 gives real headroom over the ~1000-1100
tokens actually needed (with margin for rationale-on runs and other
tokenizers) while staying well clear of 2048, which caused the KV-cache
over-reservation and PreemptionMode.RECOMPUTE thrashing (0.18 sentences/sec)
seen on the Gemma pilot.
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
    # Commit SHA for awq_repo_id — separate from `revision` because the two
    # repos are different HF repos with independent history; resolve_quant()
    # in run_judge.py returns whichever of the two actually applies to the
    # repo_id it resolved. Every judge actually run so far used the AWQ path
    # exclusively (see RUNS.md), so `revision` above was never exercised and
    # is left at "main" — run_judge.py refuses to run on an unpinned "main"
    # revision for whichever path IS resolved, so that fallback path staying
    # unpinned is inert, not a live gap.
    awq_revision: str | None = None
    # Per-model override: True everywhere except gemma-2-9b, where vLLM's
    # prefix-caching attention kernel (forward_prefix / context_attention_fwd,
    # Triton) crashes on T4/Turing with `OutOfResources: out of resource:
    # shared memory, Required: 73728, Hardware limit: 65536` — confirmed
    # against a real Kaggle run. Llama ran clean with prefix caching on the
    # same hardware, so this looks specific to Gemma-2's alternating
    # sliding-window/global attention pattern, not a blanket T4 limitation.
    enable_prefix_caching: bool = True


REGISTRY: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1536,
        dtype="float16",
        # hugging-quants is the same org (HF + community, Meta-affiliated
        # quantization partners) that publishes the Qwen-AWQ-style official
        # AWQ builds; verified to exist on the Hub.
        awq_repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        # Resolved post-hoc via HfApi().model_info().sha on 2026-08-09 — all
        # nine chunks actually ran against "main" (see PROTOCOL.md CHANGELOG
        # v1.6); repo lastModified 2024-08-07, well before every run in
        # RUNS.md, so this SHA is the one that was actually used.
        awq_revision="db1f81ad4b8c7e39777509fac66c652eb0a52f91",
    ),
    "gemma-2-9b": ModelSpec(
        repo_id="google/gemma-2-9b-it",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1536,
        dtype="float16",
        awq_repo_id="hugging-quants/gemma-2-9b-it-AWQ-INT4",
        # Resolved post-hoc via HfApi().model_info().sha on 2026-08-09 — all
        # nine chunks actually ran against "main" (see PROTOCOL.md CHANGELOG
        # v1.6); repo lastModified 2024-10-17, well before every run in
        # RUNS.md, so this SHA is the one that was actually used.
        awq_revision="6e62725da8e92309167814dad7aacc0ed8cb2484",
        enable_prefix_caching=False,
    ),
    "qwen2.5-14b": ModelSpec(
        # Official AWQ variant used for T4 (16GB) fit; the unquantized
        # 14B model does not fit on a single T4. Already AWQ, so no separate
        # awq_repo_id — there is no non-AWQ path in this registry entry for
        # --quant bnb to fall back to.
        repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
        # Resolved post-hoc via HfApi().model_info().sha on 2026-08-09 — all
        # nine chunks actually ran against "main" (see PROTOCOL.md CHANGELOG
        # v1.6); repo lastModified 2024-10-09, well before every run in
        # RUNS.md, so this SHA is the one that was actually used. Pinned
        # directly on `revision` (not awq_revision) because repo_id here is
        # already the AWQ build — there is no separate base repo for this one.
        revision="539535859b135b0244c91f3e59816150c8056698",
        quantization="awq",
        max_model_len=1536,
        dtype="float16",
    ),
    "mistral-nemo-12b": ModelSpec(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1536,
        dtype="float16",
        # casperhansen is a prolific, widely-used community AWQ quantizer
        # (775k+ downloads/month on this repo at time of writing).
        awq_repo_id="casperhansen/mistral-nemo-instruct-2407-awq",
        # Resolved post-hoc via HfApi().model_info().sha on 2026-08-09 — all
        # nine chunks actually ran against "main" (see PROTOCOL.md CHANGELOG
        # v1.6); repo lastModified 2024-09-27, well before every run in
        # RUNS.md, so this SHA is the one that was actually used.
        awq_revision="c83b6438e13051ad1c0f5683635705ee83bb8772",
    ),
    "aya-expanse-8b": ModelSpec(
        repo_id="CohereForAI/aya-expanse-8b",
        revision="main",  # PIN AT PILOT
        quantization="bitsandbytes-4bit",
        max_model_len=1536,
        dtype="float16",
        # Community (non-Cohere) AWQ build; confirmed to target this exact
        # base model and vLLM-compatible, with substantial (~197k/month)
        # download volume, but less established than the hugging-quants/
        # casperhansen repos above — worth a spot-check at pilot time.
        awq_repo_id="Orion-zhen/aya-expanse-8b-AWQ",
        # Resolved post-hoc via HfApi().model_info().sha on 2026-08-09 — all
        # nine chunks actually ran against "main" (see PROTOCOL.md CHANGELOG
        # v1.6); repo lastModified 2024-10-26, well before every run in
        # RUNS.md, so this SHA is the one that was actually used.
        awq_revision="ff52d61b71c613180581c3a4c6b3b3f636ce79e5",
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
        max_model_len=1536,
        dtype="float16",
    ),
}
