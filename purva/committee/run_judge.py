from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .models import REGISTRY, ModelSpec

BATCH_SIZE = 64
# Shared by build_llm and build_sampling_params so both actually use the same
# value, and by main()'s run_config recording so what's recorded is what ran
# rather than a second hardcoded copy that could drift from it.
SEED = 42
# 200 was marginal: guided decoding stops as soon as the JSON object closes,
# so a higher cap costs nothing on well-behaved items and only helps items
# whose sentiment_target/domain values (often Devanagari, which tokenizes
# expensively) push length past the old budget.
MAX_TOKENS = 320
FLUSH_EVERY = 50
# Recorded in every .meta.json sidecar in place of which Kaggle account ran
# the job — hardware class is reproducibility-relevant, account identity
# isn't, and every run so far has been on this same free-tier shape.
HARDWARE_DESCRIPTOR = "Kaggle T4x2 (free tier)"

# Observed pilot failure mode: judges emit one valid JSON object, then keep
# generating junk (a duplicate object, "Please provide..." loops, trailing
# notes) until max_tokens — 99% parse failures and ~3x wasted compute. These
# stop sequences halt generation as soon as that junk starts.
STOP_SEQUENCES = ["}\n\n", "\n\n\n", " | ", "|{", "Note:", "Please provide"]

BASE_SCHEMA_KEYS = {
    "subjectivity",
    "polarity",
    "confidence",
    "domain",
    "narrative_voice",
    "sentiment_target",
}
RATIONALE_KEY = "rationale"
SUBJECTIVITY_VALUES = {"objective", "subjective"}
POLARITY_VALUES = {"positive", "negative", "neutral", "mixed"}
NARRATIVE_VOICE_VALUES = {"first_person", "third_person", "mixed"}

# Appended to the rendered prompt at runtime when --no-rationale is set. The
# checked-in prompts/*.txt files stay byte-identical (frozen per
# prompts/README.md) — this is a small addendum, not an edit to those files.
RATIONALE_OFF_ADDENDUM = (
    "\n\nFor this run, omit the \"rationale\" field entirely — do not "
    "include it in the JSON object at all."
)

FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def resolve_prompt_path(prompt_arg: str) -> Path:
    p = Path(prompt_arg)
    if p.exists():
        return p
    shorthand = Path("prompts") / f"judge_prompt_{prompt_arg.strip().lower()}.txt"
    if shorthand.exists():
        return shorthand
    raise SystemExit(f"prompt file not found: {prompt_arg} (also tried {shorthand})")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def render(template: str, sentence: str, expect_rationale: bool = True) -> str:
    text = template.replace("{{SENTENCE}}", sentence.replace("\n", " ").strip())
    if not expect_rationale:
        text += RATIONALE_OFF_ADDENDUM
    return text


def strip_fences(text: str) -> str:
    t = text.strip()
    m = FENCE_RE.match(t)
    return m.group(1).strip() if m else t


def validate(obj: dict, expect_rationale: bool = True) -> bool:
    expected_keys = BASE_SCHEMA_KEYS | ({RATIONALE_KEY} if expect_rationale else set())
    if not isinstance(obj, dict) or set(obj.keys()) != expected_keys:
        return False
    if obj.get("subjectivity") not in SUBJECTIVITY_VALUES:
        return False
    polarity = obj.get("polarity")
    if polarity is not None and polarity not in POLARITY_VALUES:
        return False
    if (obj["subjectivity"] == "objective") != (polarity is None):
        return False
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False
    if not (0.0 <= float(confidence) <= 1.0):
        return False
    if obj.get("narrative_voice") not in NARRATIVE_VOICE_VALUES:
        return False
    if not isinstance(obj.get("domain"), str) or not obj["domain"].strip():
        return False
    target = obj.get("sentiment_target")
    if target is not None and not isinstance(target, str):
        return False
    if expect_rationale and (not isinstance(obj.get("rationale"), str) or not obj["rationale"].strip()):
        return False
    return True


def extract_first_json(text: str) -> str | None:
    """Return the substring from the first '{' through its matching balanced
    '}', ignoring braces that appear inside string literals. Returns None if
    no balanced object is found (e.g. truncated output)."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def try_parse(raw_text: str, expect_rationale: bool = True) -> dict | None:
    candidate = extract_first_json(strip_fences(raw_text))
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if validate(obj, expect_rationale) else None


def resolve_quant(spec: ModelSpec, quant_mode: str) -> tuple[str, str]:
    """Return (repo_id, quantization) to actually load for --quant {auto,awq,bnb}."""
    if quant_mode == "awq":
        if spec.awq_repo_id is not None:
            return spec.awq_repo_id, "awq"
        if spec.quantization == "awq":
            return spec.repo_id, "awq"
        raise SystemExit("--quant awq requested but no AWQ repo is configured for this model")

    if quant_mode == "bnb":
        if spec.quantization == "bitsandbytes-4bit":
            return spec.repo_id, "bitsandbytes-4bit"
        raise SystemExit(
            "--quant bnb requested but this model has no bitsandbytes path configured "
            f"(registry quantization={spec.quantization})"
        )

    # auto
    if spec.awq_repo_id is not None:
        return spec.awq_repo_id, "awq"
    return spec.repo_id, spec.quantization


def build_llm(spec: ModelSpec, repo_id: str, quantization: str):
    import torch
    from vllm import LLM

    tensor_parallel_size = 2 if torch.cuda.device_count() >= 2 else 1

    print(
        f"[config] repo_id={repo_id} quantization={quantization} "
        f"max_model_len={spec.max_model_len} max_num_seqs={spec.max_num_seqs} "
        f"enable_prefix_caching={spec.enable_prefix_caching} "
        f"tensor_parallel_size={tensor_parallel_size}"
    )

    kwargs = dict(
        model=repo_id,
        revision=spec.revision,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,
        max_model_len=spec.max_model_len,
        max_num_seqs=spec.max_num_seqs,
        swap_space=2,  # GB; lets residual preemption swap to CPU rather than recompute
        dtype=spec.dtype,
        seed=SEED,
        # Every request shares an identical ~1000-1100-token prompt prefix
        # (the frozen judge prompt) with only the sentence differing — cache
        # it instead of recomputing per request. Per-model override: see
        # ModelSpec.enable_prefix_caching in models.py (off for gemma-2-9b).
        enable_prefix_caching=spec.enable_prefix_caching,
        guided_decoding_backend="xgrammar",
    )
    if quantization == "awq":
        kwargs["quantization"] = "awq"
    elif quantization == "bitsandbytes-4bit":
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"

    return LLM(**kwargs)


def get_preemption_count(llm) -> int | None:
    """Best-effort: vLLM's internal stats API differs across versions and
    isn't part of its public contract, so this returns None rather than
    raising if the expected attributes aren't found."""
    try:
        scheduler = llm.llm_engine.scheduler
        schedulers = scheduler if isinstance(scheduler, (list, tuple)) else [scheduler]
        total = 0
        found = False
        for s in schedulers:
            count = getattr(s, "num_cumulative_preemption", None)
            if count is not None:
                total += count
                found = True
        return total if found else None
    except Exception:
        return None


def build_guided_schema(expect_rationale: bool) -> dict:
    """JSON schema for xgrammar-guided decoding, expressed as two mutually
    exclusive object shapes (anyOf) rather than if/then — xgrammar's
    JSON-schema-to-grammar compiler supports anyOf/const/enum reliably but
    not conditional if/then, so this is "the null-iff-objective relationship
    expressed as far as the schema language allows": one branch requires
    subjectivity="objective" with polarity=null, the other requires
    subjectivity="subjective" with polarity in the real enum. Nothing in
    JSON Schema can force "polarity is exactly null when and only when
    subjectivity is exactly objective" more precisely than this two-branch
    split without if/then.

    Two more xgrammar-specific constraints (confirmed against real errors,
    not guessed): it rejects numeric range keywords (minimum/maximum) as an
    "advanced JSON schema feature" and falls back to the buggier "outlines"
    backend, so confidence is left as a bare "number" here — the [0,1] range
    is still enforced by validate() after parsing, just not by the grammar
    itself. And list-form union types (`"type": ["string", "null"]`) crash
    outlines' schema-to-regex compiler, so nullable fields use `anyOf`
    instead, which both backends handle.
    """
    shared_props = {
        "confidence": {"type": "number"},
        "domain": {"type": "string"},
        "narrative_voice": {"enum": sorted(NARRATIVE_VOICE_VALUES)},
        "sentiment_target": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    required_keys = ["subjectivity", "polarity", "confidence", "domain", "narrative_voice", "sentiment_target"]
    if expect_rationale:
        shared_props[RATIONALE_KEY] = {"type": "string"}
        required_keys.append(RATIONALE_KEY)

    def branch(subjectivity: str, polarity_schema: dict) -> dict:
        return {
            "type": "object",
            "properties": {
                "subjectivity": {"const": subjectivity},
                "polarity": polarity_schema,
                **shared_props,
            },
            "required": required_keys,
            "additionalProperties": False,
        }

    return {
        "anyOf": [
            branch("objective", {"type": "null"}),
            branch("subjective", {"enum": sorted(POLARITY_VALUES)}),
        ]
    }


def build_sampling_params(guided: bool, expect_rationale: bool):
    from vllm import SamplingParams

    kwargs = dict(
        temperature=0.0,
        seed=SEED,
        max_tokens=MAX_TOKENS,
        stop=STOP_SEQUENCES,
        # True, not False: with False, vLLM strips the entire matched stop
        # string from the output. For the "}\n\n" stop sequence that deletes
        # the JSON object's own closing brace, leaving unbalanced output that
        # extract_first_json correctly (but unhelpfully) rejects. Keeping the
        # stop string retains the brace; extract_first_json already discards
        # anything after it via balanced-brace matching.
        include_stop_str_in_output=True,
    )

    if guided:
        from vllm.sampling_params import GuidedDecodingParams

        kwargs["guided_decoding"] = GuidedDecodingParams(
            json=build_guided_schema(expect_rationale), backend="xgrammar"
        )

    return SamplingParams(**kwargs)


def process_chunk(
    llm, sampling_params, template: str, chunk: list[dict], expect_rationale: bool = True
) -> list[tuple[dict | None, str, int]]:
    """Returns (parsed_or_None, raw_text, prompt_token_count) per item.
    prompt_token_count is carried through so failed items can report whether
    the prompt itself is crowding max_model_len, not just guessed at."""
    prompts = [render(template, row["cleaned_text"], expect_rationale) for row in chunk]
    outputs = llm.generate(prompts, sampling_params)
    raws = [o.outputs[0].text for o in outputs]
    prompt_tokens = [len(o.prompt_token_ids) for o in outputs]

    results: list[tuple[dict | None, str, int] | None] = [None] * len(chunk)
    retry_idx = []
    for i, raw in enumerate(raws):
        parsed = try_parse(raw, expect_rationale)
        if parsed is not None:
            results[i] = (parsed, raw, prompt_tokens[i])
        else:
            retry_idx.append(i)

    if retry_idx:
        retry_prompts = [prompts[i] for i in retry_idx]
        retry_outputs = llm.generate(retry_prompts, sampling_params)
        for i, out in zip(retry_idx, retry_outputs):
            raw2 = out.outputs[0].text
            results[i] = (try_parse(raw2, expect_rationale), raw2, len(out.prompt_token_ids))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--input", default="data/corpus_lid.jsonl")
    ap.add_argument("--output-dir", default="data/committee/")
    ap.add_argument("--prompt", default="prompts/judge_prompt_v1.txt")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quant", choices=["auto", "awq", "bnb"], default="auto")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="benchmark on the first N pilot sentences and exit; writes no shard")
    ap.add_argument("--guided", action=argparse.BooleanOptionalAction, default=True, help="constrain generation to the judge JSON schema via xgrammar guided decoding")
    ap.add_argument("--rationale", action=argparse.BooleanOptionalAction, default=True, help="include the rationale field in the schema/prompt (roughly doubles output length)")
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    prompt_path = resolve_prompt_path(args.prompt)
    template = prompt_path.read_text(encoding="utf-8")
    prompt_stem = prompt_path.stem
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    # --bench is fully self-contained (reads only data/pilot_set.jsonl) and
    # must not touch --input at all, so it's handled before the unconditional
    # rows-load below — bench mode should never require the full corpus file
    # to be present.
    if args.bench:
        bench_path = Path("data/pilot_set.jsonl")
        bench_rows = load_rows(bench_path)[: args.bench]
        print(f"bench mode: {len(bench_rows)} sentences from {bench_path}")

        repo_id, quantization = resolve_quant(spec, args.quant)
        llm = build_llm(spec, repo_id, quantization)
        sampling_params = build_sampling_params(args.guided, args.rationale)

        start = time.time()
        results = process_chunk(llm, sampling_params, template, bench_rows, args.rationale)
        elapsed = time.time() - start

        processed = len(bench_rows)
        rate = processed / elapsed if elapsed > 0 else 0.0
        parse_failures = sum(1 for parsed, _, _ in results if parsed is None)
        preemptions = get_preemption_count(llm)

        # Bench mode writes no shard, but failures still need to be
        # diagnosable without a full run — persist raw_response (plus
        # prompt_tokens, to check whether the prompt itself is crowding
        # max_model_len) for each failed row so the actual model output can
        # be inspected.
        failures_path = Path("data/committee") / f"bench_failures_{args.model}.jsonl"
        if parse_failures:
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("w", encoding="utf-8") as fh:
                for row, (parsed, raw, prompt_tokens) in zip(bench_rows, results):
                    if parsed is None:
                        fh.write(json.dumps({
                            "id": row["id"],
                            "cleaned_text": row["cleaned_text"],
                            "raw_response": raw,
                            "prompt_tokens": prompt_tokens,
                        }, ensure_ascii=False) + "\n")

        print("\n=== Bench summary ===")
        print(f"sentences: {processed}")
        print(f"elapsed: {elapsed:.2f}s")
        print(f"throughput: {rate:.3f} sentences/sec")
        if processed:
            print(f"parse failures: {parse_failures} ({parse_failures / processed * 100:.2f}%)")
        if parse_failures:
            print(f"failures written to {failures_path}")
        print(f"preemptions: {preemptions if preemptions is not None else 'n/a (not exposed by this vLLM version)'}")
        return

    input_path = Path("data/pilot_set.jsonl") if args.pilot else Path(args.input)
    rows = load_rows(input_path)
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        print(f"model={args.model} ({spec.repo_id})")
        print(f"prompt={prompt_path}")
        print(f"input={input_path} ({len(rows)} rows available)\n")
        for row in rows[:3]:
            print("=" * 80)
            print(f"id={row['id']}")
            print(render(template, row["cleaned_text"], args.rationale))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}__{prompt_stem}.jsonl"

    done_ids = load_done_ids(output_path)
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"{len(rows)} total, {len(done_ids)} already done, {len(todo)} to process")

    if not todo:
        return

    repo_id, quantization = resolve_quant(spec, args.quant)
    llm = build_llm(spec, repo_id, quantization)
    sampling_params = build_sampling_params(args.guided, args.rationale)

    # Recorded into every output row below. Chunks are processed weeks apart
    # across separate Kaggle sessions — this is what lets a later audit
    # confirm the conditions were actually identical rather than trusting
    # they were (see merge_shards.py's check_config_consistency).
    run_config = {
        "repo_id": repo_id,
        "revision": spec.revision,
        "quantization": quantization,
        "prompt_file": str(prompt_path),
        "seed": SEED,
        "max_model_len": spec.max_model_len,
    }

    processed = 0
    parse_failures = 0
    start = time.time()
    run_date = datetime.now(timezone.utc).date().isoformat()

    with output_path.open("a", encoding="utf-8") as fh:
        for chunk_start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[chunk_start : chunk_start + BATCH_SIZE]
            results = process_chunk(llm, sampling_params, template, chunk, args.rationale)

            for row, (parsed, raw, prompt_tokens) in zip(chunk, results):
                out_row = {"id": row["id"], "model": args.model, "prompt_variant": prompt_stem, "run_config": run_config}
                if parsed is not None:
                    out_row.update(parsed)
                else:
                    out_row["parse_failed"] = True
                    out_row["raw_response"] = raw
                    out_row["prompt_tokens"] = prompt_tokens
                    parse_failures += 1
                fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                processed += 1
                if processed % FLUSH_EVERY == 0:
                    fh.flush()

            print(f"[{min(chunk_start + BATCH_SIZE, len(todo))}/{len(todo)}] processed")
        fh.flush()

    elapsed = time.time() - start
    rate = processed / elapsed if elapsed > 0 else 0.0

    print("\n=== Final summary ===")
    print(f"processed: {processed}")
    if processed:
        print(f"parse failures: {parse_failures} ({parse_failures / processed * 100:.2f}%)")
    print(f"throughput: {rate:.2f} sentences/sec")
    print(f"elapsed: {elapsed:.1f}s")

    # One small sidecar per shard — deliberately not per-row (see run_config
    # above for the per-row fields). Chunks are processed weeks apart across
    # separate Kaggle sessions, so this is what lets a later audit confirm
    # exactly which conditions produced this shard without re-parsing the
    # kernel log.
    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "repo_id": repo_id,
        "revision": spec.revision,
        "quantization": quantization,
        "max_model_len": spec.max_model_len,
        "max_num_seqs": spec.max_num_seqs,
        "enable_prefix_caching": spec.enable_prefix_caching,
        "guided_decoding": args.guided,
        "seed": SEED,
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": prompt_sha256,
        "date": run_date,
        "hardware": HARDWARE_DESCRIPTOR,
        "total_generation_seconds": round(elapsed, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
