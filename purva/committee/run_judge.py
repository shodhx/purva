from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .models import REGISTRY, ModelSpec

BATCH_SIZE = 64
MAX_TOKENS = 200
FLUSH_EVERY = 50

# Observed pilot failure mode: judges emit one valid JSON object, then keep
# generating junk (a duplicate object, "Please provide..." loops, trailing
# notes) until max_tokens — 99% parse failures and ~3x wasted compute. These
# stop sequences halt generation as soon as that junk starts.
STOP_SEQUENCES = ["}\n\n", "\n\n\n", " | ", "|{", "Note:", "Please provide"]

SCHEMA_KEYS = {
    "subjectivity",
    "polarity",
    "confidence",
    "domain",
    "narrative_voice",
    "sentiment_target",
    "rationale",
}
SUBJECTIVITY_VALUES = {"objective", "subjective"}
POLARITY_VALUES = {"positive", "negative", "neutral", "mixed"}
NARRATIVE_VOICE_VALUES = {"first_person", "third_person", "mixed"}

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


def render(template: str, sentence: str) -> str:
    return template.replace("{{SENTENCE}}", sentence.replace("\n", " ").strip())


def strip_fences(text: str) -> str:
    t = text.strip()
    m = FENCE_RE.match(t)
    return m.group(1).strip() if m else t


def validate(obj: dict) -> bool:
    if not isinstance(obj, dict) or set(obj.keys()) != SCHEMA_KEYS:
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
    if not isinstance(obj.get("rationale"), str) or not obj["rationale"].strip():
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


def try_parse(raw_text: str) -> dict | None:
    candidate = extract_first_json(strip_fences(raw_text))
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if validate(obj) else None


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
        seed=42,
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


def build_sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=MAX_TOKENS,
        stop=STOP_SEQUENCES,
        include_stop_str_in_output=False,
    )


def process_chunk(llm, sampling_params, template: str, chunk: list[dict]) -> list[tuple[dict | None, str]]:
    prompts = [render(template, row["cleaned_text"]) for row in chunk]
    outputs = llm.generate(prompts, sampling_params)
    raws = [o.outputs[0].text for o in outputs]

    results: list[tuple[dict | None, str] | None] = [None] * len(chunk)
    retry_idx = []
    for i, raw in enumerate(raws):
        parsed = try_parse(raw)
        if parsed is not None:
            results[i] = (parsed, raw)
        else:
            retry_idx.append(i)

    if retry_idx:
        retry_prompts = [prompts[i] for i in retry_idx]
        retry_outputs = llm.generate(retry_prompts, sampling_params)
        for i, out in zip(retry_idx, retry_outputs):
            raw2 = out.outputs[0].text
            results[i] = (try_parse(raw2), raw2)

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
    args = ap.parse_args()

    spec = REGISTRY[args.model]
    prompt_path = resolve_prompt_path(args.prompt)
    template = prompt_path.read_text(encoding="utf-8")
    prompt_stem = prompt_path.stem

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
            print(render(template, row["cleaned_text"]))
        return

    if args.bench:
        bench_path = Path("data/pilot_set.jsonl")
        bench_rows = load_rows(bench_path)[: args.bench]
        print(f"bench mode: {len(bench_rows)} sentences from {bench_path}")

        repo_id, quantization = resolve_quant(spec, args.quant)
        llm = build_llm(spec, repo_id, quantization)
        sampling_params = build_sampling_params()

        start = time.time()
        results = process_chunk(llm, sampling_params, template, bench_rows)
        elapsed = time.time() - start

        processed = len(bench_rows)
        rate = processed / elapsed if elapsed > 0 else 0.0
        parse_failures = sum(1 for parsed, _ in results if parsed is None)
        preemptions = get_preemption_count(llm)

        print("\n=== Bench summary ===")
        print(f"sentences: {processed}")
        print(f"elapsed: {elapsed:.2f}s")
        print(f"throughput: {rate:.3f} sentences/sec")
        if processed:
            print(f"parse failures: {parse_failures} ({parse_failures / processed * 100:.2f}%)")
        print(f"preemptions: {preemptions if preemptions is not None else 'n/a (not exposed by this vLLM version)'}")
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
    sampling_params = build_sampling_params()

    processed = 0
    parse_failures = 0
    start = time.time()

    with output_path.open("a", encoding="utf-8") as fh:
        for chunk_start in range(0, len(todo), BATCH_SIZE):
            chunk = todo[chunk_start : chunk_start + BATCH_SIZE]
            results = process_chunk(llm, sampling_params, template, chunk)

            for row, (parsed, raw) in zip(chunk, results):
                out_row = {"id": row["id"], "model": args.model, "prompt_variant": prompt_stem}
                if parsed is not None:
                    out_row.update(parsed)
                else:
                    out_row["parse_failed"] = True
                    out_row["raw_response"] = raw
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


if __name__ == "__main__":
    main()
