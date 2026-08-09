"""Join the corpus and all five committee judges' merged output into a
single analysis-ready file — one row per sentence, judges keyed by short
name, everything else passed through unmodified — and emit a dataset-level
manifest describing how it was produced.

Pure join: no majority votes, consensus labels, or agreement scores are
computed here. That is Phase 4 (aggregation)'s job, and it needs the raw
per-judge votes untouched — including the null slots where a judge's
output failed to parse — not a version already collapsed by this step.

The manifest (data/purva_master.meta.json) is regenerated every time this
module runs, including standalone via --manifest-only against outputs that
already exist — it's cheap metadata about the (guarded, no-overwrite)
master files, not itself a guarded artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# The corpus fields carried through as-is. lid_label/lid_confidence (legacy,
# always null in data/corpus_lid.jsonl) are deliberately excluded in favor of
# the lid_model_*/lid_verdict fields that actually carry the LID result.
CORPUS_FIELDS = (
    "id", "raw_text", "cleaned_text", "source_url", "source_name",
    "scrape_timestamp", "category", "register", "text_type", "script",
    "license_class", "lid_model_label", "lid_model_confidence", "lid_verdict",
)

# The 7 fields a judge emits per sentence (see prompts/judge_prompt_v1.txt
# and run_judge.py); a parse-failed row has none of these, hence None below
# rather than a partial dict.
JUDGE_FIELDS = (
    "subjectivity", "polarity", "confidence", "domain",
    "narrative_voice", "sentiment_target", "rationale",
)

# data/committee/merged/<stem>.jsonl -> short key used in the "judges"
# object and as the Parquet column prefix (judge_<short>_<field>).
JUDGE_SHORT_NAMES = {
    "aya-expanse-8b": "aya",
    "gemma-2-9b": "gemma",
    "llama-3.1-8b": "llama",
    "mistral-nemo-12b": "mistral",
    "qwen2.5-14b": "qwen",
}

# Snapshot invariant for this corpus (RUNS.md / merge_shards.py both confirm
# full coverage at this size) — a hard assert, not a soft expectation, so a
# silently truncated or duplicated input is caught immediately.
EXPECTED_ROWS = 90_207

N_CHUNKS = 9

# Sidecar (.meta.json) keys that identify a pinned model — asserted identical
# across all 9 chunks for a given judge; a mismatch means two different
# chunks of the "same" judge were actually produced by different models.
JUDGE_CONFIG_KEYS = ("repo_id", "revision", "quantization")

# Decoding keys asserted identical across all 45 (judge, chunk) sidecars.
# enable_prefix_caching is handled separately since it's identical *within*
# a judge but intentionally differs *between* judges (the Gemma exception).
SHARED_DECODING_KEYS = ("seed", "max_model_len", "max_num_seqs", "guided_decoding")

# GlotLID v3 (cis-lmu/glotlid) — see purva/lid2/glotlid_runner.py, which pins
# this exact file (not the "model.bin" alias, which can be repointed).
GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model_v3.bin"

# Matches run_judge.py's HARDWARE_DESCRIPTOR — every run recorded in RUNS.md
# used this same free-tier shape; no Kaggle account identifier belongs here.
HARDWARE_DESCRIPTOR = "Kaggle T4x2 (free tier)"

# Label spaces defined by prompts/judge_prompt_v1.txt (judge outputs) and
# PROTOCOL.md §2 (lid_verdict's VERDICT_MAP + "other" fallback). register,
# text_type, script, and license_class have no such spec anywhere in the
# repo (checked purva/**/*.py and all docs) — only their observed values are
# reported for those, with no "unexpected" comparison to make.
EXPECTED_CATEGORICAL_VALUES = {
    "subjectivity": {"objective", "subjective"},
    "polarity": {"positive", "negative", "neutral", "mixed", None},
    "narrative_voice": {"first_person", "third_person", "mixed"},
    "lid_verdict": {"bhojpuri", "hindi", "maithili", "other"},
}

CATEGORICAL_FIELDS = (
    "subjectivity", "polarity", "narrative_voice", "lid_verdict",
    "register", "text_type", "script", "license_class",
)
_JUDGE_CATEGORICAL = ("subjectivity", "polarity", "narrative_voice")
_CORPUS_CATEGORICAL = ("lid_verdict", "register", "text_type", "script", "license_class")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_chunk_membership(chunks_dir: Path) -> dict[str, int]:
    id_to_chunk: dict[str, int] = {}
    chunk_files = sorted(chunks_dir.glob("chunk_*.jsonl"))
    if not chunk_files:
        sys.exit(f"no chunk_*.jsonl files found under {chunks_dir} — run make_chunks.py first")
    for f in chunk_files:
        chunk_n = int(f.stem.split("_")[1])
        for row in load_jsonl(f):
            id_to_chunk[row["id"]] = chunk_n
    return id_to_chunk


def load_judge_votes(path: Path) -> dict[str, dict | None]:
    """{id: {7 fields}} for a clean parse, {id: None} for parse_failed."""
    votes: dict[str, dict | None] = {}
    for row in load_jsonl(path):
        rid = row["id"]
        votes[rid] = None if row.get("parse_failed") else {k: row.get(k) for k in JUDGE_FIELDS}
    return votes


def build_and_verify_master(args) -> list[dict]:
    """The join, plus the hard-fail verification pass. Returns master_rows."""
    corpus_path = Path(args.corpus)
    corpus_rows = load_jsonl(corpus_path)
    print(f"corpus: {len(corpus_rows)} rows from {corpus_path}")

    corpus_ids = [r["id"] for r in corpus_rows]
    if len(corpus_ids) != len(set(corpus_ids)):
        dupes = [i for i, c in Counter(corpus_ids).items() if c > 1]
        sys.exit(f"corpus has {len(dupes)} duplicate id(s), e.g. {dupes[:5]}")
    corpus_id_set = set(corpus_ids)

    id_to_chunk = load_chunk_membership(Path(args.chunks_dir))
    missing_chunk = [rid for rid in corpus_ids if rid not in id_to_chunk]
    if missing_chunk:
        sys.exit(f"{len(missing_chunk)} corpus id(s) have no chunk assignment, e.g. {missing_chunk[:5]}")

    merged_dir = Path(args.merged_dir)
    judge_votes: dict[str, dict[str, dict | None]] = {}
    for stem, short in JUDGE_SHORT_NAMES.items():
        path = merged_dir / f"{stem}.jsonl"
        if not path.exists():
            sys.exit(f"missing merged judge file: {path}")
        votes = load_judge_votes(path)
        extra = set(votes) - corpus_id_set
        if extra:
            sys.exit(f"judge {short} has {len(extra)} id(s) not in the corpus, e.g. {sorted(extra)[:5]}")
        judge_votes[short] = votes
        print(f"judge {short} ({stem}): {len(votes)} rows")

    master_rows: list[dict] = []
    n_judges_counter: Counter[int] = Counter()
    for row in corpus_rows:
        rid = row["id"]
        judges_obj: dict[str, dict | None] = {}
        n = 0
        for short in JUDGE_SHORT_NAMES.values():
            v = judge_votes[short].get(rid)
            judges_obj[short] = v
            if v is not None:
                n += 1
        n_judges_counter[n] += 1
        out_row = {k: row.get(k) for k in CORPUS_FIELDS}
        out_row["chunk"] = id_to_chunk[rid]
        out_row["judges"] = judges_obj
        out_row["n_judges"] = n
        master_rows.append(out_row)

    # --- verification (hard-fail on any mismatch) ---
    print("\n=== verification ===")
    out_ids = [r["id"] for r in master_rows]
    assert len(out_ids) == len(set(out_ids)), "duplicate ids in output"
    assert set(out_ids) == corpus_id_set, "output id set does not match corpus id set"
    assert len(master_rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} rows, got {len(master_rows)}"
    print(f"output rows: {len(master_rows)} (every corpus id present exactly once, no foreign ids)")

    print("n_judges distribution:")
    for n in sorted(n_judges_counter, reverse=True):
        print(f"  {n}: {n_judges_counter[n]}")

    for short, votes in judge_votes.items():
        valid_in_file = sum(1 for v in votes.values() if v is not None)
        valid_in_master = sum(1 for r in master_rows if r["judges"][short] is not None)
        assert valid_in_file == valid_in_master, (
            f"judge {short}: merged file has {valid_in_file} valid rows but master has {valid_in_master}"
        )
        print(f"judge {short}: {valid_in_master} valid, {len(master_rows) - valid_in_master} null")

    print("all verification checks passed\n")
    return master_rows


def write_outputs(master_rows: list[dict], out_jsonl: Path, out_parquet: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in master_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl} ({out_jsonl.stat().st_size / 1e6:.1f} MB)")

    flat_rows = []
    for row in master_rows:
        flat = {k: row[k] for k in CORPUS_FIELDS}
        flat["chunk"] = row["chunk"]
        flat["n_judges"] = row["n_judges"]
        for short in JUDGE_SHORT_NAMES.values():
            v = row["judges"][short]
            for field in JUDGE_FIELDS:
                flat[f"judge_{short}_{field}"] = v[field] if v is not None else None
        flat_rows.append(flat)

    pd.DataFrame(flat_rows).to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")


# --------------------------------------------------------------------------
# Manifest generation
# --------------------------------------------------------------------------

def load_all_sidecars(committee_dir: Path) -> list[tuple[str, int, dict]]:
    """[(judge_short, chunk_n, meta_dict), ...] for all 45 (judge, chunk) sidecars."""
    out = []
    for stem, short in JUDGE_SHORT_NAMES.items():
        for chunk_n in range(1, N_CHUNKS + 1):
            meta_path = committee_dir / f"chunk_{chunk_n:02d}" / f"{stem}__judge_prompt_v1.meta.json"
            if not meta_path.exists():
                sys.exit(f"missing sidecar: {meta_path}")
            out.append((short, chunk_n, json.loads(meta_path.read_text(encoding="utf-8"))))
    return out


def build_roster_decoding_prompt(sidecars: list[tuple[str, int, dict]]) -> tuple[dict, dict, dict]:
    by_judge: dict[str, list[dict]] = defaultdict(list)
    for short, _chunk_n, meta in sidecars:
        by_judge[short].append(meta)

    roster = {}
    for short, metas in by_judge.items():
        for key in JUDGE_CONFIG_KEYS:
            values = {m[key] for m in metas}
            assert len(values) == 1, f"judge {short}: {key} differs across chunks: {values}"
        roster[short] = {"short": short, **{k: metas[0][k] for k in JUDGE_CONFIG_KEYS}}

    all_metas = [m for _, _, m in sidecars]
    decoding: dict = {}
    for key in SHARED_DECODING_KEYS:
        values = {m[key] for m in all_metas}
        assert len(values) == 1, f"decoding config {key!r} differs across sidecars: {values}"
        decoding[key] = all_metas[0][key]
    decoding["temperature"] = 0.0
    decoding["decoding_mode"] = "greedy"

    per_judge_prefix_caching = {}
    for short, metas in by_judge.items():
        values = {m["enable_prefix_caching"] for m in metas}
        assert len(values) == 1, f"judge {short}: enable_prefix_caching differs across chunks: {values}"
        per_judge_prefix_caching[short] = metas[0]["enable_prefix_caching"]
    decoding["prefix_caching"] = {
        "default": True,
        "per_judge": per_judge_prefix_caching,
        "note": (
            "gemma runs with prefix caching disabled — vLLM's Triton prefix-caching "
            "kernel OOMs on T4/Turing shared memory for Gemma-2's attention pattern "
            "(see purva/committee/models.py: enable_prefix_caching)"
        ),
    }

    prompt_shas = {m["prompt_file_sha256"] for m in all_metas}
    assert len(prompt_shas) == 1, f"prompt_file_sha256 differs across the 45 sidecars: {prompt_shas}"
    prompt_paths = {m["prompt_file"] for m in all_metas}
    assert len(prompt_paths) == 1, f"prompt_file differs across the 45 sidecars: {prompt_paths}"
    prompt_path = Path(next(iter(prompt_paths)))
    prompt_text = prompt_path.read_text(encoding="utf-8")
    actual_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    recorded_sha = next(iter(prompt_shas))
    assert actual_sha == recorded_sha, (
        f"{prompt_path} sha256 {actual_sha} does not match the value recorded in every sidecar ({recorded_sha})"
    )
    prompt_info = {
        "path": prompt_path.as_posix(),
        "sha256": actual_sha,
        "word_count": len(prompt_text.split()),
    }

    return roster, decoding, prompt_info


def _sorted_values(vals: set) -> list:
    non_null = sorted(v for v in vals if v is not None)
    return non_null + [None] if None in vals else non_null


def _infer_type(values: list) -> str:
    types: set[str] = set()
    nullable = False
    for v in values:
        if v is None:
            nullable = True
        elif isinstance(v, bool):
            types.add("bool")
        elif isinstance(v, int):
            types.add("int")
        elif isinstance(v, float):
            types.add("float")
        elif isinstance(v, str):
            types.add("str")
        elif isinstance(v, dict):
            types.add("object")
        elif isinstance(v, list):
            types.add("array")
        else:
            types.add(type(v).__name__)
    base = "|".join(sorted(types)) or "null"
    return f"{base}|null" if nullable and base != "null" else base


def build_schema(master_rows: list[dict]) -> dict:
    fields = {f: _infer_type([r.get(f) for r in master_rows]) for f in CORPUS_FIELDS}
    fields["chunk"] = _infer_type([r["chunk"] for r in master_rows])
    fields["n_judges"] = _infer_type([r["n_judges"] for r in master_rows])
    fields["judges"] = "object (see judges_fields)"

    all_judge_entries = [entry for r in master_rows for entry in r["judges"].values() if entry is not None]
    judges_fields = {f: _infer_type([e.get(f) for e in all_judge_entries]) for f in JUDGE_FIELDS}

    observed: dict[str, set] = {f: set() for f in CATEGORICAL_FIELDS}
    for row in master_rows:
        for f in _CORPUS_CATEGORICAL:
            observed[f].add(row.get(f))
        for entry in row["judges"].values():
            if entry is not None:
                for f in _JUDGE_CATEGORICAL:
                    observed[f].add(entry.get(f))

    categorical_values = {}
    for field in CATEGORICAL_FIELDS:
        obs = observed[field]
        expected = EXPECTED_CATEGORICAL_VALUES.get(field)
        entry = {"observed": _sorted_values(obs)}
        if expected is not None:
            unexpected = obs - expected
            entry["expected"] = _sorted_values(expected)
            entry["unexpected"] = _sorted_values(unexpected)
            if unexpected:
                print(f"WARNING: field {field!r} has unexpected observed value(s): {_sorted_values(unexpected)}")
        else:
            entry["expected"] = None
            entry["unexpected"] = None
        categorical_values[field] = entry

    return {"fields": fields, "judges_fields": judges_fields, "categorical_values": categorical_values}


def build_counts(master_rows: list[dict]) -> dict:
    total = len(master_rows)
    per_judge_non_null = {short: 0 for short in JUDGE_SHORT_NAMES.values()}
    for r in master_rows:
        for short, entry in r["judges"].items():
            if entry is not None:
                per_judge_non_null[short] += 1
    n_judges_dist = Counter(r["n_judges"] for r in master_rows)
    per_chunk = Counter(r["chunk"] for r in master_rows)
    per_source = Counter(r["source_name"] for r in master_rows)
    per_register = Counter(r["register"] for r in master_rows)
    total_failures = sum(total - c for c in per_judge_non_null.values())

    return {
        "total_rows": total,
        "unique_ids": len({r["id"] for r in master_rows}),
        "n_judges_distribution": {str(k): v for k, v in sorted(n_judges_dist.items(), reverse=True)},
        "per_judge_non_null": per_judge_non_null,
        "per_chunk_counts": {str(k): v for k, v in sorted(per_chunk.items())},
        "per_source_counts": dict(sorted(per_source.items(), key=lambda kv: -kv[1])),
        "per_register_counts": dict(sorted(per_register.items(), key=lambda kv: -kv[1])),
        "total_judge_decisions": total * len(JUDGE_SHORT_NAMES) - total_failures,
    }


def read_protocol_version(protocol_path: Path) -> str:
    text = protocol_path.read_text(encoding="utf-8")
    m = re.search(r"^Version:\s*(\S+)", text, re.MULTILINE)
    if not m:
        sys.exit(f"could not find a 'Version:' line in {protocol_path}")
    return m.group(1)


def build_provenance(master_rows: list[dict], protocol_path: Path) -> dict:
    return {
        "protocol_version": read_protocol_version(protocol_path),
        "lid_model": {"repo_id": GLOTLID_REPO_ID, "filename": GLOTLID_FILENAME},
        "corpus_source_count": len({r["source_name"] for r in master_rows}),
        "chunking_scheme": {
            "n_chunks": N_CHUNKS,
            "stratified_on": ["source_name", "register", "text_type"],
            "seed": 42,
            "method": "largest-remainder allocation (see purva/committee/make_chunks.py)",
        },
        "hardware": HARDWARE_DESCRIPTOR,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_integrity(out_jsonl: Path, out_parquet: Path) -> dict:
    result = {}
    for name, path in (("jsonl", out_jsonl), ("parquet", out_parquet)):
        if not path.exists():
            sys.exit(f"cannot compute integrity hash — {path} does not exist")
        result[name] = {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def generate_manifest(
    master_rows: list[dict],
    committee_dir: Path,
    protocol_path: Path,
    out_jsonl: Path,
    out_parquet: Path,
    manifest_path: Path,
) -> dict:
    sidecars = load_all_sidecars(committee_dir)
    roster, decoding, prompt_info = build_roster_decoding_prompt(sidecars)

    manifest = {
        "judges": roster,
        "prompt": prompt_info,
        "decoding": decoding,
        "schema": build_schema(master_rows),
        "counts": build_counts(master_rows),
        "provenance": build_provenance(master_rows, protocol_path),
        "integrity": build_integrity(out_jsonl, out_parquet),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({manifest_path.stat().st_size / 1e3:.1f} KB)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus_lid.jsonl")
    ap.add_argument("--chunks-dir", default="data/chunks")
    ap.add_argument("--merged-dir", default="data/committee/merged")
    ap.add_argument("--committee-dir", default="data/committee")
    ap.add_argument("--protocol-file", default="PROTOCOL.md")
    ap.add_argument("--output-jsonl", default="data/purva_master.jsonl")
    ap.add_argument("--output-parquet", default="data/purva_master.parquet")
    ap.add_argument("--manifest-path", default="data/purva_master.meta.json")
    ap.add_argument(
        "--manifest-only", action="store_true",
        help="regenerate the manifest from the existing master files without rebuilding them",
    )
    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_parquet = Path(args.output_parquet)
    manifest_path = Path(args.manifest_path)
    committee_dir = Path(args.committee_dir)
    protocol_path = Path(args.protocol_file)

    if args.manifest_only:
        for p in (out_jsonl, out_parquet):
            if not p.exists():
                sys.exit(f"--manifest-only requires an existing {p} to read from")
        print(f"--manifest-only: reading existing {out_jsonl}")
        master_rows = load_jsonl(out_jsonl)
        print(f"loaded {len(master_rows)} rows\n")
    else:
        for p in (out_jsonl, out_parquet):
            if p.exists():
                sys.exit(f"refusing to overwrite existing {p} — remove it first if you want to rebuild")
        master_rows = build_and_verify_master(args)
        write_outputs(master_rows, out_jsonl, out_parquet)

    manifest = generate_manifest(master_rows, committee_dir, protocol_path, out_jsonl, out_parquet, manifest_path)
    print("\n=== manifest ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
