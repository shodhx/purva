"""Merge per-chunk judge shards into one file per judge, with mechanical
verification that no chunk has a silent gap and no sentence was double-voted.

Expected layout (written by kaggle_run.py / kaggle/kernel/main.py --chunk N):

    data/chunks/chunk_01.jsonl ... chunk_09.jsonl   (from make_chunks.py)
    data/committee/chunk_01/{judge}__{prompt}.jsonl ... chunk_09/...

Chunks and judges accumulate independently over weeks — a given (chunk,
judge) shard simply won't exist yet until that run happens, and that's
expected, not an error. What's NOT expected: a shard file that exists but is
missing some of the sentence IDs it should cover (a run that silently
dropped rows), or the same ID showing up twice for one judge. Those are the
failure modes this script exists to catch mechanically, because with 9
chunks x 5+ judges accumulating over weeks nobody is going to notice a gap
by eye.

A chunk counts as "completed" only once every judge that has produced a
shard for ANY chunk has also produced one for this chunk — i.e. the roster
of judges is inferred from what's actually on disk, not hardcoded. Only
completed chunks are checked for per-ID coverage; chunks still in progress
are reported but not gated on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

CONFIG_KEYS = ("repo_id", "revision", "quantization", "prompt_file", "seed", "max_model_len")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_chunk_id_sets(chunks_dir: Path) -> dict[str, set[str]]:
    chunk_files = sorted(chunks_dir.glob("chunk_*.jsonl"))
    if not chunk_files:
        sys.exit(f"no chunk_*.jsonl files found under {chunks_dir} — run make_chunks.py first")
    out = {}
    for f in chunk_files:
        chunk_name = f.stem  # e.g. "chunk_01"
        out[chunk_name] = {r["id"] for r in load_rows(f)}
    return out


def discover_shards(committee_dir: Path) -> dict[str, dict[str, Path]]:
    """Returns {judge: {chunk_name: shard_path}}. Judge name is the part of
    the filename before '__'; chunk name is the parent directory name.
    Only looks one level down (data/committee/chunk_NN/*.jsonl) — top-level
    data/committee/*.jsonl files from pre-chunking runs are intentionally
    not part of this merge."""
    by_judge: dict[str, dict[str, Path]] = defaultdict(dict)
    for chunk_dir in sorted(committee_dir.glob("chunk_*")):
        if not chunk_dir.is_dir():
            continue
        chunk_name = chunk_dir.name
        for shard_path in sorted(chunk_dir.glob("*.jsonl")):
            judge = shard_path.name.split("__", 1)[0]
            if chunk_name in by_judge[judge]:
                sys.exit(f"multiple shard files for judge={judge} in {chunk_dir} — ambiguous, refusing to merge")
            by_judge[judge][chunk_name] = shard_path
    return by_judge


def check_no_duplicate_ids(judge: str, chunk_rows: dict[str, list[dict]]) -> None:
    """Hard-fail if any ID appears twice — either within one chunk's shard
    (a resumed run that re-appended already-written rows) or across two
    different chunks for the same judge (should be impossible given
    make_chunks.py's disjoint partition, but is exactly the kind of thing
    that must be verified mechanically rather than assumed)."""
    seen: dict[str, str] = {}
    for chunk_name, rows in chunk_rows.items():
        local_seen: set[str] = set()
        for r in rows:
            rid = r["id"]
            if rid in local_seen:
                sys.exit(f"judge={judge} chunk={chunk_name}: ID {rid} appears twice within the same shard file")
            local_seen.add(rid)
            if rid in seen:
                sys.exit(
                    f"judge={judge}: ID {rid} appears in both {seen[rid]} and {chunk_name} — "
                    "a sentence was processed under two different chunks for this judge"
                )
            seen[rid] = chunk_name


def check_config_consistency(judge: str, chunk_rows: dict[str, list[dict]]) -> None:
    """Hard-fail if the recorded run_config (model repo, revision,
    quantization, prompt file, seed, max_model_len) differs across this
    judge's chunks — chunks are processed weeks apart, and this is the
    mechanical check that conditions were actually identical rather than
    trusted to be."""
    seen_configs: dict[tuple, str] = {}
    for chunk_name, rows in chunk_rows.items():
        for r in rows:
            cfg = r.get("run_config")
            if cfg is None:
                sys.exit(f"judge={judge} chunk={chunk_name}: row id={r['id']} has no run_config recorded")
            key = tuple(cfg.get(k) for k in CONFIG_KEYS)
            if key not in seen_configs:
                seen_configs[key] = chunk_name
    if len(seen_configs) > 1:
        lines = "\n".join(f"    {dict(zip(CONFIG_KEYS, k))}  (first seen in {c})" for k, c in seen_configs.items())
        sys.exit(f"judge={judge}: run_config differs across chunks — conditions were not identical:\n{lines}")


def check_chunk_coverage(judge: str, chunk_name: str, rows: list[dict], expected_ids: set[str]) -> None:
    got_ids = {r["id"] for r in rows}
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    if missing:
        sys.exit(
            f"judge={judge} chunk={chunk_name}: {len(missing)} of {len(expected_ids)} sentence(s) missing a "
            f"vote, e.g. {sorted(missing)[:5]} — this chunk is marked complete (every judge has a shard for "
            "it) but this judge's shard has a silent gap"
        )
    if extra:
        sys.exit(
            f"judge={judge} chunk={chunk_name}: shard contains {len(extra)} ID(s) not in this chunk, "
            f"e.g. {sorted(extra)[:5]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default="data/chunks")
    ap.add_argument("--committee-dir", default="data/committee")
    ap.add_argument("--output-dir", default="data/committee/merged")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    committee_dir = Path(args.committee_dir)
    out_dir = Path(args.output_dir)

    chunk_id_sets = load_chunk_id_sets(chunks_dir)
    all_chunk_names = sorted(chunk_id_sets)
    print(f"chunks known: {all_chunk_names}")

    by_judge = discover_shards(committee_dir)
    if not by_judge:
        sys.exit(f"no shard files found under {committee_dir}/chunk_*/*.jsonl — nothing to merge")
    judges = sorted(by_judge)
    print(f"judges discovered: {judges}")

    completed_chunks = [c for c in all_chunk_names if all(c in by_judge[j] for j in judges)]
    pending_chunks = [c for c in all_chunk_names if c not in completed_chunks]

    print("\nper-judge, per-chunk shard status:")
    header = f"  {'judge':16s} " + " ".join(f"{c[-2:]:>4s}" for c in all_chunk_names)
    print(header)
    for j in judges:
        marks = " ".join(f"{'X':>4s}" if c in by_judge[j] else f"{'.':>4s}" for c in all_chunk_names)
        print(f"  {j:16s} {marks}")

    print(f"\ncompleted chunks (all {len(judges)} judges present): {completed_chunks or '(none yet)'}")
    print(f"pending chunks (still missing some judges): {pending_chunks or '(none)'}")

    merged: dict[str, list[dict]] = {}
    for j in judges:
        chunk_rows = {c: load_rows(p) for c, p in by_judge[j].items()}

        check_no_duplicate_ids(j, chunk_rows)
        check_config_consistency(j, chunk_rows)

        for c in completed_chunks:
            if c in chunk_rows:
                check_chunk_coverage(j, c, chunk_rows[c], chunk_id_sets[c])

        all_rows = [r for c in sorted(chunk_rows) for r in chunk_rows[c]]
        all_rows.sort(key=lambda r: r["id"])
        merged[j] = all_rows

    print("\nverified: no duplicate IDs, no cross-chunk double-votes, config consistent per judge, "
          "no coverage gaps in any completed chunk")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\nwriting merged shards:")
    for j in judges:
        path = out_dir / f"{j}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in merged[j]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path} ({len(merged[j])} rows across {len(by_judge[j])} chunk(s))")


if __name__ == "__main__":
    main()
