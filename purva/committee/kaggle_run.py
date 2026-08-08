"""Local driver that wraps the Kaggle CLI to run purva/committee/run_judge.py
on Kaggle GPU from the terminal, instead of a human copying files through
the notebook UI.

Flow: copy kaggle/kernel/ to a scratch dir -> patch
MODEL_NAME/BENCH_N/QUANT/CHUNK constants in the copy's main.py -> `kaggle
kernels push` -> poll `kaggle kernels status` until terminal -> `kaggle
kernels output` into data/committee/ (full/chunk run) or parse the
downloaded log for the bench summary (bench mode). On failure, fetch and
print the kernel log either way.

--chunk N (1-9) targets a single data/chunks/chunk_NN.jsonl shard produced
by make_chunks.py, instead of the default full/subset input — this is the
chunk-major processing mode, where every chunk is labeled by all judges
before moving to the next chunk. The kernel stages that run's output under
committee/chunk_NN/ so downloaded shards keep their chunk subdirectory
(merge_shards.py depends on that layout).

The checked-in kaggle/kernel/main.py is never modified — only the scratch
copy is patched, so every push is reproducible purely from this script's
CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..lid.env import load_env
from .models import REGISTRY

load_env()

KERNEL_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "kaggle" / "kernel"
# No account name is hardcoded here — each operator's Kaggle username lives
# in their own .env (KAGGLE_OWNER), never in source control. --owner exists
# to override this per-invocation when rotating between accounts for fresh
# GPU quota.
DEFAULT_OWNER = os.environ.get("KAGGLE_OWNER", "")
KERNEL_SLUG = "purva-judge-committee"
DATASET_SLUG = "purva-corpus"

TERMINAL_OK = {"complete"}
TERMINAL_FAIL = {"error", "cancelacknowledged", "cancelrequested"}

# The `kaggle` CLI opens downloaded log/output files with the platform default
# text encoding (cp1252 on Windows) and crashes on non-ASCII content — pip's
# unicode progress-bar characters, or (once real runs happen) Devanagari
# rationale/domain text in shard output. Forcing UTF-8 mode in the child
# `kaggle` process's env avoids that; PYTHONUTF8 must be set before a Python
# process starts, so it's only applied to subprocesses, not this process
# retroactively (see main() for this process's own stdout handling).
_SUBPROCESS_ENV = {**os.environ, "PYTHONUTF8": "1"}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    kwargs.setdefault("env", _SUBPROCESS_ENV)
    # This host intermittently fails to spawn a child process at the OS level
    # (WinError 5 / access denied, unrelated to `kaggle` itself) — retry a
    # few times rather than letting a transient spawn failure kill a
    # multi-minute poll loop and orphan an already-running kernel session.
    last_exc = None
    for attempt in range(5):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
        except OSError as e:
            last_exc = e
            print(f"  (subprocess spawn failed, attempt {attempt + 1}/5: {e})")
            time.sleep(2)
    raise last_exc


def patch_main(text: str, model: str, bench: int, quant: str, guided: bool, rationale: bool, chunk: int) -> str:
    patched, n1 = re.subn(r'^MODEL_NAME = .*$', f'MODEL_NAME = "{model}"', text, count=1, flags=re.MULTILINE)
    patched, n2 = re.subn(r'^BENCH_N = .*$', f'BENCH_N = {bench}', patched, count=1, flags=re.MULTILINE)
    patched, n3 = re.subn(r'^QUANT = .*$', f'QUANT = "{quant}"', patched, count=1, flags=re.MULTILINE)
    patched, n4 = re.subn(r'^GUIDED = .*$', f'GUIDED = {guided}', patched, count=1, flags=re.MULTILINE)
    patched, n5 = re.subn(r'^RATIONALE = .*$', f'RATIONALE = {rationale}', patched, count=1, flags=re.MULTILINE)
    patched, n6 = re.subn(r'^CHUNK = .*$', f'CHUNK = {chunk}', patched, count=1, flags=re.MULTILINE)
    if (n1, n2, n3, n4, n5, n6) != (1, 1, 1, 1, 1, 1):
        raise RuntimeError(f"expected to patch exactly 1 of each constant, got {(n1, n2, n3, n4, n5, n6)}")
    return patched


def prepare_scratch_dir(model: str, bench: int, quant: str, guided: bool, rationale: bool, chunk: int, owner: str) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="purva_kaggle_push_"))
    for item in KERNEL_TEMPLATE_DIR.iterdir():
        dest = scratch / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy(item, dest)

    main_path = scratch / "main.py"
    patched = patch_main(main_path.read_text(encoding="utf-8"), model, bench, quant, guided, rationale, chunk)
    main_path.write_text(patched, encoding="utf-8")

    # kernel-metadata.json's "id" determines which account's kernel `kaggle
    # kernels push` targets, and "dataset_sources" which account's dataset
    # gets mounted at /kaggle/input — both must move together, or the push
    # would go to one account while reading stale/no data from another's
    # dataset. The checked-in template holds a placeholder (no real account
    # name belongs in source control), so this always rewrites both fields
    # rather than only when --owner deviates from some hardcoded default.
    meta_path = scratch / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["id"] = f"{owner}/{KERNEL_SLUG}"
    meta["dataset_sources"] = [f"{owner}/{DATASET_SLUG}"]
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return scratch


def push(scratch_dir: Path) -> None:
    result = run(["kaggle", "kernels", "push", "-p", str(scratch_dir)])
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"kaggle kernels push failed (exit {result.returncode})")


def poll_status(kernel_ref: str, timeout_s: int, poll_interval_s: int) -> str:
    deadline = time.time() + timeout_s
    consecutive_query_failures = 0

    while time.time() < deadline:
        result = run(["kaggle", "kernels", "status", kernel_ref])
        text = (result.stdout + result.stderr).strip()

        if result.returncode != 0:
            consecutive_query_failures += 1
            print(f"status query failed ({consecutive_query_failures}): {text}")
            if consecutive_query_failures >= 10:
                raise SystemExit("kaggle kernels status kept failing — giving up")
            time.sleep(poll_interval_s)
            continue

        consecutive_query_failures = 0
        lowered = text.lower()
        print(f"status: {text}")

        if any(s in lowered for s in TERMINAL_OK):
            return "complete"
        if any(s in lowered for s in TERMINAL_FAIL):
            return "error"

        time.sleep(poll_interval_s)

    raise SystemExit(f"timed out after {timeout_s}s waiting for {kernel_ref} to finish")


def fetch_output(kernel_ref: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    result = run(["kaggle", "kernels", "output", kernel_ref, "-p", str(dest_dir), "--force"])
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return dest_dir


def read_log_text(log_file: Path) -> str:
    """Kaggle's downloaded .log file is a JSON array of
    {"stream_name": "stdout"|"stderr", "time": float, "data": str} entries,
    not plain text — reconstruct the actual console output from it. Falls
    back to the raw file content if it isn't in that format."""
    raw = log_file.read_text(encoding="utf-8", errors="replace")
    try:
        entries = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw
    return "".join(e.get("data", "") for e in entries if isinstance(e, dict))


def print_log(dest_dir: Path) -> None:
    log_files = sorted(dest_dir.glob("*.log"))
    if not log_files:
        print(f"(no .log file found in {dest_dir})")
        return
    for log_file in log_files:
        print(f"\n=== {log_file} ===")
        print(read_log_text(log_file))


def print_bench_summary(dest_dir: Path) -> None:
    log_files = sorted(dest_dir.glob("*.log"))
    summary_keys = ("[config]", "sentences:", "elapsed:", "throughput:", "parse failures:", "failures written to", "preemptions:")
    found_any = False
    for log_file in log_files:
        for line in read_log_text(log_file).splitlines():
            if any(line.strip().startswith(k) for k in summary_keys):
                print(line)
                found_any = True
    if not found_any:
        print("(bench summary lines not found in log — printing full log instead)")
        print_log(dest_dir)


def main():
    # Downloaded log/shard content can contain Devanagari; make sure printing
    # it can't crash this process on a non-UTF-8 console (e.g. Windows cp1252).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--quant", choices=["auto", "awq", "bnb"], default="auto")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="benchmark on the first N pilot sentences instead of a full run")
    ap.add_argument("--guided", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rationale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--chunk", type=int, default=0, metavar="N", choices=range(0, 10), help="process only data/chunks/chunk_NN.jsonl (1-9); 0 = legacy full/subset run over the default input")
    ap.add_argument("--timeout", type=int, default=2400, help="max seconds to wait for the kernel to finish")
    ap.add_argument("--poll-interval", type=int, default=20)
    ap.add_argument("--output-dir", default="data/committee")
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Kaggle account that owns the kernel/dataset (defaults to $KAGGLE_OWNER from .env) — switch when rotating accounts for fresh GPU quota")
    args = ap.parse_args()

    if not args.owner:
        ap.error("no --owner given and KAGGLE_OWNER is not set in .env — set one of the two to a Kaggle username")

    if args.bench and args.chunk:
        ap.error("--bench and --chunk are mutually exclusive — bench mode always runs over data/pilot_set.jsonl")

    kernel_ref = f"{args.owner}/{KERNEL_SLUG}"

    scratch_dir = prepare_scratch_dir(args.model, args.bench, args.quant, args.guided, args.rationale, args.chunk, args.owner)
    print(f"scratch push dir: {scratch_dir}")

    try:
        push(scratch_dir)
        status = poll_status(kernel_ref, args.timeout, args.poll_interval)

        mode = f"bench{args.bench}" if args.bench else (f"chunk{args.chunk:02d}" if args.chunk else "full")
        tag = f"{mode}__{args.quant}__g{int(args.guided)}r{int(args.rationale)}"
        fetch_dir = Path(args.output_dir) / "kaggle_out" / f"{args.model}__{tag}"
        fetch_output(kernel_ref, fetch_dir)

        if status == "error":
            print(f"\nkernel run FAILED for model={args.model} quant={args.quant} bench={args.bench}")
            print_log(fetch_dir)
            raise SystemExit(1)

        print(f"\nkernel run complete for model={args.model} quant={args.quant} bench={args.bench}")
        if args.bench:
            print_bench_summary(fetch_dir)
        else:
            shard_files = (
                list(fetch_dir.glob("committee/*/*.jsonl"))
                or list(fetch_dir.glob("committee/*.jsonl"))
                or list(fetch_dir.glob("*.jsonl"))
            )
            print(f"shards downloaded to {fetch_dir}: {[str(f.relative_to(fetch_dir)) for f in shard_files]}")
            if args.chunk:
                print(
                    f"copy these into data/committee/chunk_{args.chunk:02d}/ before running merge_shards.py "
                    "(chunk subdirectory structure must be preserved)"
                )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
