"""Local driver that wraps the Kaggle CLI to run purva/benchmark/train.py on
Kaggle GPU — same flow as purva/committee/kaggle_run.py: copy
kaggle/kernel_benchmark/ to a scratch dir -> patch constants in the copy's
main.py -> `kaggle kernels push` -> poll status -> `kaggle kernels output`
into data/benchmark_models/ (checkpoint dir + .meta.json sidecar).

The checked-in kaggle/kernel_benchmark/main.py is never modified — only the
scratch copy is patched, so every push is reproducible purely from this
script's CLI arguments.
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

KERNEL_TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "kaggle" / "kernel_benchmark"
DEFAULT_OWNER = os.environ.get("KAGGLE_OWNER", "")
KERNEL_SLUG = "purva-benchmark-train"
DATASET_SLUG = "purva-benchmark-data"

TERMINAL_OK = {"complete"}
TERMINAL_FAIL = {"error", "cancelacknowledged", "cancelrequested"}

_SUBPROCESS_ENV = {**os.environ, "PYTHONUTF8": "1"}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    kwargs.setdefault("env", _SUBPROCESS_ENV)
    last_exc = None
    for attempt in range(5):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
        except OSError as e:
            last_exc = e
            print(f"  (subprocess spawn failed, attempt {attempt + 1}/5: {e})")
            time.sleep(2)
    raise last_exc


def patch_main(text: str, model: str, epochs: int, lr: float, batch_size: int, max_len: int, seed: int, fp16: bool, limit: int, hindi_set: str, early_stopping_patience: int) -> str:
    patched, n1 = re.subn(r'^MODEL_NAME = .*$', f'MODEL_NAME = "{model}"', text, count=1, flags=re.MULTILINE)
    patched, n2 = re.subn(r'^EPOCHS = .*$', f'EPOCHS = {epochs}', patched, count=1, flags=re.MULTILINE)
    patched, n3 = re.subn(r'^LR = .*$', f'LR = {lr}', patched, count=1, flags=re.MULTILINE)
    patched, n4 = re.subn(r'^BATCH_SIZE = .*$', f'BATCH_SIZE = {batch_size}', patched, count=1, flags=re.MULTILINE)
    patched, n5 = re.subn(r'^MAX_LEN = .*$', f'MAX_LEN = {max_len}', patched, count=1, flags=re.MULTILINE)
    patched, n6 = re.subn(r'^SEED = .*$', f'SEED = {seed}', patched, count=1, flags=re.MULTILINE)
    patched, n7 = re.subn(r'^FP16 = .*$', f'FP16 = {fp16}', patched, count=1, flags=re.MULTILINE)
    patched, n8 = re.subn(r'^LIMIT = .*$', f'LIMIT = {limit}', patched, count=1, flags=re.MULTILINE)
    patched, n9 = re.subn(r'^HINDI_SET = .*$', f'HINDI_SET = "{hindi_set}"', patched, count=1, flags=re.MULTILINE)
    patched, n10 = re.subn(r'^EARLY_STOPPING_PATIENCE = .*$', f'EARLY_STOPPING_PATIENCE = {early_stopping_patience}', patched, count=1, flags=re.MULTILINE)
    if (n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) != (1, 1, 1, 1, 1, 1, 1, 1, 1, 1):
        raise RuntimeError(f"expected to patch exactly 1 of each constant, got {(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)}")
    return patched


def prepare_scratch_dir(model: str, epochs: int, lr: float, batch_size: int, max_len: int, seed: int, fp16: bool, limit: int, owner: str, hindi_set: str, early_stopping_patience: int) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="purva_kaggle_bench_push_"))
    for item in KERNEL_TEMPLATE_DIR.iterdir():
        dest = scratch / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy(item, dest)

    main_path = scratch / "main.py"
    patched = patch_main(main_path.read_text(encoding="utf-8"), model, epochs, lr, batch_size, max_len, seed, fp16, limit, hindi_set, early_stopping_patience)
    main_path.write_text(patched, encoding="utf-8")

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


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=0, metavar="N", help="debug only: cap training set size for a smoke test")
    ap.add_argument("--timeout", type=int, default=2400, help="max seconds to wait for the kernel to finish")
    ap.add_argument("--poll-interval", type=int, default=20)
    ap.add_argument("--output-dir", default="data/benchmark_models")
    ap.add_argument("--owner", default=DEFAULT_OWNER, help="Kaggle account (defaults to $KAGGLE_OWNER from .env)")
    ap.add_argument("--hindi-set", default="hindi_validation_set.jsonl",
                     help="hindi_baseline only: filename (must be in the pushed Kaggle dataset) to train on")
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                     help="hindi_baseline only: >0 enables dev-loss-driven early stopping; --epochs becomes a max-epochs cap")
    args = ap.parse_args()

    if not args.owner:
        ap.error("no --owner given and KAGGLE_OWNER is not set in .env — set one of the two to a Kaggle username")

    kernel_ref = f"{args.owner}/{KERNEL_SLUG}"

    scratch_dir = prepare_scratch_dir(args.model, args.epochs, args.lr, args.batch_size, args.max_len, args.seed, args.fp16, args.limit, args.owner, args.hindi_set, args.early_stopping_patience)
    print(f"scratch push dir: {scratch_dir}")

    try:
        push(scratch_dir)
        status = poll_status(kernel_ref, args.timeout, args.poll_interval)

        fetch_dir = Path(args.output_dir) / "kaggle_out" / args.model
        fetch_output(kernel_ref, fetch_dir)

        if status == "error":
            print(f"\nkernel run FAILED for model={args.model}")
            print_log(fetch_dir)
            raise SystemExit(1)

        print(f"\nkernel run complete for model={args.model}")
        ckpt_src = fetch_dir / "benchmark_models" / args.model
        meta_src = fetch_dir / "benchmark_models" / f"{args.model}.meta.json"
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if ckpt_src.exists():
            dest = out_dir / args.model
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ckpt_src, dest)
            print(f"copied checkpoint -> {dest}")
        else:
            print(f"WARNING: no checkpoint found at {ckpt_src}")
        if meta_src.exists():
            shutil.copy(meta_src, out_dir / meta_src.name)
            print(f"copied meta -> {out_dir / meta_src.name}")
        else:
            print(f"WARNING: no meta.json found at {meta_src}")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
