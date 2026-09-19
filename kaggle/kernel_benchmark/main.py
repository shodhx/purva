"""Kaggle kernel entry point for benchmark fine-tuning (purva/benchmark/train.py).

Pushed via purva/benchmark/kaggle_run.py, which copies this whole
kaggle/kernel_benchmark/ directory to a scratch folder and patches the
constants below in the COPY before pushing — this file in source control
always stays at its template defaults. Do not hand-edit these values for a
one-off run; use kaggle_run.py so every push is reproducible from its CLI
arguments.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# --- PATCHABLE CONSTANTS: kaggle_run.py rewrites these lines ---
MODEL_NAME = "muril"
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 32
MAX_LEN = 128
SEED = 42
FP16 = True
LIMIT = 0  # 0 = full training set; >0 = debug cap, e.g. for a smoke test
HINDI_SET = "hindi_validation_set.jsonl"  # hindi_baseline only
EARLY_STOPPING_PATIENCE = 0  # hindi_baseline only; 0 = disabled (fixed EPOCHS)
# --- END PATCHABLE CONSTANTS ---

REPO_URL = "https://github.com/shodhx/purva.git"
# Cloned outside /kaggle/working on purpose — see kaggle/kernel/main.py's
# identical rationale (anything left in /kaggle/working becomes downloadable
# kernel "output" whole, including .git internals, if the repo lives there).
REPO_DIR = Path("/kaggle/tmp/purva")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
DATA_FILES = ["purva_master.parquet", "purva_aggregated.jsonl", "splits.json", HINDI_SET]


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def find_input_file(name: str) -> Path | None:
    """Locate a dataset file anywhere under /kaggle/input — Kaggle's mount
    layout for dataset_sources has varied across API/UI versions, so search
    rather than assume a fixed path (same approach as kaggle/kernel/main.py)."""
    if not KAGGLE_INPUT_ROOT.exists():
        return None
    matches = list(KAGGLE_INPUT_ROOT.rglob(name))
    return matches[0] if matches else None


def main():
    if KAGGLE_INPUT_ROOT.exists():
        all_files = sorted(str(p.relative_to(KAGGLE_INPUT_ROOT)) for p in KAGGLE_INPUT_ROOT.rglob("*") if p.is_file())
        print(f"/kaggle/input tree ({len(all_files)} files): {all_files}")
    else:
        print("/kaggle/input does not exist")

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle-benchmark.txt"], cwd=str(REPO_DIR))

    try:
        import torch

        print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
              f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception as e:
        print(f"[env] could not import torch to report device info: {e!r}")

    data_dir = REPO_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for name in DATA_FILES:
        src = find_input_file(name)
        if src is not None:
            shutil.copy(src, data_dir / name)
            print(f"copied {src} -> {data_dir / name}")
        else:
            print(f"skip (not found anywhere under {KAGGLE_INPUT_ROOT}): {name}")

    cmd = [
        sys.executable, "-m", "purva.benchmark.train",
        "--model", MODEL_NAME,
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--batch-size", str(BATCH_SIZE),
        "--max-len", str(MAX_LEN),
        "--seed", str(SEED),
        "--output-dir", "data/benchmark_models",
        "--hindi-set", f"data/{HINDI_SET}",
    ]
    cmd += ["--fp16"] if FP16 else ["--no-fp16"]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    if EARLY_STOPPING_PATIENCE:
        cmd += ["--early-stopping-patience", str(EARLY_STOPPING_PATIENCE)]

    run(cmd, cwd=str(REPO_DIR))

    src_dir = data_dir / "benchmark_models"
    out_dir = Path("/kaggle/working/benchmark_models")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoint directory (data/benchmark_models/<model>/) plus its sidecar
    # (data/benchmark_models/<model>.meta.json) — copy both whole.
    model_ckpt_dir = src_dir / MODEL_NAME
    if model_ckpt_dir.exists():
        shutil.copytree(model_ckpt_dir, out_dir / MODEL_NAME, dirs_exist_ok=True)
        print(f"staged checkpoint {model_ckpt_dir} -> {out_dir / MODEL_NAME}")
    meta_src = src_dir / f"{MODEL_NAME}.meta.json"
    if meta_src.exists():
        shutil.copy(meta_src, out_dir / meta_src.name)
        print(f"staged {meta_src} -> {out_dir / meta_src.name}")


if __name__ == "__main__":
    main()
