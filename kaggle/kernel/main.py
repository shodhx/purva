"""Kaggle kernel entry point for the PURVA judge committee.

Pushed via purva/committee/kaggle_run.py, which copies this whole
kaggle/kernel/ directory to a scratch folder and patches the constants
below in the COPY before pushing — this file in source control always
stays at its template defaults. Do not hand-edit these values for a
one-off run; use kaggle_run.py so every push is reproducible from its CLI
arguments.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- PATCHABLE CONSTANTS: kaggle_run.py rewrites these three lines ---
MODEL_NAME = "llama-3.1-8b"
BENCH_N = 0  # 0 = full run over INPUT_FILE; >0 = bench mode, first N pilot sentences, writes no shard
QUANT = "auto"  # auto | awq | bnb
# --- END PATCHABLE CONSTANTS ---

REPO_URL = "https://github.com/shodhx/purva.git"
# Cloned outside /kaggle/working on purpose: anything left in /kaggle/working
# at the end of the run becomes kernel "output" and gets pulled down whole by
# `kaggle kernels output` (including .git internals and __pycache__) if the
# repo lives there. /kaggle/working/committee/ is the only thing we want
# downloadable, staged there explicitly at the end of a full run.
REPO_DIR = Path("/kaggle/tmp/purva")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
INPUT_FILE = "data/label_subset.jsonl"  # full-run input; ignored in bench mode
DATA_FILES = ["corpus_lid.jsonl", "pilot_set.jsonl", "label_subset.jsonl"]


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def set_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("HF_TOKEN loaded from Kaggle secrets")
    except Exception as e:
        print(f"no HF_TOKEN secret available ({e}); proceeding without it "
              "(fine for ungated repos, will fail to download gated ones)")


def find_input_file(name: str) -> Path | None:
    """Locate a dataset file anywhere under /kaggle/input. Kaggle's mount
    layout for dataset_sources has varied across API/UI versions (flat
    /kaggle/input/<slug>/ vs nested /kaggle/input/datasets/<owner>/<slug>/),
    so search rather than assume a fixed path."""
    if not KAGGLE_INPUT_ROOT.exists():
        return None
    matches = list(KAGGLE_INPUT_ROOT.rglob(name))
    return matches[0] if matches else None


def main():
    set_hf_token()

    if KAGGLE_INPUT_ROOT.exists():
        all_files = sorted(str(p.relative_to(KAGGLE_INPUT_ROOT)) for p in KAGGLE_INPUT_ROOT.rglob("*") if p.is_file())
        print(f"/kaggle/input tree ({len(all_files)} files): {all_files}")
    else:
        print("/kaggle/input does not exist")

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], cwd=str(REPO_DIR))

    data_dir = REPO_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for name in DATA_FILES:
        src = find_input_file(name)
        if src is not None:
            shutil.copy(src, data_dir / name)
            print(f"copied {src} -> {data_dir / name}")
        else:
            print(f"skip (not found anywhere under {KAGGLE_INPUT_ROOT}): {name}")

    cmd = [sys.executable, "-m", "purva.committee.run_judge", "--model", MODEL_NAME, "--quant", QUANT]
    if BENCH_N:
        cmd += ["--bench", str(BENCH_N)]
    else:
        cmd += ["--input", INPUT_FILE]

    run(cmd, cwd=str(REPO_DIR))

    if not BENCH_N:
        out_dir = Path("/kaggle/working/committee")
        out_dir.mkdir(exist_ok=True)
        for f in (data_dir / "committee").glob("*.jsonl"):
            shutil.copy(f, out_dir / f.name)
            print(f"staged output {f} -> {out_dir / f.name}")


if __name__ == "__main__":
    main()
