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

# --- PATCHABLE CONSTANTS: kaggle_run.py rewrites these lines ---
MODEL_NAME = "llama-3.1-8b"
BENCH_N = 0  # 0 = full run over INPUT_FILE; >0 = bench mode, first N pilot sentences, writes no shard
QUANT = "auto"  # auto | awq | bnb
GUIDED = True  # xgrammar guided decoding, constrains output to the judge JSON schema
RATIONALE = True  # include the rationale field in the schema/prompt
CHUNK = 0  # 0 = legacy run over INPUT_FILE below; 1-9 = process only data/chunks/chunk_NN.jsonl
# --- END PATCHABLE CONSTANTS ---

REPO_URL = "https://github.com/shodhx/purva.git"
# Cloned outside /kaggle/working on purpose: anything left in /kaggle/working
# at the end of the run becomes kernel "output" and gets pulled down whole by
# `kaggle kernels output` (including .git internals and __pycache__) if the
# repo lives there. /kaggle/working/committee/ is the only thing we want
# downloadable, staged there explicitly at the end of a full run.
REPO_DIR = Path("/kaggle/tmp/purva")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
INPUT_FILE = "data/label_subset.jsonl"  # legacy full-run input; ignored in bench mode and when CHUNK is set
DATA_FILES = ["corpus_lid.jsonl", "pilot_set.jsonl", "label_subset.jsonl"] + [
    f"chunk_{n:02d}.jsonl" for n in range(1, 10)
]


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


def print_hf_diagnostics():
    """Prints whether an HF token is actually populated in the environment
    and what identity (if any) huggingface_hub resolves it to. This is a
    pure diagnostic — it never raises — so a gated-repo access question
    (e.g. ai4bharat/Airavata for the 'indic' judge) resolves as a side
    effect of the next run's log instead of needing its own investigation."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[hf diagnostic] HF token env var populated: {bool(token)}")
    try:
        from huggingface_hub import whoami

        print(f"[hf diagnostic] huggingface_hub.whoami() -> {whoami()}")
    except Exception as e:
        print(f"[hf diagnostic] huggingface_hub.whoami() failed: {e!r}")


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
    if not (0 <= CHUNK <= 9):
        raise ValueError(f"CHUNK must be 0 (legacy) or 1-9, got {CHUNK}")

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

    # Only meaningful once huggingface_hub is actually installed (pinned in
    # requirements-kaggle.txt), hence run after the pip install above.
    print_hf_diagnostics()

    data_dir = REPO_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    for name in DATA_FILES:
        src = find_input_file(name)
        if src is not None:
            shutil.copy(src, data_dir / name)
            print(f"copied {src} -> {data_dir / name}")
        else:
            print(f"skip (not found anywhere under {KAGGLE_INPUT_ROOT}): {name}")

    # CHUNK>0 targets a single data/chunks/chunk_NN.jsonl shard (chunk-major
    # processing: every judge labels one chunk before any judge moves on to
    # the next) instead of the legacy INPUT_FILE. Its output is staged under
    # a chunk_NN/ subdirectory so merge_shards.py can tell which chunk each
    # shard belongs to and detect gaps per (chunk, judge) pair.
    chunk_tag = f"chunk_{CHUNK:02d}" if CHUNK else None
    run_input_file = f"data/{chunk_tag}.jsonl" if CHUNK else INPUT_FILE
    output_dir = f"data/committee/{chunk_tag}" if CHUNK else "data/committee"

    cmd = [sys.executable, "-m", "purva.committee.run_judge", "--model", MODEL_NAME, "--quant", QUANT]
    cmd += ["--guided"] if GUIDED else ["--no-guided"]
    cmd += ["--rationale"] if RATIONALE else ["--no-rationale"]
    if BENCH_N:
        cmd += ["--bench", str(BENCH_N)]
    else:
        cmd += ["--input", run_input_file, "--output-dir", output_dir]

    run(cmd, cwd=str(REPO_DIR))

    if BENCH_N:
        # No shard in bench mode, but any bench_failures_{model}.jsonl (see
        # run_judge.py) needs to reach /kaggle/working to be downloadable
        # via `kaggle kernels output`.
        failures_src = data_dir / "committee" / f"bench_failures_{MODEL_NAME}.jsonl"
        if failures_src.exists():
            shutil.copy(failures_src, Path("/kaggle/working") / failures_src.name)
            print(f"staged bench failures {failures_src} -> /kaggle/working/{failures_src.name}")
    else:
        src_dir = data_dir / output_dir[len("data/") :]
        out_dir = Path("/kaggle/working/committee") / (chunk_tag or "")
        out_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.jsonl", "*.meta.json"):
            for f in src_dir.glob(pattern):
                shutil.copy(f, out_dir / f.name)
                print(f"staged output {f} -> {out_dir / f.name}")


if __name__ == "__main__":
    main()
