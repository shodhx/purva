"""Fine-tune benchmark sequence classifiers on silver (Dawid-Skene consensus)
labels (PROTOCOL.md §8).

TRAIN NOW, EVALUATE LATER: human gold labels for the Bhojpuri corpus do not
exist yet (Phase-6 human annotation is still pending), so this script only
produces trained checkpoints + provenance sidecars. It never computes or
prints an accuracy figure, and it never touches data/splits.json's test
split — see assert_no_test_leakage() below, which is a hard failure, not a
warning.

Two training sources (purva/benchmark/models.py's BenchmarkModelSpec.training_source):

  "corpus" — muril, indicbert, xlmr. Training data is the standard
  (unstratified) Dawid-Skene four-class consensus label
  (data/purva_aggregated.jsonl's "dawid_skene"."label" — the validated
  primary method per data/purva_aggregated.meta.json), restricted to
  data/splits.json's "train" ids. The "dev" ids are tokenized and evaluated
  every epoch purely to report a loss trajectory; they are NEVER used for
  checkpoint selection (load_best_model_at_end=False, save_strategy="no"
  during training, and the final-epoch weights are what gets saved — see
  PROTOCOL.md §8 and the task instruction this script was written against:
  dev contamination of the eventual gold evaluation is exactly what must be
  avoided here).

  "hindi_baseline" — muril-hindi-baseline. Training data is a pooled
  multi-source Hindi sentiment set (default data/hindi_baseline_pool.jsonl,
  built by purva/benchmark/build_hindi_baseline_pool.py from OdiaGenAI's set
  plus additional public HF datasets — see that script's docstring for the
  full source list, sizes, and licences), with NO Bhojpuri data at all —
  this is the ablation testing whether transfer from a high-resource
  neighbour language would have sufficed on its own. The gold label space
  for every pooled source is 3-class polarity with no objective/subjectivity
  stage, so this model trains a 3-class head over HINDI_LABELS
  (negative/neutral/positive) rather than forcing a 4-class head with a
  permanently dead "objective" class — at Bhojpuri evaluation time (a later,
  separate task), Bhojpuri "objective" items are simply items this baseline
  structurally cannot predict, not a bug.

  That pooled file carries no frozen dev split of its own (data/splits.json
  is Bhojpuri-specific), so a 10%, gold-label-stratified hold-out (seed 42)
  is carved out here purely for loss monitoring — a deliberate, disclosed
  deviation from the "corpus" flow; see build_hindi_baseline_data()'s
  returned meta and the printed report. Unlike "corpus", this hold-out MAY
  drive an early-stopping decision on epoch count (--early-stopping-patience)
  when requested: the Bhojpuri gold test set this ablation will eventually be
  scored against is untouched by this file, so there is no leakage risk in
  letting Hindi-side dev loss pick when to stop — see main()'s early-stopping
  block and the meta's "checkpoint_selection" field for what was actually
  used on a given run.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..aggregate._common import LABELS as _DS_LABELS_5
from .models import REGISTRY

# The four-class fallback label space (PROTOCOL.md CHANGELOG v1.7) — the
# order data/purva_aggregated.meta.json confirms as the primary consensus's
# actual label space ("label_space_path": "four_class_fallback"). Sliced
# from the 5-class tuple rather than redeclared, so this can never silently
# drift from purva/aggregate/_common.py's canonical order. Used only by the
# "corpus" training source.
LABELS = _DS_LABELS_5[:4]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

# The "hindi_baseline" training source's label space: every pooled Hindi
# sentiment source is 3-class polarity with no subjectivity stage, so this
# is a 3-class head, not the 4-class LABELS above (an "objective" output
# that never sees a training example is a dead class, not a fair transfer
# test — see the module docstring and build_hindi_baseline_pool.py).
HINDI_GOLD_LABELS = ("negative", "neutral", "positive")
HINDI_LABELS = HINDI_GOLD_LABELS
HINDI_LABEL_TO_IDX = {label: i for i, label in enumerate(HINDI_LABELS)}

DEFAULT_OUTPUT_DIR = "data/benchmark_models"
DEFAULT_HINDI_SET = "data/hindi_baseline_pool.jsonl"


def load_splits(path: str) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_aggregated_ds_labels(path: str) -> dict[str, str]:
    """{id: standard Dawid-Skene four-class label}, skipping any row whose
    'dawid_skene' block is absent (there is none in practice — dawid_skene
    is written for every row by run_aggregation.py — but this fails loudly
    rather than silently if that ever changes)."""
    label_by_id: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            block = row.get("dawid_skene")
            if block is None:
                continue
            label_by_id[row["id"]] = block["label"]
    return label_by_id


def build_corpus_data(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], list[int], list[str], dict]:
    """Returns (train_texts, train_labels, dev_texts, dev_labels, train_ids, meta)."""
    splits = load_splits(args.splits)
    train_ids, dev_ids, test_ids = set(splits["train"]), set(splits["dev"]), set(splits["test"])
    assert train_ids.isdisjoint(test_ids), "data/splits.json itself has train/test overlap — corrupt split file"
    assert train_ids.isdisjoint(dev_ids), "data/splits.json itself has train/dev overlap — corrupt split file"

    label_by_id = load_aggregated_ds_labels(args.aggregated)
    master = pd.read_parquet(args.master, columns=["id", "cleaned_text"])
    text_by_id = dict(zip(master["id"], master["cleaned_text"]))

    def build(ids: set[str]) -> tuple[list[str], list[str], list[int]]:
        ordered = sorted(ids)
        missing_label = [i for i in ordered if i not in label_by_id]
        missing_text = [i for i in ordered if i not in text_by_id]
        assert not missing_label, f"{len(missing_label)} id(s) missing a dawid_skene consensus label, e.g. {missing_label[:3]}"
        assert not missing_text, f"{len(missing_text)} id(s) missing cleaned_text in {args.master}, e.g. {missing_text[:3]}"
        texts = [text_by_id[i] for i in ordered]
        labels = [LABEL_TO_IDX[label_by_id[i]] for i in ordered]
        return ordered, texts, labels

    train_ids_l, train_texts, train_labels = build(train_ids)
    dev_ids_l, dev_texts, dev_labels = build(dev_ids)

    if args.limit:
        train_ids_l = train_ids_l[: args.limit]
        train_texts = train_texts[: args.limit]
        train_labels = train_labels[: args.limit]

    meta = {
        "training_label_source": (
            "standard (unstratified) Dawid-Skene four-class consensus label, "
            f"{args.aggregated}['dawid_skene']['label'], restricted to {args.splits}['train']"
        ),
        "splits_file": str(args.splits),
        "n_train_ids_total": len(train_ids),
        "n_dev_ids_total": len(dev_ids),
        "n_test_ids_total": len(test_ids),
        "dev_usage": "loss monitoring only, every epoch — never used for checkpoint selection",
        "checkpoint_selection": "final-epoch checkpoint (no best-on-dev selection)",
        "test_split_touched": False,
        "deviation_notes": [],
    }
    return train_texts, train_labels, dev_texts, dev_labels, train_ids_l, meta


def build_hindi_baseline_data(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], list[int], list[str], dict]:
    lines = [x for x in Path(args.hindi_set).read_text(encoding="utf-8").splitlines() if x.strip()]
    header = json.loads(lines[0])
    rows = [json.loads(x) for x in lines[1:]]

    ids = [r["id"] for r in rows]
    texts = [r["cleaned_text"] for r in rows]
    gold = [r["gold_label"] for r in rows]
    assert set(gold) <= set(HINDI_GOLD_LABELS), f"unexpected gold label(s): {set(gold) - set(HINDI_GOLD_LABELS)}"
    labels = [HINDI_LABEL_TO_IDX[g] for g in gold]

    # 10% stratified (on gold_label) hold-out, seed 42, for loss monitoring
    # only — see module docstring. Deterministic: ids are already in the
    # file's stable sampling order (make_hindi_validation_set.py), so the
    # per-label index lists below are stable before the seeded shuffle.
    rng = random.Random(42)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gold):
        by_label[g].append(i)
    dev_idx: set[int] = set()
    for g in sorted(by_label):
        idxs = by_label[g][:]
        rng.shuffle(idxs)
        k = max(1, round(0.1 * len(idxs)))
        dev_idx.update(idxs[:k])

    train_idx = [i for i in range(len(rows)) if i not in dev_idx]
    dev_idx_sorted = sorted(dev_idx)

    train_ids_l = [ids[i] for i in train_idx]
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    dev_texts = [texts[i] for i in dev_idx_sorted]
    dev_labels = [labels[i] for i in dev_idx_sorted]

    if args.limit:
        train_ids_l = train_ids_l[: args.limit]
        train_texts = train_texts[: args.limit]
        train_labels = train_labels[: args.limit]

    meta = {
        "training_label_source": (
            "third-party Hindi gold labels pooled from multiple public HF datasets "
            f"(see hindi_set_header for the full per-source breakdown; {header.get('_n_sources', '?')} source(s), "
            f"{header.get('_pooled_total', len(rows))} items after cross-source dedup), trained as a 3-class "
            "head over HINDI_LABELS (negative/neutral/positive) — no 'objective' class, since none of the "
            "pooled sources have a subjectivity stage — and NO Bhojpuri data of any kind used in this run"
        ),
        "hindi_set_file": str(args.hindi_set),
        "hindi_set_header": header,
        "n_hindi_items_total": len(rows),
        "dev_usage": "loss monitoring only, every epoch" + (
            " — drives early stopping on epoch count (see checkpoint_selection), but never picks a checkpoint "
            "beyond the epoch that stopping rule already selects"
            if args.early_stopping_patience else " — never used for checkpoint selection"
        ),
        "checkpoint_selection": (
            f"best-epoch by dev loss, early stopping patience={args.early_stopping_patience} "
            f"(max_epochs={args.epochs})"
            if args.early_stopping_patience else "final-epoch checkpoint (no best-on-dev selection)"
        ),
        "test_split_touched": "n/a — this ablation uses no Bhojpuri data, so data/splits.json's test split is not applicable",
        "deviation_notes": [
            "No frozen dev split exists for this pooled dataset (data/splits.json is Bhojpuri-specific). A 10% "
            "hold-out, stratified on gold_label with seed 42, was carved from the pooled file purely for loss "
            "monitoring.",
            "Every pooled source is 3-class polarity (negative/neutral/positive) with no subjectivity stage. "
            "Unlike the earlier single-source run, this is trained as a genuine 3-class head (HINDI_LABELS), "
            "not a 4-class head with a permanently dead 'objective' class — see the module docstring. At "
            "Bhojpuri evaluation time (a separate, later task), Bhojpuri gold 'objective' items are items "
            "this baseline structurally cannot predict; that is a disclosed limitation of the transfer "
            "approach being tested, not a bug in this run.",
            "Unlike the 'corpus' training source's fixed-epoch, never-select-on-dev discipline, this run's "
            "epoch count IS chosen from the Hindi-side dev loss curve (early stopping) when requested via "
            "--early-stopping-patience — safe here because this dev split has no relationship to the "
            "Bhojpuri gold test set this ablation will eventually be scored against.",
        ],
    }
    return train_texts, train_labels, dev_texts, dev_labels, train_ids_l, meta


def assert_no_test_leakage(train_ids: list[str], splits_path: str | None) -> None:
    """Hard failure, not a warning, if any training id is in data/splits.json's
    test split. For the hindi_baseline source there is no Bhojpuri splits
    file at all, so this is a no-op by construction (nothing to leak into)."""
    if splits_path is None:
        print("test-leakage check: n/a (no Bhojpuri split file involved in this run)")
        return
    test_ids = set(load_splits(splits_path)["test"])
    leaked = set(train_ids) & test_ids
    assert not leaked, f"TEST LEAKAGE: {len(leaked)} training id(s) are in the test split, e.g. {sorted(leaked)[:3]}"
    print(f"test-leakage check PASSED: 0 of {len(train_ids)} training ids found in the "
          f"{len(test_ids)}-item test split ({splits_path})")


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--limit", type=int, default=0, help="debug only: cap training set size, e.g. for a smoke test")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--hindi-set", default=DEFAULT_HINDI_SET)
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                     help="hindi_baseline only: >0 stops training when dev loss hasn't improved for this many "
                          "eval epochs, and saves the best (not final) epoch's weights; --epochs then acts as a "
                          "max-epochs cap. 0 (default) preserves the fixed-epoch, never-select-on-dev behaviour.")
    args = ap.parse_args()

    model_key = args.model
    spec = REGISTRY[model_key]

    if args.early_stopping_patience and spec.training_source != "hindi_baseline":
        raise SystemExit("--early-stopping-patience is only supported for training_source='hindi_baseline' "
                          "— the 'corpus' models must keep the fixed-epoch, never-select-on-dev discipline")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if spec.training_source == "corpus":
        train_texts, train_labels, dev_texts, dev_labels, train_ids, data_meta = build_corpus_data(args)
        splits_path_for_check = args.splits
        label_space = LABELS
    elif spec.training_source == "hindi_baseline":
        train_texts, train_labels, dev_texts, dev_labels, train_ids, data_meta = build_hindi_baseline_data(args)
        splits_path_for_check = None
        label_space = HINDI_LABELS
    else:
        raise SystemExit(f"unknown training_source {spec.training_source!r} for model {model_key!r}")

    assert_no_test_leakage(train_ids, splits_path_for_check)

    print(f"model={model_key} repo_id={spec.repo_id} revision={spec.revision} label_space={label_space}")
    print(f"train_set_size={len(train_texts)} dev_set_size={len(dev_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.repo_id, revision=spec.revision, num_labels=len(label_space)
    )

    train_enc = tokenizer(train_texts, truncation=True, max_length=args.max_len)
    train_ds = ClassificationDataset(train_enc, train_labels)
    dev_ds = None
    if dev_texts:
        dev_enc = tokenizer(dev_texts, truncation=True, max_length=args.max_len)
        dev_ds = ClassificationDataset(dev_enc, dev_labels)

    collator = DataCollatorWithPadding(tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(args.fp16 and device == "cuda")
    if args.fp16 and not use_fp16:
        print("fp16 requested but no CUDA device is available — training in fp32 on CPU instead")

    use_early_stopping = args.early_stopping_patience > 0
    if use_early_stopping:
        assert dev_ds is not None, "--early-stopping-patience requires a dev set to monitor"

    scratch_dir = Path(tempfile.mkdtemp(prefix=f"purva_bench_trainer_{model_key}_"))
    training_args = TrainingArguments(
        output_dir=str(scratch_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        fp16=use_fp16,
        eval_strategy="epoch" if dev_ds is not None else "no",
        logging_strategy="epoch",
        # "corpus" (and hindi_baseline with early stopping off): final-epoch
        # weights only, saved manually below, dev never drives selection.
        # hindi_baseline WITH early stopping: dev loss IS allowed to pick the
        # stopping epoch (see module docstring) — save_strategy must then
        # match eval_strategy so load_best_model_at_end can restore it.
        save_strategy="epoch" if use_early_stopping else "no",
        save_total_limit=(args.early_stopping_patience + 1) if use_early_stopping else None,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        report_to=[],
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if use_early_stopping else []
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    wall_clock_seconds = time.time() - t0

    train_loss_trajectory = [
        {"epoch": e["epoch"], "loss": e["loss"]} for e in trainer.state.log_history if "loss" in e
    ]
    dev_loss_trajectory = [
        {"epoch": e["epoch"], "eval_loss": e["eval_loss"]} for e in trainer.state.log_history if "eval_loss" in e
    ]
    chosen_epoch = None
    if use_early_stopping and dev_loss_trajectory:
        chosen_epoch = min(dev_loss_trajectory, key=lambda e: e["eval_loss"])["epoch"]
        print(f"early stopping: ran {len(dev_loss_trajectory)} epoch(s), best dev loss at epoch {chosen_epoch} "
              f"(patience={args.early_stopping_patience}, max_epochs={args.epochs})")

    final_dir = Path(args.output_dir) / model_key
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved {'best-epoch (early stopping)' if use_early_stopping else 'final-epoch'} checkpoint to {final_dir}")

    meta = {
        "model_key": model_key,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "seed": args.seed,
            "fp16": use_fp16,
        },
        "device": device,
        "train_set_size": len(train_labels),
        "dev_set_size": len(dev_labels),
        "label_space": list(label_space),
        "train_loss_trajectory": train_loss_trajectory,
        "dev_loss_trajectory": dev_loss_trajectory,
        "final_train_loss": train_loss_trajectory[-1]["loss"] if train_loss_trajectory else None,
        "final_dev_loss": dev_loss_trajectory[-1]["eval_loss"] if dev_loss_trajectory else None,
        "early_stopping_patience": args.early_stopping_patience,
        "stopped_epoch": dev_loss_trajectory[-1]["epoch"] if dev_loss_trajectory else None,
        "chosen_epoch": chosen_epoch,
        "wall_clock_seconds": round(wall_clock_seconds, 1),
        "checkpoint_dir": str(final_dir),
        **data_meta,
    }
    meta_path = Path(args.output_dir) / f"{model_key}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")

    print("\n=== summary ===")
    print(f"train_set_size: {meta['train_set_size']}")
    print(f"dev_set_size: {meta['dev_set_size']}")
    print(f"train_loss_trajectory: {train_loss_trajectory}")
    print(f"dev_loss_trajectory: {dev_loss_trajectory}")
    print(f"wall_clock_seconds: {meta['wall_clock_seconds']}")
    print(f"test split touched: {data_meta.get('test_split_touched')}")


if __name__ == "__main__":
    main()
