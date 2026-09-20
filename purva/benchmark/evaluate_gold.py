"""Evaluate the four fine-tuned benchmark checkpoints, plus the aggregation
methods themselves, against the first human gold labels for this corpus
(data/gold_unanimous_91.jsonl).

Gold set provenance: 91 items surviving from an original 100-item sample
drawn from data/routing_low_entropy_control.jsonl (the zero/near-zero
Dawid-Skene entropy pool, i.e. items where all five judges' derived votes
agreed) — 7 of the 100 were judged not-Bhojpuri and 1 unclear, both
excluded before this file was produced. This is the EASY end of the
corpus's difficulty distribution by construction (unanimous committee
agreement was the selection criterion): every figure in this report is an
upper bound on true corpus-wide performance, not an estimate of it, and
none of it is extrapolated to the full corpus.

Two things this script checks that were not explicitly asked for, because
they are necessary to interpret the requested numbers honestly:

  1. One gold item's human_label is "mixed", a class outside every model's
     label space (mixed is excluded from the aggregation label space
     entirely — PROTOCOL.md CHANGELOG v1.7). It is excluded from every
     metric in this report (base n=90, not 91), consistently across every
     model, DS, and MV, rather than silently forced into an error bucket
     for some models and not others.
  2. 73 of the 91 gold ids are in data/splits.json's TRAIN split — the
     split muril/indicbert/xlmr were fine-tuned on. Only 10 are in test
     and 8 in dev. This means the headline n=90 figures for those three
     corpus models are dominated by items those models may have
     memorized, not generalized to. This script reports both the full-90
     figures (as requested) and a supplementary test-split-only cut
     (n=10 per corpus model) as the only leak-free signal available for
     them; muril-hindi-baseline was trained on zero Bhojpuri data, so no
     part of the gold set is memorizable for it and this caveat does not
     apply to it.

Checkpoint caveat (recorded here, not re-derived): per
data/benchmark_models/{indicbert,xlmr}.meta.json, both took a final-epoch
(epoch 4) checkpoint past their own dev-loss minima (epoch 2 for indicbert,
epoch 3 for xlmr) — see PROTOCOL.md §8's "TRAIN NOW" checkpoint discipline
in purva/benchmark/train.py. Their figures below understate what those
architectures would achieve with early stopping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CORPUS_LABELS = ("objective", "positive", "negative", "neutral")
HINDI_LABELS = ("negative", "neutral", "positive")
CORPUS_MODEL_KEYS = ("muril", "indicbert", "xlmr")
HINDI_BASELINE_KEY = "muril-hindi-baseline"

N_BOOTSTRAP = 1000
SEED = 42
MIN_CELL_N = 10

INDICBERT_XLMR_CHECKPOINT_NOTE = (
    "final-epoch (epoch 4) checkpoint saved, but dev loss actually bottomed out earlier "
    "(epoch {best_epoch}, dev_loss={best_loss:.4f}) and rose by epoch 4 (dev_loss={final_loss:.4f}) "
    "— see data/benchmark_models/{model}.meta.json. This model's figures below understate what it "
    "would achieve with early stopping."
)


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_gold(path: str) -> tuple[list[dict], dict]:
    rows = load_jsonl(path)
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate ids in gold set"
    mixed = [r for r in rows if r["human_label"] == "mixed"]
    usable = [r for r in rows if r["human_label"] != "mixed"]
    bad = {r["human_label"] for r in usable} - set(CORPUS_LABELS)
    assert not bad, f"unexpected gold label(s) outside the 4-class scheme and not 'mixed': {bad}"
    meta = {
        "n_raw": len(rows),
        "n_mixed_excluded": len(mixed),
        "mixed_ids": [r["id"] for r in mixed],
        "n_used": len(usable),
        "human_label_distribution_used": dict(Counter(r["human_label"] for r in usable)),
        "human_label_distribution_all_91": dict(Counter(r["human_label"] for r in rows)),
        "committee_unanimous_label_distribution_all_91": dict(Counter(r["committee_unanimous_label"] for r in rows)),
        "n_human_vs_committee_unanimous_disagreement": sum(
            1 for r in rows if r["human_label"] != r["committee_unanimous_label"]
        ),
    }
    return usable, meta


def load_splits(path: str) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: set(v) for k, v in raw.items()}


def load_aggregated_for_ids(path: str, ids: set[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    remaining = set(ids)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not remaining:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in remaining:
                out[row["id"]] = {
                    "dawid_skene": row["dawid_skene"]["label"],
                    "majority_vote": row["majority_vote"]["label"],
                }
                remaining.discard(row["id"])
    missing = ids - set(out)
    assert not missing, f"{len(missing)} gold id(s) not found in {path}, e.g. {sorted(missing)[:3]}"
    return out


def predict(model_dir: str, texts: list[str], label_space: tuple[str, ...], max_len: int = 128, batch_size: int = 16) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            logits = model(**enc).logits
            idx = logits.argmax(dim=-1).tolist()
            preds.extend(label_space[j] for j in idx)
    return preds


def accuracy(preds: list[str], gold: list[str]) -> float:
    assert len(preds) == len(gold) and len(gold) > 0
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def macro_f1(preds: list[str], gold: list[str], labels: tuple[str, ...]) -> float:
    assert len(preds) == len(gold) and len(gold) > 0
    per_class = []
    for lbl in labels:
        tp = sum(1 for p, g in zip(preds, gold) if p == lbl and g == lbl)
        fp = sum(1 for p, g in zip(preds, gold) if p == lbl and g != lbl)
        fn = sum(1 for p, g in zip(preds, gold) if p != lbl and g == lbl)
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class.append(f1)
    return float(np.mean(per_class))


def bootstrap_ci(preds: list[str], gold: list[str], metric_fn, labels: tuple[str, ...] | None,
                  n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    n = len(gold)
    point = metric_fn(preds, gold, labels) if labels is not None else metric_fn(preds, gold)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = [preds[i] for i in idx]
        g = [gold[i] for i in idx]
        stats[b] = metric_fn(p, g, labels) if labels is not None else metric_fn(p, g)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": round(float(point), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4), "n": n, "n_bootstrap": n_boot}


def score_block(preds: list[str], gold: list[str], labels: tuple[str, ...]) -> dict:
    return {
        "accuracy": bootstrap_ci(preds, gold, accuracy, None),
        "macro_f1": bootstrap_ci(preds, gold, macro_f1, labels),
    }


def slice_accuracy(preds: list[str], gold: list[str], keys: list[str], min_n: int = MIN_CELL_N) -> dict:
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    out = {}
    for k, idxs in sorted(by_key.items()):
        n = len(idxs)
        if n < min_n:
            out[k] = {"n": n, "accuracy": None, "note": f"n={n} < {min_n}, rate not computed"}
        else:
            p = [preds[i] for i in idxs]
            g = [gold[i] for i in idxs]
            out[k] = {"n": n, "accuracy": round(accuracy(p, g), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold_unanimous_91.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--models-dir", default="data/benchmark_models")
    ap.add_argument("--output-json", default="data/benchmark_results.json")
    ap.add_argument("--output-md", default="data/benchmark_results.md")
    args = ap.parse_args()

    gold_rows, gold_meta = load_gold(args.gold)
    n = len(gold_rows)
    ids = [r["id"] for r in gold_rows]
    texts = [r["cleaned_text"] for r in gold_rows]
    gold4 = [r["human_label"] for r in gold_rows]
    registers = [r["register"] for r in gold_rows]
    text_types = [r["text_type"] for r in gold_rows]
    print(f"gold: {gold_meta['n_raw']} raw, {gold_meta['n_mixed_excluded']} 'mixed' excluded, n={n} used")
    print(f"label distribution (n={n}): {gold_meta['human_label_distribution_used']}")

    splits = load_splits(args.splits)
    split_membership = {}
    for sname, sids in splits.items():
        split_membership[sname] = [i for i in ids if i in sids]
    n_in_any_split = sum(len(v) for v in split_membership.values())
    print(f"gold ids by data/splits.json membership: "
          f"train={len(split_membership.get('train', []))} "
          f"dev={len(split_membership.get('dev', []))} "
          f"test={len(split_membership.get('test', []))} "
          f"(total accounted for: {n_in_any_split}/{n})")
    test_id_set = set(split_membership.get("test", []))
    test_idx = [i for i, iid in enumerate(ids) if iid in test_id_set]
    print(f"test-split-only subset for leak-free corpus-model comparison: n={len(test_idx)}")

    agg = load_aggregated_for_ids(args.aggregated, set(ids))
    ds_preds = [agg[i]["dawid_skene"] for i in ids]
    mv_preds = [agg[i]["majority_vote"] for i in ids]
    n_ds_mv_disagree = sum(1 for a, b in zip(ds_preds, mv_preds) if a != b)

    results: dict = {
        "gold_set": {
            "file": args.gold,
            **gold_meta,
            "note": (
                "Drawn from the unanimous-agreement (near-zero Dawid-Skene entropy) control pool — "
                "the EASY end of the corpus difficulty distribution by construction. Every figure below "
                "is an upper bound on true corpus-wide performance, not an estimate of it."
            ),
        },
        "split_membership_of_gold_ids": {
            k: len(v) for k, v in split_membership.items()
        },
        "leakage_caveat": (
            f"{len(split_membership.get('train', []))} of {n} usable gold items are in data/splits.json's "
            "TRAIN split, which muril/indicbert/xlmr were fine-tuned on. Their full-n figures below are "
            "dominated by items they may have memorized, not generalized to. The 'test_split_only' block "
            f"per corpus model (n={len(test_idx)}) is the only leak-free signal available for them from this "
            "gold set. muril-hindi-baseline was trained on zero Bhojpuri data, so this caveat does not apply "
            "to it — all 90 items are equally unseen."
        ),
        "models": {},
        "aggregation": {},
    }

    # --- corpus models: muril, indicbert, xlmr (4-class) ---
    for key in CORPUS_MODEL_KEYS:
        model_dir = str(Path(args.models_dir) / key)
        print(f"\n=== {key} ===")
        preds = predict(model_dir, texts, CORPUS_LABELS)
        block: dict = {
            "label_space": list(CORPUS_LABELS),
            "full": score_block(preds, gold4, CORPUS_LABELS),
        }
        if test_idx:
            preds_test = [preds[i] for i in test_idx]
            gold_test = [gold4[i] for i in test_idx]
            block["test_split_only"] = score_block(preds_test, gold_test, CORPUS_LABELS)
        block["by_register"] = slice_accuracy(preds, gold4, registers)
        block["by_text_type"] = slice_accuracy(preds, gold4, text_types)
        meta_path = Path(args.models_dir) / f"{key}.meta.json"
        if meta_path.exists():
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            dev_traj = m.get("dev_loss_trajectory", [])
            if dev_traj:
                best = min(dev_traj, key=lambda e: e["eval_loss"])
                final = dev_traj[-1]
                if best["epoch"] != final["epoch"]:
                    block["checkpoint_caveat"] = INDICBERT_XLMR_CHECKPOINT_NOTE.format(
                        best_epoch=best["epoch"], best_loss=best["eval_loss"],
                        final_loss=final["eval_loss"], model=key,
                    )
        results["models"][key] = block
        print(f"  accuracy(full,n={n}): {block['full']['accuracy']}")
        print(f"  macro_f1(full,n={n}): {block['full']['macro_f1']}")

    # --- muril-hindi-baseline (3-class head, no objective) ---
    print(f"\n=== {HINDI_BASELINE_KEY} ===")
    hb_dir = str(Path(args.models_dir) / HINDI_BASELINE_KEY)
    hb_preds = predict(hb_dir, texts, HINDI_LABELS)
    polarity_idx = [i for i, g in enumerate(gold4) if g in HINDI_LABELS]
    hb_preds_polarity = [hb_preds[i] for i in polarity_idx]
    gold_polarity = [gold4[i] for i in polarity_idx]
    hb_block = {
        "label_space": list(HINDI_LABELS),
        "structural_limitation": (
            "3-class head (negative/neutral/positive) with no 'objective' output, trained on Hindi data "
            "with no subjectivity stage in any source. 'polarity_subset' scores it only on gold items it "
            "can structurally address; 'overall' scores it on every non-mixed gold item, counting every "
            "'objective' item as an automatic error because the model has no way to express that label. "
            "These measure DIFFERENT things: polarity_subset asks 'how good is it at polarity when polarity "
            "is the question', overall asks 'can it participate in the full subjectivity+polarity task at "
            "all' — the second is the substantive finding about cross-lingual transfer, since 47/90 gold "
            "items are objective and this baseline is structurally guaranteed wrong on all of them."
        ),
        "polarity_subset": {
            **score_block(hb_preds_polarity, gold_polarity, HINDI_LABELS),
            "n_objective_excluded_from_this_view": n - len(polarity_idx),
        },
        "overall": {
            **score_block(hb_preds, gold4, CORPUS_LABELS),
            "n_objective_counted_as_error": sum(1 for g in gold4 if g == "objective"),
        },
        "by_register_overall": slice_accuracy(hb_preds, gold4, registers),
        "by_text_type_overall": slice_accuracy(hb_preds, gold4, text_types),
    }
    results["models"][HINDI_BASELINE_KEY] = hb_block
    print(f"  polarity_subset accuracy (n={len(polarity_idx)}): {hb_block['polarity_subset']['accuracy']}")
    print(f"  overall accuracy (n={n}, {hb_block['overall']['n_objective_counted_as_error']} objective auto-wrong): {hb_block['overall']['accuracy']}")

    # --- aggregation itself: DS consensus, majority vote ---
    print(f"\n=== aggregation (DS, MV) ===")
    ds_block = {"label_space": list(CORPUS_LABELS), "full": score_block(ds_preds, gold4, CORPUS_LABELS),
                "by_register": slice_accuracy(ds_preds, gold4, registers),
                "by_text_type": slice_accuracy(ds_preds, gold4, text_types)}
    mv_block = {"label_space": list(CORPUS_LABELS), "full": score_block(mv_preds, gold4, CORPUS_LABELS),
                "by_register": slice_accuracy(mv_preds, gold4, registers),
                "by_text_type": slice_accuracy(mv_preds, gold4, text_types)}
    results["aggregation"] = {
        "dawid_skene": ds_block,
        "majority_vote": mv_block,
        "ds_vs_mv_agreement_note": (
            f"DS and MV consensus labels differ on {n_ds_mv_disagree}/{n} gold items on this "
            "unanimous-judge-agreement subset. On a set selected FOR unanimous agreement, the two methods "
            "coincide by construction (all 5 votes identical -> both trivially pick that label) — this is "
            "not a finding about DS vs MV in general, it is a property of how this subset was sampled."
        ),
        "n_ds_vs_mv_disagreement": n_ds_mv_disagree,
    }
    print(f"  DS accuracy(full,n={n}): {ds_block['full']['accuracy']}")
    print(f"  MV accuracy(full,n={n}): {mv_block['full']['accuracy']}")
    print(f"  DS vs MV disagree: {n_ds_mv_disagree}/{n}")

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_json}")

    write_markdown(results, Path(args.output_md))
    print(f"wrote {args.output_md}")


def fmt_ci(block: dict) -> str:
    return f"{block['point']:.3f} [{block['ci95_lo']:.3f}, {block['ci95_hi']:.3f}] (n={block['n']}, {block['n_bootstrap']} bootstrap resamples)"


def fmt_slice_table(slices: dict) -> str:
    lines = ["| cell | n | accuracy |", "|---|---|---|"]
    for k, v in slices.items():
        if v["accuracy"] is None:
            lines.append(f"| {k} | {v['n']} | *n<{MIN_CELL_N}, not computed* |")
        else:
            lines.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} |")
    return "\n".join(lines)


def write_markdown(results: dict, path: Path) -> None:
    gm = results["gold_set"]
    lines = []
    lines.append("# Benchmark model evaluation against human gold (unanimous-agreement control sample)\n")
    lines.append(f"Gold file: `{gm['file']}`. {gm['n_raw']} items scored by a human annotator after LID "
                 f"screening (7/100 not-Bhojpuri, 1/100 unclear, excluded upstream of this file). "
                 f"{gm['n_mixed_excluded']} further item(s) labeled 'mixed' excluded here — outside every "
                 f"model's label space. **n={gm['n_used']} used for every metric below.**\n")
    lines.append(f"> {gm['note']}\n")
    lines.append(f"Human label distribution (n={gm['n_used']}): `{gm['human_label_distribution_used']}`\n")
    lines.append(f"Human annotator disagreed with the judges' own unanimous call on "
                 f"{gm['n_human_vs_committee_unanimous_disagreement']}/{gm['n_raw']} of the original 91 items "
                 "— worth keeping in mind when calling this subset the pipeline's \"easy end\": even here, "
                 "human/model-committee disagreement is not zero.\n")

    lines.append("## Leakage caveat (checked, not requested — but load-bearing)\n")
    lines.append(f"{results['leakage_caveat']}\n")
    sm = results["split_membership_of_gold_ids"]
    lines.append(f"Split membership of the {gm['n_used']} usable gold ids: "
                 f"train={sm.get('train', 0)}, dev={sm.get('dev', 0)}, test={sm.get('test', 0)}.\n")

    labels_str = ", ".join(CORPUS_LABELS)
    lines.append("## Fine-tuned model results\n")
    lines.append(f"4-class label space: {labels_str}. Accuracy and macro-F1 with bootstrap 95% CI "
                 "(1,000 resamples, percentile method). With n=90 (and n=10 for the test-only cut) these "
                 "intervals are wide — reported as such, not as point estimates alone.\n")

    for key in CORPUS_MODEL_KEYS:
        b = results["models"][key]
        lines.append(f"### {key}\n")
        lines.append(f"- Full gold set (n={b['full']['accuracy']['n']}): "
                     f"accuracy {fmt_ci(b['full']['accuracy'])}; macro-F1 {fmt_ci(b['full']['macro_f1'])}")
        if "test_split_only" in b:
            t = b["test_split_only"]
            lines.append(f"- **Test-split-only (leak-free), n={t['accuracy']['n']}**: "
                         f"accuracy {fmt_ci(t['accuracy'])}; macro-F1 {fmt_ci(t['macro_f1'])}")
        if "checkpoint_caveat" in b:
            lines.append(f"- **Checkpoint caveat**: {b['checkpoint_caveat']}")
        lines.append("\nBy register:\n")
        lines.append(fmt_slice_table(b["by_register"]))
        lines.append("\nBy text_type:\n")
        lines.append(fmt_slice_table(b["by_text_type"]))
        lines.append("")

    hb = results["models"][HINDI_BASELINE_KEY]
    lines.append(f"### {HINDI_BASELINE_KEY}\n")
    lines.append(f"3-class label space: {', '.join(HINDI_LABELS)}. {hb['structural_limitation']}\n")
    p = hb["polarity_subset"]
    lines.append(f"- **Polarity subset** (gold restricted to positive/negative/neutral, n={p['accuracy']['n']}, "
                 f"{p['n_objective_excluded_from_this_view']} objective items excluded from this view): "
                 f"accuracy {fmt_ci(p['accuracy'])}; macro-F1 {fmt_ci(p['macro_f1'])}")
    o = hb["overall"]
    lines.append(f"- **Overall** (n={o['accuracy']['n']}, {o['n_objective_counted_as_error']} objective items "
                 "counted as automatic errors — the model cannot express that label): "
                 f"accuracy {fmt_ci(o['accuracy'])}; macro-F1 {fmt_ci(o['macro_f1'])}")
    lines.append("\nBy register (overall figure):\n")
    lines.append(fmt_slice_table(hb["by_register_overall"]))
    lines.append("\nBy text_type (overall figure):\n")
    lines.append(fmt_slice_table(hb["by_text_type_overall"]))
    lines.append("")

    lines.append("## Aggregation itself (DS consensus, majority vote) vs gold\n")
    agg = results["aggregation"]
    lines.append(f"{agg['ds_vs_mv_agreement_note']}\n")
    for name, key in (("Dawid-Skene consensus", "dawid_skene"), ("Majority vote", "majority_vote")):
        b = agg[key]
        lines.append(f"### {name}\n")
        lines.append(f"- Full gold set (n={b['full']['accuracy']['n']}): "
                     f"accuracy {fmt_ci(b['full']['accuracy'])}; macro-F1 {fmt_ci(b['full']['macro_f1'])}")
        lines.append("\nBy register:\n")
        lines.append(fmt_slice_table(b["by_register"]))
        lines.append("\nBy text_type:\n")
        lines.append(fmt_slice_table(b["by_text_type"]))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
