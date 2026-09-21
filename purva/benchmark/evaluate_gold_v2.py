"""Evaluate benchmark models against the clean, leak-free test gold set
(data/gold_test_120.jsonl) — SUPERSEDES purva/benchmark/evaluate_gold.py and
data/benchmark_results.md, which are INVALID for muril/indicbert/xlmr:
73 of that gold set's 90 usable items were in those models' own TRAIN
split, so those figures are dominated by memorized items, not generalized
performance. data/benchmark_results.md must not be cited for those three
models. (muril-hindi-baseline's earlier figures were leak-free already,
since it was trained on zero Bhojpuri data — but this script re-derives
them on the new gold set anyway, since the two gold sets are different
samples with different difficulty composition.)

This run also follows the MV-over-DS aggregator decision: muril/indicbert/
xlmr here are the MAJORITY-VOTE-trained checkpoints (data/benchmark_models/
{model}/), not the earlier Dawid-Skene-trained ones (now archived under
data/benchmark_models/{model}_ds_silver/). Both are evaluated here so the
DS-trained-vs-MV-trained comparison is itself reportable.

Gold set: data/gold_test_120.jsonl, 111 usable items (already excludes 9
originally-unusable items from the underlying 120-item draw), difficulty-
balanced. 3 further items are labeled 'mixed', outside every model's label
space, and excluded from every metric (base n=108) — consistent with
evaluate_gold.py's convention. Verified here (not just asserted): all 111
gold ids fall in data/splits.json's TEST split, zero overlap with train/dev.

On this gold set, unlike the unanimous-only set evaluate_gold.py used, DS
and MV consensus do NOT trivially coincide (these are difficulty-balanced,
contested-inclusive items) — this is the clean head-to-head the earlier
evaluation could not provide.

Entropy-half split: items are ranked by Dawid-Skene posterior entropy
(data/purva_aggregated.jsonl['dawid_skene']['entropy_norm'] — the entropy
measure used throughout this project for difficulty stratification, e.g.
the routing sets) and split into the 60 lowest-entropy ("low") and 51
highest-entropy ("high") items, exactly reproducing the 60/51 split this
task specifies; ties are broken by id for a fully deterministic split.
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
DS_SILVER_SUFFIX = "_ds_silver"
HINDI_BASELINE_KEY = "muril-hindi-baseline"

N_BOOTSTRAP = 1000
SEED = 42
MIN_CELL_N = 10
N_LOW_ENTROPY = 60


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
        "human_label_distribution_all_111": dict(Counter(r["human_label"] for r in rows)),
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
                    "ds_entropy": row["dawid_skene"]["entropy_norm"],
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


def bootstrap_diff_ci(preds_a: list[str], preds_b: list[str], gold: list[str], metric_fn,
                       labels: tuple[str, ...] | None, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Paired bootstrap: same resampled item indices used for both a and b on every
    resample, so the difference's CI reflects correlated performance on shared items."""
    n = len(gold)
    point_a = metric_fn(preds_a, gold, labels) if labels is not None else metric_fn(preds_a, gold)
    point_b = metric_fn(preds_b, gold, labels) if labels is not None else metric_fn(preds_b, gold)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ga = [gold[i] for i in idx]
        pa = [preds_a[i] for i in idx]
        pb = [preds_b[i] for i in idx]
        sa = metric_fn(pa, ga, labels) if labels is not None else metric_fn(pa, ga)
        sb = metric_fn(pb, ga, labels) if labels is not None else metric_fn(pb, ga)
        diffs[b] = sa - sb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "a": round(float(point_a), 4), "b": round(float(point_b), 4),
        "diff_a_minus_b": round(float(point_a - point_b), 4),
        "diff_ci95_lo": round(float(lo), 4), "diff_ci95_hi": round(float(hi), 4),
        "n": n, "n_bootstrap": n_boot,
    }


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


def confusion_matrix(preds: list[str], gold: list[str], labels: tuple[str, ...]) -> dict:
    counts = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, preds):
        counts[g][p] += 1
    per_class_recall = {g: (counts[g][g] / s if (s := sum(counts[g].values())) else None) for g in labels}
    return {"labels": list(labels), "counts_gold_row_pred_col": counts, "per_gold_recall": per_class_recall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold_test_120.jsonl")
    ap.add_argument("--splits", default="data/splits.json")
    ap.add_argument("--aggregated", default="data/purva_aggregated.jsonl")
    ap.add_argument("--models-dir", default="data/benchmark_models")
    ap.add_argument("--output-json", default="data/benchmark_results_v2.json")
    ap.add_argument("--output-md", default="data/benchmark_results_v2.md")
    args = ap.parse_args()

    gold_rows, gold_meta = load_gold(args.gold)
    n = len(gold_rows)
    ids = [r["id"] for r in gold_rows]
    texts = [r["cleaned_text"] for r in gold_rows]
    gold4 = [r["human_label"] for r in gold_rows]
    registers = [r["register"] for r in gold_rows]
    print(f"gold: {gold_meta['n_raw']} raw, {gold_meta['n_mixed_excluded']} 'mixed' excluded, n={n} used")
    print(f"label distribution (n={n}): {gold_meta['human_label_distribution_used']}")

    all_ids_incl_mixed = [r["id"] for r in load_jsonl(args.gold)]
    splits = load_splits(args.splits)
    for sname in ("train", "dev", "test"):
        overlap = len(set(all_ids_incl_mixed) & splits[sname])
        print(f"gold ids (incl. mixed) in splits.json['{sname}']: {overlap}")
    leak_free = set(all_ids_incl_mixed) <= splits["test"]
    print(f"ALL {len(all_ids_incl_mixed)} gold ids (incl. mixed) are in the TEST split: {leak_free}")
    assert leak_free, "gold_test_120.jsonl is NOT leak-free against data/splits.json — aborting"

    agg = load_aggregated_for_ids(args.aggregated, set(all_ids_incl_mixed))
    ds_preds_all = [agg[i]["dawid_skene"] for i in ids]
    mv_preds_all = [agg[i]["majority_vote"] for i in ids]
    n_ds_mv_disagree = sum(1 for a, b in zip(ds_preds_all, mv_preds_all) if a != b)
    print(f"DS vs MV disagree on gold (n={n}): {n_ds_mv_disagree}/{n}")

    order = sorted(all_ids_incl_mixed, key=lambda i: (agg[i]["ds_entropy"], i))
    low_ids = set(order[:N_LOW_ENTROPY])
    high_ids = set(order[N_LOW_ENTROPY:])
    entropy_half = ["low" if i in low_ids else "high" for i in ids]
    print(f"entropy halves (incl. mixed in the split, excl. from scoring): "
          f"low={len(low_ids)} high={len(high_ids)}")
    print(f"entropy halves among the {n} usable (non-mixed) items: "
          f"low={sum(1 for h in entropy_half if h == 'low')} high={sum(1 for h in entropy_half if h == 'high')}")

    results: dict = {
        "gold_set": {
            "file": args.gold,
            **gold_meta,
            "note": "Difficulty-balanced test-split gold, verified zero training-set overlap. SUPERSEDES data/benchmark_results.md (73/90 leaky) for muril/indicbert/xlmr.",
        },
        "leakage_check": {
            "all_gold_ids_in_test_split": leak_free,
            "note": "Verified programmatically, not just asserted from the task description.",
        },
        "entropy_split": {
            "method": "sorted ascending by data/purva_aggregated.jsonl['dawid_skene']['entropy_norm'], ties broken by id; first 60 = low, remaining 51 = high (over all 111 items incl. mixed)",
            "n_low_incl_mixed": len(low_ids),
            "n_high_incl_mixed": len(high_ids),
            "n_low_usable": sum(1 for h in entropy_half if h == "low"),
            "n_high_usable": sum(1 for h in entropy_half if h == "high"),
        },
        "models": {},
        "aggregation": {},
        "ds_trained_vs_mv_trained": {},
    }

    # --- corpus models: muril, indicbert, xlmr (MV-trained, primary) ---
    corpus_preds: dict[str, list[str]] = {}
    for key in CORPUS_MODEL_KEYS:
        model_dir = str(Path(args.models_dir) / key)
        print(f"\n=== {key} (MV-trained) ===")
        preds = predict(model_dir, texts, CORPUS_LABELS)
        corpus_preds[key] = preds
        block: dict = {
            "label_space": list(CORPUS_LABELS),
            "checkpoint": model_dir,
            "full": score_block(preds, gold4, CORPUS_LABELS),
            "by_register": slice_accuracy(preds, gold4, registers),
            "by_entropy_half": slice_accuracy(preds, gold4, entropy_half),
        }
        results["models"][key] = block
        print(f"  accuracy(n={n}): {block['full']['accuracy']}")
        print(f"  macro_f1(n={n}): {block['full']['macro_f1']}")

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
            "3-class head (negative/neutral/positive) with no 'objective' output. 'polarity_subset' scores it "
            "only on gold items it can structurally address; 'overall' scores it on every non-mixed gold item, "
            "counting every 'objective' item as an automatic error. These measure different things."
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
        "by_entropy_half_overall": slice_accuracy(hb_preds, gold4, entropy_half),
    }
    results["models"][HINDI_BASELINE_KEY] = hb_block
    print(f"  polarity_subset accuracy (n={len(polarity_idx)}): {hb_block['polarity_subset']['accuracy']}")
    print(f"  overall accuracy (n={n}): {hb_block['overall']['accuracy']}")

    # --- aggregation itself: DS consensus, MV consensus, head-to-head ---
    print(f"\n=== aggregation (DS vs MV, clean head-to-head) ===")
    ds_block = {"label_space": list(CORPUS_LABELS), "full": score_block(ds_preds_all, gold4, CORPUS_LABELS),
                "by_register": slice_accuracy(ds_preds_all, gold4, registers),
                "by_entropy_half": slice_accuracy(ds_preds_all, gold4, entropy_half),
                "confusion_matrix": confusion_matrix(ds_preds_all, gold4, CORPUS_LABELS)}
    mv_block = {"label_space": list(CORPUS_LABELS), "full": score_block(mv_preds_all, gold4, CORPUS_LABELS),
                "by_register": slice_accuracy(mv_preds_all, gold4, registers),
                "by_entropy_half": slice_accuracy(mv_preds_all, gold4, entropy_half),
                "confusion_matrix": confusion_matrix(mv_preds_all, gold4, CORPUS_LABELS)}
    acc_diff = bootstrap_diff_ci(mv_preds_all, ds_preds_all, gold4, accuracy, None)
    f1_diff = bootstrap_diff_ci(mv_preds_all, ds_preds_all, gold4, macro_f1, CORPUS_LABELS)
    results["aggregation"] = {
        "dawid_skene": ds_block,
        "majority_vote": mv_block,
        "n_ds_vs_mv_disagreement": n_ds_mv_disagree,
        "mv_minus_ds_accuracy": acc_diff,
        "mv_minus_ds_macro_f1": f1_diff,
        "note": "On this difficulty-balanced (contested-inclusive) gold set, DS and MV genuinely differ — this is the clean head-to-head the unanimous-only gold set (data/gold_unanimous_91.jsonl) could not provide.",
    }
    print(f"  DS accuracy: {ds_block['full']['accuracy']}")
    print(f"  MV accuracy: {mv_block['full']['accuracy']}")
    print(f"  MV-DS accuracy diff: {acc_diff}")

    # --- DS-trained vs MV-trained models, if cheap (it is: CPU inference, n=108) ---
    print(f"\n=== DS-trained vs MV-trained model comparison ===")
    for key in CORPUS_MODEL_KEYS:
        ds_ckpt_dir = str(Path(args.models_dir) / f"{key}{DS_SILVER_SUFFIX}")
        print(f"  predicting with {key}{DS_SILVER_SUFFIX} ...")
        ds_trained_preds = predict(ds_ckpt_dir, texts, CORPUS_LABELS)
        acc_d = bootstrap_diff_ci(corpus_preds[key], ds_trained_preds, gold4, accuracy, None)
        f1_d = bootstrap_diff_ci(corpus_preds[key], ds_trained_preds, gold4, macro_f1, CORPUS_LABELS)
        results["ds_trained_vs_mv_trained"][key] = {
            "mv_trained_checkpoint": str(Path(args.models_dir) / key),
            "ds_trained_checkpoint": ds_ckpt_dir,
            "mv_minus_ds_accuracy": acc_d,
            "mv_minus_ds_macro_f1": f1_d,
        }
        print(f"  {key}: MV-trained acc={acc_d['a']} DS-trained acc={acc_d['b']} diff={acc_d['diff_a_minus_b']} "
              f"CI=[{acc_d['diff_ci95_lo']}, {acc_d['diff_ci95_hi']}]")

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out_json}")

    write_markdown(results, Path(args.output_md))
    print(f"wrote {args.output_md}")


def fmt_ci(block: dict) -> str:
    return f"{block['point']:.3f} [{block['ci95_lo']:.3f}, {block['ci95_hi']:.3f}] (n={block['n']}, {block['n_bootstrap']} bootstrap resamples)"


def fmt_diff(block: dict, name_a: str, name_b: str) -> str:
    return (f"{name_a}={block['a']:.3f}, {name_b}={block['b']:.3f}, diff={block['diff_a_minus_b']:+.3f} "
            f"[{block['diff_ci95_lo']:+.3f}, {block['diff_ci95_hi']:+.3f}] (n={block['n']}, {block['n_bootstrap']} bootstrap resamples)")


def fmt_slice_table(slices: dict) -> str:
    lines = ["| cell | n | accuracy |", "|---|---|---|"]
    for k, v in slices.items():
        if v["accuracy"] is None:
            lines.append(f"| {k} | {v['n']} | *n<{MIN_CELL_N}, not computed* |")
        else:
            lines.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} |")
    return "\n".join(lines)


def fmt_confusion(cm: dict) -> str:
    labels = cm["labels"]
    header = "| gold \\ pred | " + " | ".join(labels) + " | recall |"
    sep = "|---" * (len(labels) + 2) + "|"
    lines = [header, sep]
    for g in labels:
        row = cm["counts_gold_row_pred_col"][g]
        recall = cm["per_gold_recall"][g]
        recall_str = f"{recall:.3f}" if recall is not None else "n/a"
        lines.append(f"| {g} | " + " | ".join(str(row[p]) for p in labels) + f" | {recall_str} |")
    return "\n".join(lines)


def write_markdown(results: dict, path: Path) -> None:
    gm = results["gold_set"]
    lines = []
    lines.append("# Benchmark model evaluation v2 — clean leak-free test gold\n")
    lines.append("**This supersedes `data/benchmark_results.md` for muril/indicbert/xlmr.** That earlier "
                 "evaluation used a gold set where 73/90 usable items were in those models' own training "
                 "split — its figures are invalid for those three models and must not be cited. "
                 "`data/benchmark_results.md`'s muril-hindi-baseline figures were leak-free (zero Bhojpuri "
                 "training data), but this run re-derives them on a different, difficulty-balanced sample.\n")
    lines.append(f"Gold file: `{gm['file']}`. {gm['n_raw']} usable items (already post-exclusion from the "
                 f"original 120-item difficulty-balanced draw). {gm['n_mixed_excluded']} further item(s) "
                 f"labeled 'mixed' excluded here — outside every model's label space. **n={gm['n_used']} used "
                 "for every metric below.**\n")
    lines.append(f"Human label distribution (n={gm['n_used']}): `{gm['human_label_distribution_used']}`\n")
    lc = results["leakage_check"]
    lines.append(f"**Leakage check** (verified programmatically): all {gm['n_raw']} gold ids "
                 f"(including 'mixed') are in `data/splits.json`'s TEST split = {lc['all_gold_ids_in_test_split']}. "
                 "Zero overlap with train/dev.\n")
    es = results["entropy_split"]
    lines.append(f"**Entropy-half split**: {es['method']}. "
                 f"low={es['n_low_incl_mixed']} (n={es['n_low_usable']} usable), "
                 f"high={es['n_high_incl_mixed']} (n={es['n_high_usable']} usable).\n")

    labels_str = ", ".join(CORPUS_LABELS)
    lines.append("## Fine-tuned model results (majority-vote-trained, primary)\n")
    lines.append(f"4-class label space: {labels_str}. Accuracy and macro-F1 with bootstrap 95% CI "
                 f"(1,000 resamples, percentile method), n={gm['n_used']}.\n")

    for key in CORPUS_MODEL_KEYS:
        b = results["models"][key]
        lines.append(f"### {key}\n")
        lines.append(f"- Full gold set: accuracy {fmt_ci(b['full']['accuracy'])}; macro-F1 {fmt_ci(b['full']['macro_f1'])}")
        lines.append("\nBy register:\n")
        lines.append(fmt_slice_table(b["by_register"]))
        lines.append("\nBy entropy half:\n")
        lines.append(fmt_slice_table(b["by_entropy_half"]))
        lines.append("")

    hb = results["models"][HINDI_BASELINE_KEY]
    lines.append(f"### {HINDI_BASELINE_KEY}\n")
    lines.append(f"3-class label space: {', '.join(HINDI_LABELS)}. {hb['structural_limitation']}\n")
    p = hb["polarity_subset"]
    lines.append(f"- **Polarity subset** (n={p['accuracy']['n']}, {p['n_objective_excluded_from_this_view']} "
                 f"objective items excluded): accuracy {fmt_ci(p['accuracy'])}; macro-F1 {fmt_ci(p['macro_f1'])}")
    o = hb["overall"]
    lines.append(f"- **Overall** (n={o['accuracy']['n']}, {o['n_objective_counted_as_error']} objective items "
                 "counted as automatic errors): accuracy " + fmt_ci(o["accuracy"]) + "; macro-F1 " + fmt_ci(o["macro_f1"]))
    lines.append("\nBy register (overall figure):\n")
    lines.append(fmt_slice_table(hb["by_register_overall"]))
    lines.append("\nBy entropy half (overall figure):\n")
    lines.append(fmt_slice_table(hb["by_entropy_half_overall"]))
    lines.append("")

    lines.append("## Aggregation itself: DS consensus vs majority vote — clean head-to-head\n")
    agg = results["aggregation"]
    lines.append(f"{agg['note']}\n")
    lines.append(f"DS and MV consensus labels differ on {agg['n_ds_vs_mv_disagreement']}/{gm['n_used']} gold items.\n")
    for name, key in (("Dawid-Skene consensus", "dawid_skene"), ("Majority vote", "majority_vote")):
        b = agg[key]
        lines.append(f"### {name}\n")
        lines.append(f"- accuracy {fmt_ci(b['full']['accuracy'])}; macro-F1 {fmt_ci(b['full']['macro_f1'])}")
        lines.append("\nBy register:\n")
        lines.append(fmt_slice_table(b["by_register"]))
        lines.append("\nBy entropy half:\n")
        lines.append(fmt_slice_table(b["by_entropy_half"]))
        lines.append("\nConfusion matrix (rows=gold, cols=predicted):\n")
        lines.append(fmt_confusion(b["confusion_matrix"]))
        lines.append("")

    lines.append("### Majority vote vs Dawid-Skene — direct comparison\n")
    lines.append(f"- Accuracy: {fmt_diff(agg['mv_minus_ds_accuracy'], 'MV', 'DS')}")
    lines.append(f"- Macro-F1: {fmt_diff(agg['mv_minus_ds_macro_f1'], 'MV', 'DS')}")
    lines.append("")

    lines.append("## DS-trained vs MV-trained models — did the silver-label choice affect downstream models?\n")
    for key in CORPUS_MODEL_KEYS:
        d = results["ds_trained_vs_mv_trained"][key]
        lines.append(f"### {key}\n")
        lines.append(f"- Accuracy: {fmt_diff(d['mv_minus_ds_accuracy'], 'MV-trained', 'DS-trained')}")
        lines.append(f"- Macro-F1: {fmt_diff(d['mv_minus_ds_macro_f1'], 'MV-trained', 'DS-trained')}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
