"""Driver for Phase 4 aggregation (PROTOCOL.md §5). Runs every aggregator
over the full corpus and writes:

  data/purva_aggregated.jsonl — per sentence: id, the five raw votes
    (unchanged), and each aggregator's posterior/argmax/entropy.
  data/purva_aggregated.meta.json — aggregator configs, iteration counts,
    convergence status, and runtime, same discipline as the committee's
    per-shard .meta.json sidecars.

"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import baselines as bl
from . import dawid_skene as ds
from ._common import JUDGES, LABELS, argmax_labels, build_strata, build_vote_matrix, load_master, posterior_entropy

DEFAULT_OUTPUT_JSONL = "data/purva_aggregated.jsonl"
DEFAULT_OUTPUT_META = "data/purva_aggregated.meta.json"


def _raw_votes_per_item(df, i: int) -> dict:
    votes = {}
    for judge in JUDGES:
        subj = df.at[df.index[i], f"judge_{judge}_subjectivity"]
        if subj is None or (isinstance(subj, float) and np.isnan(subj)):
            votes[judge] = None
            continue
        votes[judge] = {
            "subjectivity": subj,
            "polarity": df.at[df.index[i], f"judge_{judge}_polarity"],
            "confidence": df.at[df.index[i], f"judge_{judge}_confidence"],
            "domain": df.at[df.index[i], f"judge_{judge}_domain"],
            "narrative_voice": df.at[df.index[i], f"judge_{judge}_narrative_voice"],
            "sentiment_target": df.at[df.index[i], f"judge_{judge}_sentiment_target"],
            "rationale": df.at[df.index[i], f"judge_{judge}_rationale"],
        }
    return votes


def _method_block(posteriors: np.ndarray) -> list[dict]:
    labels = argmax_labels(posteriors)
    ent_nat, ent_norm = posterior_entropy(posteriors)
    return [
        {
            "posterior": {LABELS[k]: float(posteriors[i, k]) for k in range(len(LABELS))},
            "label": labels[i],
            "entropy_nat": float(ent_nat[i]),
            "entropy_norm": float(ent_norm[i]),
        }
        for i in range(posteriors.shape[0])
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", default="data/purva_master.parquet")
    ap.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    ap.add_argument("--output-meta", default=DEFAULT_OUTPUT_META)
    ap.add_argument("--mace-restarts", type=int, default=3)
    ap.add_argument("--glad-iter", type=int, default=50)
    ap.add_argument("--skip-mace", action="store_true")
    ap.add_argument("--skip-glad", action="store_true")
    args = ap.parse_args()

    out_jsonl = Path(args.output_jsonl)
    out_meta = Path(args.output_meta)

    print(f"loading {args.master}")
    df = load_master(args.master)
    votes = build_vote_matrix(df)
    strata, strata_keys = build_strata(df)
    item_ids = df["id"].to_numpy()
    print(f"{len(df)} items, {votes.shape[1]} judges, {len(strata_keys)} strata")

    meta: dict = {"n_items": len(df), "judges": list(JUDGES), "labels": list(LABELS), "methods": {}}

    methods: dict[str, list[dict]] = {}

    print("\n=== standard Dawid-Skene ===")
    t0 = time.time()
    ds_config = ds.DSConfig()
    ds_result = ds.run_dawid_skene(votes, ds_config)
    runtime = time.time() - t0
    print(f"iters={ds_result.n_iter} converged={ds_result.converged} runtime={runtime:.1f}s "
          f"final_ll={ds_result.log_likelihood[-1]:.2f}")
    methods["dawid_skene"] = _method_block(ds_result.posteriors)
    meta["methods"]["dawid_skene"] = {
        "config": {"alpha": ds_config.alpha, "max_iter": ds_config.max_iter, "tol": ds_config.tol},
        "n_iter": ds_result.n_iter,
        "converged": ds_result.converged,
        "final_log_likelihood": ds_result.log_likelihood[-1],
        "log_likelihood_trajectory": ds_result.log_likelihood,
        "runtime_seconds": round(runtime, 1),
    }

    print("\n=== stratified Dawid-Skene ===")
    t0 = time.time()
    sds_config = ds.StratifiedDSConfig()
    sds_result = ds.run_stratified_dawid_skene(votes, strata, strata_keys, sds_config)
    runtime = time.time() - t0
    print(f"iters={sds_result.n_iter} converged={sds_result.converged} runtime={runtime:.1f}s "
          f"final_ll={sds_result.log_likelihood[-1]:.2f}")
    methods["stratified_dawid_skene"] = _method_block(sds_result.posteriors)
    meta["methods"]["stratified_dawid_skene"] = {
        "config": {
            "alpha": sds_config.alpha, "max_iter": sds_config.max_iter, "tol": sds_config.tol,
            "shrinkage_k0": sds_config.shrinkage_k0,
        },
        "strata": [f"{r}/{t}" for r, t in strata_keys],
        "strata_sizes": {f"{r}/{t}": int(n) for (r, t), n in zip(strata_keys, sds_result.strata_sizes)},
        "n_iter": sds_result.n_iter,
        "converged": sds_result.converged,
        "final_log_likelihood": sds_result.log_likelihood[-1],
        "log_likelihood_trajectory": sds_result.log_likelihood,
        "runtime_seconds": round(runtime, 1),
    }

    print("\n=== majority vote ===")
    t0 = time.time()
    mv_result = bl.majority_vote(votes)
    runtime = time.time() - t0
    print(f"tie_count={mv_result.extra['tie_count']} tie_rate={mv_result.extra['tie_rate']:.4f}")
    methods["majority_vote"] = _method_block(mv_result.posteriors)
    meta["methods"]["majority_vote"] = {**mv_result.extra, "runtime_seconds": round(runtime, 1)}

    if args.skip_mace:
        meta["methods"]["mace"] = {"available": False, "unavailable_reason": "skipped via --skip-mace"}
    else:
        print("\n=== MACE ===")
        t0 = time.time()
        mace_result = bl.run_mace(votes, item_ids, n_restarts=args.mace_restarts)
        runtime = time.time() - t0
        print(f"available={mace_result.available} runtime={runtime:.1f}s")
        if mace_result.available:
            methods["mace"] = _method_block(mace_result.posteriors)
            meta["methods"]["mace"] = {**mace_result.extra, "runtime_seconds": round(runtime, 1), "available": True}
        else:
            print(f"MACE unavailable: {mace_result.unavailable_reason}")
            meta["methods"]["mace"] = {"available": False, "unavailable_reason": mace_result.unavailable_reason}

    if args.skip_glad:
        meta["methods"]["glad"] = {"available": False, "unavailable_reason": "skipped via --skip-glad"}
    else:
        print("\n=== GLAD ===")
        t0 = time.time()
        glad_result = bl.run_glad(votes, item_ids, n_iter=args.glad_iter)
        runtime = time.time() - t0
        print(f"available={glad_result.available} runtime={runtime:.1f}s")
        if glad_result.available:
            methods["glad"] = _method_block(glad_result.posteriors)
            meta["methods"]["glad"] = {**glad_result.extra, "runtime_seconds": round(runtime, 1), "available": True}
        else:
            print(f"GLAD unavailable: {glad_result.unavailable_reason}")
            meta["methods"]["glad"] = {"available": False, "unavailable_reason": glad_result.unavailable_reason}

    print(f"\nwriting {out_jsonl}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for i in range(len(df)):
            row = {
                "id": item_ids[i],
                "votes": _raw_votes_per_item(df, i),
            }
            for method_name, blocks in methods.items():
                row[method_name] = blocks[i]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out_jsonl} ({out_jsonl.stat().st_size / 1e6:.1f} MB)")

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_meta}")


if __name__ == "__main__":
    main()
