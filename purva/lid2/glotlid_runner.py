from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import fasttext
from huggingface_hub import hf_hub_download

# GlotLID v3 (cis-lmu/glotlid), Apache 2.0.
# Model card: https://huggingface.co/cis-lmu/glotlid
# Exact file pinned for reproducibility (not the "model.bin" alias, which can be
# repointed to a future version): https://huggingface.co/cis-lmu/glotlid/blob/main/model_v3.bin
GLOTLID_REPO_ID = "cis-lmu/glotlid"
GLOTLID_FILENAME = "model_v3.bin"

# IndicLID (AI4Bharat) was the originally planned model (PROTOCOL.md v1.0) but covers
# only the 22 Eighth-Schedule Indian languages and has no Bhojpuri class, so it cannot
# separate Bhojpuri from Hindi. GlotLID provides bho_Deva/hin_Deva/mai_Deva as distinct
# classes. See PROTOCOL.md CHANGELOG v1.1.
VERDICT_MAP = {
    "bho_Deva": "bhojpuri",
    "hin_Deva": "hindi",
    "mai_Deva": "maithili",
}

BATCH_SIZE = 2000


def load_model() -> fasttext.FastText._FastText:
    path = hf_hub_download(repo_id=GLOTLID_REPO_ID, filename=GLOTLID_FILENAME)
    print(f"Loaded GlotLID model from {path}")
    return fasttext.load_model(path)


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done


def predict_batch(model, texts: list[str]) -> list[tuple[str, float]]:
    labels, probs = model.predict(texts, k=1)
    out = []
    for lbl, prob in zip(labels, probs):
        raw = lbl[0].replace("__label__", "")
        out.append((raw, float(prob[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus_filtered.jsonl")
    ap.add_argument("--output", default="data/corpus_lid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = [json.loads(x) for x in in_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = load_done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} total, {len(done)} already done, {len(todo)} to process\n")

    if todo:
        model = load_model()
        with out_path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(todo), BATCH_SIZE):
                batch = todo[start : start + BATCH_SIZE]
                texts = [r["cleaned_text"].replace("\n", " ").strip() or " " for r in batch]
                preds = predict_batch(model, texts)
                for row, (raw_label, confidence) in zip(batch, preds):
                    row["lid_model_label"] = raw_label
                    row["lid_model_confidence"] = round(confidence, 4)
                    row["lid_verdict"] = VERDICT_MAP.get(raw_label, "other")
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done_so_far = min(start + BATCH_SIZE, len(todo))
                print(f"[{done_so_far}/{len(todo)}] processed")

    summarize(out_path)


def summarize(out_path: Path):
    rows = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    total = len(rows)
    verdict_counts = Counter(r["lid_verdict"] for r in rows)

    print("\n=== Final summary ===")
    print(f"total records: {total}\n")
    print("lid_verdict counts and percentages:")
    for verdict in ("bhojpuri", "hindi", "maithili", "other"):
        n = verdict_counts.get(verdict, 0)
        pct = (n / total * 100) if total else 0.0
        print(f"  {verdict:10s}: {n:7d}  ({pct:5.1f}%)")

    by_source = defaultdict(Counter)
    for r in rows:
        by_source[r["source_name"]][r["lid_verdict"]] += 1

    print("\nper-source breakdown (% of source's rows, by lid_verdict):")
    header = f"  {'source_name':25s} {'n':>7s} {'bhojpuri':>9s} {'hindi':>9s} {'maithili':>9s} {'other':>9s}"
    print(header)
    for source in sorted(by_source, key=lambda s: -sum(by_source[s].values())):
        counts = by_source[source]
        n = sum(counts.values())
        pcts = [counts.get(v, 0) / n * 100 for v in ("bhojpuri", "hindi", "maithili", "other")]
        print(f"  {source:25s} {n:7d} {pcts[0]:8.1f}% {pcts[1]:8.1f}% {pcts[2]:8.1f}% {pcts[3]:8.1f}%")


if __name__ == "__main__":
    main()
