"""Build the Hindi cross-lingual validation set (data/hindi_validation_set.jsonl).

Purpose (see data/hindi_validation_report.md when written): validate the
judge-committee + Dawid-Skene aggregation pipeline against EXISTING third-party
gold labels in a related language, because Bhojpuri gold is limited. This
validates the PIPELINE, not our Bhojpuri labels.

Dataset selection (assessed 2026-09-17, HuggingFace Hub candidates checked):
  - mteb/sentiment_analysis_hindi: 2,497 rows but split into train 1,249 /
    test 1,248 — each split alone is below the 2,000-item requirement, and the
    card carries no licence. REJECTED.
  - sepidmnorozy/Hindi_sentiment: 37 downloads, no dataset card/licence at
    all, "n<1K" size tag. REJECTED.
  - iam-tsr/hindi-sentiments: MIT, 100K+ rows, but a 2026 personal re-upload
    ("indic_sentiment_data.csv") of unknown original provenance — no source
    paper, no annotation documentation. Unverifiable provenance. REJECTED.
  - OdiaGenAI/sentiment_analysis_hindi: 2,497 sentence-level product/movie
    review snippets with gold polarity labels neg/pos/neu (neg 351, pos 1,147,
    neu 999 — all classes present, largest class only 3.3x the smallest, so
    proportional stratification keeps every class well represented).
    Contributor: Kusumlata Patiyal (IIT Patna — Akshar Bharati group), i.e.
    the IIT Patna Hindi review annotation line of work. Public, ungated,
    single 1.0 MB file, card documents the label-adjudication conventions.
    SELECTED.
  - IIT Patna's larger product/Yelp corpora (SAXR/Amazon-Review-Ir line,
    35k-67k sentences): not located on the HuggingFace Hub under any
    searchable name; not used here.
  - Process-Venue/Movie_Review_Sentiment_Hindi: apache-2.0 but ~1,000 rows.
    Below the 2,000 requirement. REJECTED.

Label mapping (their scheme -> ours, PROTOCOL.md §3):
  their "neg" -> our "negative"
  their "pos" -> our "positive"
  their "neu" -> our "neutral" (a mild/hedged subjective stance in our
  Stage-B scheme, not our Stage-A "objective" — their scheme has no
  subjectivity stage at all, see the mapping-loss note in the report).

Sampling: proportional stratification on the mapped gold label (largest-
remainder allocation, the same method as purva/committee/make_chunks.py),
seed 42, n = 2,000. Output schema is exactly what
purva/committee/run_judge.py expects from a --input file: an `id` field and
a `cleaned_text` field per row, plus metadata columns for the report stage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

SEED = 42
N_SAMPLE = 2_000

SOURCE_REPO_ID = "OdiaGenAI/sentiment_analysis_hindi"
SOURCE_FILE = "sentiment_analysis_term_train.jsonl"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_REPO_ID}/resolve/main/{SOURCE_FILE}"

# their label -> our label (PROTOCOL.md §3 Stage-B polarity space)
LABEL_MAP = {"neg": "negative", "pos": "positive", "neu": "neutral"}

DEFAULT_OUTPUT = Path("data/hindi_validation_set.jsonl")


def fetch_rows(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "purva-research-bot/0.1 (academic)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8")
    rows = json.loads(raw)
    if isinstance(rows, dict):  # tolerate {"data": [...]} wrappers
        rows = rows.get("data")
    assert isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows), "unexpected file shape"
    return rows


def allocate_proportional(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` across label strata,
    proportional to stratum size (same method as make_chunks.py)."""
    n_all = sum(counts.values())
    raw = {k: v * total / n_all for k, v in counts.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(base.values())
    order = sorted(counts, key=lambda k: (-(raw[k] - base[k]), k))
    for k in order[:remainder]:
        base[k] += 1
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"refusing to overwrite existing {out_path} — a validation set must stay fixed once judges have run against it")

    rows = fetch_rows(SOURCE_URL)
    print(f"downloaded {len(rows)} rows from {SOURCE_URL}")

    mapped: list[dict] = []
    for r in rows:
        gold = LABEL_MAP[r["label"]]
        mapped.append({"text": r["text"].strip(), "gold_label_raw": r["label"], "gold_label": gold})
    assert all(m["text"] for m in mapped), "empty text after strip"

    dupes = [t for t, c in Counter(m["text"] for m in mapped).items() if c > 1]
    assert not dupes, f"{len(dupes)} duplicate text(s) in source, e.g. {dupes[:3]}"

    by_label: dict[str, list[dict]] = {}
    for m in mapped:
        by_label.setdefault(m["gold_label"], []).append(m)
    counts = {k: len(v) for k, v in sorted(by_label.items())}
    print(f"gold label distribution (mapped): {counts}")

    alloc = allocate_proportional(counts, args.n)
    print(f"stratified allocation: {alloc}")

    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for label in sorted(by_label):
        pool = by_label[label][:]
        rng.shuffle(pool)
        take = alloc[label]
        assert take <= len(pool), f"not enough items in stratum {label!r}: {take} > {len(pool)}"
        sampled.extend(pool[:take])

    # sort by (gold label, original source order) for a stable, inspectable file
    order = {t: i for i, t in enumerate((m["text"] for m in mapped))}
    sampled.sort(key=lambda m: (m["gold_label"], order[m["text"]]))

    n_all = len(mapped)
    out_rows = []
    for i, m in enumerate(sampled):
        out_rows.append({
            "id": f"hin_{i:05d}",
            "cleaned_text": m["text"],
            "gold_label": m["gold_label"],
            "gold_label_source_raw": m["gold_label_raw"],
            "language": "hindi",
            "source_name": "odia_genai_hindi_sentiment",
            "source_url": f"https://huggingface.co/datasets/{SOURCE_REPO_ID}",
            "register": "product_review",
            "text_type": "prose",
            "script": "devanagari",
            "license_class": "cc_by_nc_sa_4.0_assumed_iit_patna_lineage",
            "license_note": (
                "dataset card declares no licence; contributor Kusumlata Patiyal "
                "(IIT Patna / Akshar Bharati annotation group) and content "
                "(short Amazon product & movie review snippets) place it in the "
                "IIT Patna Hindi review corpora line (CC BY-NC-SA 4.0). "
                "Research/non-commercial use only."
            ),
            "dataset_repo_id": SOURCE_REPO_ID,
            "dataset_file": SOURCE_FILE,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "_dataset_repo_id": SOURCE_REPO_ID,
            "_dataset_file": SOURCE_FILE,
            "_source_url": SOURCE_URL,
            "_sampled": len(out_rows),
            "_source_size": n_all,
            "_seed": args.seed,
            "_stratified_on": "gold_label",
            "_allocation": alloc,
            "_source_distribution": counts,
            "_label_map": LABEL_MAP,
        }, ensure_ascii=False) + "\n")
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}: {len(out_rows)} rows")
    final = Counter(r["gold_label"] for r in out_rows)
    print(f"final gold distribution: {dict(final)}")


if __name__ == "__main__":
    main()
