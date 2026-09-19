"""Build the pooled Hindi sentiment training set for the hindi_baseline
retrain (data/hindi_baseline_pool.jsonl).

The original muril-hindi-baseline run (see RUNS.md-adjacent
data/benchmark_models/muril-hindi-baseline.meta.json, superseded by this
pool) trained on only OdiaGenAI/sentiment_analysis_hindi's 2,000-item
validation sample (1,800 train / 200 dev) — barely past initialisation
(train loss 2.73 -> 2.55 over 4 epochs). This script assembles a larger,
still ungated/public/license-documented pool by adding every other Hindi
sentiment dataset that survives the same provenance bar
purva/validation/make_hindi_validation_set.py already applied.

Dataset search (re-assessed 2026-09-20, HuggingFace Hub candidates checked
live — not from memory):

  SELECTED:
  - OdiaGenAI/sentiment_analysis_hindi (already used): 2,497 sentence-level
    product/movie review snippets, gold neg/pos/neu. Card declares no
    licence; contributor Kusumlata Patiyal (IIT Patna / Akshar Bharati
    group) places it in the IIT Patna Hindi review corpora line (inferred
    CC BY-NC-SA 4.0, research/non-commercial). Same inference already
    recorded in data/hindi_validation_set.jsonl's header.
  - ai4bharat/indic_glue, config iitp-mr.hi (IIT Patna Movie Reviews):
    3,100 rows (train 2,480 / validation 310 / test 310), native 3-class
    ClassLabel(["negative","neutral","positive"]) per the dataset loading
    script (verified against
    github.com/huggingface/datasets tag 2.0.0, datasets/indic_glue/indic_glue.py).
    Card license tag is "other" with no supplementary text; underlying
    corpus is Akhtar et al. 2016 (COLING), "A Hybrid Deep Learning
    Architecture for Sentiment Analysis" — the same IIT Patna academic
    lineage as OdiaGenAI's set, aggregated into AI4Bharat's IndicGLUE
    benchmark. No explicit commercial-use grant found; treated as
    research/non-commercial use only, same inference basis as OdiaGenAI.
  - ai4bharat/indic_glue, config iitp-pr.hi (IIT Patna Product Reviews):
    5,228 rows (train 4,182 / validation 523 / test 523), same label
    scheme, same licence basis as iitp-mr.hi above (same paper, same
    benchmark).
  - Process-Venue/Movie_Review_Sentiment_Hindi: apache-2.0 (card-stated,
    confirmed via the HF API license tag), 1,000 rows with per-item
    Annotator_1/Annotator_2/IAA/Answer columns; "Answer" is the adjudicated
    label. Label values are Devanagari सकारात्मक/नकारात्मक/मिश्रित
    (positive/negative/mixed) — "मिश्रित" (mixed, 174 rows) has no home in
    a 3-class positive/negative/neutral scheme and is DROPPED, not mapped,
    same as the corpus's own "mixed" treatment (PROTOCOL.md CHANGELOG v1.7
    excludes mixed from the aggregation label space for the identical
    reason: no natural home in a scheme without it). Rejected only for
    being ~1,000 rows on its own (below the single-validation-set
    threshold at pipeline-validation time); that threshold does not apply
    to a pooled training set.

  REJECTED (re-verified live, not from the earlier validation-set note):
  - mteb/sentiment_analysis_hindi: confirmed 2,497 rows total (1,249+1,248
    split) — the same size as OdiaGenAI's source, near-certainly a resplit
    of the same underlying data, no licence. REJECTED as duplicate.
  - sepidmnorozy/Hindi_sentiment: 630 rows, no licence/card, unattributed
    Bollywood movie-review scrapes, binary labels only. REJECTED, no
    licence/provenance.
  - iam-tsr/hindi-sentiments: MIT tag but content inspection shows
    machine-translated multilingual data (columns eng_text/label/
    hindi_text) including garbled non-Hindi source text crudely
    translated. REJECTED, low quality / unverifiable provenance confirmed
    by content, not just the card.
  - EmmadiVishnu/sentiment_analysis_hindi: identical filename and byte
    size to OdiaGenAI's file, same contributor. REJECTED as exact reupload.
  - ai4bharat/IndicSentiment (hi): only 156 (validation) + 1,000 (test) =
    1,156 rows, binary only (Positive/Negative, no neutral), and text is
    machine-translated from English reviews, not native Hindi. REJECTED —
    translated text and no neutral signal would dilute a native-Hindi
    transfer baseline.
  - cardiffnlp/tweet_sentiment_multilingual: script-based loader, no
    explicit licence, Hindi is a small slice of a combined multilingual
    file with tweet-redistribution risk. REJECTED.
  - Abhishek4896/hindi-english-code-mixed-tweets-sentiment,
    AjStar101/adaption-hindi-english-sentiment: code-mixed Hindi-English.
    EXCLUDED by scope (this is a monolingual Hindi transfer baseline).

Cross-source dedup: exact full-text string match (stripped), first-
occurrence-wins, in the SOURCE_PRIORITY order below (OdiaGenAI first,
since it's the incumbent training source). 124 exact-duplicate texts were
found between OdiaGenAI and the two IITP configs (same IIT Patna review
corpus, different extraction granularity — OdiaGenAI is aspect-term-level,
IITP is full-review-level, but some full reviews appear verbatim in both);
0 duplicates were found between iitp-mr and iitp-pr.

Result: 11,525 items after dedup (positive 5,106 / neutral 4,059 / negative
2,360) — short of the 15,000-20,000 target. Every other candidate found is
either a duplicate, unlicensed/unverifiable, code-mixed, or a low-quality
machine-translated scrape; the target is not stretched to hit a round
number by lowering the same provenance bar the rest of this project uses.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

DEFAULT_OUTPUT = Path("data/hindi_baseline_pool.jsonl")

TARGET_LABELS = ("negative", "neutral", "positive")

ODIA_REPO_ID = "OdiaGenAI/sentiment_analysis_hindi"
ODIA_FILE = "sentiment_analysis_term_train.jsonl"
ODIA_URL = f"https://huggingface.co/datasets/{ODIA_REPO_ID}/resolve/main/{ODIA_FILE}"
ODIA_LABEL_MAP = {"neg": "negative", "pos": "positive", "neu": "neutral"}
ODIA_LICENSE_NOTE = (
    "dataset card declares no licence; contributor Kusumlata Patiyal (IIT Patna / "
    "Akshar Bharati annotation group) and content (short Amazon product & movie "
    "review snippets) place it in the IIT Patna Hindi review corpora line "
    "(CC BY-NC-SA 4.0 inferred). Research/non-commercial use only."
)

IITP_LICENSE_NOTE = (
    "HF dataset card tags license as 'other' with no supplementary text. Underlying "
    "corpus is Akhtar et al. 2016 (COLING), \"A Hybrid Deep Learning Architecture for "
    "Sentiment Analysis\" (IIT Patna), aggregated into AI4Bharat's IndicGLUE benchmark "
    "— the same academic lineage as OdiaGenAI's set above. No explicit commercial-use "
    "grant found; treated as research/non-commercial use only, same inference basis."
)
IITP_LABEL_NAMES = ("negative", "neutral", "positive")  # verified against the indic_glue
# loading script's ClassLabel(names=[...]) for config names starting with "iitp"
# (github.com/huggingface/datasets, tag 2.0.0, datasets/indic_glue/indic_glue.py)

PV_REPO_ID = "Process-Venue/Movie_Review_Sentiment_Hindi"
PV_FILE = "PROJ_MOVIE_REVIEW'S_SENTIMENT_1000_V1_0004.csv"
PV_URL = "https://huggingface.co/datasets/" + PV_REPO_ID + "/resolve/main/" + PV_FILE.replace("'", "%27")
PV_LABEL_MAP = {"सकारात्मक": "positive", "नकारात्मक": "negative"}  # "मिश्रित" (mixed) dropped
PV_LICENSE_NOTE = "card states apache-2.0 explicitly (confirmed via the HF API license tag)."

# Cross-source dedup priority: earlier source wins a duplicate text.
SOURCE_PRIORITY = ("odia_genai_hindi_sentiment", "iitp_movie_reviews", "iitp_product_reviews",
                    "process_venue_movie_reviews")

DATASETS_SERVER_ROOT = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100


def _get(url: str, retries: int = 6) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "purva-research-bot/0.1 (academic)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  429 rate-limited, retrying in {wait}s ({attempt + 1}/{retries})...")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("unreachable")


def fetch_odia() -> list[dict]:
    raw = json.loads(_get(ODIA_URL))
    if isinstance(raw, dict):
        raw = raw.get("data")
    assert isinstance(raw, list) and raw, "unexpected OdiaGenAI file shape"
    out = []
    for r in raw:
        gold = ODIA_LABEL_MAP[r["label"]]
        text = r["text"].strip()
        if not text:
            continue
        out.append({
            "cleaned_text": text,
            "gold_label": gold,
            "gold_label_source_raw": r["label"],
            "source_name": "odia_genai_hindi_sentiment",
            "source_url": f"https://huggingface.co/datasets/{ODIA_REPO_ID}",
            "register": "product_review",
            "dataset_repo_id": ODIA_REPO_ID,
            "dataset_file": ODIA_FILE,
            "license_class": "cc_by_nc_sa_4.0_assumed_iit_patna_lineage",
            "license_note": ODIA_LICENSE_NOTE,
        })
    print(f"OdiaGenAI: fetched {len(out)} rows")
    return out


def fetch_indic_glue_config(config: str, source_name: str, register: str) -> list[dict]:
    size_info = json.loads(_get(f"{DATASETS_SERVER_ROOT}/size?dataset=ai4bharat/indic_glue&config={config}"))
    splits = [s["split"] for s in size_info["size"]["splits"]]
    rows_out: list[dict] = []
    for split in splits:
        offset = 0
        while True:
            url = (f"{DATASETS_SERVER_ROOT}/rows?dataset=ai4bharat/indic_glue&config={config}"
                   f"&split={split}&offset={offset}&length={PAGE_SIZE}")
            page = json.loads(_get(url))
            page_rows = page.get("rows", [])
            if not page_rows:
                break
            for entry in page_rows:
                row = entry["row"]
                text = (row.get("text") or "").strip()
                label_idx = row.get("label")
                if not text or label_idx is None:
                    continue
                gold = IITP_LABEL_NAMES[label_idx]
                rows_out.append({
                    "cleaned_text": text,
                    "gold_label": gold,
                    "gold_label_source_raw": label_idx,
                    "source_name": source_name,
                    "source_url": f"https://huggingface.co/datasets/ai4bharat/indic_glue/viewer/{config}",
                    "register": register,
                    "dataset_repo_id": "ai4bharat/indic_glue",
                    "dataset_file": f"config={config} split={split}",
                    "license_class": "other_inferred_research_use_ai4bharat_indicglue",
                    "license_note": IITP_LICENSE_NOTE,
                })
            offset += len(page_rows)
            if len(page_rows) < PAGE_SIZE:
                break
            time.sleep(0.5)
    print(f"{source_name} ({config}): fetched {len(rows_out)} rows")
    return rows_out


def fetch_process_venue() -> list[dict]:
    txt = _get(PV_URL)
    rows = list(csv.DictReader(io.StringIO(txt)))
    out = []
    n_mixed_dropped = 0
    for r in rows:
        answer = r.get("Answer", "").strip()
        text = r.get("Movie Review's", "").strip()
        if not text:
            continue
        if answer not in PV_LABEL_MAP:
            n_mixed_dropped += 1
            continue
        out.append({
            "cleaned_text": text,
            "gold_label": PV_LABEL_MAP[answer],
            "gold_label_source_raw": answer,
            "source_name": "process_venue_movie_reviews",
            "source_url": f"https://huggingface.co/datasets/{PV_REPO_ID}",
            "register": "movie_review",
            "dataset_repo_id": PV_REPO_ID,
            "dataset_file": PV_FILE,
            "license_class": "apache_2.0",
            "license_note": PV_LICENSE_NOTE,
        })
    print(f"Process-Venue: fetched {len(out)} rows ({n_mixed_dropped} 'मिश्रित'/mixed dropped, no home in 3-class scheme)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()

    out_path = Path(args.output)
    if out_path.exists():
        sys.exit(f"refusing to overwrite existing {out_path} — delete it first if you intend to rebuild the pool")

    sources = {
        "odia_genai_hindi_sentiment": fetch_odia(),
        "iitp_movie_reviews": fetch_indic_glue_config("iitp-mr.hi", "iitp_movie_reviews", "movie_review"),
        "iitp_product_reviews": fetch_indic_glue_config("iitp-pr.hi", "iitp_product_reviews", "product_review"),
        "process_venue_movie_reviews": fetch_process_venue(),
    }

    for name, rows in sources.items():
        bad = {r["gold_label"] for r in rows} - set(TARGET_LABELS)
        assert not bad, f"{name}: unexpected label(s) {bad}"

    seen_text: set[str] = set()
    pooled: list[dict] = []
    dedup_dropped: dict[str, int] = {}
    for name in SOURCE_PRIORITY:
        dropped = 0
        for r in sources[name]:
            key = r["cleaned_text"]
            if key in seen_text:
                dropped += 1
                continue
            seen_text.add(key)
            pooled.append(r)
        dedup_dropped[name] = dropped
        print(f"{name}: {dropped} cross/intra-source duplicate(s) dropped")

    label_counts = Counter(r["gold_label"] for r in pooled)
    print(f"pooled total: {len(pooled)}; label distribution: {dict(label_counts)}")

    out_rows = []
    for i, r in enumerate(pooled):
        out_rows.append({
            "id": f"hinpool_{i:05d}",
            "cleaned_text": r["cleaned_text"],
            "gold_label": r["gold_label"],
            "gold_label_source_raw": r["gold_label_source_raw"],
            "language": "hindi",
            "source_name": r["source_name"],
            "source_url": r["source_url"],
            "register": r["register"],
            "text_type": "prose",
            "script": "devanagari",
            "license_class": r["license_class"],
            "license_note": r["license_note"],
            "dataset_repo_id": r["dataset_repo_id"],
            "dataset_file": r["dataset_file"],
        })

    header = {
        "_pool_version": 1,
        "_n_sources": len(SOURCE_PRIORITY),
        "_source_priority_order": list(SOURCE_PRIORITY),
        "_source_raw_counts": {name: len(rows) for name, rows in sources.items()},
        "_source_dedup_dropped": dedup_dropped,
        "_pooled_total": len(out_rows),
        "_label_distribution": dict(label_counts),
        "_dedup_method": "exact full-text string match (stripped), first-occurrence-wins in _source_priority_order",
        "_label_space": list(TARGET_LABELS),
        "_note": (
            "3-class polarity pool (no objective/subjectivity stage in any source) built to retrain "
            "muril-hindi-baseline on more than the original 1,800-item OdiaGenAI-only training set. "
            "See this script's module docstring for full source selection/rejection reasoning."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {out_path}: {len(out_rows)} rows")


if __name__ == "__main__":
    main()
