from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .env import load_env
from .groq_judge import GroqJudge
from .gemini_judge import GeminiJudge


def combine(groq: dict, gemini: dict) -> dict:
    g, m = groq["label"], gemini["label"]
    note = None

    if g == m == "bhojpuri":
        label, conf = "bhojpuri", (groq["confidence"] + gemini["confidence"]) / 2
    elif g == m == "hindi":
        label, conf = "hindi", (groq["confidence"] + gemini["confidence"]) / 2
    elif g == m == "other":
        label, conf = "other", (groq["confidence"] + gemini["confidence"]) / 2
        gr, mr = groq["reason"], gemini["reason"]
        if "error" in gr or "error" in mr or "retries" in gr or "retries" in mr:
            note = "api_error"
        elif "unparseable" in gr or "unparseable" in mr:
            note = "unparseable"
        else:
            note = "both_judges_other"
    else:
        label = "disagree"
        conf = min(groq["confidence"], gemini["confidence"])
        note = f"{g}_vs_{m}"

    return {"label": label, "confidence": round(conf, 3), "note": note}


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/purva_pilot_candidates.jsonl")
    ap.add_argument("--output", default="data/purva_pilot_lid.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_env()
    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = [json.loads(x) for x in in_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done = load_done(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} total, {len(done)} already done, {len(todo)} to process\n")

    groq = GroqJudge()
    gemini = GeminiJudge()

    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo, 1):
            text = row["cleaned_text"]
            gj = groq.judge(text)
            mj = gemini.judge(text)
            c = combine(gj, mj)
            row["lid_label"] = c["label"]
            row["lid_confidence"] = c["confidence"]
            row["lid_note"] = c["note"]
            row["lid_groq"] = gj
            row["lid_gemini"] = mj
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {c['label']:9s} conf={c['confidence']:.2f}"
                  f"{' note=' + c['note'] if c['note'] else ''}")

    tally(out_path, len(rows))


def tally(out_path: Path, total: int):
    rows = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    labels = Counter(r["lid_label"] for r in rows)
    notes = Counter(r["lid_note"] for r in rows if r.get("lid_note"))

    print("\n--- check 1 tally ---")
    print(f"processed         : {len(rows)}/{total}")
    for k in ("bhojpuri", "disagree", "hindi", "other"):
        print(f"{k:18s}: {labels.get(k, 0)}")

    if notes:
        print("\nother / disagree breakdown:")
        for k, v in notes.most_common():
            print(f"  {k:22s}: {v}")

    processed = len(rows)
    if processed:
        bho = labels.get("bhojpuri", 0)
        errors = sum(v for k, v in notes.items() if k in ("api_error", "unparseable"))
        clean = processed - errors
        print(f"\nbhojpuri agreement (both judges): {bho}/{processed} = {bho/processed:.1%}")
        if clean:
            print(f"excluding {errors} unprocessed: {bho}/{clean} = {bho/clean:.1%}")


if __name__ == "__main__":
    main()