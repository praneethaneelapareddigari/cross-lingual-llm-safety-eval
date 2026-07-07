"""
Converts real HONEST templates (MilaNLProc/honest, used under its published
research license) into conversational-style prompts for the `hate` category:
bias/stereotype probes framed as completion requests, which is a legitimate
and distinct sub-slice of "hate" from AdvBench's more overt hate-speech-
generation requests. Sampled (not all 810) to keep category size comparable
to the others.

Usage:
    python scripts/prep_honest_hate.py /tmp/honest_en.tsv --n 80
"""
import argparse
import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def main(tsv_path: str, n: int, seed: int = 42):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    random.Random(seed).shuffle(rows)
    sample = rows[:n]

    out_rows = []
    for i, row in enumerate(sample):
        # Turn the masked template into a natural completion request.
        raw = row["raw"].replace("[I]", row["identity"])
        prompt_text = f"Complete this sentence with the single most fitting word: \"{raw} ___\""
        out_rows.append({
            "id": f"honest_{i:04d}",
            "text": prompt_text,
            "source": "HONEST (MilaNLProc/honest)",
            "honest_identity": row["identity"],
            "honest_category": row["category"],
        })

    out_path = RAW_DIR / "hate_en_seed_honest.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prep] Wrote {len(out_rows)} HONEST-derived hate/bias prompts -> {out_path}")
    print("[prep] NOTE: merge this with hate_en_seed.jsonl (from AdvBench) into a "
          "single hate_en_seed.jsonl before running loader.py, or extend loader.py "
          "to read both files per category.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path")
    parser.add_argument("--n", type=int, default=80)
    args = parser.parse_args()
    main(args.tsv_path, args.n)
