"""
One-time data-prep script: maps the public AdvBench harmful_behaviors.csv
into our 9-category taxonomy using keyword matching, and writes
data/raw/<category>_en_seed.jsonl files in the format loader.py expects.

AdvBench itself has no category labels, so this is a lightweight heuristic
classifier — NOT rigorous enough for the final paper. Before running
translation/evaluation at scale, a human pass should spot-check ~10% of the
category assignments per data/sourcing_plan.md's review standard.

Usage:
    python scripts/prep_advbench_seeds.py /tmp/advbench_raw.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Keyword -> category heuristic. Order matters: first match wins.
CATEGORY_KEYWORDS = {
    "malware": ["hack", "exploit", "malware", "virus", "ransomware", "phishing",
                "ddos", "botnet", "backdoor", "keylogger", "trojan", "worm",
                "software vulnerabilit", "cyberattack", "cyber attack"],
    "violence": ["bomb", "weapon", "explosive", "firearm", "gun", "kill",
                 "attack", "terroris", "assassinat", "violence", "shoot",
                 "poison to harm others", "mass casualt"],
    "drugs": ["drug", "narcotic", "meth", "cocaine", "heroin", "opioid",
              "illegal substance"],
    "self_harm": ["suicide", "self-harm", "self harm", "eating disorder"],
    "privacy": ["stalk", "track someone", "surveil", "private information",
                "steal identity", "identity theft", "dox", "spy on"],
    "hate": ["hate speech", "racist", "discriminat", "ethnic", "supremac",
             "slur", "genocide propaganda"],
    "finance": ["launder", "counterfeit", "fraud", "scam", "ponzi",
                "insider trading", "tax evasion", "credit card"],
    "medical": ["prescription", "medical advice", "overdose", "medication",
                "diagnose", "untested treatment"],
    "politics": ["election", "propaganda", "political", "voter", "disinformation",
                 "government database", "protest"],
}

DEFAULT_CATEGORY = "violence"  # AdvBench skews toward violent/illegal-act framing


def categorize(text: str) -> str:
    lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return cat
    return DEFAULT_CATEGORY


def main(csv_path: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    buckets = {cat: [] for cat in CATEGORY_KEYWORDS}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = row["goal"].strip()
            if not text:
                continue
            cat = categorize(text)
            buckets[cat].append({
                "id": f"advbench_{i:04d}",
                "text": text,
                "source": "AdvBench (llm-attacks/llm-attacks)",
            })

    for cat, rows in buckets.items():
        if not rows:
            continue
        out_path = RAW_DIR / f"{cat}_en_seed.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[prep] {cat}: {len(rows)} prompts -> {out_path}")

    total = sum(len(v) for v in buckets.values())
    print(f"[prep] Total categorized: {total} / {i + 1} source rows")
    print("[prep] NOTE: keyword-based categorization is a heuristic starting "
          "point only. Human review required before treating as final "
          "(see data/sourcing_plan.md).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/advbench_raw.csv")
