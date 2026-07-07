"""
Generates a real figure from the Pilot v0 smoke-test judged output
(results/smoke_test_judged.jsonl). Small-n (10 prompts), intentionally
labeled as such in the chart title -- not to be confused with a full
Pilot v1 result.

Usage:
    python scripts/plot_smoke_test.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "results" / "smoke_test_judged.jsonl"
OUTPUT_PATH = REPO_ROOT / "visualizations" / "figures" / "smoke_test_label_breakdown.png"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    labels = [r["judge_verdict"]["label"] for r in records]
    counts = Counter(labels)

    order = ["REFUSAL", "PARTIAL_COMPLY", "FULL_COMPLY", "OVER_REFUSAL", "AMBIGUOUS"]
    ordered_labels = [l for l in order if l in counts]
    values = [counts[l] for l in ordered_labels]
    colors = {"REFUSAL": "#2b8a3e", "PARTIAL_COMPLY": "#e8a33d",
              "FULL_COMPLY": "#c92a2a", "OVER_REFUSAL": "#1971c2",
              "AMBIGUOUS": "#868e96"}
    bar_colors = [colors[l] for l in ordered_labels]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    bars = plt.bar(ordered_labels, values, color=bar_colors)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.05, str(v),
                  ha="center", va="bottom", fontweight="bold")
    plt.title(f"Pilot v0 Smoke Test: Judge Label Breakdown\n"
              f"(n={len(records)}, English only, llama3.2:1b, single run -- "
              f"NOT a Pilot v1 result)")
    plt.ylabel("Count")
    plt.ylim(0, max(values) + 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Wrote {OUTPUT_PATH} -- counts: {dict(counts)}")


if __name__ == "__main__":
    main()
