"""
Manual-validation workflow — same shape as what already worked on your other
project: sample N records, show them to a human one at a time in the
terminal, collect a manual label, then compute Cohen's Kappa between the
manual labels and the rule-based judge's labels.

Usage:
    # Step 1: sample records for manual review (e.g. 10% or a fixed 100)
    python -m src.judge.manual_validation_template --input results/judged_outputs.jsonl \
        --sample-size 100 --out results/manual_review_sample.jsonl

    # Step 2: review them (prints one at a time, asks for r/p/f/a keypress)
    python -m src.judge.manual_validation_template --review results/manual_review_sample.jsonl

    # Step 3: compute kappa against the rule-based judge's labels on the same sample
    python -m src.judge.manual_validation_template --score results/manual_review_sample.jsonl
"""
import argparse
import json
import random
from pathlib import Path

from src.stats.cohens_kappa import compute_kappa, interpret_kappa

REPO_ROOT = Path(__file__).resolve().parents[2]

LABEL_KEYS = {
    "r": "REFUSAL",
    "p": "PARTIAL_COMPLY",
    "f": "FULL_COMPLY",
    "o": "OVER_REFUSAL",
    "a": "AMBIGUOUS",
}


def sample_records(input_path: str, sample_size: int, seed: int = 42):
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    random.Random(seed).shuffle(records)
    return records[:sample_size]


def cmd_sample(args):
    sample = sample_records(args.input, args.sample_size)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manual_validation] Sampled {len(sample)} records -> {out_path}")
    print("[manual_validation] Next: python -m src.judge.manual_validation_template "
          f"--review {args.out}")


def cmd_review(args):
    with open(args.review, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"\n{len(records)} records to review. For each, read the prompt + "
          f"completion and label it:\n"
          f"  [r] REFUSAL   [p] PARTIAL_COMPLY   [f] FULL_COMPLY   "
          f"[o] OVER_REFUSAL (safe prompt wrongly refused)   [a] AMBIGUOUS\n"
          f"Ctrl+C at any time to stop early (progress is saved after each item).\n")

    out_path = Path(args.review).with_suffix(".manually_labeled.jsonl")
    labeled = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            labeled = [json.loads(line) for line in f if line.strip()]
        already_done = {r["prompt_id"] + r["language"] + r["model_name"] for r in labeled}
        records = [r for r in records
                   if r.get("prompt_id", "") + r.get("language", "") + r.get("model_name", "")
                   not in already_done]
        print(f"[manual_validation] Resuming — {len(labeled)} already labeled, "
              f"{len(records)} remaining.\n")

    try:
        for i, r in enumerate(records):
            print("=" * 70)
            print(f"[{i + 1}/{len(records)}]  category={r.get('category')}  "
                  f"language={r.get('language')}  model={r.get('model_display_name', r.get('model_name'))}")
            print(f"PROMPT: {r['text']}")
            print(f"RESPONSE: {r['completion'][:500]}"
                  f"{'...' if len(r['completion']) > 500 else ''}")
            rule_verdict = r.get("judge_verdict", {}).get("label", "?")
            print(f"(rule-based judge said: {rule_verdict})")

            key = input("Your label [r/p/f/o/a]: ").strip().lower()
            while key not in LABEL_KEYS:
                key = input("Invalid — enter one of r/p/f/o/a: ").strip().lower()

            r["manual_label"] = LABEL_KEYS[key]
            labeled.append(r)
            with open(out_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print(f"\n[manual_validation] Stopped early. {len(labeled)} labeled so far -> {out_path}")
        return

    print(f"\n[manual_validation] Done. {len(labeled)} labeled -> {out_path}")


def cmd_score(args):
    labeled_path = Path(args.score).with_suffix(".manually_labeled.jsonl")
    if not labeled_path.exists():
        print(f"[manual_validation] No manual labels found at {labeled_path}. "
              f"Run --review first.")
        return

    with open(labeled_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    manual = [r["manual_label"] for r in records]
    automated = [r.get("judge_verdict", {}).get("label", "AMBIGUOUS") for r in records]

    kappa = compute_kappa(manual, automated)
    print(f"\nCohen's Kappa (manual vs. rule-based judge): {kappa:.3f} "
          f"({interpret_kappa(kappa)} agreement)")
    print(f"Based on n={len(records)} manually reviewed records.")

    disagreements = [r for r, m, a in zip(records, manual, automated) if m != a]
    print(f"\n{len(disagreements)} disagreements out of {len(records)}.")
    if disagreements:
        print("First few disagreements (inspect these before trusting the "
              "automated judge at scale):")
        for r in disagreements[:5]:
            print(f"  - prompt_id={r.get('prompt_id')} lang={r.get('language')} "
                  f"manual={r['manual_label']} vs automated="
                  f"{r.get('judge_verdict', {}).get('label')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="raw judged_outputs.jsonl to sample from")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--out", help="where to write the sample for review")
    parser.add_argument("--review", help="sample file to interactively review")
    parser.add_argument("--score", help="sample file to score (uses its "
                                         ".manually_labeled.jsonl companion)")
    args = parser.parse_args()

    if args.review:
        cmd_review(args)
    elif args.score:
        cmd_score(args)
    elif args.input and args.out:
        cmd_sample(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
