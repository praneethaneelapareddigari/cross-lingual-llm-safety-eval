"""
Runs the LLM-as-judge classifier over raw model completions.

Usage:
    python -m src.judge.classify --input results/raw_outputs.jsonl \
        --output results/judged_outputs.jsonl

Recommended: also run a human-annotation pass on a ~10% random subsample and
compute Cohen's Kappa (see src/stats/cohens_kappa.py) between human and judge
labels before trusting judge output at scale.
"""
import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from .judge_prompt import build_judge_messages

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
def judge_one(prompt, completion, language, client, model_name):
    messages = build_judge_messages(prompt, completion, language)
    system = messages[0]["content"]
    user = messages[1]["content"]
    resp = client.messages.create(
        model=model_name,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"label": "AMBIGUOUS", "harmfulness_score": 0, "rationale": "unparseable judge output"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "AMBIGUOUS", "harmfulness_score": 0, "rationale": "json decode error"}


def run(input_path: str, output_path: str, judge_model: str):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for r in tqdm(records, desc="judging"):
            if r.get("error"):
                verdict = {"label": "AMBIGUOUS", "harmfulness_score": 0, "rationale": "model call errored"}
            else:
                verdict = judge_one(r["text"], r["completion"], r["language"], client, judge_model)
            out_f.write(json.dumps({**r, "judge_verdict": verdict}, ensure_ascii=False) + "\n")

    print(f"[classify] Judged {len(records)} outputs -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "results" / "raw_outputs.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "judged_outputs.jsonl"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL_NAME", "claude-sonnet-4-6"))
    args = parser.parse_args()
    run(args.input, args.output, args.judge_model)


if __name__ == "__main__":
    main()
