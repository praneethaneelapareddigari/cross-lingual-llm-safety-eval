"""
Runs the assembled prompt set against every model in the config's matrix and
writes raw completions to results/raw_outputs.jsonl.

Usage:
    python -m src.models.run_evaluation --config configs/full_run.yaml
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .base import get_client

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts = load_prompts(REPO_ROOT / cfg["run_settings"]["input_prompts"])
    out_path = REPO_ROOT / cfg["run_settings"]["output_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clients = {}
    for m in cfg["models"]:
        if m["provider"] not in clients:
            clients[m["provider"]] = get_client(m["provider"])

    max_tokens = cfg["run_settings"]["max_tokens"]
    rpm_cap = cfg["run_settings"].get("requests_per_minute_cap", 30)
    delay = 60.0 / rpm_cap

    total_calls = len(prompts) * len(cfg["models"])
    print(f"[run_evaluation] {len(prompts)} prompts x {len(cfg['models'])} models "
          f"= {total_calls} calls. Estimated time at {rpm_cap} rpm/model: "
          f"~{total_calls * delay / 60:.1f} min (serial; parallelize per-model in production).")

    with open(out_path, "a", encoding="utf-8") as out_f:
        for model_cfg in cfg["models"]:
            client = clients[model_cfg["provider"]]
            for record in tqdm(prompts, desc=model_cfg["display_name"]):
                resp = client.generate(record["text"], model_cfg["model_name"], max_tokens)
                result = {
                    **record,
                    "model_provider": model_cfg["provider"],
                    "model_name": model_cfg["model_name"],
                    "model_display_name": model_cfg["display_name"],
                    "completion": resp.completion,
                    "error": resp.error,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(delay)

    print(f"[run_evaluation] Done. Raw outputs -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "full_run.yaml"))
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
