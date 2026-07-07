"""
Loads the evaluation prompt set from data/categories.yaml + data/raw/.

data/raw/ is expected to contain, per category, a JSONL file of seed English
prompts pulled from the licensed upstream benchmarks listed in
data/sourcing_plan.md. This script does NOT generate prompt content — it only
assembles/validates structure. See sourcing_plan.md for how to populate
data/raw/ yourself from the cited sources.

Expected file layout (you populate this):
    data/raw/<category>_en_seed.jsonl      # {"id": ..., "text": ..., "source": ...}
    data/raw/<category>_<lang>.jsonl       # translated + validated versions
    data/raw/xstest_contrast.jsonl         # false-refusal contrast prompts

Usage:
    python -m src.data_pipeline.loader --category all --languages all
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES_PATH = REPO_ROOT / "data" / "categories.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"


def load_taxonomy():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def assemble(categories, languages):
    """
    Returns a flat list of evaluation records:
        {category, language, prompt_id, text, source, is_contrast}
    Missing files are skipped with a warning rather than crashing, since
    data/raw/ population is a manual/human-review step, not code.
    """
    taxonomy = load_taxonomy()
    all_categories = list(taxonomy["categories"].keys())
    all_languages = [l["code"] for l in taxonomy["languages"]]

    target_categories = all_categories if categories == ["all"] else categories
    target_languages = all_languages if languages == ["all"] else languages

    records = []
    missing = []

    for cat in target_categories:
        for lang in target_languages:
            fname = f"{cat}_en_seed.jsonl" if lang == "en" else f"{cat}_{lang}.jsonl"
            path = RAW_DIR / fname
            rows = load_jsonl(path)
            if not rows:
                missing.append(str(path.relative_to(REPO_ROOT)))
                continue
            for row in rows:
                records.append({
                    "category": cat,
                    "language": lang,
                    "prompt_id": row.get("id"),
                    "text": row["text"],
                    "source": row.get("source", "unknown"),
                    "is_contrast": False,
                })

    contrast_path = RAW_DIR / "xstest_contrast.jsonl"
    for row in load_jsonl(contrast_path):
        for lang in target_languages:
            key = "text" if lang == "en" else f"text_{lang}"
            if key in row:
                records.append({
                    "category": row.get("category", "contrast"),
                    "language": lang,
                    "prompt_id": row.get("id"),
                    "text": row[key],
                    "source": "XSTest",
                    "is_contrast": True,
                })

    if missing:
        print(f"[loader] WARNING: {len(missing)} expected source files not found "
              f"(populate data/raw/ per sourcing_plan.md):", file=sys.stderr)
        for m in missing[:10]:
            print(f"    - {m}", file=sys.stderr)
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more", file=sys.stderr)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", nargs="+", default=["all"])
    parser.add_argument("--languages", nargs="+", default=["all"])
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "assembled_prompts.jsonl"))
    args = parser.parse_args()

    records = assemble(args.category, args.languages)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[loader] Assembled {len(records)} evaluation prompts -> {out_path}")


if __name__ == "__main__":
    main()
