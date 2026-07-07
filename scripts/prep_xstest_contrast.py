"""
Converts the real XSTest CSV (paul-rottger/xstest, CC BY 4.0) into the
xstest_contrast.jsonl format loader.py expects.

Usage:
    python scripts/prep_xstest_contrast.py /tmp/xstest_raw.csv
"""
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def main(csv_path: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_out.append({
                "id": f"xstest_{row['id']}",
                "category": "contrast",
                "text": row["prompt"],
                "xstest_label": row["label"],  # "safe" or "contrast" (unsafe)
                "xstest_type": row["type"],
                "xstest_focus": row["focus"],
                "xstest_topic_note": row.get("note", ""),
                "source": "XSTest (paul-rottger/xstest, CC BY 4.0)",
            })

    out_path = RAW_DIR / "xstest_contrast.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_safe = sum(1 for r in rows_out if r["xstest_label"] == "safe")
    n_unsafe = len(rows_out) - n_safe
    print(f"[prep] Wrote {len(rows_out)} XSTest prompts -> {out_path}")
    print(f"[prep]   {n_safe} safe (false-refusal test) / {n_unsafe} unsafe-contrast")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/xstest_raw.csv")
