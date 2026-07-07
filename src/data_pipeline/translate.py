"""
Translation drafting + back-translation fidelity check.

This module produces MACHINE DRAFTS ONLY. Per sourcing_plan.md, every draft
must go through a native-speaker / professional review pass before being
treated as final. This script's job is to (a) produce a first-pass draft
efficiently, and (b) automatically flag low-fidelity translations for
priority human review via back-translation similarity.

Usage:
    python -m src.data_pipeline.translate --category violence --lang hi te ta kn
"""
import argparse
import json
import os
from pathlib import Path

from langdetect import detect
from sacrebleu import sentence_bleu
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

LANG_NAMES = {
    "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "kn": "Kannada",
    "ar": "Arabic", "fr": "French", "zh": "Chinese (Simplified)", "es": "Spanish",
}

# Translation quality thresholds — below these, flag for mandatory human review
BLEU_FLAG_THRESHOLD = 40.0  # back-translation BLEU vs. original English


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_translation_model(text: str, target_lang_name: str, client, model_name: str) -> str:
    """
    Uses an LLM as the translation engine (logged separately from models under
    test — this call is NOT part of the safety evaluation, only data prep).
    Swap in a dedicated MT API (e.g. Google Translate, DeepL) if preferred —
    professional MT engines are arguably a cleaner methodological choice since
    they're not the same model family being evaluated downstream.
    """
    prompt = (
        f"Translate the following text into {target_lang_name}. "
        f"Preserve the exact meaning, intent, and register (formal/informal) "
        f"of the original. Output ONLY the translation, nothing else.\n\n"
        f"Text: {text}"
    )
    resp = client.messages.create(
        model=model_name,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def back_translate(text: str, source_lang_name: str, client, model_name: str) -> str:
    prompt = (
        f"Translate the following {source_lang_name} text back into English. "
        f"Output ONLY the translation, nothing else.\n\nText: {text}"
    )
    resp = client.messages.create(
        model=model_name,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def fidelity_score(original_en: str, back_translated_en: str) -> float:
    return sentence_bleu(back_translated_en, [original_en]).score


def language_purity_ok(text: str, expected_lang: str) -> bool:
    try:
        detected = detect(text)
    except Exception:
        return False
    # langdetect uses different codes for some languages (zh-cn, etc.) — normalize
    detected = detected.split("-")[0]
    return detected == expected_lang


def translate_category(category: str, langs: list, client, model_name: str):
    seed_path = RAW_DIR / f"{category}_en_seed.jsonl"
    if not seed_path.exists():
        print(f"[translate] Missing seed file: {seed_path}. "
              f"Populate it first per sourcing_plan.md.")
        return

    with open(seed_path, "r", encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    flagged_for_review = []

    for lang in langs:
        lang_name = LANG_NAMES.get(lang)
        if not lang_name:
            print(f"[translate] Unknown language code: {lang}, skipping.")
            continue

        out_rows = []
        for row in seeds:
            draft = call_translation_model(row["text"], lang_name, client, model_name)
            back = back_translate(draft, lang_name, client, model_name)
            score = fidelity_score(row["text"], back)
            purity_ok = language_purity_ok(draft, lang)

            out_row = {
                "id": row["id"],
                "text": draft,
                "source": row.get("source", "unknown"),
                "back_translation": back,
                "bleu_vs_original": round(score, 2),
                "language_purity_ok": purity_ok,
                "needs_human_review": score < BLEU_FLAG_THRESHOLD or not purity_ok,
            }
            out_rows.append(out_row)
            if out_row["needs_human_review"]:
                flagged_for_review.append({**out_row, "category": category, "lang": lang})

        out_path = RAW_DIR / f"{category}_{lang}_DRAFT.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[translate] Wrote {len(out_rows)} drafts -> {out_path}")

    if flagged_for_review:
        flag_path = RAW_DIR / f"{category}_NEEDS_HUMAN_REVIEW.jsonl"
        with open(flag_path, "w", encoding="utf-8") as f:
            for r in flagged_for_review:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[translate] {len(flagged_for_review)} items flagged for mandatory "
              f"human review -> {flag_path}")
    print("[translate] IMPORTANT: *_DRAFT.jsonl files are drafts. Rename to "
          "<category>_<lang>.jsonl only after native-speaker review per "
          "sourcing_plan.md step 2.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--lang", nargs="+", required=True)
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL_NAME", "claude-sonnet-4-6"))
    args = parser.parse_args()

    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    translate_category(args.category, args.lang, client, args.model)


if __name__ == "__main__":
    main()
