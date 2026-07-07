"""
Local translation using NLLB-200 (facebook/nllb-200-distilled-600M by default),
via Hugging Face transformers. Fully local, no API key, runs on CPU (slower)
or GPU if available. Downloaded once (~2.5GB for the distilled-600M variant),
cached locally afterward.

This replaces the LLM-API-based translate.py approach — same role in the
pipeline (produce translated prompt drafts + flag low-fidelity ones for
human review), different engine.

Usage:
    python -m src.data_pipeline.translate_local --category violence --lang hi te ta kn

Install (not in requirements.txt by default since it pulls in torch):
    pip install --break-system-packages transformers torch sentencepiece sacremoses
"""
import argparse
import json
from pathlib import Path

from sacrebleu import sentence_bleu
from langdetect import detect

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# NLLB-200 uses FLORES-200 language codes, not ISO 639-1 — map ours to theirs.
NLLB_LANG_CODES = {
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "en": "eng_Latn",
}

BLEU_FLAG_THRESHOLD = 40.0  # same threshold as the API-based pipeline, for consistency

_model = None
_tokenizer = None
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # ~600M params, CPU-friendly.
# Swap to "facebook/nllb-200-1.3B" or "facebook/nllb-200-3.3B" for better
# quality if you have the RAM/GPU for it.


def _load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        print(f"[translate_local] Loading {MODEL_NAME} (first run downloads "
              f"~2.5GB, cached afterward)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return _model, _tokenizer


def translate_text(text: str, target_lang_code: str, source_lang_code: str = "eng_Latn") -> str:
    model, tokenizer = _load_model()
    tokenizer.src_lang = source_lang_code
    inputs = tokenizer(text, return_tensors="pt")
    target_id = tokenizer.convert_tokens_to_ids(target_lang_code)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=target_id,
        max_length=200,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def back_translate(text: str, source_lang_code: str) -> str:
    return translate_text(text, "eng_Latn", source_lang_code=source_lang_code)


def language_purity_ok(text: str, expected_lang_iso: str) -> bool:
    try:
        detected = detect(text)
    except Exception:
        return False
    return detected.split("-")[0] == expected_lang_iso


def translate_category(category: str, langs: list):
    seed_path = RAW_DIR / f"{category}_en_seed.jsonl"
    if not seed_path.exists():
        print(f"[translate_local] Missing seed file: {seed_path}.")
        return

    with open(seed_path, "r", encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    flagged = []

    for lang in langs:
        nllb_code = NLLB_LANG_CODES.get(lang)
        if not nllb_code:
            print(f"[translate_local] Unknown language code: {lang}, skipping.")
            continue

        out_rows = []
        for i, row in enumerate(seeds):
            draft = translate_text(row["text"], nllb_code)
            back = back_translate(draft, nllb_code)
            score = sentence_bleu(back, [row["text"]]).score
            purity_ok = language_purity_ok(draft, lang)

            out_row = {
                "id": row["id"],
                "text": draft,
                "source": row.get("source", "unknown"),
                "back_translation": back,
                "bleu_vs_original": round(score, 2),
                "language_purity_ok": purity_ok,
                "needs_human_review": score < BLEU_FLAG_THRESHOLD or not purity_ok,
                "translation_engine": MODEL_NAME,
            }
            out_rows.append(out_row)
            if out_row["needs_human_review"]:
                flagged.append({**out_row, "category": category, "lang": lang})

            if (i + 1) % 20 == 0:
                print(f"[translate_local] {category}/{lang}: {i + 1}/{len(seeds)} done")

        out_path = RAW_DIR / f"{category}_{lang}_DRAFT.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[translate_local] Wrote {len(out_rows)} drafts -> {out_path}")

    if flagged:
        flag_path = RAW_DIR / f"{category}_NEEDS_HUMAN_REVIEW.jsonl"
        with open(flag_path, "w", encoding="utf-8") as f:
            for r in flagged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[translate_local] {len(flagged)} items flagged for human review -> {flag_path}")

    print("[translate_local] *_DRAFT.jsonl files are drafts. Rename to "
          "<category>_<lang>.jsonl only after native-speaker review.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--lang", nargs="+", required=True)
    args = parser.parse_args()
    translate_category(args.category, args.lang)


if __name__ == "__main__":
    main()
