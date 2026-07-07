# Cross-Lingual LLM Safety Evaluation

![python](https://img.shields.io/badge/python-3.10+-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-pilot%20in%20progress-yellow) ![pipeline](https://img.shields.io/badge/pipeline-fully%20local-brightgreen)

**Status: 🚧 Active Pilot** — Full writeup: [`paper/pilot_report.md`](paper/pilot_report.md)

---

## Abstract

We present a multilingual benchmark for evaluating whether large language model safety behavior transfers consistently across languages. The benchmark spans 9 languages, 9 safety-critical domains, and 1,050 curated prompts, and combines local translation (NLLB-200), model evaluation (Ollama), paired statistical testing (McNemar, Wilson CI, Bootstrap, Cohen's h/κ), and a fully reproducible zero-cost infrastructure. The current pilot validates the evaluation pipeline and establishes the methodology for large-scale multilingual safety benchmarking.

This repository documents the complete research lifecycle — from benchmark design and infrastructure validation to statistical methodology and reproducible experimentation — rather than presenting only final results.

---

## Repository Highlights

| | |
|---|---|
| 🌍 Languages | 9 (English, Hindi, Telugu, Tamil, Kannada, Arabic, French, Chinese, Spanish) |
| 📂 Harm categories | 9 |
| 📝 Seed prompts | 1,050 (sourced from AdvBench, XSTest, HONEST) |
| 🤖 Models | **Implemented:** Llama ✓ · Gemma ✓ · Qwen ✓ &nbsp;&nbsp; **Planned:** GPT ○ · Claude ○ · Gemini ○ |
| 📊 Statistical tests | 5 (McNemar, Wilson CI, Bootstrap CI, Cohen's h, Cohen's κ) |
| 💻 Pipeline | Fully local — NLLB-200 translation + Ollama generation, zero API keys |
| 📄 Report | [Pilot v0 technical report](paper/pilot_report.md) |

---

## Repository Maturity

```
Infrastructure  ██████████  100%
Dataset         ███████░░░   70%
Experiments     ███░░░░░░░   30%
Paper           ████░░░░░░   40%
Documentation   █████████░   90%
```

---

## Project Status

| Component | Status |
|---|---|
| Research design + taxonomy | ✅ Complete |
| Seed dataset (1,050 prompts) | ✅ Complete |
| Translation pipeline (NLLB-200) | ✅ Built + tested on real hardware |
| Generation pipeline (Ollama) | ✅ Built + tested on real hardware |
| Judge (rule-based + manual validation) | ✅ Built — bug found and fixed during smoke test |
| Statistical toolkit | ✅ All 5 tests implemented + unit-verified |
| Multilingual translation run | 🚧 In progress (`hate` category → Hindi, interrupted) |
| Full multi-language experiment | 🚧 Pending |
| Manual judge validation (Cohen's κ) | 🚧 Pending |
| Statistical analysis on real data | 🚧 Pending |
| `medical` category data | ❌ Gated (SORRY-Bench data-use agreement required) |

---

## Why This Matters

- Most safety benchmarks evaluate only English, despite global LLM deployment
- Prior cross-lingual work typically covers 3–5 languages and reports raw percentages without paired significance testing
- South Indian languages (Telugu, Tamil, Kannada) are systematically under-represented — even South Asia-focused work is usually Hindi/Bengali-first

## Research Question

> Do LLMs exhibit consistent safety behavior across languages when presented with equivalent prompts, and does any inconsistency correlate with a language's training-data resource level?

## Hypotheses

| Hypothesis | Description |
|---|---|
| H1 | Lower-resource languages (Telugu, Tamil, Kannada) exhibit measurably lower refusal rates than higher-resource languages (English, Hindi, Chinese) |
| H2 | Over-refusal on benign prompts does not increase proportionally with any refusal-rate drop — i.e., a genuine safety-transfer failure, not general capability degradation |
| H3 | The cross-lingual safety gap varies by harm category, not uniformly |

## Contributions

This project contributes:

- A multilingual LLM safety benchmark spanning 9 languages and 9 harm categories
- A fully local, reproducible evaluation pipeline (zero API keys, zero cost)
- A paired statistical evaluation methodology (McNemar + effect sizes, not raw percentages)
- A curated, source-traceable multilingual prompt dataset
- An openly documented pilot methodology including threats to validity

## How This Compares to Prior Work

 | Feature | AdvBench | HarmBench | XSTest | XSafety | This Work |
|---|---|---|---|---|---|
| Languages | 1 | 1 | 1 | 10 | **9** |
| Harm categories | Broad | 10+ | 2 | 10 | **9** |
| Paired statistics | ❌ | Limited | ❌ | ❌ | **✅** |
| Effect sizes | ❌ | ❌ | ❌ | ❌ | **✅** |
| Statistical significance testing | ❌ | Limited | ❌ | ❌ | **✅** |
| Open-source reproducible pipeline | ❌ | ❌ | ❌ | ❌ | **✅** |
| Technical report included | ❌ | ❌ | ❌ | ❌ | **✅** |
| South Indian languages | ❌ | ❌ | ❌ | ❌ | **✅** |

## Architecture

```mermaid
graph TD
    A[Seed] --> B[Translate]
    B --> C[Generate]
    C --> D[Judge]
    D --> E[Statistics]
    E --> F[Visualize]
    F --> G[Report]
```

The evaluation pipeline consists of five stages: (1) dataset construction from published benchmarks, (2) NLLB-200 translation with back-translation fidelity QA, (3) model inference via Ollama at temperature 0, (4) safety evaluation via rule-based judge with manual validation, and (5) paired statistical analysis across language pairs.

## Pilot Infrastructure Findings

![Pilot v0 smoke test label breakdown](visualizations/figures/smoke_test_label_breakdown.png)

Two real issues found through actually running the pipeline — not assumed:

1. **The rule-based judge silently mis-scored refusals using curly apostrophes.** A response with a curly `'` (U+2019) failed to match the phrase list written with a straight `'`, causing a genuine refusal to be misclassified as `FULL_COMPLY`. Found via manual inspection of 10 smoke-test outputs, fixed in `src/judge/rule_based_judge.py`. This is exactly the class of silent measurement error a smoke test is designed to catch before it contaminates results at scale.

2. **BLEU-based fidelity checking over-flags valid paraphrases.** A correct Hindi translation was flagged for human review because the back-translation used "gender-based" instead of "sexist" — a valid paraphrase, not a translation error. The fidelity check is intentionally conservative, but flagged items should be manually spot-checked rather than assumed to be genuine failures.

## Threats to Validity

| Threat | Description |
|---|---|
| Translation ambiguity | NLLB-200 may introduce semantic drift; back-translation BLEU-checking catches the worst cases but not all |
| Translation quality drift | NLLB-200 performance varies by domain — safety-critical phrasing may translate less reliably than general text |
| Judge reliability | Rule-based classifier has sparse non-English phrase lists; manual κ validation is required before trusting non-English numbers |
| Sample size | Current partial run validates pipeline functionality only — insufficient to confirm or reject H1 |
| Model version drift | Ollama model tags may update; exact versions are logged at evaluation time |
| Hardware constraints | 8GB-RAM machine constrained experiment scale and caused one translation run interruption |

## Open Questions

- Does multilingual RLHF coverage explain refusal-rate gaps across languages?
- Can back-translation quality scores predict per-language safety measurement reliability?
- Does the cross-lingual safety gap correlate with harm category severity, or is it uniform?
- Are refusal patterns consistent across open-weight model families, or model-family-specific?

## Roadmap

| Version | Milestone | Status |
|---|---|---|
| Pilot v0 | Infrastructure validation + smoke test | ✅ |
| Pilot v1 | `hate` category, 2 languages, 3 models, real κ | 🚧 |
| Benchmark v1 | All 9 categories, 8 languages, full statistical analysis | ⬜ |
| Paper Draft | Results writeup + threats to validity | ⬜ |
| Dataset Release | Public release under data-use agreement | ⬜ |

## Reproducibility

**Local path (zero cost, no API keys):**
```bash
pip install -r requirements.txt -r requirements-local.txt
python -m src.models.ollama_client        # confirm Ollama is reachable
ollama pull llama3.2:1b

python -m src.data_pipeline.loader --category hate --languages en
python -m src.data_pipeline.translate_local --category hate --lang hi
python -m src.models.run_evaluation --config configs/local_run.yaml
python -m src.judge.rule_based_judge --input results/raw_outputs_local.jsonl \
    --output results/judged_outputs.jsonl
python -m src.judge.manual_validation_template --input results/judged_outputs.jsonl \
    --sample-size 50 --out results/manual_review_sample.jsonl
python -m src.stats.run_all_tests --input results/judged_outputs.jsonl
```

See [`paper/pilot_report.md`](paper/pilot_report.md) for the full experimental protocol and [`data/sourcing_plan.md`](data/sourcing_plan.md) for dataset documentation.

## Repository Structure

```
cross-lingual-llm-safety-eval/
├── data/
│   ├── categories.yaml         # 9-category harm taxonomy
│   ├── sourcing_plan.md        # upstream source per category/language
│   └── raw/                   # seed prompts (AdvBench/HONEST/XSTest)
├── src/
│   ├── data_pipeline/           # NLLB-200 translation + loader
│   ├── models/                 # Ollama client + paid-API providers
│   ├── judge/                 # rule-based judge + manual validation
│   └── stats/                  # McNemar, Wilson CI, bootstrap, Cohen's h/κ
├── visualizations/             # dashboard + generated figures
├── paper/
│   ├── pilot_report.md         # Pilot v0 technical report
│   ├── proposal.md             # full research protocol
│   └── related_work.md         # annotated bibliography
├── configs/                    # local/full/smoke-test run configs
└── scripts/                   # data-prep + figure-generation scripts
```

## Ethics

This project evaluates existing public model behavior — no models are fine-tuned to be more harmful, and no novel harmful prompts are authored. All seed prompts are sourced from and traceable to already-published, ethically-released benchmarks. See [`data/sourcing_plan.md`](data/sourcing_plan.md) for full provenance.

## Citation

```bibtex
@misc{neelapareddigari2026crosslingual,
  author = {Neelapareddigari, Praneetha},
  title  = {Cross-Lingual LLM Safety Evaluation},
  year   = {2026},
  url    = {https://github.com/praneethaneelapareddigari/cross-lingual-llm-safety-eval}
}
```

## License

MIT
