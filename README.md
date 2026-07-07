# Cross-Lingual / Cross-Regional LLM Safety Evaluation

**Goal:** Measure whether frontier LLM safety behavior (refusal, harmfulness, over-refusal,
bias) differs systematically across languages and regions — with statistically rigorous,
reproducible comparisons rather than anecdotal spot-checks.

> Built for: AI Safety Fellowship submission (Singapore track)

## Why this project, given prior work

Several benchmarks already touch this space: XSafety, MultiJail, SORRY-Bench, XSTest,
PolygloToxicityPrompts, RTP-LX, CIVICS, XTRUST, and very recent South-Asia-specific work
(IndicSafe, IndicGuard). See `paper/related_work.md` for the full mapping.

**This project's distinct contribution:**
1. Simultaneous coverage of Hindi, Telugu, Tamil, Kannada (South Indian languages are
   under-represented as a *group* — most prior South Asian work is Hindi/Bengali-first)
   alongside Arabic, French, Chinese, Spanish and English, for direct cross-regional
   comparison, not just cross-language.
2. A full 9-category harm taxonomy tested consistently across all languages (most prior
   work uses 3-5 categories or is single-domain, e.g. only toxicity or only jailbreak).
3. Current-generation frontier models (GPT-5-class, Gemini, Claude, Llama, Gemma, Qwen)
   evaluated on the same day/week to avoid model-drift confounds.
4. A rigorous statistical layer (McNemar, Wilson CI, bootstrap CI, Cohen's h, Cohen's
   Kappa) applied uniformly — most prior work reports raw percentages without paired
   significance testing or effect sizes.

## Repo layout

```
data/
  categories.yaml           # 9-category harm taxonomy + severity tiers
  sourcing_plan.md          # which upstream benchmark feeds each category/language
  raw/                      # (gitignored) licensed/gated source prompts go here
src/
  data_pipeline/            # load, translate, back-translate, validate
  models/                   # unified API client for all 6 model families
  judge/                    # LLM-as-judge harmfulness/refusal classifier
  stats/                    # McNemar, Wilson CI, bootstrap, Cohen's h/kappa
paper/
  proposal.md               # full research protocol (methods section ready to submit)
  related_work.md           # annotated bibliography of prior benchmarks
visualizations/
  dashboard.py              # generates the core paper figures from results/
results/                    # (gitignored) raw model outputs + judge scores land here
```

## IMPORTANT: what this repo does NOT contain

This repo does **not** ship a novel list of harmful prompts. The data pipeline is built
to *pull from and adapt already-published, ethically-released safety benchmarks*
(AdvBench, HarmBench, XSTest, SORRY-Bench, etc. — see `data/sourcing_plan.md`), not to
generate fresh harmful content. Raw prompt/response pairs and judge verdicts are kept out
of version control (`results/`, `data/raw/`) and should be shared only under a data-use
agreement, consistent with how HarmBench/AdvBench-derived work is released. Only
aggregated statistics, code, and category-level metadata are meant to be public.

## Two ways to run this: local (free) or API-based (paid)

**Local path (recommended to start — zero cost, no API keys):**
Uses NLLB-200 (translation) + Ollama (Llama/Gemma/Qwen generation) + a
rule-based judge, validated against manual review on a subsample via
Cohen's Kappa. This mirrors a workflow already proven to work end-to-end on
similar projects, and sidesteps the two failure points paid APIs commonly
hit (incomplete billing setup, or a 0-quota free tier).

```bash
pip install -r requirements.txt -r requirements-local.txt
python -m src.models.ollama_client              # sanity-check Ollama is reachable
ollama pull llama3.3 && ollama pull gemma2 && ollama pull qwen2.5

python -m src.data_pipeline.loader --category all --languages en
python -m src.data_pipeline.translate_local --category hate --lang hi te ta kn ar fr zh es
# review data/raw/*_NEEDS_HUMAN_REVIEW.jsonl, then rename *_DRAFT.jsonl -> final names
python -m src.data_pipeline.loader --category all --languages all
python -m src.models.run_evaluation --config configs/local_run.yaml
python -m src.judge.rule_based_judge --input results/raw_outputs_local.jsonl \
    --output results/judged_outputs.jsonl

# Validate the automated judge against manual review (recommended before
# trusting numbers at scale — same workflow that already worked elsewhere):
python -m src.judge.manual_validation_template --input results/judged_outputs.jsonl \
    --sample-size 100 --out results/manual_review_sample.jsonl
python -m src.judge.manual_validation_template --review results/manual_review_sample.jsonl
python -m src.judge.manual_validation_template --score results/manual_review_sample.jsonl

python -m src.stats.run_all_tests --input results/judged_outputs.jsonl
python visualizations/dashboard.py
```

**API path (paid, needed only for GPT/Gemini/Claude — no local equivalent exists):**
```bash
pip install -r requirements.txt
cp .env.example .env        # add your API keys
python -m src.data_pipeline.translate --category violence --lang hi te ta kn ar fr zh es
python -m src.models.run_evaluation --config configs/full_run.yaml
python -m src.judge.classify --input results/raw_outputs.jsonl
python -m src.stats.run_all_tests --input results/judged_outputs.jsonl
python visualizations/dashboard.py
```

Both paths write to the same `results/judged_outputs.jsonl` schema, so
`src/stats/run_all_tests.py` and the dashboard work unmodified regardless of
which path produced the data — you can even run local models first, add
GPT/Gemini/Claude later via the API path, and merge both output files before
the stats step.

