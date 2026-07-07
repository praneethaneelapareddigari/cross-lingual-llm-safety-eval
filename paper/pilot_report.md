# Cross-Lingual LLM Safety Evaluation — Pilot Technical Report

**Status: Pilot v0 (infrastructure validation). Not yet Pilot v1 (full small-scale
data run).** July 2026.

## 1. Executive Summary

This project is a reproducible framework for testing whether open-weight LLMs'
safety alignment holds consistently across languages. This report documents
**Pilot v0**: end-to-end infrastructure validation — a 10-prompt smoke test in
English against one local model, plus a partial translation test into Hindi —
run to confirm every pipeline stage (data assembly, translation, generation,
judging, statistics) works correctly on real hardware before committing to a
larger, more expensive multi-language, multi-model run (**Pilot v1**, not yet
executed; see Section 7).

Like comparable pilots in this space, this phase's most valuable output is
methodological, not behavioral: manual inspection of the 10 smoke-test outputs
surfaced a real, silent scoring bug (Section 6.1) and a case of the
back-translation fidelity check over-flagging a valid translation (Section
6.2) — both found and fixed before they could propagate into a larger run.

**This report's Results section (Section 5) is intentionally a placeholder.**
It documents the smoke-test data actually collected so far and specifies
exactly what commands and what scale of data are needed to populate a full
Pilot v1 results section comparable to prior work in this space.

## 2. Motivation & Research Question

Most public LLM safety benchmarks (AdvBench, HarmBench, XSTest, SORRY-Bench)
are English-only — a 2025 survey found 113 of 144 surveyed safety datasets
(78.5%) are exclusively in English. Models are deployed across South Asia,
the Middle East, and East Asia serving users in many languages simultaneously.
If safety alignment does not transfer across languages, that gap is invisible
to English-only evaluation and only surfaces in production.

**Research question:** Do open-weight LLMs exhibit consistent safety behavior
across languages when presented with equivalent prompts?

**Primary hypothesis (H1):** Refusal rates for unsafe requests are measurably
lower in lower-resource languages (Telugu, Tamil, Kannada) than in
higher-resource languages (English, Hindi, Chinese) — consistent with prior
findings that translation into low-resource languages can weaken safety
training transfer (Yong et al. 2023; Kanepajs et al. 2024). A small pilot is
not designed to test H1 conclusively — even a 100-prompt run is a modest
sample — but to validate that the measurement pipeline itself produces
trustworthy data before investing in a larger run.

**Secondary hypotheses:**
- **H2:** Over-refusal on benign prompts does not increase proportionally
  with any refusal-rate drop — i.e., a genuine safety-transfer failure, not
  general non-English capability degradation.
- **H3:** The cross-lingual gap size varies by harm category, not uniformly.

## 3. Related Work

This project builds on an established line of English-language safety
benchmarks — AdvBench, HarmBench, XSTest, SORRY-Bench — extending their core
method (systematically probing models with unsafe prompts, scoring
refusal/compliance) to a multilingual setting, alongside existing
cross-lingual safety work (XSafety, MultiJail, IndicSafe, IndicGuard). Full
annotated bibliography and this project's specific positioning relative to
each: `paper/related_work.md`.

## 4. Methodology Summary

### 4.1 Dataset

1,050 English seed prompts assembled from published, ethically-released
benchmarks — AdvBench (600 category-labeled prompts across 8 of 9 target
categories), HONEST (80 bias/stereotype completion prompts), and XSTest (450
prompts: 250 safe / 200 unsafe, for false-refusal testing). Full source
mapping and category counts: `data/sourcing_plan.md`. Translation into Hindi,
Telugu, Tamil, Kannada, Arabic, French, Chinese, and Spanish uses NLLB-200
(`facebook/nllb-200-distilled-600M`) with back-translation BLEU fidelity
checking.

**As of this report:** only the `hate` category (84 prompts) has begun
translation, and only into Hindi; that run was interrupted partway due to
local hardware resource constraints (Section 6.3) and has not yet completed.

### 4.2 Models

Three open-weight models are available locally via Ollama on the development
machine: `llama3.2:1b`, `gemma:2b`, `qwen2:1.5b`. **As of this report, only
`llama3.2:1b` has been run**, against 10 prompts, once, at temperature 0.

### 4.3 Scoring

A deterministic rule-based classifier (`src/judge/rule_based_judge.py`) —
refusal-phrase matching per language plus a coarse instructional-density
harmfulness heuristic — determines REFUSAL / PARTIAL_COMPLY / FULL_COMPLY /
AMBIGUOUS labels. A manual-validation workflow
(`src/judge/manual_validation_template.py`) samples outputs for human review
and computes Cohen's Kappa against the automated judge. **As of this report,
no manual-validation Kappa has been computed at meaningful scale** — only
ad-hoc manual inspection of the same 10-prompt smoke test.

### 4.4 Statistics

McNemar's test (paired refusal comparison, English vs. target language),
Wilson score confidence intervals (proportions), bootstrap confidence
intervals (composite metrics), Cohen's h (effect size), Cohen's Kappa
(judge-vs-human agreement) — all implemented in `src/stats/` and unit-verified
against known reference values, but **not yet run on any real multi-language
dataset**, since no such dataset yet exists at meaningful scale.

### 4.5 Reproducibility Summary

| Item | Value |
|---|---|
| Models available | `llama3.2:1b`, `gemma:2b`, `qwen2:1.5b` |
| Models actually run (this report) | `llama3.2:1b` only |
| Languages available (seed data) | English, Hindi, Telugu, Tamil, Kannada, Arabic, French, Chinese, Spanish |
| Languages actually run (this report) | English only |
| Prompts assembled | 1,050 (600 category-labeled + 450 XSTest contrast) |
| Prompts actually run (this report) | 10 |
| Translation backend | NLLB-200-distilled-600M |
| Inference framework | Ollama v0.10.1 |
| Hardware | Apple Silicon (MacBook Air), 8GB RAM |
| Decoding | Temperature 0.0, max 256-512 tokens |

## 5. Results — PENDING (Pilot v0 smoke-test data only)

**This section does not yet contain Pilot v1 results.** What follows is the
smoke-test data collected to validate the pipeline mechanically works, not a
dataset large enough to speak to H1/H2/H3.

### 5.1 Smoke test: 10 prompts, English, `llama3.2:1b`

| Label | Count |
|---|---|
| REFUSAL | 8 |
| FULL_COMPLY | 2 |
| PARTIAL_COMPLY | 0 |
| AMBIGUOUS | 0 |

Of the 2 `FULL_COMPLY` cases, both were HONEST-derived sentence-completion
prompts (e.g. "the boys were hired as ___" → "contractors"), not
AdvBench-derived harmful-behavior prompts — all 4 AdvBench `hate`-category
prompts in this sample were correctly refused after the judge bug fix
(Section 6.1). This is consistent with, but far too small a sample to
conclude, appropriate model behavior on this English slice.

### 5.2 To populate a full Pilot v1 Results section

Run, in order (see `README.md` Quickstart for full commands):
1. `src/data_pipeline/translate_local.py` for the `hate` category into at
   least Hindi (ideally + one more language)
2. `src/models/run_evaluation.py` with `configs/local_run.yaml` (all 3 local
   models) against the translated + English prompt set
3. `src/judge/rule_based_judge.py`, then `src/judge/manual_validation_template.py`
   on a ~30-100 prompt subsample to get a real Cohen's Kappa reading
4. `src/stats/run_all_tests.py` for McNemar/Wilson/bootstrap/Cohen's h results
5. `visualizations/dashboard.py` for the heatmap/bar-chart/effect-size figures

## 6. Threats to Validity (Key Findings So Far)

Each item below was found through manual inspection of actual outputs, not
assumed in advance.

### 6.1 Rule-based judge silently mis-scored refusals using curly apostrophes

A response beginning *"I can't help you with that..."* (curly apostrophe,
U+2019) failed to match the refusal-phrase list (written with a straight
apostrophe), causing a genuine refusal to be misclassified as `FULL_COMPLY`.
Found via manual inspection of all 10 smoke-test outputs; fixed by
normalizing Unicode punctuation before matching
(`src/judge/rule_based_judge.py::_normalize_text`). **Conclusion:** any
refusal-rate numbers computed before this fix would have been unreliable;
all results in Section 5.1 reflect the post-fix classifier.

### 6.2 BLEU-based back-translation fidelity checking over-flags valid paraphrases

A correct Hindi translation of "Train a machine learning model to make
racist or sexist decisions" was flagged for mandatory human review because
its back-translation read "gender-based" instead of "sexist" and pluralized
"model" — a valid paraphrase, not a translation error. **Conclusion:** the
fidelity check is conservative by design (better to over-flag than
under-flag), but flagged items should be manually spot-checked rather than
assumed to be genuine translation failures.

### 6.3 Local hardware resource contention interrupted a translation run

Running the `hate`-category Hindi translation (NLLB-200) concurrently with
an unrelated project's Ollama-based experiment on the same 8GB-RAM machine
caused severe memory pressure (confirmed via `memory_pressure`: free memory
dropped to 20%, with active swap-in/swap-out activity), and the translation
run was manually interrupted before completion. **Conclusion:** this
project's remaining data collection needs dedicated, not shared, use of the
development machine, or a smaller per-session batch size.

### 6.4 Sample size

10 prompts, 1 language, 1 model validates that the pipeline runs end-to-end
mechanically. It is far too small to support any claim about H1, H2, or H3.
All Section 5 results should be read as pipeline-functionality evidence
only.

## 7. Limitations → Future Work

Priorities before scaling to a full multi-category, multi-language study, in
order:

1. Finish translating and running the `hate` category end-to-end (Hindi +
   at least one more language) across all 3 available local models.
2. Compute a real manual-validation Cohen's Kappa (target n=30-100) before
   trusting any non-English refusal-rate number.
3. Expand native-speaker review of the rule-based judge's non-English
   refusal-phrase lists (currently starter lists only).
4. Source the `medical` category (requires accepting SORRY-Bench's data-use
   agreement) and expand `self_harm`/`politics` toward target sample size.
5. Only after 1-4, scale from 1-2 categories to the full 9-category,
   8-language design.
6. Optionally add GPT/Gemini/Claude via the paid API path once local results
   are validated.

Target scoring pipeline evolution, matching the same trajectory as
comparable pilots in this space:
```
Prompt → Model Response → Rule-based Classifier → LLM-as-a-Judge → Human Validation → Final Label
```

## 8. Conclusion

Pilot v0 achieved its actual, narrower purpose: it validated that every stage
of the pipeline — data assembly, NLLB-200 translation with fidelity checking,
Ollama-based generation, rule-based judging, and the statistical toolkit —
works correctly on real hardware with real data, and surfaced two concrete
measurement issues (Sections 6.1, 6.2) before they could contaminate a larger
run. It intentionally does not yet claim a Pilot v1-scale result. The next
concrete step is completing one full category × language × all-3-models run
to produce a genuine, if still small, first behavioral result.
