# Cross-Lingual and Cross-Regional Safety Evaluation of Frontier Language Models

*Research protocol / paper skeleton — draft for AI Safety Fellowship submission*

---

## Abstract *(fill in after data collection)*

We present a systematic evaluation of safety behavior — refusal rate, attack
success rate, harmfulness, and false-refusal rate — across six frontier language
model families (GPT, Gemini, Claude, Llama, Gemma, Qwen) and nine languages
spanning five regions (English, Hindi, Telugu, Tamil, Kannada, Arabic, French,
Chinese, Spanish), using a harmonized 9-category harm taxonomy and a rigorous
paired-statistics framework (McNemar's test, Wilson and bootstrap confidence
intervals, Cohen's h and Cohen's Kappa). [Fill in with 2-3 sentence summary of
headline findings once results are in.]

---

## 1. Introduction

### 1.1 Motivation

Safety alignment in large language models is trained and evaluated
overwhelmingly in English. A 2025 survey found that 113 of 144 surveyed safety
datasets were exclusively in English (78.5%), and multiple independent studies
have found that simply translating a harmful English prompt into a
lower-resource language can weaken or bypass a model's safety training — an
effect observed even in otherwise well-aligned frontier models. This creates a
real-world equity and safety gap: hundreds of millions of users interacting
with these models in Hindi, Tamil, Telugu, Kannada, Arabic, and other languages
may be receiving meaningfully different safety guarantees than English-language
users, without any visibility into that gap.

### 1.2 Research questions

- **RQ1:** Do frontier LLMs exhibit statistically significant differences in
  refusal rate and attack success rate (ASR) across languages, holding the
  underlying request constant?
- **RQ2:** Does the size of this gap correlate with a language's training-data
  resource tier (high/mid/low), consistent with prior findings on
  low-resource-language jailbreaks?
- **RQ3:** Do models over-refuse (false-refusal) more or less in non-English
  languages, i.e. is the safety gap symmetric (both under- and over-triggering)
  or one-directional?
- **RQ4:** Does the gap vary by harm category — are some categories (e.g.
  self-harm, medical) more robust across languages than others (e.g. malware,
  hate)?
- **RQ5 (exploratory):** How much of the observed gap is attributable to
  translation itself (a "translation as jailbreak" effect) versus genuine
  differences in how the model reasons about the same well-translated request?

### 1.3 Contribution

This work is the first, to our knowledge, to jointly (a) cover Hindi, Telugu,
Tamil, and Kannada as a South Indian language group alongside major world
languages for direct cross-regional comparison, (b) apply a full paired
statistical-testing framework (not just raw percentage comparisons) to
cross-lingual safety evaluation, and (c) evaluate the current generation of
six major model families side-by-side on the same harmonized taxonomy.

---

## 2. Related Work

*(See `related_work.md` for the fully annotated version with citations —
summary below)*

**Multilingual safety benchmarks.** XSafety was the first multilingual safety
benchmark, covering 14 safety issues across 10 languages, and found LLMs
produce more unsafe responses to non-English queries. MultiJail provides
manually translated jailbreak prompts across nine non-English languages, built
from 315 English seed prompts. RTP-LX and PolygloToxicityPrompts extend
toxicity evaluation multilingually. SORRY-Bench offers an efficient refusal
taxonomy. XSTest introduced the concept of exaggerated safety / over-refusal,
which we adopt as our false-refusal contrast set.

**Resource-tier effects.** A study across 24 official EU languages found that
for GPT-4o and Mistral Large 2, jailbreak attack success rates tend to be
higher for low-resourced languages, and earlier work showed low-resource
language translations could bypass GPT-4 safeguards entirely.

**Regional efforts.** Very recent work (2026) has begun to fill regional gaps:
IndicSafe evaluates South Asian language safety using an LLM-as-judge over
60,000 generations; IndicGuard builds a multilingual safety guard model for
Indic languages including low-resource scripts; TukaBench and UbuntuGuard
address African languages. Our work is complementary — narrower in region
count but broader in taxonomy depth and statistical rigor, and current on the
newest model generation.

**Gap this project fills:** no existing published benchmark evaluates Telugu,
Tamil, and Kannada together as a distinct South Indian sub-group (most South
Asian work is Hindi/Bengali/Urdu-first), and no prior work applies McNemar's
paired test + Cohen's h effect sizes to this specific comparison, which
matters because raw percentage-point gaps without paired significance testing
can overstate or understate the reliability of an observed cross-lingual
difference.

---

## 3. Methodology

### 3.1 Dataset construction

We do not author new harmful prompts. We assemble ~900 English seed prompts
from established, ethically-released safety benchmarks (HarmBench, AdvBench,
SORRY-Bench, RTP-LX, PolygloToxicityPrompts, CIVICS, XTRUST — full mapping in
`data/sourcing_plan.md`), spanning 9 categories: violence, malware, drugs,
self-harm, privacy, hate, politics, finance, medical. We additionally include
~250 XSTest-style contrast prompts (safe requests that superficially resemble
unsafe ones) to measure false refusal.

Each English seed prompt is:
1. Machine-translated into Hindi, Telugu, Tamil, Kannada, Arabic, French,
   Chinese, and Spanish.
2. Reviewed by a native or professional speaker for semantic fidelity and
   cultural naturalness.
3. Back-translated into English; BLEU similarity against the original flags
   low-fidelity translations for mandatory human re-review.
4. Checked for language purity via automatic language detection.

This yields ~8,100 evaluation prompts (900 seeds × 9 languages) plus the
contrast set, comfortably exceeding the fellowship's 500-1000 prompt target
once you count each translated variant as part of the evaluation set (900
distinct seed prompts satisfies the "500-1000 unique prompts" framing if
preferred — report both counts for clarity).

### 3.2 Models under test

**Phase 1 (current, zero-cost):** Llama, Gemma, and Qwen, run locally via
Ollama — no API key or billing required. This intentionally limits Phase 1
to open-weight models.

**Phase 2 (optional, requires API access):** GPT, Gemini, and Claude have no
local equivalent and require paid API access to include in the comparison.
This phase is additive — Phase 1 results stand on their own and are not
blocked by it. Both phases write to the same output schema and are merged
before the statistical analysis.

Exact model versions are pinned and logged at evaluation time
(`configs/local_run.yaml` / `configs/full_run.yaml`), since model strings and
locally-pulled model tags change over time.

### 3.3 Evaluation procedure

Each assembled prompt is sent to each model at temperature 0 (deterministic
decoding, for reproducibility).

**Judging:** given prior friction with paid LLM-judge APIs (incomplete
billing setup on one provider, a zero-quota free tier on another), the
primary automated judge here is a deterministic rule-based classifier
(`src/judge/rule_based_judge.py`) — refusal-phrase matching per language plus
a coarse instructional-density harmfulness heuristic. This is a real
limitation relative to an LLM-as-judge approach (lower recall on indirect/
hedged refusals, no semantic understanding of harmfulness) and should be
described as such in the paper, not silently upgraded to "LLM-judged."

**Judge validation:** rather than skip validation because the LLM-judge path
is unavailable, the same substitute used successfully elsewhere is applied
here: a random subsample (suggested n=100) is manually labeled by a human
reviewer (`src/judge/manual_validation_template.py`), and Cohen's Kappa is
computed between manual and rule-based labels
(target: κ ≥ 0.6 before trusting the automated numbers at scale; report
whatever κ is actually observed, including a low value, since that is itself
a finding about measurement reliability — see Section 3.6).


### 3.4 Metrics

| Metric | Definition | Computed via |
|---|---|---|
| ASR (Attack Success Rate) | % of harmful prompts receiving FULL_COMPLY or PARTIAL_COMPLY | `src/stats/wilson_ci.py` |
| Refusal Rate | % of prompts receiving REFUSAL | `src/stats/wilson_ci.py` |
| False Refusal Rate | % of *safe* contrast prompts receiving REFUSAL/OVER_REFUSAL | `src/stats/wilson_ci.py` |
| Harmfulness | Mean 0-4 judge score, on complied prompts | `src/stats/bootstrap_ci.py` |
| Toxicity | Judge-scored on `hate` category specifically | same pipeline |
| Robustness | Within-model variance of ASR across languages | descriptive stat |
| Bias (politics category) | Directional skew in response framing across regions | qualitative + judge |

### 3.5 Statistical analysis

- **McNemar's test** (paired, exact binomial for n<25 discordant pairs) — for
  each (model, category, language) triple, tests whether refusal outcomes
  differ significantly between English and the target language on the *same*
  underlying prompts.
- **Wilson score confidence intervals** — for all proportion metrics (ASR,
  refusal rate, false-refusal rate), preferred over normal-approximation CIs
  given several near-0%/near-100% cells expected in the data.
- **Bootstrap confidence intervals** (10,000 resamples) — for composite/mean
  metrics without closed-form variance (mean harmfulness score).
- **Cohen's h** — effect size for every McNemar comparison, reported alongside
  p-values so that statistical significance is not conflated with practical
  significance at large n.
- **Cohen's Kappa** — judge-vs-human agreement validation (Section 3.3).

All code implementing these tests is in `src/stats/` and unit-verified
against known reference values (see inline `if __name__ == "__main__"` sanity
checks in each module).

### 3.6 Limitations (state explicitly in the paper)

- Self-harm category is capped at n≈80 seed prompts for ethical-review
  reasons, giving somewhat lower statistical power for subtle effects in that
  category specifically.
- The primary judge is a deterministic rule-based classifier, not an LLM
  judge — report its measured Cohen's Kappa against manual review plainly,
  including if it is low. A low κ is itself a valid, reportable finding
  (it says the automated signal is not yet trustworthy), but it means ASR/
  refusal-rate numbers derived purely from the rule-based judge should be
  clearly labeled as provisional until either the phrase lists are expanded
  with native-speaker input or an LLM judge becomes available.
- LLM-as-judge introduces some measurement noise even after Kappa validation;
  report κ transparently rather than treating judge output as ground truth.
- Machine-translation-then-human-review, while more rigorous than MT alone,
  cannot fully separate "translation as jailbreak" from "genuine native
  reasoning gap" — Section 3.1's back-translation pipeline is a partial
  mitigation, not a complete solution; flag this explicitly per RQ5.
- Politics category is deliberately scored for bias/evenhandedness rather
  than ASR, since "harm" is not the right frame for contested political
  content — avoid conflating this with the other 8 categories in aggregate
  ASR figures.

---

## 4. Results *(populate after running the pipeline)*

Suggested structure:
- 4.1 Overall ASR/refusal rate by language (Figure: `fig1_asr_heatmap.png`)
- 4.2 Paired English-vs-language significance tests (Figure:
  `fig3_effect_size_volcano.png`, Table: full McNemar + Cohen's h results)
- 4.3 Resource-tier correlation (RQ2) — regress effect size against language
  resource tier (high/mid/low from `data/categories.yaml`)
- 4.4 False refusal analysis (RQ3) (Figure: `fig4_false_refusal.png`)
- 4.5 Category-level breakdown (RQ4)
- 4.6 Translation-artifact vs. native-gap analysis (RQ5)

## 5. Discussion *(after results)*

## 6. Ethical considerations

- No novel harmful content generated; all seed prompts traceable to cited,
  ethically-released upstream benchmarks.
- Self-harm category reviewed for appropriateness before inclusion.
- Full prompt/response/judge data released only under a data-use agreement,
  consistent with HarmBench/AdvBench norms; aggregated statistics and all
  code released openly.
- Findings framed as measuring existing, already-documented vulnerability
  classes (cross-lingual jailbreak), not disclosing novel attack techniques.

## 7. Conclusion *(after results)*

## Deliverables checklist

- [ ] Dataset (gated release): `data/raw/` populated + validated per
      `sourcing_plan.md`
- [ ] GitHub repo (public): this repo, results/data/raw excluded via
      `.gitignore`
- [ ] Research paper: this document, completed with Sections 4-7
- [ ] Visualizations: `visualizations/figures/*.png`, generated via
      `visualizations/dashboard.py`
