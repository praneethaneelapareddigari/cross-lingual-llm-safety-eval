# Annotated Bibliography — Cross-Lingual LLM Safety

*Use this to populate the paper's Related Work / References section. Verify
exact citation format (venue, year, page numbers) against each paper's
official listing before submission — some of these are very recent (2026)
preprints and details may update.*

## Foundational multilingual safety benchmarks

- **XSafety** (Wang et al.) — first multilingual safety benchmark for LLMs,
  14 safety issues across 10 languages; found LLMs produce more unsafe
  responses to non-English queries. Uses translated data (no native prompts).

- **MultiJail** (Deng et al., 2024, EMNLP) — 315 English unsafe prompts,
  manually translated into Chinese, Italian, Vietnamese, Arabic, Korean,
  Thai, Bengali, Swahili, Javanese. Studies cross-lingual jailbreak attacks
  specifically.

- **XSTest** (Röttger et al.) — identified "exaggerated safety" / over-refusal
  as a distinct failure mode; source of our false-refusal contrast
  methodology.

- **SORRY-Bench** (Xie et al., 2025) — efficient benchmark for assessing
  refusal capabilities across a structured harm taxonomy; one of our primary
  seed sources for privacy/finance/medical/self-harm categories.

- **PolygloToxicityPrompts** (Jain et al., 2024) — native toxic content
  across 17 languages; emphasizes that native data captures nuances
  machine-translated data misses. Relevant to our RQ5 (translation artifact
  vs. genuine gap).

- **RTP-LX** (De Wynter et al., 2025) — toxicity evaluation dataset,
  European-language-heavy.

- **CIVICS** (Pistilli et al., 2024) — hand-crafted multilingual dataset of
  value-laden/socially-sensitive prompts; seed source for our politics
  category.

- **XTRUST** (Li et al., 2024) — multilingual benchmark across diverse
  topics including privacy and trust.

- **HONEST** (Nozza et al., 2021) — manually created templates for measuring
  hurtful sentence completions; seed source for hate category.

## Resource-tier / jailbreak-vulnerability findings

- **Yong, Menghini & Bach (2023)** — "Low-Resource Languages Jailbreak GPT-4"
  — showed low-resource-language translation alone can bypass GPT-4
  safeguards. Directly motivates RQ2.

- **Kanepajs et al. (2024)** — studied 24 official EU languages; found
  jailbreak ASR tends to be higher for low-resourced languages in GPT-4o and
  Mistral Large 2, via logistic regression against resource level. Closest
  prior methodology to our RQ2 analysis — cite as primary comparison point.

- **Deng et al. (2023/2024)** — multilingual jailbreak challenges more
  broadly; proposes mitigation methods (Self-Defence framework).

- **Röttger et al. (2025)** — survey finding 113/144 surveyed safety datasets
  (78.5%) are English-only. Use this figure directly in the Introduction to
  motivate the project.

## Regional / South Asian and other underrepresented-region work (2026, very recent)

- **IndicSafe** — South Asian LLM safety benchmark; uses GPT-4o as automated
  multilingual judge (SAFE/UNSAFE/REFUSAL/AMBIGUOUS) over 60,000 generations
  across chat models (Claude, GPT-4o, Grok-3) and instruction models (LLaMA,
  Mistral, Cohere). Closest prior work to ours in region — cite prominently
  and differentiate (they don't include Telugu/Tamil/Kannada as a distinct
  group with paired significance testing).

- **IndicGuard** — multilingual safety guard model + dataset for Indic
  languages, including low-resource scripts (Dogri, Konkani, Sanskrit);
  compares against CultureGuard and AEGIS 2.0. Relevant as a potential
  baseline classifier to compare your LLM-judge against.

- **TukaBench** — culturally grounded jailbreak benchmark for African
  languages; methodologically relevant (culturally-grounded adaptation, not
  pure MT) even though different region.

- **UbuntuGuard** — 10 African languages, human-verified adversarial queries
  built from GPT-5-generated seeds.

- **SomaliBench Eval** — measures English-to-Somali refusal gaps
  specifically; reuses HarmBench/AdvBench tradition without proposing new
  attacks, same design philosophy as this project ("we measure baseline
  refusal transfer across languages," not novel attacks).

- **LinguaSafe** — comprehensive multilingual safety benchmark; discusses
  the native-data-vs-translation distinction (PTP vs. MultiJail/XSafety) —
  useful citation for justifying our back-translation validation pipeline.

## How to position this project relative to the above

State explicitly in the paper: "IndicSafe and IndicGuard represent the
closest prior work geographically. Our contribution is (1) explicit inclusion
of Telugu, Tamil, and Kannada as a distinct South Indian sub-group alongside
Hindi, rather than treating South Asia as Hindi-representative; (2) a
harmonized 9-category taxonomy applied identically across all languages
rather than category-specific datasets; and (3) a full paired-statistics
layer (McNemar, Wilson CI, Cohen's h) rather than raw percentage reporting,
which lets us make calibrated claims about which language gaps are
statistically robust versus which may be noise at current sample sizes."
