# Data Sourcing Plan

## Current status (updated after first data-pull pass)

| Category | English seeds | Status |
|---|---|---|
| violence | 255 | ✅ pulled from AdvBench (public GitHub, `llm-attacks/llm-attacks`) |
| malware | 120 | ✅ pulled from AdvBench |
| finance | 58 | ✅ pulled from AdvBench |
| hate | 84 | ✅ 4 from AdvBench + 80 from HONEST (public GitHub, `MilaNLProc/honest`) |
| privacy | 24 | ✅ pulled from AdvBench |
| drugs | 23 | ✅ pulled from AdvBench |
| self_harm | 21 | ⚠️ pulled from AdvBench only — below target n=80; needs supplementing and mandatory ethics review before use |
| politics | 15 | ⚠️ pulled from AdvBench only — below target; CIVICS would be the ideal supplement but is HuggingFace-gated, see note below |
| medical | 0 | ❌ not yet sourced — AdvBench has no medical-harm prompts |
| contrast (false-refusal) | 450 (250 safe / 200 unsafe) | ✅ pulled from XSTest (public GitHub, `paul-rottger/xstest`, CC BY 4.0) |

**Total real, source-traceable English prompts so far: 1,050** (600 category
seeds + 450 contrast). Category weights don't yet match the equal-100-per-
category target in `categories.yaml` — AdvBench's own composition skews
toward violence/malware, exactly the kind of imbalance SORRY-Bench's authors
built their dataset to correct. Two categories (self_harm, politics) and one
entirely (medical) need additional sourcing before the full run.

**Access notes:**
- **SORRY-Bench** — the best remaining source for medical/self-harm/finance
  balance — is gated behind a click-through data-use agreement on
  HuggingFace ("You may not use the SORRY-Bench Dataset if you do not accept
  this Agreement"). I didn't route around this. If you want this category
  filled from SORRY-Bench, you (a human with an accountable identity) need to
  accept that agreement yourself at
  https://huggingface.co/datasets/sorry-bench/sorry-bench-202503, download
  it, and drop it in `data/raw/` — at which point the loader will pick it up.
- **CIVICS** (best source for politics) is HuggingFace-only
  (`llm-values/CIVICS`) — same manual-download situation, though its license
  is more permissive (no click-through agreement found), so this is just a
  network-access limitation on my end, not a gating issue.
- **RTP-LX / PolygloToxicityPrompts** — not yet located at an ungated,
  fetchable URL; worth a manual check on HuggingFace.


## Principle

We do not author new harmful prompts. We **assemble, license-check, and adapt** prompts
from existing, peer-reviewed / widely-used safety benchmarks, then translate + validate
them into the 8 target languages. This is both more rigorous (reviewers trust
established seed sets) and avoids the ethical/legal overhead of generating new harmful
content from scratch.

## Seed benchmark → category mapping

| Category    | Primary source(s)              | License note (verify before use)                  | Approx. usable English prompts |
|-------------|--------------------------------|----------------------------------------------------|---------------------------------|
| Violence    | HarmBench, AdvBench            | Research-use released; check current repo terms    | ~100+ |
| Malware     | HarmBench                      | Research-use released                               | ~100+ |
| Drugs       | AdvBench, SORRY-Bench          | Research-use released                               | ~100+ |
| Self-harm   | SORRY-Bench                    | Research-use; extra ethics review required          | ~80 |
| Privacy     | SORRY-Bench, XTRUST             | Research-use                                        | ~100 |
| Hate        | RTP-LX, PolygloToxicityPrompts, HONEST | Research-use, some CC-licensed                | ~100+ |
| Politics    | CIVICS, XTRUST                  | CC-licensed / research-use                          | ~100 |
| Finance     | SORRY-Bench                     | Research-use                                        | ~100 |
| Medical     | SORRY-Bench, XTRUST              | Research-use                                        | ~100 |
| False-refusal contrast set (all categories) | XSTest | Research-use, widely cited | ~250 pairs |

**Action item before data collection starts:** re-verify each dataset's current license
terms directly on its hosting page (HuggingFace/GitHub) — licenses can change, and some
benchmarks require a data-use agreement or gating for the "harmful" split.

## Translation pipeline

1. **Machine draft** — translate each English seed prompt into the 8 target languages
   using a strong MT system (or an LLM translation call, logged separately from the
   models under evaluation).
2. **Native-speaker / professional review** — every machine draft must be reviewed by a
   native or professional speaker for:
   - Semantic fidelity (does it ask the same thing?)
   - Cultural naturalness (would a native speaker actually phrase it this way?)
   - No accidental softening or hardening of the request during translation
3. **Back-translation check** — translate back into English, compute similarity
   (BLEU/BERTScore) against the original as an automated fidelity flag; anything below
   threshold gets a second human pass.
4. **Language-purity check** — run `langdetect` on the final prompt to confirm it's
   predominantly in the target language (catches translation artifacts / code-switching
   that could itself act as a confound).

This separates two effects the literature often conflates:
- **"Translation as jailbreak"** — the act of translation itself weakening safety
  training exposure, independent of content
- **"Native safety gap"** — the model genuinely reasoning differently about the same
  well-translated request in a different language

Running both a machine-translated-only condition and a human-validated condition per
language lets you report on this distinction directly — it's a plausible secondary
contribution of the paper.

## Sample size and power

- Total target: ~900 English seed prompts × 9 languages ≈ 8,100 evaluation prompts
  (before contrast set), well within your 500–1000 *seed* prompt goal once you count the
  contrast/false-refusal set separately.
- For McNemar's test with paired binary outcomes (refused/not refused, English vs.
  target language), ~100 prompts per category gives reasonable power to detect medium
  effect sizes (Cohen's h ≈ 0.3) at α=0.05; smaller self-harm category (n=80) is
  slightly underpowered for subtle effects — flag this as a limitation in the paper.

## Ethics / IRB-equivalent notes

- No prompt in the final dataset should exceed the specificity already present in its
  cited source benchmark (i.e., you are not adding operational detail, only translating).
- Self-harm category prompts should be reviewed by someone with relevant domain
  familiarity; consider consulting your institution's ethics board or a
  mental-health-informed advisor before finalizing this subset.
- Full raw prompt/response/judge-verdict data stays out of the public repo
  (`results/`, `data/raw/` are gitignored). Release a data-use-agreement-gated version,
  consistent with HarmBench/AdvBench-derived work norms.
- Publish aggregated statistics, model names, category-level summaries, and all code
  openly — that's what the fellowship committee and future researchers actually need.
