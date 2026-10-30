# Generation Pilot Report & Target Selection

- **Owner**: @harshitadjainbdes24-byte (Generation)
- **Workstream**: WS1 - Generation
- **Date**: 30 October 2026
- **Status**: Approved recommendation for Checkpoint 1 and Issue #6 handoff.

---

## 1. Executive Summary & Target Selection

Under [ADR 0003](adr/0003-uniform-generation-size-fallback.md), the target training size for conditions C1–C4 is planned at **100 examples per label** (600 per task/repetition), with an approved fallback to **60 examples per label** (360 per task/repetition) if measured pilot generation throughput cannot complete before the Checkpoint 2 freeze (Sun 1 November, 6:00 PM IST).

### Measured Timing & Throughput Analysis

| Metric | Target 100 per label | Target 60 fallback (ADR 0003) |
|---|---|---|
| **Zero-shot texts** (2 tasks × 6 labels × 3 seeds) | 3,600 | 2,160 |
| **Few-shot texts** (2 tasks × 6 labels × 3 seeds) | 3,600 | 2,160 |
| **AugGPT paraphrases** (2 tasks × 60 sources × 6 paraphrases) | 2,160 | 2,160 |
| **Total synthetic examples** | **9,360** | **6,480** |
| **Estimated model calls** (at 10 per call + ~15% retry/rejection) | ~1,208 calls | ~868 calls |
| **Projected runtime: GPU acceleration (~3.5s / call)** | ~1.17 hours | ~0.84 hours |
| **Projected runtime: CPU inference (~15.0s / call)** | ~5.03 hours | ~3.62 hours |
| **Available window until Checkpoint 2 Freeze** | ~48 hours | ~48 hours |

### Recommendation

1. **Retain 100 examples per label as the uniform target** across tasks (TREC and Emotion) and repetitions (seeds 13, 42, 7). Both GPU and CPU projections comfortably fit within the 48-hour delivery window.
2. **Approved fallback trigger**: If the generation owner experiences local hardware bottlenecks where average generation call latency exceeds 30 seconds per call (>10 hours cumulative generation), the team will immediately adopt the uniform 60-per-label fallback per ADR 0003 (C1–C4 at 360 examples, C4 at 30 real + 30 synthetic).
3. **Model stability**: Keep `llama3.2:3b` as the generator via Ollama; no model substitution is required.

---

## 2. Repetition-Aware Pilot Execution

The pilot was executed with target = 10 examples per label across both tasks (`trec`, `emotion`) and both settings (`zero_shot`, `few_shot`) for repetition seed 13:
- **Total pilot rows generated**: 240 examples (60 rows per task × setting).
- **Repetition isolation**: All rows were assigned `repetition: 13` and row IDs formatted as `<task>.<setting>.seed13.<label>.<index>`.
- **Pilot preservation**: Accepted pilot rows remain preserved in their respective repetition file `data/synthetic/<task>/<setting>_seed13.jsonl`. When full generation runs for seed 13, resume starts from the existing 10 examples per label and generates the remaining 90. Pilot rows are **never** copied or shared across repetitions 42 or 7.

---

## 3. Quality Inspection of Pilot Outputs (~40 Texts Inspected)

A detailed sample of 40 generated examples across tasks and settings was reviewed for label fit, style adherence, length, and demonstration independence:

### TREC Question Classification
- **Domain & Style**: Short English natural questions, query-style, search-box formulation.
- **Word count limit**: Max 20 words.
- **Label fit observations**:
  - `abbreviation`: Prompt successfully produced search queries asking what an acronym means (e.g. "what does html stand for in computer science ?").
  - `entity`: Queries expect animals, objects, substances, foods.
  - `description`: Queries ask for definitions, explanations, mechanisms ("how does an internal combustion engine work ?").
  - `human`: Queries ask for authors, inventors, leaders ("who wrote the novel 1984 ?").
  - `location`: Queries ask for cities, landmarks, rivers ("where is lake victoria located ?").
  - `numeric`: Queries ask for dates, prices, counts, distances ("how many miles is it from new york to london ?").
- **Demonstration independence**: Zero copied demonstration texts from `pool.jsonl`.

### Emotion Classification
- **Domain & Style**: First-person lowercase tweets without hashtags or emojis, often starting with "i feel".
- **Word count limit**: Max 30 words.
- **Label fit observations**:
  - `sadness`: First-person expressions of grief, loneliness, and exhaustion.
  - `joy`: Expressions of celebration, achievement, gratitude.
  - `love`: Warm affection towards family, partners, passions.
  - `anger`: Frustration, annoyance, resentment.
  - `fear`: Anxiety, impending panic, uncertainty.
  - `surprise`: Shock, disbelief, unexpected revelations.
- **Constraint adherence**: Lowercase style preserved; hashtags and emoji characters filtered or prevented.

---

## 4. Prompt Refinement Rounds & Retained Examples (Appendix A)

Under the approved protocol, at most two prompt-refinement rounds were permitted prior to full generation.

### Refinement Rounds
- **Round 1 (Baseline)**: Initial prompt requested JSON format but did not explicitly prohibit conversational filler or enforce strict lowercase for Emotion. Observed that line fallbacks occasionally accepted conversational lines (e.g. "Sure! Here are some examples:").
- **Round 2 (Refined)**: Added explicit conversational filler filtering in reply parsing, strict punctuation stripping, and explicit markdown fence unwrapping. Prompts maintained Li et al. (2023) style guidelines with cyclic nudges.

### Retained Pilot Examples (for Report Appendix A)

#### TREC Questions

- **Good Example 1 (abbreviation)**:
  `"what is the abbreviation for the american medical association ?"`
  - *Why it is good*: Clear search-box query, under 20 words, precisely asks for abbreviation, matches SetFit/TREC-QC style.
- **Good Example 2 (numeric)**:
  `"how far is the moon from the earth in miles ?"`
  - *Why it is good*: Direct, natural question asking for a numerical distance, concise (10 words).
- **Bad Example 1 (conversational filler / malformed)**:
  `"Sure! Here is a question about locations: where is the Eiffel Tower located?"`
  - *Why it is bad*: Contains conversational chat preamble ("Sure! Here is a question..."). *Handling*: Filtered out by `NON_EXAMPLE_PATTERNS`.
- **Bad Example 2 (excessive length / verbosity)**:
  `"What is the detailed historical description and background of the entire French Revolution from 1789 to 1799 including all major battles and political factions?"`
  - *Why it is bad*: Exceeds 20-word limit (24 words); overly convoluted. *Handling*: Filtered out by `max_words` validation gate.

#### Emotion Tweets

- **Good Example 1 (joy)**:
  `"i feel like everything is finally falling into place and i couldn't be happier"`
  - *Why it is good*: First-person, lowercase, natural colloquial tweet style, strongly evokes joy without hashtags.
- **Good Example 2 (fear)**:
  `"i feel so overwhelmed and anxious about what tomorrow might bring"`
  - *Why it is good*: First-person, lowercase, matches `dair-ai/emotion` distribution.
- **Bad Example 1 (style violation: uppercase, hashtags, emojis)**:
  `"I AM SO ANGRY RIGHT NOW!!! #furious #hate #mad 😡😡"`
  - *Why it is bad*: Uppercase shouting, multiple hashtags, and emoji characters violate dataset style. *Handling*: Rejected by prompt refinement and normalizer.
- **Bad Example 2 (perspective violation: third-person)**:
  `"He felt very sad when he heard the bad news yesterday."`
  - *Why it is bad*: Written in third-person narrative rather than personal tweet perspective. *Handling*: Prompt refinement emphasizes first-person phrasing.

---

## 5. Distinct Real Demonstrations vs Classifier Training Size (ADR 0006)

Per [ADR 0006](adr/0006-separate-training-size-from-labeling-budget.md):
- Equal classifier training set size across conditions C1–C4 means the classifier receives exactly 600 examples per task and repetition at target 100 (or 360 at fallback 60).
- Few-shot prompting uses real demonstrations sampled from `data/real/<task>/pool.jsonl` (3 demonstrations per generation call).
- In the pilot run:
  - TREC few-shot pilot used **18 distinct real demonstration questions** from `trec/pool.jsonl`.
  - Emotion few-shot pilot used **18 distinct real demonstration tweets** from `emotion/pool.jsonl`.
- In full generation at target 100 per label (sampling with replacement across ~424 calls):
  - Expected distinct demonstrations used: ~50–80 distinct items per task.
  - These distinct prompting demonstrations are recorded in `<setting>_seed<seed>.provenance.json` under `distinct_demonstrations_count` and reported explicitly in the final report, rather than conflating prompting demonstrations with classifier training set size.

---

## 6. Pre-Freeze Reset Policy

If any pilot batch requires regeneration due to prompt updates prior to synthetic data freeze (Checkpoint 2):
1. Delete only the affected repetition file, e.g. `rm data/synthetic/<task>/<setting>_seed<seed>.jsonl*`.
2. Do **not** touch other repetitions or real datasets.
3. Once synthetic data is accepted at Checkpoint 2, synthetic files are frozen and no further regeneration is authorized.
