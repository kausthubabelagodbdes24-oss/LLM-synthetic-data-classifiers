## 3a. Model and Generation Method

*Generation workstream owner / section author: Harshita (@harshitadjainbdes24-byte); implementation and integration performed by Kausthuba Belagod (@kausthubabelagodbdes24-oss) on her behalf.*

### Generator Identity, Environment, and Provenance

All synthetic data for the accepted `scope30-v1` study was generated locally and offline through Ollama. The committed record of the hardware and Ollama version is the pilot's ([`docs/PILOT_REPORT.md`](../PILOT_REPORT.md)): Ollama 0.40.13 on an NVIDIA RTX 4060 Laptop GPU (8 GiB). The final scope30-v1 sidecars record the model identity and digest, but they are not committed and the committed evidence does not record the hardware or Ollama version of those final runs.

Following the initial generation pilot, the original candidate generator (`llama3.2:3b`, digest `a80c4f17…`) showed clear label-fit problems in the operator's pilot review (documented in [`docs/PILOT_REPORT.md`](../PILOT_REPORT.md)). Under [ADR 0010](../adr/0010-test-qwen-fallback-after-pilot-failure.md) and [ADR 0011](../adr/0011-accept-qwen-pilot-and-uniform-100-target.md), the project lead tested and accepted `qwen3:4b` as the replacement generator across all conditions:
- **Configured Model Tag:** `qwen3:4b`
- **Model Digest:** `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` (pinned in `ws1_generation.generate.APPROVED_GENERATORS`)
- **Sampling Temperature:** $T = 0.9$
- **Inference Configuration:** Executed via Ollama's HTTP REST API (`/api/generate`) with JSON formatting (`format: "json"`). For zero-shot, few-shot, and standard AugGPT, model default thinking was retained. For corrected TREC AugGPT calls, thinking was explicitly disabled (`"think": false`) and output tokens were capped at `num_predict: 1024` under [ADR 0013 Amendment 1](../adr/0013-trec-auggpt-answer-type.md).
- **Provenance Tracking:** Every synthetic artifact is paired with an append-only JSON sidecar (`<setting>_seed<S>.provenance.json`) recording the model identity, resolved digest, generation fingerprint, prompt dependency hashes, exact source hashes, invocation sessions with start/finish timestamps, elapsed seconds, accepted counts, and cumulative statistics.

### Generation Prompt Strategies

We implemented three distinct prompting strategies in `ws1_generation/generate.py`:

1. **Zero-Shot (`prompt_zero_shot`)**:
   Generates synthetic text from scratch conditioned on task context, target label definitions, and boundary guidelines. The prompt incorporates:
   - System/context framing: `You are building a dataset of {context}. Write {n} different examples of {style}.`
   - Canonical label definition and boundaries: Explicit instructions distinguish easily confused classes (e.g., in TREC, requiring open search-style questions and separating `abbreviation` expansions from `description` definitions; in Emotion, requiring the target category to be the dominant emotion rather than a secondary reaction).
   - Rotating lexical nudges (`NUDGES`): Encourages diverse syntactic openings and varied sentence lengths.
   - Format enforcement: Strict JSON reply constraint: `Reply only with JSON: {"examples": ["...", "..."]}`.

2. **Few-Shot (`prompt_few_shot`)**:
   Inherits the zero-shot task framing, label boundaries, and constraints, but additionally prepends three real in-domain demonstration exemplars ($k=3$) sampled from an isolated demonstration pool split (`data/real/<task>/pool.jsonl`):
   ```
   Here are real examples with this label:
   - "..."
   - "..."
   - "..."
   Use these for style, while following the requested label's requirements above.
   Write NEW examples in the same style. Do not copy or paraphrase these.
   ```
   The pool split is strictly disjoint from both the real training split (`train.jsonl`) and the official test set (`test.jsonl`).

3. **AugGPT Paraphrasing (`prompt_auggpt` and Variants)**:
   Performs sentence-level paraphrasing on canonical low-resource real sources ($n=10$ per label from `lowres_seed13.jsonl`):
   - **Base AugGPT (`prompt_auggpt`)**:
     `Please rephrase the following sentence {k} different ways, keeping its meaning: "{text}"` requesting $k=6$ paraphrases per source example.
   - **Length Guard Variant (`prompt_auggpt_length_guard`)**:
     Sources whose whitespace word count exceeds the task's configured limit (`max_words`: 30 words for Emotion) append:
     `Each rephrasing must be at most {max_words} words, so shorten it: keep the main meaning and the writer's feeling, and leave out minor details.`
     This prevents unparseable length rejections on long source tweets.
   - **Corrected TREC Answer-Type Variant (`prompt_auggpt_trec_answer_type`, ADR 0013 v2)**:
     To eliminate label drift observed during initial quality reviews, TREC seed-13 paraphrasing supplies the source question's label and definition, and requires:
     `Every rephrasing must keep the same expected answer type, so it keeps this label. ... Keep the same facts and relationships between them, the same entities and participants (who did what to whom), any negation, and the same intent. Each rephrasing must be a question or request for information. Do not answer the question, and do not add facts, names, dates or details that are not in it.`
     It includes an explicit `human` question rule: `"Who is X?"` questions must remain who-questions about the same referent and retain the `human` label, preventing drift toward biographical description. Calls are bounded with `"think": false` and `num_predict: 1024`.

### Parsing, Filtering, and Fail-Closed Safeguards

Candidate generations are processed through a multi-stage filtering pipeline before admission to synthetic corpora:
- **Structured JSON Parsing**: Parses the model reply into an array of string candidates. If parsing fails, the call is retried up to three times.
- **Conversational Chatter Filter**: Regex patterns (`NON_EXAMPLE_PATTERNS`) reject assistant conversational filler (e.g., *"Here are your examples:"*, *"Sure, below are..."*, *"As an AI..."*).
- **Exact-Match Test Leakage Exclusion**: Every candidate is normalized with `common.norm` (lower-cased and whitespace-collapsed; punctuation is kept) and checked against the untouched official held-out test split (`data/real/<task>/test.jsonl`). Any candidate matching a test item is immediately rejected. Across all accepted synthetic corpora, zero test leaks occurred.
- **In-Corpus Deduplication**: Duplicate normalized texts within the same artifact are rejected.
- **Copy Rejection**: Few-shot candidates copying any of the prompt's three demonstrations are rejected. AugGPT candidates identical to the source text are rejected.
- **Write-Ahead Journaling**: Each batch of accepted rows is written to an identity-bound write-ahead journal before appending to the JSONL corpus, followed by atomic provenance metadata replacement. Interrupted runs resume cleanly from the shortfall without duplicate rows or count inflation.

### Scope30-v1 Selected Counts and Demonstration Costs

Under canonical [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), the experiment executes under **scope30-v1**: seed 13 only, 30 training examples per label for Conditions C1–C4 (180 total per task), 10 real sources per label for C5a (60 total), and 420 examples for C5b (60 real sources + 360 AugGPT paraphrases).

- **Deterministic Selection**: Zero-shot and few-shot corpora were selected from compatible excess generated output by ranking row IDs using the ascending SHA-256 hash of UTF-8 `scope30-v1|13|<task>|<setting>|<row-id>` (row ID as tie-breaker) and taking the first 30 per label. The selection is recorded in deterministic manifests (`*.scope30-v1.selection.json`).
- **Corpora Counts**:
  - Four selected zero/few-shot artifacts (`*.scope30-v1.selected.jsonl`): exactly 180 rows each (30 per label $\times$ 6 labels), totaling 720 selected texts.
  - Two AugGPT artifacts (`auggpt_seed13.jsonl`): exactly 360 paraphrases each (6 per source $\times$ 60 sources), totaling 720 paraphrases.
  - Grand total selected synthetic examples: **1,440**.
- **Demonstration and Source Costs** (disclosed separately from classifier training counts):
  - **C2 (Zero-shot)**: 0 real demonstrations used.
  - **C3 (Few-shot)**: Draws on **59** distinct real pool demonstrations for TREC and **54** for Emotion.
  - **C4 (50/50 Mix)**: 90 real training rows + 90 few-shot rows; draws on at most 59 (TREC) and 54 (Emotion) real demonstrations.
  - **C5a (Low-resource real)**: 60 real rows (10 per label from `lowres_seed13.jsonl`).
  - **C5b (AugGPT augmented)**: 60 real rows + 360 synthetic paraphrases (420 total rows; 0 additional real demonstrations, as paraphrases derive from the same 60 sources).
- **Cross-Setting Overlap**: 18 normalized TREC texts are shared between the selected zero-shot and few-shot corpora (e.g., *"Who founded the Red Cross?"*, *"What river runs through London?"*), whereas Emotion has 0. This is an informational indicator of lower semantic diversity in TREC question generation.

### Quality Review, Findings, Corrections, and Freeze

Synthetic data quality was audited under a reproducible screening protocol ([ADR 0008](../adr/0008-audit-generated-content-before-freezing.md), [ADR 0009](../adr/0009-define-audit-investigation-trigger.md), [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md)):

- **Audit Method**: Model-assisted review conducted using Claude Opus 5.5 (`claude-opus-5-5`), operated by Kausthuba Belagod in Claude Code CLI. The initial quality gate sampled 10 examples in each of the six batches (2 tasks $\times$ 3 settings; 60 examples total) using reproducible ascending SHA-256 hash ordering of row IDs (`issue11-quality-v1`). It did not model-review all 1,440 rows. Subsequent evaluation stages expanded this evidence with TREC round 2 (10 fresh rows), targeted rechecks of round-1 IDs, a supplementary Emotion targeted check (12 rows across two sources), and heuristic scope scans across all 360 TREC AugGPT paraphrases.
- **Explicit Limitation**: This was a model-assisted and heuristic screening protocol; it is not independent human review, not human consensus, and not a population error-rate estimate. Ten-item sampling batches served as an operational quality gate to detect systemic prompt failure, not as a measurement of population dataset accuracy.
- **Trigger Rule**: Any batch with $\ge 2$ clear errors (defined as wrong intended label or, for AugGPT, changed source meaning) in its 10-item sample triggers a `FLAGGED` status, blocking freeze until the underlying cause is corrected and a fresh random round passes.
- **Round 1 Findings (31 October 2026)**:
  - 5 batches passed: `trec/zero_shot` (0/10), `trec/few_shot` (0/10), `emotion/zero_shot` (0/10), `emotion/few_shot` (0/10, 1 ambiguous), `emotion/auggpt` (0/10, 6 ambiguous).
  - `trec/auggpt` was **FLAGGED** with 2 clear errors: `trec.train.04257.seed13.aug5` (abbreviation source reworded as a definition request) and `trec.train.00845.seed13.aug3` (description source reworded as an event question).
- **Correction and Promotion (ADR 0013 and Amendment 1)**:
  - The root cause of the reviewed answer-type drift was that `prompt_auggpt` did not constrain the expected answer type.
  - The v2 correction (`auggpt-trec-answer-type-v2`) added explicit answer-type rules and the "Who is X?" human-label rule.
  - Request controls `"think": false` and `num_predict: 1024` were additional operational controls to prevent thinking and runaway-output behavior; they were not established as the cause of the answer-type drift.
  - All 360 paraphrases were regenerated in an isolated candidate directory and promoted via `promote_auggpt` (promotion record `89487383…`), preserving the original artifact (`4711ebe6…`) under `superseded/auggpt-trec-answer-type-v2/`.
- **Round 2 Findings (1 November 2026)**:
  - Fresh random sample of 10 rows from the promoted TREC AugGPT corpus (excluding round 1 IDs): **1 clear error** (`trec.train.04163.seed13.aug4`, where "Hot Wheels" became "Hot and Wheels", changing named entity meaning) and 3 ambiguous. Outcome: **PASS** (1 < 2, below trigger).
  - Targeted re-check of 10 round-1 IDs: `04257.aug5` resolved (asks for acronym expansion); `00845.aug3` unresolved (still asks "what specific incident").
  - Supplementary Emotion targeted check: Re-adjudication confirmed 12 clear errors across all 6 paraphrases of `emotion.train.07790` (agent reversed: writer's actions attributed to husband) and all 6 of `emotion.train.15223` (third party's strength made self-attributed).
  - Heuristic scope scan of TREC AugGPT: Found ~13 likely problematic rows and 6 borderline rows out of 360 (corrupted names, label word leaks, one reversed relation).
- **Freeze Manifest (Issue #12, Lead Option A)**:
  - Project lead decision: Under **Option A**, the verified corpora were accepted without further regeneration or row deletion, with all known flaws explicitly disclosed in the freeze manifest (`docs/freeze/scope30-v1-seed13.freeze.json`, SHA-256 `dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1`).
  - The freeze binds every artifact SHA-256, ordered row IDs, provenance sidecars, prompt hashes, and all 16 Issue #11 review files. Downstream classifier training (#13) consumes strictly these frozen files.
