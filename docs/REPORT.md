# SynthText: Evaluation of Synthetic Data Classifiers Across Objective and Subjective Tasks

## Contributors & Workstream Responsibilities

- **Kausthuba Belagod (@kausthubabelagodbdes24-oss)**: Project lead, training owner (Workstream 2), and sole operator — real data preparation, classifiers, experimental baseline conditions C1 & C5a, report integration, combined slides, and submission.
- **Harshita Jain (@harshitadjainbdes24-byte)**: Generation owner (Workstream 1; historical/role assignment) — prompt templates, local model generation (zero-shot, few-shot, AugGPT), generation filtering, Appendix A, limitations and conclusion.
- **Hementh M (@hem4456)**: Evaluation owner (Workstream 3; historical/role assignment) — in-repository evaluation design, reproducible test item selection, macro-F1 tables, paired-gap relative analysis, report assembly and editing.

*Operating model note (ADR 0012):* Under direct project lead instruction, Kausthuba is the sole operator executing generation, training, and evaluation workflows in separate workstream branches. Historical role ownership is preserved for attribution and modularity. Independent human annotation and external file distribution are superseded by in-repository evaluation.

---

## 1. Introduction

*Owner: Hementh (@hem4456); Operator: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.5 pages*

- **Problem context**: Labelled data acquisition is slow, costly, and resource-intensive. LLMs present an appealing mechanism for automated synthetic data synthesis, yet prior findings (Li et al., 2023) suggest their effectiveness degrades on subjective classification tasks compared to objective benchmarks.
- **Research question & hypothesis**: Does training classifiers on LLM-generated synthetic text incur a larger performance gap relative to real-data training on subjective tasks than on objective tasks? (Subjectivity hypothesis: Emotion will exhibit a larger negative gap than TREC, with few-shot prompting mitigating part of the gap).
- **Study overview (scope30-v1)**: Under canonical [ADR 0012](adr/0012-eod-exploratory-30-seed13.md), we conduct an exploratory single-seed comparison (seed 13) across six training conditions (C1 real-only, C2 zero-shot synthetic, C3 few-shot synthetic, C4 50/50 real-synthetic mix, C5a low-resource real baseline, C5b low-resource real augmented with AugGPT) on two 6-class benchmarks: TREC (question classification; objective) and Emotion (sentiment/emotion classification; subjective). Evaluation is performed on untouched held-out real test sets using frozen Sentence-BERT (`all-MiniLM-L6-v2`) embeddings and logistic regression classifiers.

---

## 2. Related Work

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.5 pages*

- Concise review on synthetic data generation and task subjectivity (Li et al., 2023).
- AugGPT: dialogue- and prompt-driven text augmentation for low-resource intent classification (Dai et al., 2023).
- Few-shot in-context learning with large language models (Brown et al., 2020).
- Sentence representations with Sentence-BERT (Reimers and Gurevych, 2019).

---

## 3. Tasks, Data and Generation

### 3a. Model and Generation Method
*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.75 pages*

- Generation model identity: local execution via Ollama using `qwen3:4b` (accepted in ADR 0011, superseding original Llama pilot; observed digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`), inference temperature $T = 0.9$.
- Prompting strategies: zero-shot with descriptive label guidance, few-shot with 3 real in-domain demonstrations sampled from the reserved pool, AugGPT paraphrasing with 6 meaning-preserving augmentations per source example.
- Validation, filtering, and rejection policies (JSON parse recovery, exact-match test deduplication, label consistency checks, identity-bound write-ahead journals).

### 3b. Tasks and Real Data Preparation
*Owner: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.5 pages*

- Objective task: TREC (`SetFit/TREC-QC`), 6 coarse question types (abbreviation, description, entity, human, location, numeric).
- Subjective task: Emotion (`dair-ai/emotion`), 6 emotion categories (sadness, joy, love, anger, fear, surprise).
- Partitioning methodology: 90% real training split, 10% demonstration and low-resource pool split (isolated via deterministic seed 0 shuffle), untouched official test split.
- Low-resource source sample preparation: exactly 10 examples per label for seed 13 (`lowres_seed13.jsonl`; exactly 60 examples per task).
- Verified split sizes and line counts bit-for-bit against SHA-256 hashes using canonical LF JSONL serialization.

---

## 4. Classifier and Experimental Conditions

*Owner: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.75 pages*

- Feature representation: Frozen `all-MiniLM-L6-v2` embeddings with L2 normalization (`normalize_embeddings=True`).
- Classification head: Logistic regression with $C = 1.0$, maximum 2,000 iterations, initialized with seed 13.
- Six training conditions (scope30-v1 under [ADR 0012](adr/0012-eod-exploratory-30-seed13.md)):
  - **C1**: Real training data only (30 examples per label; 180 total per task).
  - **C2**: Zero-shot synthetic data only (30 examples per label; 180 total).
  - **C3**: Few-shot synthetic data only (3 demonstrations per prompt; 30 per label; 180 total).
  - **C4**: 50% real + 50% few-shot synthetic mix (15 real + 15 synthetic per label; 180 total).
  - **C5a**: Low-resource real data only (10 examples per label from `lowres_seed13.jsonl`; 60 total).
  - **C5b**: C5a real data augmented with 6 AugGPT paraphrases per source example (420 total).
- Execution scope: Seed 13 only; twelve unique result keys overall (2 tasks × 6 conditions × 1 seed).

---

## 5. Results and Evaluation

*Owner: Hementh (@hem4456); Operator: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~1.0–1.5 pages*

### 5a. In-Repository Evaluation Design (ADR 0012)
- **Evaluation Item Selection & Tooling**:
  - Sampled exactly 5 test items per label per task from verified real test splits (`data/real/trec/test.jsonl` with 500 items; `data/real/emotion/test.jsonl` with 2,000 items).
  - Total 60 items (30 items for TREC, 30 items for Emotion).
  - Deterministically shuffled within each task using `ANNOTATION_SEED = 0` so that item order does not leak gold label groups.
  - Recorded canonical reference key at `results/annotation_key.csv` (contains gold labels for audit; kept separate from blind evaluation templates).
- **Revised Evaluation Protocol**:
  - Per canonical [ADR 0012](adr/0012-eod-exploratory-30-seed13.md), independent human annotation, external file distribution (email/drive), and external document sharing (Google Docs) are removed under sole-operator execution.
  - These optional 60-item held-out test annotation utilities are preserved as historical response tooling and are strictly distinct from the required 60 generated-text quality reviews conducted in Workstream 1/3 (#11/#12).
  - No human consensus or inter-annotator agreement metrics are fabricated. Evaluation procedures are documented in [`docs/ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md).

### 5b. Accepted Real-Data Baseline Performance (C1 and C5a)
The four seed-13 baseline classifiers were executed on verified real datasets using frozen `all-MiniLM-L6-v2` embeddings and logistic regression, accepted and merged through PR #25 into `results/scope30-seed13/runs.csv` (mirrored at `results/runs.csv`). Macro-F1 and accuracy on the untouched held-out real test sets:

| Task | Condition | Seed | Training Size ($n$) | Macro-F1 | Accuracy | Status |
|------|-----------|------|---------------------|----------|----------|--------|
| **TREC** | **C1** (Real 30/label) | 13 | 180 | **0.6009** | **0.6520** | Accepted (PR #25) |
| **TREC** | **C5a** (Low-res 10/label) | 13 | 60 | **0.5974** | **0.6400** | Accepted (PR #25) |
| **Emotion** | **C1** (Real 30/label) | 13 | 180 | **0.4082** | **0.4420** | Accepted (PR #25) |
| **Emotion** | **C5a** (Low-res 10/label) | 13 | 60 | **0.3381** | **0.3730** | Accepted (PR #25) |

*Baseline observations & withdrawn scores:*
- **Single-run scores without repetition SD:** Under ADR 0012, scores are reported as single-run metrics for seed 13 without cross-seed standard deviations, uncertainty bars, or statistical significance claims.
- **Historical and withdrawn scores:** `results/archive/runs_target100.csv` preserves nine earlier runs: Emotion C1 at 100/label and both tasks' C5a results across seeds 13, 42 and 7. It contains no TREC C1 row. Earlier undersized TREC C1 results and conclusions based on them were withdrawn. All superseded results are excluded from the current analysis.
- **Baseline performance:** On TREC (objective), real baseline performance is 0.6009 macro-F1 at 30 examples per label and 0.5974 at 10 examples per label. On Emotion (subjective), baseline performance is 0.4082 macro-F1 at 30 examples per label and 0.3381 at 10 examples per label. Both benchmarks perform substantially above the 6-class random guessing reference ($\approx 0.167$).

### 5c. Upstream Dependencies and Blocked Comparisons
The remaining eight condition results require upstream synthetic data generation and classifier training:
- **C2 (Zero-shot synthetic)** & **C3 (Few-shot synthetic)**: Depend on zero-shot and few-shot generation across 30 examples per label for seed 13 via `qwen3:4b` (Issue #10).
- **C5b (AugGPT augmented)**: Depends on generation of 720 paraphrases (6 per source) for seed 13 sources (Issue #9; PR #24).
- **C4 (50/50 mix)** & Full Grid Training: Depends on synthetic data selection, quality audit, and freeze (Issue #12) and training across the remaining 8 keys (Issue #13).
- **Relative Gap Analysis & Core Hypotheses**:
  - Comparison 1 (Synthetic gap on Emotion vs TREC): Blocked pending C2/C3 runs.
  - Comparison 2 (Few-shot vs Zero-shot mitigation): Blocked pending C3 vs C2 runs.
  - Comparison 3 (50/50 mix parity with C1): Blocked pending C4 runs.
  - Comparison 4 (AugGPT augmentation gain over C5a): Blocked pending C5b runs.
- In accordance with the shared protocol, fake sample fixtures from `data/sample/` are strictly excluded from research findings.

---

## 6. Limitations and Conclusions

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.5 pages*

- **Scope constraints**: Exploratory single-seed comparison (seed 13) at 30 examples per label (C1–C4), compact local 4B generator (`qwen3:4b`), frozen Sentence-BERT embeddings, and logistic regression classifiers.
- **Operating model**: Under sole-operator execution ([ADR 0012](adr/0012-eod-exploratory-30-seed13.md)), evaluation relies on benchmark metrics on held-out real test splits rather than multi-human annotator consensus.
- **Statistical limits**: Single-repetition findings cannot establish statistical significance across seeds or prove that task differences are causally driven by subjectivity.
- **Synthetic data quality (issue #12 freeze, lead Option A)**: The frozen seed-13 corpora are accepted with known errors. They were disclosed, not removed or corrected; details are in [docs/freeze/README.md](freeze/README.md).
  - **TREC AugGPT** (ADR 0013 v2, 360 rows): the random review passed with 1 clear error in 10 (`04163.aug4`, where "Hot Wheels" becomes "Hot and Wheels") and 3 ambiguous rows.
    - A targeted re-check found `00845.aug3` still changes the expected answer type.
    - A heuristic scan found about 13 likely problematic rows plus 6 borderline ones: answer-type changes, corrupted names (seen only in the v2 output), one reversed relation, and abbreviation questions drifting toward definitions. Around 3–4 of these may change the intended label.
  - **Emotion AugGPT**: all 12 paraphrases of two long sources change the source meaning while keeping their label.
  - These findings come from model-assisted review (one model, one judgment per item) and heuristics. They are not independent-human judgments or population error rates, and the retained errors are not claimed to be harmless. The acceptance holds only for this exploratory scope30-v1, seed-13 study. It does not establish corpus correctness, robustness, stability or generalization.
- **Interim conclusion**: Accepted baseline models provide reference baselines (0.6009 F1 on TREC, 0.4082 F1 on Emotion) at 30 examples per label. Synthetic data classifier performance will be evaluated once upstream generation and training complete under the quality audit and freeze gates.

---

## Appendix A. Prompt Templates and Generation Examples

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~1.0 page*

- Exact system prompts and generation instructions for zero-shot, few-shot, and AugGPT settings.
- Representative qualitative audit: two high-quality examples and two error/low-fidelity examples per task from the accepted Qwen generation pilot ([ADR 0011](adr/0011-accept-qwen-pilot-and-uniform-100-target.md)).
