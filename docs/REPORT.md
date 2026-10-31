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
- **Study overview**: We compare six training conditions (C1 real-only, C2 zero-shot synthetic, C3 few-shot synthetic, C4 50/50 real-synthetic mix, C5a low-resource real baseline, C5b low-resource real augmented with AugGPT) across two 6-class benchmarks: TREC (question classification; objective) and Emotion (sentiment/emotion classification; subjective). Evaluation is performed on held-out real test sets using frozen Sentence-BERT (`all-MiniLM-L6-v2`) embeddings and logistic regression classifiers across three independent generation replicates (seeds 13, 42, 7).

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
- Low-resource source sample preparation: exactly 10 examples per label for seeds 13, 42, 7 (exactly 60 examples per task-replicate).
- Verified split sizes and line counts bit-for-bit against SHA-256 hashes using canonical LF JSONL serialization.

---

## 4. Classifier and Experimental Conditions

*Owner: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.75 pages*

- Feature representation: Frozen `all-MiniLM-L6-v2` embeddings with L2 normalization (`normalize_embeddings=True`).
- Classification head: Logistic regression with $C = 1.0$, maximum 2,000 iterations, initialized with replication seed.
- Six training conditions:
  - **C1**: Real training data only (uniform target of 100 per label; 600 total per task).
  - **C2**: Zero-shot synthetic data only (100 per label; 600 total).
  - **C3**: Few-shot synthetic data only (3 demonstrations per prompt; 100 per label; 600 total).
  - **C4**: 50% real + 50% few-shot synthetic mix (50 real + 50 synthetic per label; 600 total).
  - **C5a**: Low-resource real data only (10 examples per label; 60 total).
  - **C5b**: C5a real data augmented with 6 AugGPT paraphrases per source example (420 total).
- Three independent replicates per condition identified by seeds 13, 42, and 7.

---

## 5. Results and Evaluation

*Owner: Hementh (@hem4456); Operator: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~1.0–1.5 pages*

### 5a. In-Repository Evaluation Design (ADR 0012)
- **Evaluation Item Selection**:
  - Sampled exactly 5 test items per label per task from verified real test splits (`data/real/trec/test.jsonl` with 500 items; `data/real/emotion/test.jsonl` with 2,000 items).
  - Total 60 items (30 items for TREC, 30 items for Emotion).
  - Deterministically shuffled within each task using `ANNOTATION_SEED = 0` so that item order does not leak gold label groups.
  - Recorded canonical reference key at `results/annotation_key.csv` (contains gold labels for audit; kept separate from blind evaluation templates).
- **Revised Evaluation Protocol**:
  - Per [ADR 0012](adr/0012-in-repo-evaluation-without-human-annotation.md), independent human annotation and external file distribution (email/drive/Google Docs) are no longer required and are superseded by in-repository evaluation under sole-operator execution.
  - No human consensus or inter-annotator agreement metrics are fabricated.
  - Evaluation procedures and data validation tools are documented in [`docs/ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md).

### 5b. Real-Data Baseline Performance (C1 and C5a)
Real-data baseline classifiers were executed on verified real datasets using frozen `all-MiniLM-L6-v2` embeddings and logistic regression across repetitions 13, 42, and 7 (from Issue #7 / PR #25). Macro-F1 and accuracy on the untouched held-out real test sets:

| Task | Condition | Seed 13 (F1 / Acc) | Seed 42 (F1 / Acc) | Seed 7 (F1 / Acc) | Mean Macro-F1 | Mean Accuracy |
|------|-----------|--------------------|--------------------|-------------------|---------------|---------------|
| **TREC** | **C1** (Real 100/label) | 0.747 / 0.784 | 0.747 / 0.784 | 0.759 / 0.776 | **0.751 ± 0.007** | **0.781 ± 0.015** |
| **TREC** | **C5a** (Low-res 10/label) | 0.597 / 0.640 | 0.538 / 0.590 | 0.492 / 0.518 | **0.542 ± 0.053** | **0.583 ± 0.061** |
| **Emotion** | **C1** (Real 100/label) | 0.463 / 0.506 | 0.459 / 0.496 | 0.462 / 0.499 | **0.461 ± 0.002** | **0.500 ± 0.005** |
| **Emotion** | **C5a** (Low-res 10/label) | 0.338 / 0.373 | 0.324 / 0.357 | 0.308 / 0.316 | **0.324 ± 0.015** | **0.349 ± 0.030** |

*Observations from baseline runs:*
- Both benchmarks perform substantially above the 6-class random guessing reference ($\approx 0.167$).
- Low-resource training (C5a with 60 examples) exhibits an expected performance drop relative to full real training (C1 with $\approx 600$ examples): $-27.8\%$ relative drop on TREC and $-29.7\%$ relative drop on Emotion.
- Variation across seeds is higher in the low-resource setting ($\pm 0.053$ for TREC C5a vs $\pm 0.007$ for C1), consistent with sample sensitivity in low-data regimes.

### 5c. Upstream Dependencies and Blocked Comparisons
The remaining conditions require upstream synthetic data and training runs that are currently in progress:
- **C2 (Zero-shot synthetic)** & **C3 (Few-shot synthetic)**: Depend on full generation across seeds 13, 42, 7 via `qwen3:4b` (Issue #10).
- **C5b (AugGPT augmented)**: Depends on generation of 2,160 paraphrases across repetitions (Issue #9; PR #24).
- **C4 (50/50 mix)** & Full Grid Training: Depends on synthetic data audit and freeze (Issue #12) and 36-run grid training (Issue #13).
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

- **Scope constraints**: Evaluation limited to two tasks (TREC, Emotion), a compact local 4B generator (`qwen3:4b`), frozen Sentence-BERT embeddings, and logistic regression classifiers.
- **Operating model**: Under sole-operator execution (ADR 0012), evaluation relies on benchmark metrics on held-out test splits and reproducible in-repo evaluation sets rather than multi-human annotator consensus.
- **Confounding factors**: Differences between TREC and Emotion include text length, syntactic structure, vocabulary diversity, and class balance in source pools, which may influence classifier accuracy independently of semantic subjectivity.
- **Interim conclusion**: Baseline real models provide stable, reproducible anchor points ($0.751$ F1 on TREC, $0.461$ F1 on Emotion). Synthetic data classifier viability will be evaluated once upstream generation and training complete under the audit and freeze gates.

---

## Appendix A. Prompt Templates and Generation Examples

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~1.0 page*

- Exact system prompts and generation instructions for zero-shot, few-shot, and AugGPT settings.
- Representative qualitative audit: two high-quality examples and two error/low-fidelity examples per task from the accepted Qwen generation pilot (ADR 0011).
