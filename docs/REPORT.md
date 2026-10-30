# SynthText: Evaluation of Synthetic Data Classifiers Across Objective and Subjective Tasks

## Contributors & Workstream Responsibilities

- **Kausthuba Belagod (@kausthubabelagodbdes24-oss)**: Project lead & training owner (Workstream 2) — tasks, real data preparation, classifiers, experimental conditions C1–C5b, combined slides, submission.
- **Harshita Jain (@harshitadjainbdes24-byte)**: Generation owner (Workstream 1) — prompt templates, local model generation (zero-shot, few-shot, AugGPT), generation filtering, Appendix A, limitations and conclusion.
- **Hementh M (@hem4456)**: Evaluation owner (Workstream 3) — subjectivity evaluation, nominal Krippendorff's alpha, macro-F1 tables, paired-gap relative analysis, report assembly and editing.

---

## 1. Introduction

*Owner: Hementh (@hem4456)*
*Planning length: ~0.5 pages*

- **Problem context**: Labelled data acquisition is slow, costly, and resource-intensive. LLMs present an appealing mechanism for automated synthetic data synthesis, yet prior findings (Li et al., 2023) suggest their effectiveness degrades on subjective classification tasks compared to objective benchmarks.
- **Research question & hypothesis**: Does training classifiers on LLM-generated synthetic text incur a larger performance gap relative to real-data training on subjective tasks than on objective tasks? (Subjectivity hypothesis: Emotion will exhibit a larger negative gap than TREC, with few-shot prompting mitigating part of the gap).
- **Study overview**: We compare six training conditions (C1 real-only, C2 zero-shot synthetic, C3 few-shot synthetic, C4 50/50 real-synthetic mix, C5a low-resource real baseline, C5b low-resource real augmented with AugGPT) across two 6-class benchmarks: TREC (question classification; objective) and Emotion (sentiment/emotion classification; subjective). Evaluation is performed on held-out real test sets using frozen Sentence-BERT (`all-MiniLM-L6-v2`) embeddings and logistic regression classifiers across three independent generation replicates (seeds 13, 42, 7).

---

## 2. Related Work

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.5 pages*

- Concise paragraph on synthetic data generation and task subjectivity (Li et al., 2023).
- AugGPT: dialogue- and prompt-driven text augmentation for low-resource intent classification (Dai et al., 2023).
- Few-shot in-context learning with large language models (Brown et al., 2020).
- Sentence representations with Sentence-BERT (Reimers and Gurevych, 2019).

---

## 3. Tasks, Data and Generation

### 3a. Model and Generation Method
*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.75 pages*

- Generation model identity, local execution via Ollama (`llama3.2:3b`), inference temperature ($T = 0.9$).
- Prompting strategies: zero-shot with descriptive label guidance, few-shot with 3 real in-domain demonstrations sampled from the reserved pool, AugGPT paraphrasing with 6 meaning-preserving augmentations per source example.
- Validation, filtering, and rejection policies (JSON parse recovery, exact-match test deduplication, label consistency checks).

### 3b. Tasks and Real Data Preparation
*Owner: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.5 pages*

- Objective task: TREC (`SetFit/TREC-QC`), 6 coarse question types (abbreviation, description, entity, human, location, numeric).
- Subjective task: Emotion (`dair-ai/emotion`), 6 emotion categories (sadness, joy, love, anger, fear, surprise).
- Partitioning methodology: 90% real training split, 10% demonstration and low-resource pool split (isolated via deterministic seed 0 shuffle), untouched official test split.
- Low-resource source sample preparation (10 examples per label for seeds 13, 42, 7; exactly 60 examples per task-replicate).

---

## 4. Classifier and Experimental Conditions

*Owner: Kausthuba (@kausthubabelagodbdes24-oss)*
*Planning length: ~0.75 pages*

- Feature representation: Frozen `all-MiniLM-L6-v2` embeddings with L2 normalization.
- Classification head: Logistic regression with $C = 1.0$, maximum 2,000 iterations.
- Six training conditions:
  - **C1**: Real training data only (balanced target per label).
  - **C2**: Zero-shot synthetic data only.
  - **C3**: Few-shot synthetic data only (3 demonstrations per prompt).
  - **C4**: 50% real + 50% few-shot synthetic mix.
  - **C5a**: Low-resource real data only (10 examples per label).
  - **C5b**: C5a real data augmented with 6 AugGPT paraphrases per source example.
- Three independent replicates per condition identified by seeds 13, 42, and 7.

---

## 5. Results and Evaluation

*Owner: Hementh (@hem4456)*
*Planning length: ~1.0–1.5 pages*

- **Task Subjectivity Verification**:
  - Independent human annotation on 30 common test items per task (60 total items) by three annotators (Kausthuba, Harshita, Hementh) without access to gold labels or peer submissions.
  - Inter-annotator agreement metrics: Nominal Krippendorff's alpha ($\alpha$) and percentage pairwise agreement.
- **Classification Performance**:
  - Macro-F1 scores (mean $\pm$ standard deviation across replicates 13, 42, 7) across conditions C1 through C5b for TREC and Emotion.
- **Relative Performance Gap Analysis**:
  - Paired relative gap calculation per replicate: $\text{Gap} = (\text{Macro-F1}_{\text{synthetic}} - \text{Macro-F1}_{\text{C1}}) / \text{Macro-F1}_{\text{C1}}$.
  - Reported as mean $\pm$ standard deviation over matching seeds.
- **Four Core Comparisons**:
  1. Synthetic-versus-real performance gap: Is the degradation more severe on Emotion than on TREC?
  2. Prompting regime: Does few-shot generation (C3) shrink the gap relative to zero-shot (C2)?
  3. Augmentation parity: How close does the 50/50 mix (C4) approach full real-data performance (C1)?
  4. Low-resource augmentation: Does AugGPT augmentation (C5b) improve upon the 10-per-label real baseline (C5a)?

---

## 6. Limitations and Conclusions

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~0.5 pages*

- Scope constraints: Evaluation across two tasks, compact 3B local generator, fixed sentence embedding representations, small human annotator sample ($N = 3$).
- Confounding factors: Domain differences, class distributions, sentence lengths, and phrasing nuance vs. pure subjectivity.
- Evidence-based conclusions on the viability of synthetic text classifiers for subjective NLP tasks.

---

## Appendix A. Annotation File Distribution Guide

*Owner: Hementh (@hem4456) — operational reference for distributing annotation response files.*

The `python -m ws3_evaluation.annotate make` command produces the following files.
Distribute them as described below; **never share `annotation_key.csv` with any annotator**.

| File | Recipient | Contains gold labels? | Safe to share? |
|------|-----------|----------------------|----------------|
| `results/annotation_key.csv` | Internal only — do NOT share | **Yes** | **No — private audit record** |
| `results/annotation_kausthuba.csv` | @kausthubabelagodbdes24-oss only | No | Yes — only to this annotator |
| `results/annotation_harshita.csv` | @harshitadjainbdes24-byte only | No | Yes — only to this annotator |
| `results/annotation_hementh.csv` | @hem4456 only | No | Yes — only to this annotator |

**Human process requirement — Hementh completes first:**
Hementh (@hem4456) must complete and return `annotation_hementh.csv` independently before any
other annotator's labels are viewed or before `combine` is run. This order is required by
ADR-0005 (independent annotation) but is **not enforced by the code**. The team lead must
confirm Hementh's submission separately (e.g., a comment on Issue #8) before running
`python -m ws3_evaluation.annotate combine`.

**Overwrite protection:**
Running `make` a second time without `--force` will abort rather than overwrite completed judgments.
Only use `--force` after explicitly confirming that no completed responses will be lost.

---

## Appendix B. Prompt Templates and Generation Examples

*Owner: Harshita (@harshitadjainbdes24-byte)*
*Planning length: ~1.0 page*

- Exact system prompts and generation instructions for zero-shot, few-shot, and AugGPT settings.
- Representative qualitative audit: two high-quality examples and two error/low-fidelity examples per task from the generation pilot.
