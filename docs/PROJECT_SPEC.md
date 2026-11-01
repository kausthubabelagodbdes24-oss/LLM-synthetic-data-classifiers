# SynthText final experiment specification

**Status:** Completed and accepted on 1 November 2026
**Study identifier:** `scope30-v1`, seed 13

## Research question

Can synthetic text produced by a compact local language model support text-classifier training, and does the observed relationship between synthetic-only and real-only training differ between an objective task and a subjective task?

The study is exploratory. It compares observed directions across two tasks; it does not isolate subjectivity as the cause of any difference and does not estimate statistical significance or cross-seed stability.

## Tasks and data

| Task | Dataset | Labels | Prepared train / pool / official test |
| --- | --- | --- | ---: |
| TREC | `SetFit/TREC-QC` | abbreviation, description, entity, human, location, numeric | 4,907 / 545 / 500 |
| Emotion | `dair-ai/emotion` | sadness, joy, love, anger, fear, surprise | 14,400 / 1,600 / 2,000 |

The real training split and demonstration pool are deterministic and disjoint. Official test sets remain unchanged and are used only for final evaluation. The canonical low-resource sample contains ten real examples per label, or 60 examples per task.

## Experimental conditions

| Condition | Construction per label | Rows per task |
| --- | --- | ---: |
| C1 | 30 real training examples | 180 |
| C2 | 30 zero-shot synthetic examples | 180 |
| C3 | 30 few-shot synthetic examples | 180 |
| C4 | 15 real and 15 selected few-shot synthetic examples | 180 |
| C5a | 10 real low-resource source examples | 60 |
| C5b | The same 10 sources plus six distinct AugGPT paraphrases per source | 420 |

Two tasks multiplied by six conditions gives twelve result keys. C1–C4 have equal classifier input counts but not equal real-labelling costs. C5a and C5b intentionally differ in training size, so their comparison combines augmentation with an increase in the number of rows.

The accepted selected synthetic set contains 1,440 texts: 720 zero-shot/few-shot examples and 720 AugGPT paraphrases.

## Generation

- **Model:** `qwen3:4b`
- **Recorded digest:** `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`
- **Temperature:** 0.9
- **Modes:** zero-shot, few-shot with three same-label demonstrations, and AugGPT with six paraphrases per source
- **TREC correction:** the accepted AugGPT prompt preserves the source question's answer type

Generation identity, prompts, options, source hashes, selected row IDs, and model provenance are recorded in the committed manifests. Raw generated corpora and private real data are not committed.

## Quality and freeze

The selected handoff was checked for exact counts, all-label coverage, unique identifiers and texts, source matching, held-out-test exclusion, accepted model identity, and file/provenance hashes.

A disclosed model-assisted quality screen reviewed ten examples per task and generation mode. It is not independent-human annotation, human agreement, or a population error estimate. Known quality limitations remain documented rather than silently removed. The accepted freeze manifest is [freeze/scope30-v1-seed13.freeze.json](freeze/scope30-v1-seed13.freeze.json); its readable overview is [freeze/README.md](freeze/README.md).

## Classifier and evaluation

- **Embedder:** frozen `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- **Features:** 384-dimensional L2-normalized sentence embeddings
- **Classifier:** scikit-learn logistic regression, `C=1.0`, `max_iter=2000`, `random_state=13`
- **Metrics:** macro-F1 and accuracy on the unchanged official test sets

The training system accepts only the approved freeze and model identity. Exactly twelve valid result records are required before analysis.

## Planned comparisons

1. Synthetic-only versus real-only training on each task.
2. Few-shot versus zero-shot synthetic training.
3. Mixed real/few-shot training versus real-only training.
4. AugGPT-augmented training versus the same real sources without paraphrases.

Relative synthetic-only gaps are calculated against the matching same-task C1 score. A missing C1 blocks analysis; a zero C1 retains the task but leaves the relative gap undefined.

## Evidence and claim limits

The study reports one run for each condition. It therefore provides no standard deviation, confidence interval, significance test, replication estimate, or stability claim. Cross-task differences cannot be attributed to subjectivity alone because the datasets also differ in label semantics, text style, balance, and test-set size.

The accepted results, charts, and limitations are available in:

- [REPORT.pdf](REPORT.pdf) — reviewed final report
- [analysis/scope30-v1-seed13/summary.md](analysis/scope30-v1-seed13/summary.md) — result tables and comparisons
- [analysis/scope30-v1-seed13/analysis.json](analysis/scope30-v1-seed13/analysis.json) — machine-readable analysis
- [../notebooks/SynthText_scope30_walkthrough.ipynb](../notebooks/SynthText_scope30_walkthrough.ipynb) — Colab walkthrough

## Ownership

| Workstream | Accountable owner |
| --- | --- |
| Data generation | Harshita (`@harshitadjainbdes24-byte`) |
| Data preparation, training, and project lead | Kausthuba Belagod (`@kausthubabelagodbdes24-oss`) |
| Evaluation, analysis, and report assembly | Hementh (`@hem4456`) |

Kausthuba implemented and integrated some generation and evaluation work on the relevant owners' behalf. This is execution provenance, not a transfer of workstream ownership.

## Presentation boundary

The final professor-facing presentation and submission logistics are outside the repository and are not tracked as development work. The reviewed report, tables, charts, and Colab walkthrough are the repository materials intended to support that presentation.

Development history, earlier plans, migration notes, and internal operator protocols remain available through the [documentation archive](README.md#development-archive), but they are not required to understand the final study.
