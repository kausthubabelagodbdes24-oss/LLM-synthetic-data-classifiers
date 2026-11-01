# Methods: Data Preparation, Classification Heads, and Experimental Training (scope30-v1)

**Workstream 2: Data and Classifier Methods**<br>
**Accountable Lead:** Kausthuba Belagod (@kausthubabelagodbdes24-oss)<br>
**Execution Authority:** [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0014](../adr/0014-presentation-separation-colab.md), [PROJECT_SPEC.md](../PROJECT_SPEC.md), [CONTRACTS.md](../agents/CONTRACTS.md)<br>
**Target Scope:** Sections 3b ("Tasks and Real Data Preparation") and 4 ("Classifier and Experimental Conditions") of the repository report scaffold.

---

## 1. Tasks and Real Data Preparation (Report Section 3b)

### 1.1 Benchmark Datasets and Task Formulations

The SynthText investigation benchmarks synthetic data effectiveness across two distinct text classification domains designed to reflect differences in semantic subjectivity:

1. **Objective Benchmark: TREC Question Classification (`SetFit/TREC-QC`)**
   - **Task Domain:** Fact-oriented question categorization over 6 coarse semantic question types:
     - `ABBR` (abbreviation)
     - `DESC` (description and definition)
     - `ENTY` (entities)
     - `HUM` (human beings)
     - `LOC` (locations)
     - `NUM` (numeric and quantitative values)
   - **Canonical Split Sizes:** The official benchmark provides 5,452 training examples and 500 test examples.
   - **Target Label Distribution:** In the training split, class frequencies vary naturally (`DESC`, `ENTY`, `HUM` are dominant; `ABBR` is low-frequency with 76 total training rows). The 76 abbreviation examples exceed our target of 30 per label, enabling balanced class extraction without undersampling.

2. **Subjective Benchmark: Twitter Emotion Classification (`dair-ai/emotion`, split configuration)**
   - **Task Domain:** Affective sentiment classification over 6 emotional states:
     - `sadness` (class 0)
     - `joy` (class 1)
     - `love` (class 2)
     - `anger` (class 3)
     - `fear` (class 4)
     - `surprise` (class 5)
   - **Affective Subsumption Mapping:** Sub-facet affective states are deterministically subsumed under standard psychological primaries: *annoyance* is categorized under `anger`; *anxiety* is categorized under `fear`.
   - **Canonical Split Sizes:** The official benchmark provides 16,000 training examples, 2,000 validation examples, and 2,000 test examples. (The validation split is excluded from this study to preserve canonical test comparability).

---

### 1.2 Deterministic Partitioning and Data Separation

To support both supervised training and few-shot in-context demonstration sampling without data leakage, the official training split of each task was deterministically partitioned into disjoint training and demonstration sets:

- **Partitioning Procedure:** The official training dataset was shuffled using a fixed seed (`seed = 0`) and split into a 90% primary training set and a 10% demonstration/pool set.
- **Partition Counts:**
  - **TREC:**
    - `data/real/trec/train.jsonl`: 4,907 rows (90% of 5,452).
    - `data/real/trec/pool.jsonl`: 545 rows (10% of 5,452). Pool label distribution, as recorded in the Issue #5 data contract: `ABBR`: 10, `DESC`: 126, `ENTY`: 128, `HUM`: 124, `LOC`: 78, `NUM`: 79. (Per-label pool counts are not stored in the freeze manifest or other committed artifacts; the contract is the producer's record.)
    - `data/real/trec/test.jsonl`: 500 rows (untouched official test set).
  - **Emotion:**
    - `data/real/emotion/train.jsonl`: 14,400 rows (90% of 16,000).
    - `data/real/emotion/pool.jsonl`: 1,600 rows (10% of 16,000). Pool label distribution, as recorded in the Issue #5 data contract: `anger`: 252, `fear`: 178, `joy`: 539, `love`: 144, `sadness`: 430, `surprise`: 57.
    - `data/real/emotion/test.jsonl`: 2,000 rows (untouched official test set).
- **Serialization and Cryptographic Verification:** All datasets are stored using line-delimited JSON (JSONL) with canonical UTF-8 encoding and LF (`\n`) line endings. The canonical SHA-256 digests below are the values enforced by `ws2_training/prepare_data.py` (`CHECKSUMS`); the freeze manifest records the same `pool`, `test` and `lowres_seed13` hashes:
  - `trec/train.jsonl`: `b587c84d26b361bbe6f4126aff136e9d737c6c349059e89d2a3ce79e226f4331`
  - `trec/pool.jsonl`: `0a78fbe6e38b8b9eb4f7d9e24e619d63f855e073bffb1e00edb23508490d3a6e`
  - `trec/test.jsonl`: `415f526277d0f23f9bfe903a2a6d994529ec183820b6d9d0d55d47c3195aa0c0`
  - `emotion/train.jsonl`: `4c6e92df9649661ab1c6fefe06f83206382322e6997fcb70ca53d22d302fac57`
  - `emotion/pool.jsonl`: `dde4702b5f5168f5fad977ba6b659f9304ae02a3cb37c18281e208271f847d40`
  - `emotion/test.jsonl`: `d3b7176e8ab6be052151511a3e9843fa633c5747d0dae73980a2e52a4fb6cd88`

---

### 1.3 Test Set Isolation and Leakage Boundaries

The official benchmark test sets (500 items for TREC, 2,000 items for Emotion) are held strictly separate to evaluate generalization:
- **No Direct Sampling of Test Rows:** No official test row was directly sampled into training sets, low-resource source samples, or few-shot demonstration pools.
- **Generation-Time Exclusion:** During LLM synthetic generation, every candidate synthetic sentence was checked against the normalized text of the official test set. Any exact normalized test match was immediately rejected and excluded from candidate corpora.
- **Pinned Official Duplicate Boundary:** As detailed in Section 1.4, one normalized official TREC train/test text duplicate (`trec.train.05262` = `trec.test.00378`) was retained in conditions C1 and C4 under the approved "Pin + Disclose" decision. All other prohibited overlaps—including any test overlap in synthetic corpora, low-resource samples, or Emotion conditions—fail closed and abort execution.
- **Evaluation Integrity:** Classifiers are evaluated solely against these fixed official test sets, preserving direct comparability to published external benchmarks and avoiding optimistic evaluation artifacts.

---

### 1.4 Disclosure of the Pinned TREC Train/Test Duplicate

The official TREC release contains question texts that appear in both its training and test splits (the exact number of such duplicates is not recorded in committed evidence). During our canonical seed-13 sampling:
- The sampled real subset for condition C1 (and the real half of C4) drew question `trec.train.05262` (*"What is the abbreviation for Texas ?"*).
- This text is identical under whitespace/case normalization to official test item `trec.test.00378` (label `ABBR`, normalized SHA-256 `ba6af17c366034f8dca2abff920ba01cc773fc77eda6188a1127b8190b324d2e`).
- **Lead Governance Decision ("Pin + Disclose", 1 November 2026):** To avoid post-hoc data tampering or non-canonical split filtering, the lead approved retaining this sampled row while strictly disclosing it in the training record (`training.disclosed_test_text_overlap`).
- **Limited Scope:** This duplicate appears **only** in TREC C1 and the real partition of TREC C4. It does **not** appear in TREC C2, C3, C5a, C5b, nor in any Emotion condition. All other test-ID or test-text overlaps (including any overlap in synthetic corpora) fail closed and abort training.

---

### 1.5 Canonical Low-Resource Real Sampling (C5a)

To evaluate low-resource classification and augmentation baselines, a deterministic low-resource sample of exactly 10 examples per label across all 6 classes (60 examples total per task) was drawn from `pool.jsonl` using seed 13 (`ws2_training/prepare_data.py`), so it is disjoint from `train.jsonl`:
- File paths: `data/real/trec/lowres_seed13.jsonl` and `data/real/emotion/lowres_seed13.jsonl`.
- Canonical hashes:
  - `trec/lowres_seed13.jsonl` (60 lines): `901c0596a42b6fd0fa128cb1dd05b26bb1dbdcbd0c1a0a3dca9011beb7a03a29`
  - `emotion/lowres_seed13.jsonl` (60 lines): `7bec4826ca96fa056ec5defc13a051f1c2ac25ef0270164219af13e1471af27f`
- These exact 60 instances serve as both the standalone low-resource real baseline (Condition C5a) and the exact seed sources for AugGPT paraphrasing (Condition C5b).

---

## 2. Classifier and Experimental Conditions (Report Section 4)

### 2.1 Feature Representation via Frozen Sentence Embeddings

To ensure that downstream classification differences reflect synthetic training data quality rather than variation in language-model fine-tuning dynamics, all sentences are transformed into fixed dense representations using a frozen Sentence-BERT model:
- **Model Identity:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Git Commit / Revision:** `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- **Snapshot File-Hash Mapping Identity (`identity_sha256`):** SHA-256 `36bb95116de1358079c1fab49ac634493101e484c99edbd882691fbf96859cad` (the SHA-256 digest of the canonical snapshot file-hash mapping in `model_identity`; individual snapshot-file SHA-256 values are recorded separately).
- **Embedding Configuration:**
  - Embedding dimensionality: 384 dimensions.
  - Frozen weights: strictly inference-only; zero parameters are updated during training.
  - Feature normalization: L2 unit-sphere normalization (`normalize_embeddings=True`).
  - Execution runtime: CPU execution, batch size 64, offline mode (no network requests).

---

### 2.2 Classification Head Configuration

On top of the frozen 384-dimensional normalized embeddings, a multi-class linear classifier is trained using scikit-learn's `LogisticRegression`:
- **Recorded settings:** `LogisticRegression(C=1.0, max_iter=2000, random_state=13)`, exactly as constructed in `ws2_training/train_scope30.py` and recorded in every result binding (`analysis.json` → `grid.classifier.classifier`).
- **Unrecorded settings:** every other argument (for example the solver, tolerance and multi-class handling) is the scikit-learn default of the installed version. These defaults are not recorded in the grid binding, so this report does not state them. Whether each fit converged is recorded per key in the uncommitted training records (`n_iter`/`converged`) and is not reported here.
- **Fit order:** rows are shuffled with `random.Random(13)` before fitting, as in Issue #7.

---

### 2.3 The scope30-v1 Six-Condition Experimental Design

Under project lead decision [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), the study implements the **scope30-v1** design: an exploratory single-seed (seed 13) comparison evaluating 6 experimental conditions across both benchmark tasks (12 unique result keys total):

| Condition | Description | Construction per Label | Total Training Rows ($n_{\text{train}}$) |
|---|---|---|---:|
| **C1** | Real data only (primary reference baseline) | 30 real training examples | 180 |
| **C2** | Zero-shot synthetic data only | 30 zero-shot LLM examples | 180 |
| **C3** | Few-shot synthetic data only | 30 few-shot LLM examples (3 demos/prompt) | 180 |
| **C4** | 50/50 Real + few-shot synthetic mix | 15 real + 15 few-shot synthetic examples | 180 |
| **C5a** | Low-resource real baseline | 10 real source examples (`lowres_seed13.jsonl`) | 60 |
| **C5b** | AugGPT-augmented low-resource | 10 real sources + 6 AugGPT paraphrases per source | 420 |

#### Condition Composition Notes:
- **C1–C4 Balance:** Conditions C1, C2, C3, and C4 maintain an identical classifier training size of exactly 180 examples per task (30 per label across all 6 classes).
- **C4 Composition:** Condition C4 combines an independent seed-13 `per_label_sample(train, 15, seed 13, exact)` draw of 15 real rows per label (a separate draw; it is not constructed as a subset of the C1 sample) with the first 15 few-shot rows per label of the C3 corpus in the ADR 0012 rank order (ascending SHA-256 of `scope30-v1|13|<task>|few_shot|<row-id>`).
- **C5a/C5b Unequal Size Disclosure:** Condition C5a contains 60 examples (10 per label), while C5b contains 420 examples (60 real sources + 360 paraphrases). The size difference between C5a ($n=60$) and C5b ($n=420$) is an intentional, fundamental property of data augmentation via paraphrasing. As disclosed in Section 2.6, any performance change between C5b and C5a combines the semantic quality of the paraphrases with the effect of a $7\times$ increase in training set size.
- **Generated Corpus Totals:** Across both tasks, the experiment selects 720 zero/few-shot texts (360 TREC, 360 Emotion) and 720 AugGPT paraphrases (360 TREC, 360 Emotion), totaling 1,440 selected synthetic texts.

---

### 2.4 Deterministic Selection, Freeze Integrity, and Provenance

To transition from raw generation outputs (which initially targeted up to 100 rows per label during exploratory generation) to the approved 30-per-label scope without biased sampling:
1. **Deterministic Selection Rule (ADR 0012):** Within each label group of a candidate synthetic artifact, valid rows are ordered by ascending hexadecimal SHA-256 hash of the UTF-8 string:
   $$\text{SHA-256}(\text{"scope30-v1|13|} \,\|\, \text{task} \,\|\, \text{"|"} \,\|\, \text{setting} \,\|\, \text{"|"} \,\|\, \text{row\_id})$$
   with row ID as a deterministic tie-breaker. The first 30 ranked rows are extracted into `<setting>_seed13.scope30-v1.selected.jsonl`. This rule depends solely on stable row identifiers and never references text content or test performance.
2. **AugGPT Source Matching:** C5b AugGPT generation requires strict 1-to-1 matching to the 60 canonical C5a sources, with exactly 6 distinct valid paraphrases per source. For TREC AugGPT, semantic answer-type drifting was corrected under ADR 0013 (v2 candidate promoted via PR #31).
3. **Cryptographic Freeze Manifest:** The complete synthetic corpus was validated and frozen under Issue #12 in `docs/freeze/scope30-v1-seed13.freeze.json` (canonical SHA-256: `dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1`). The freeze binds all file paths, SHA-256 digests, row counts, ordered ID lists, model identity (`qwen3:4b` digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`), prompt hashes, and model-assisted review outcomes (`docs/reviews/issue11/`).
4. **Self-Hashed Provenance Enforcement:** The training harness (`ws2_training/train_scope30.py`) produces an immutable provenance record for every result key (`records/<task>.<condition>.seed13.json`). Each record binds the freeze hash, code commit SHA (`1354118b4b05bec31973c268057d8a713907f3a6`), exact ordered ID list, embedder snapshot file-hash mapping identity (`model_identity.identity_sha256`), classifier hyperparameters, and test evaluation scores.

---

### 2.5 Training Workflow, Resume, Replacement, and Validation Behavior

The training execution pipeline (`ws2_training/train_scope30.py`) enforces strict fail-closed reproducibility:
- **Preflight Freeze Check:** Before constructing any training batch or invoking scikit-learn, the trainer independently verifies that `scope30-v1-seed13.freeze.json` matches SHA-256 `dd2e52ee...`. If verification fails, the process immediately aborts.
- **Storage Isolation:** Results are written to a scope-dedicated directory `results/scope30-v1-seed13-freeze-v1/`. Legacy result paths (e.g. `results/scope30-seed13/` or historical directories) are explicitly refused to prevent mixing across experimental scopes.
- **Idempotent Resume:** When executing training, existing records are inspected. A key is reused if and only if its recorded provenance cryptographically matches current code, freeze, input data, and model configuration bit-for-bit.
- **Targeted Replacement:** Any modified or unverified record halts execution unless explicitly authorized via `--replace <key>`, which archives superseded records to a `superseded/` directory.
- **Grid Validation (`--validate`):** Following execution, a dedicated validation routine confirms that exactly 12 unique keys are present, re-verifies all self-hashes, checks `runs.csv` against `grid.json`, and validates the aggregate completion token `grid_complete.json` (SHA-256 `3b12bbb878d8191e387e3abc0f77fc02d9f081b98f0216f9b64dbb144bf40032`).
- **Legacy Trainer Guard:** The legacy training script `ws2_training/train.py` refuses real execution for conditions C2, C3, C4, and C5b, preventing accidental execution using outdated configurations.

---

### 2.6 Classifier Input Counts vs. Real Labelling Costs

A central methodological requirement of this investigation is distinguishing the number of training examples supplied to the classifier ($n_{\text{train}}$) from the actual human labelling investment required to produce that data:

| Task | Condition | $n_{\text{train}}$ | Real Training Rows | Synthetic Training Rows | Real Demos for Generation | Labelling Cost Accounting Note |
|---|---|---:|---:|---:|---:|---|
| **TREC** | **C1** | 180 | 180 | 0 | 0 | 180 real annotations required |
| **TREC** | **C2** | 180 | 0 | 180 | 0 | 0 real annotations (zero-shot prompt uses label definitions only) |
| **TREC** | **C3** | 180 | 0 | 180 | 59 | 59 real pool annotations required as 3-shot prompt demonstrations |
| **TREC** | **C4** | 180 | 90 | 90 | ≤ 59 | 90 real annotations + up to 59 demonstration annotations ($\le 149$ total) |
| **TREC** | **C5a** | 60 | 60 | 0 | 0 | 60 real annotations required |
| **TREC** | **C5b** | 420 | 60 | 360 | 0 | Exactly the same 60 real source annotations as C5a ($7\times$ training size) |
| **Emotion** | **C1** | 180 | 180 | 0 | 0 | 180 real annotations required |
| **Emotion** | **C2** | 180 | 0 | 180 | 0 | 0 real annotations (zero-shot prompt uses label definitions only) |
| **Emotion** | **C3** | 180 | 0 | 180 | 54 | 54 real pool annotations required as 3-shot prompt demonstrations |
| **Emotion** | **C4** | 180 | 90 | 90 | ≤ 54 | 90 real annotations + up to 54 demonstration annotations ($\le 144$ total) |
| **Emotion** | **C5a** | 60 | 60 | 0 | 0 | 60 real annotations required |
| **Emotion** | **C5b** | 420 | 60 | 360 | 0 | Exactly the same 60 real source annotations as C5a ($7\times$ training size) |

#### Critical Methodological Takeaways:
1. **Input Parity $\neq$ Cost Parity:** While C1, C2, and C3 all feed 180 rows to the classifier, C2 represents an automated pipeline requiring 0 real data annotations, whereas C3 consumed 59 (TREC) or 54 (Emotion) real human-annotated examples to serve as in-context prompt exemplars.
2. **C5a vs. C5b Scaling:** C5a and C5b incur the exact same human labelling cost (60 real examples). However, C5b feeds 420 rows to the classifier ($60\text{ real} + 360\text{ paraphrases}$). Consequently, C5b evaluates the utility of augmenting a fixed labelling budget with LLM paraphrases, but its raw classifier input count is not matched to C5a.

---

### 2.7 Metric Reporting and Paired-Gap Calculations

Classification effectiveness is evaluated on the untouched official test splits:
- **Macro-Averaged F1 Score:** The unweighted arithmetic mean of per-class F1 scores:
  $$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} F_{1, c}$$
  Macro-F1 is our primary evaluation metric, preventing dominant majority classes from masking poor performance on minority labels.
- **Classification Accuracy:** The overall proportion of correct test predictions.
- **Paired Relative Gap:** For synthetic-only conditions C2 and C3, performance is evaluated relative to the matching real baseline C1 trained on the same task and seed:
  $$\text{Relative Gap} = \frac{\text{Macro-F1}_{\text{synthetic}} - \text{Macro-F1}_{\text{C1}}}{\text{Macro-F1}_{\text{C1}}}$$
- **Paired Absolute Gap:**
  $$\text{Absolute Gap} = \text{Macro-F1}_{\text{synthetic}} - \text{Macro-F1}_{\text{C1}}$$
- **Zero Denominator Handling:** If a baseline score $\text{Macro-F1}_{\text{C1}} = 0$, the relative gap is recorded as `null` (displayed as `"undefined"`), the task is preserved, and the absolute difference is reported. A missing C1 result blocks analysis execution entirely.
- **Single-Run Scientific Boundary:** Because scope30-v1 evaluates a single random seed (seed 13), all reported values are exact single-run measurements. **No cross-seed standard deviations, confidence intervals, standard errors, zero-uncertainty error bars, or statistical significance tests ($p$-values) are calculated or claimed.**

---

### 2.8 Reference to Accepted Experimental Results

The twelve-key scope30-v1 training grid was executed once on 1 November 2026 under merge commit `1354118b4b05bec31973c268057d8a713907f3a6` (PR #33) and accepted by lead review (Issue #13). Downstream analysis was conducted by `ws3_evaluation/analyze_scope30.py` and committed in `docs/analysis/scope30-v1-seed13/` (PR #35):

- **Accepted Aggregate Grid Checksum:** `grid_complete.json` SHA-256 `3b12bbb878d8191e387e3abc0f77fc02d9f081b98f0216f9b64dbb144bf40032`.
- **Accepted Analysis Checksums:** `analysis.json` SHA-256 `d0486efdceb6cec23d47fc0a943f3307a65fe9230913c38f7f50af5dc26177f6`; `summary.md` SHA-256 `a95abe66758878a5b5e2bbb29eca8a203f4bb00d93443915a93236869d09b989`.

#### Summary of Accepted Evaluation Scores:
- **TREC (Objective Benchmark):**
  - C1 (Real 30/label): Macro-F1 = **0.6009**, Accuracy = **0.6520**
  - C2 (Zero-shot synthetic): Macro-F1 = **0.6257**, Accuracy = **0.6680** (Relative Gap: **+4.12%**, Absolute Gap: **+0.0247**)
  - C3 (Few-shot synthetic): Macro-F1 = **0.6411**, Accuracy = **0.6880** (Relative Gap: **+6.69%**, Absolute Gap: **+0.0402**)
  - C4 (50/50 Real + few-shot): Macro-F1 = **0.6638**, Accuracy = **0.7100** (C4 vs. C1: **+0.0629**)
  - C5a (Low-res real 10/label): Macro-F1 = **0.5974**, Accuracy = **0.6400**
  - C5b (Low-res + AugGPT): Macro-F1 = **0.6251**, Accuracy = **0.6700** (C5b vs. C5a: **+0.0277**)
- **Emotion (Subjective Benchmark):**
  - C1 (Real 30/label): Macro-F1 = **0.4082**, Accuracy = **0.4420**
  - C2 (Zero-shot synthetic): Macro-F1 = **0.3804**, Accuracy = **0.4305** (Relative Gap: **-6.80%**, Absolute Gap: **-0.0278**)
  - C3 (Few-shot synthetic): Macro-F1 = **0.3841**, Accuracy = **0.4285** (Relative Gap: **-5.90%**, Absolute Gap: **-0.0241**)
  - C4 (50/50 Real + few-shot): Macro-F1 = **0.4033**, Accuracy = **0.4425** (C4 vs. C1: **-0.0049**)
  - C5a (Low-res real 10/label): Macro-F1 = **0.3381**, Accuracy = **0.3730**
  - C5b (Low-res + AugGPT): Macro-F1 = **0.3432**, Accuracy = **0.3705** (C5b vs. C5a: **+0.0051**)

---

### 2.9 Scientific Boundaries and Methodological Guardrails

All interpretations of these methods and results are subject to strict scientific boundaries:
1. **Single-Seed Limitation:** All findings are derived from a single seed (`seed = 13`). Differences of a few hundredths of macro-F1 can arise from sample selection alone. These results do not establish statistical significance, cross-seed stability, or generalizability across random seeds.
2. **Non-Causal Task Comparison:** While TREC and Emotion represent objective and subjective classification tasks respectively, they also differ across text length, syntactic complexity, vocabulary distribution, class balance, and test set size (500 vs. 2,000 instances). Performance differences cannot be causally attributed to semantic subjectivity alone.
3. **Quality Review Boundaries:** Synthetic dataset audit was conducted via model-assisted evaluation (Claude Opus 5.5, 60 samples across 6 batches; Issue #11). Known semantic drift and answer-type alterations in AugGPT paraphrases were accepted under lead Option A and disclosed in `docs/freeze/README.md`. These reviews do not constitute independent human consensus or population error bounds.
4. **Distinction Between Demonstration Fixtures and Evidence:** Unit tests and continuous integration workflows employ fake-data fixtures (`data/sample/`) and hashing embedders. Fake fixture scores are exclusively test harness checks and are never reported as scientific evidence.
