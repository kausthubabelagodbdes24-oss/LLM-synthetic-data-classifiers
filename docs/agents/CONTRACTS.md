# Shared contract register

The specification defines experimental meaning. Record exact accepted interfaces here or link to their authoritative definitions. Proposed formats are not accepted contracts.

| Interface | Current state / agreement required | Owners |
| --- | --- | --- |
| Real data splits and low-resource samples | Accepted in Issue #5. Real splits and 10-per-label lowres source samples (seeds 13, 42, 7) prepared reproducibly with strict class coverage validation and verified SHA-256 checksums. | Training and lead (@kausthubabelagodbdes24-oss) |
| Repetition-specific generation | Legacy artifacts are shared across seeds. Agree naming/layout, repetition identity, IDs, source provenance, generation provenance, resume boundaries and legacy handling before the generation producer and training consumer migrate. | Generation + training; lead approves shared changes |
| Completion, audit and freeze evidence | Exact targets and the audit rule are agreed. Define how accepted artifacts identify the frozen revision and how training verifies approval/completeness. | Generation + training; evaluation consumes provenance |
| Run records and paired analysis | Preserve unique task/condition/repetition keys and exact matching baselines. Agree precision, provenance and schema changes before changing writer/reader. Zero-baseline policy remains unresolved. | Training + evaluation |
| Annotation exchange | Independent responses must combine without changed judgments. Agree stable item matching, label validation, missing responses and actual active-annotator reporting; preserve original responses. | Evaluation + all human annotators |

For each accepted change record: exact contract, approving issue/PR and lead decision, affected producers/consumers, migration/compatibility behavior and verification evidence. Do not mark a format agreed without recording details. Keep consumers working during migration or coordinate a bounded integration change explicitly. Update this register as implementation lands; do not freeze an invented schema just to make another model's task easier.

## Real data splits and repetition-matched source samples

- **Approving issue & lead decision**: Issue #5 ([Training and project lead] Prepare reproducible real data and repetition-matched source samples), approved by @kausthubabelagodbdes24-oss.
- **Producer**: `ws2_training/prepare_data.py` (Owner: @kausthubabelagodbdes24-oss).
- **Consumers**:
  - `ws1_generation/generate.py`: reads `data/real/<task>/pool.jsonl` (for few-shot demonstration sampling) and `data/real/<task>/lowres_seed<S>.jsonl` (for AugGPT source example paraphrasing).
  - `ws2_training/train.py`: reads `data/real/<task>/train.jsonl` (C1/C4 real training), `data/real/<task>/test.jsonl` (held-out test evaluations), and `data/real/<task>/lowres_seed<S>.jsonl` (C5a/C5b low-resource training).
  - `ws3_evaluation/annotate.py`: reads `data/real/<task>/test.jsonl` (for sampling human annotation items).
- **File paths and structure**:
  - Written under `data/real/<task>/` (where `<task>` is `trec` or `emotion`):
    - `train.jsonl`: 90% real training split.
    - `pool.jsonl`: 10% demonstration and low-resource pool split (isolated via deterministic seed 0 shuffle). Disjoint from `train.jsonl`.
    - `test.jsonl`: Untouched official test split.
    - `lowres_seed13.jsonl`, `lowres_seed42.jsonl`, `lowres_seed7.jsonl`: exactly 10 examples per label (60 examples per file for 6 labels) sampled deterministically from `pool.jsonl`.
  - Schema: UTF-8 JSON Lines with keys:
    - `id`: unique identifier (`<task>.<split>.<index>`)
    - `text`: normalized whitespace text string
    - `label`: canonical label name string
- **Observed split sizes**:
  - **TREC** (`SetFit/TREC-QC`):
    - Official train: 5,452 -> `pool.jsonl`: 545 rows, `train.jsonl`: 4,907 rows.
    - Official test -> `test.jsonl`: 500 rows.
    - Pool label distribution (545 rows): abbreviation: 10, description: 126, entity: 128, human: 124, location: 78, numeric: 79. (Note: `abbreviation` has exactly 10 examples, meeting the threshold).
    - Low-resource samples (`lowres_seed13.jsonl`, `lowres_seed42.jsonl`, `lowres_seed7.jsonl`): exactly 60 rows each (10 per label across all 6 labels).
  - **Emotion** (`dair-ai/emotion`, split configuration):
    - Official train: 16,000 -> `pool.jsonl`: 1,600 rows, `train.jsonl`: 14,400 rows.
    - Official test -> `test.jsonl`: 2,000 rows.
    - Pool label distribution (1,600 rows): anger: 252, fear: 178, joy: 539, love: 144, sadness: 430, surprise: 57.
    - Low-resource samples (`lowres_seed13.jsonl`, `lowres_seed42.jsonl`, `lowres_seed7.jsonl`): exactly 60 rows each (10 per label across all 6 labels).
- **Validation and rejection policy**:
  - Rejection gate: `split_data` and `per_label_sample(..., exact=True)` strictly validate that all expected labels are present in the pool with at least `lowres_per_label` (10) examples. If any label has fewer than 10 examples or is missing, a `ValueError` is raised immediately. Silent undersampling is forbidden.
  - Held-out test set is retained untouched from the official dataset split.
- **Verification**:
  - Rebuild command: `python -m ws2_training.prepare_data`
  - Verification command: `python -m ws2_training.prepare_data --verify` checks existing files against the SHA-256 hashes and line counts.
