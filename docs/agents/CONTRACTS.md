# Shared contract register

The specification defines experimental meaning. Record exact accepted interfaces here or link to their authoritative definitions. Proposed formats are not accepted contracts.

| Interface | Current state / agreement required | Owners |
| --- | --- | --- |
| Real data splits and low-resource samples | Accepted in Issue #5. Real splits and 10-per-label lowres source samples (seeds 13, 42, 7) prepared reproducibly with strict class coverage validation, canonical UTF-8 JSONL with LF newlines, and verified SHA-256 checksums. | Training and lead (@kausthubabelagodbdes24-oss) |
| Repetition-specific generation | Proposed in PR #22 for Issue #6; acceptance is pending owner-run Ollama smoke/pilot evidence and review. Real and fake artifacts are isolated, and resume fails closed on incompatible row, model, prompt, option or source identity. | Generation (@harshitadjainbdes24-byte) + training (@kausthubabelagodbdes24-oss) |
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
  - Serialization: Canonical UTF-8 JSON Lines with explicit LF newlines (`\n`, byte `0x0A`, no OS-specific CRLF translation) using `common.write_jsonl`.
  - Record schema:
    - `id`: unique identifier string (`<task>.<split>.<index>`)
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
- **Canonical SHA-256 Manifest**:
  - `trec/train.jsonl`: 4,907 lines, `b587c84d26b361bbe6f4126aff136e9d737c6c349059e89d2a3ce79e226f4331`
  - `trec/pool.jsonl`: 545 lines, `0a78fbe6e38b8b9eb4f7d9e24e619d63f855e073bffb1e00edb23508490d3a6e`
  - `trec/test.jsonl`: 500 lines, `415f526277d0f23f9bfe903a2a6d994529ec183820b6d9d0d55d47c3195aa0c0`
  - `trec/lowres_seed13.jsonl`: 60 lines, `901c0596a42b6fd0fa128cb1dd05b26bb1dbdcbd0c1a0a3dca9011beb7a03a29`
  - `trec/lowres_seed42.jsonl`: 60 lines, `fce20620829492d339fea73ea89eca3f00df2bb4d84c93d6c4b399bbc1eb5ae3`
  - `trec/lowres_seed7.jsonl`: 60 lines, `cc790390509218422206fa1a1fb8c3afffce93297c12a62b66a32b047a7a3fee`
  - `emotion/train.jsonl`: 14,400 lines, `4c6e92df9649661ab1c6fefe06f83206382322e6997fcb70ca53d22d302fac57`
  - `emotion/pool.jsonl`: 1,600 lines, `dde4702b5f5168f5fad977ba6b659f9304ae02a3cb37c18281e208271f847d40`
  - `emotion/test.jsonl`: 2,000 lines, `d3b7176e8ab6be052151511a3e9843fa633c5747d0dae73980a2e52a4fb6cd88`
  - `emotion/lowres_seed13.jsonl`: 60 lines, `7bec4826ca96fa056ec5defc13a051f1c2ac25ef0270164219af13e1471af27f`
  - `emotion/lowres_seed42.jsonl`: 60 lines, `4af40bceb9cb819ceb76ac3a6d99bc6495aec7501fc541d579adfb2649c5f192`
  - `emotion/lowres_seed7.jsonl`: 60 lines, `4134701559a61d2a288119a1bca4d2a84227ee48fca0962821959e59b30b10d3`
- **Verification**:
  - Rebuild command: `python -m ws2_training.prepare_data`
  - Verification command: `python -m ws2_training.prepare_data --verify` checks existing files against the SHA-256 hashes and line counts.

## Repetition-specific generation & provenance

- **Approval status**: Proposed by PR #22 for Issue #6 ([Generation] Run a repetition-aware generation pilot and choose the uniform target), owned by @harshitadjainbdes24-byte. Not yet accepted for integration; the real Ollama smoke, pilot, inspection and target decision remain pending.
- **Producer**: `ws1_generation/generate.py` (Owner: @harshitadjainbdes24-byte).
- **Consumers**:
  - `ws2_training/train.py` (C2, C3, C4, C5b model training)
  - `ws3_evaluation/` (provenance audit, reporting, appendix documentation)
- **File paths and structure**:
  - Real model output is written under `data/synthetic/<task>/`; fake dry-run output is written under `data/synthetic-fixtures/<task>/` and cannot satisfy a real resume target.
  - In either mode:
    - Repetition-specific files: `<setting>_seed<seed>.jsonl` where `<setting>` is `zero_shot`, `few_shot`, or `auggpt`, and `<seed>` is in `[13, 42, 7]`.
    - Sidecar provenance manifest: `<setting>_seed<seed>.provenance.json`.
    - Legacy unseeded compatibility: `<setting>.jsonl` (when `--seed` is omitted).
  - Serialization: Canonical UTF-8 JSON Lines with explicit LF newlines (`\n`) using `common.write_jsonl`.
  - Record schema:
    ```json
    {
      "id": "<task>.<setting>.seed<seed>.<label>.<index:04d>",
      "task": "<task>",
      "text": "<generated text>",
      "label": "<canonical label>",
      "setting": "<zero_shot|few_shot|auggpt>",
      "repetition": 13,
      "source_ids": ["<id1>", "<id2>", "<id3>"],
      "generation_mode": "real",
      "schema_version": 2,
      "generation_fingerprint": "<sha256>"
    }
    ```
    - `source_ids`: empty list `[]` for `zero_shot`; contains exactly three real demonstration IDs from `data/real/<task>/pool.jsonl` for `few_shot`; contains single source ID from `lowres_seed<seed>.jsonl` for `auggpt`.
- **Sidecar Provenance Manifest Schema**:
  - Path: `data/synthetic/<task>/<setting>_seed<seed>.provenance.json`
  - Keys:
    - `task`: task name (`trec` or `emotion`)
    - `setting`: generation setting (`zero_shot`, `few_shot`, `auggpt`)
    - `repetition`: seed (`13`, `42`, `7`, or `null` for legacy)
    - `generation_mode`: `real` or `fake`; fake and real sidecars never share a path.
    - `model_identity`: provider, configured tag, resolved model name and actual Ollama digest (or the explicit fixture identity).
    - `generation_identity` / `generation_fingerprint`: stable task, setting, repetition, mode, model digest, prompt dependency hashes, options hash and source-input hashes used to authorize resume. Few-shot identity covers both its own template and the inherited zero-shot template.
    - `target_per_label`: target count per class
    - `counts_by_label`: dictionary mapping each class label to accepted count
    - `total_count`: total accepted rows in file
    - `is_complete`: boolean gate indicating if all classes meet target
    - `distinct_demonstrations_count`: count of unique real demonstration items used
    - `distinct_demonstration_ids`: sorted list of distinct pool source IDs
    - `sessions`: append-only invocation history containing a UUID run ID, start/finish, actual elapsed seconds, target, before/after counts, accepted row IDs, per-session statistics and completion/no-op status.
    - `cumulative_stats`: diagnostic counters accumulated without erasing prior sessions.
    - `created_at` / `updated_at`: ISO 8601 UTC timestamps.
- **Resume and isolation policy**:
  - Resume boundaries are isolated to mode plus `(task, setting, repetition)` and the complete generation fingerprint.
  - When re-running, existing rows in `<setting>_seed<seed>.jsonl` are read; if already at or above target, 0 calls are made.
  - Rows from one repetition are never copied, reshuffled, or mixed into another repetition.
  - Missing/null identity fields, corrupt JSONL, missing/corrupt sidecars, incompatible model digests, prompt/options changes, and changed or misplaced source identities fail before existing rows are counted.
    - Each accepted labeled-generation batch uses an identity-bound write-ahead journal before JSONL append, followed by atomic provenance replacement and journal removal. Restart reconciles missing, partial or completed batch appends against the exact persisted prefix and journal rows, records the prior session as `interrupted`, preserves accepted rows/history and resumes the shortfall. Missing or incompatible journals do not bypass ordinary provenance validation.
- **Migration & backward compatibility**:
  - `common.synth_path(task, setting, seed=None, mode=None)` defaults to real storage. `mode="fake"` selects fixture-only storage; the offline pipeline test opts into it with `SYNTHTEXT_GENERATION_MODE=fake`.
  - Unseeded file naming remains available for the legacy smoke seam, but legacy rows without the required identity fields and matching sidecar are intentionally not resumable. Regenerate them in the selected mode instead of relabeling or silently counting them.
- **Verification**:
  - Unit tests: `tests/test_generation.py` verifies repetition isolation, fake-to-real isolation, model-call interruption, crash recovery across initial and resumed JSONL/provenance writes, stale temporary checkpoint handling, fail-closed corrupt/legacy/misplaced rows, model/prompt dependency incompatibility, no-op history preservation, demonstration source tracking, AugGPT fixture isolation, non-example/malformed reply filtering, and mock connection recovery.
  - Full pipeline test: `tests/test_pipeline.py` uses fake-only storage and offline embeddings. Passing it is smoke validation, not Ollama or experimental evidence.
