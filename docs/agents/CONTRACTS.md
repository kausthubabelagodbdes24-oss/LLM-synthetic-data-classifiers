# Shared contract register

**Current authority: [ADR0012](../adr/0012-eod-exploratory-30-seed13.md), approved 31 October 2026.** Both tasks, six conditions, seed13 only, 30/label in C1–C4; totals180/180/180/180/60/420. Twelve result keys. Human annotation, distribution, external sharing and alpha are not required. Read [migration](../SCOPE_MIGRATION.md).

Historical schemas/canonical hashes below remain valid. Target100 and three-seed statements describe earlier deliveries and are superseded for current execution. Producer sidecars preserve original targets/counts. Selected manifests record exact scope30-v1 IDs, original hashes/fingerprints; use ADR0012's deterministic excess-selection rule. Producers/consumers must implement and verify this in #10/#13; documentation does not establish implementation.


The specification defines experimental meaning. Record exact accepted interfaces here or link to their authoritative definitions. Proposed formats are not accepted contracts.

**30 October fallback-pilot decision:** the sole operator/lead Kausthuba approved an isolated `qwen3:4b` pilot using unchanged final prompts and data interfaces; see [ADR 0010](../adr/0010-test-qwen-fallback-after-pilot-failure.md). This approves testing the shared model setting, not dataset acceptance or full generation. Preserve all earlier Llama artifacts and reviews; model digests/fingerprints remain incompatible across resume. Runtime option changes need a separate decision if compatibility problems arise.

**Historical ADR0011 acceptance; target superseded by ADR0012:** Kausthuba accepted Qwen with the documented quality limitations and chose uniform 100 per label. See [ADR 0011](../adr/0011-accept-qwen-pilot-and-uniform-100-target.md). Lead scientific acceptance and final implementation review are recorded; consult PR #22 for current CI/merge state. Exact full-data completeness, content audit and freeze acceptance are still future gates.

| Interface | Current state / agreement required | Owners |
| --- | --- | --- |
| Real data splits and low-resource samples | Accepted in Issue #5. Real splits and 10-per-label lowres source samples (seeds 13, 42, 7) prepared reproducibly with strict class coverage validation, canonical UTF-8 JSONL with LF newlines, and verified SHA-256 checksums. | Training and lead (@kausthubabelagodbdes24-oss) |
| Repetition-specific generation | Accepted Qwen pilot for Issue #6; current scope30/seed13 under ADR0012; implementation delivery and CI/merge evidence are in PR #22. Real/fake storage and identity-bound seeded resume are unchanged. | Sole operator and lead: @kausthubabelagodbdes24-oss |
| Completion, audit and freeze evidence | Exact targets and the audit rule are agreed. Define how accepted artifacts identify the frozen revision and how training verifies approval/completeness. | Generation + training; evaluation consumes provenance |
| Run records and paired analysis | Four seed-13 baseline result keys (C1=180, C5a=60 for trec and emotion); fresh results/scope30-seed13 storage. ADR0012 defines current single-seed exploratory scope; 12 total keys. | Training (@kausthubabelagodbdes24-oss) + evaluation |
| Annotation exchange | Historical optional response tooling only. No human collection/agreement gate. Current model-assisted quality reviews disclose method and preserve originals. | Evaluation + all human annotators |

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

- **Historical pilot approval (100 target superseded by ADR0012)**: Lead scientific acceptance recorded on 31 October 2026 in ADR 0011 for Issue #6 and PR #22. Kausthuba accepted the real Qwen pilot with its disclosed quality limitations and uniform 100-per-label target. Final implementation review is clear; integration requires current-head CI at merge (see PR #22). Full-data audit/freeze acceptance is not implied.
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
    - Each accepted labeled-generation batch uses an identity-bound write-ahead journal before JSONL append, followed by atomic provenance replacement and journal removal. Restart consults the journal before strict JSONL parsing, verifies the exact raw-byte prefix and authorized batch serialization (including a partial UTF-8 sequence), completes only the missing authorized bytes, then validates the recovered rows normally. It records the prior session as `interrupted`, preserves accepted rows/history and resumes the shortfall. Missing or incompatible journals do not bypass ordinary provenance validation.
- **scope30-v1 zero/few-shot selection (Issue #10, ADR 0012)**:
  - Producer: `ws1_generation/select_scope30.py`, seed 13 only. The original `<setting>_seed13.jsonl` and `.provenance.json` files are read-only inputs and are never truncated or rewritten. Producer sidecar targets (pilot 10, historical 100, resume 30) stay in their append-only `sessions`, and the selected 30 is validated independently of them.
  - Shortfalls: resume compatible generation with `python -m ws1_generation.generate --setting <zero_shot|few_shot> --task <task> --seed 13 --target 30`. The generation fingerprint excludes the target, so this appends only the missing rows under the unchanged identity.
  - Validation before selection (fail-closed): approved model pin; fingerprint recomputed from current prompts, options and sources; every row passes the producer's resume rules (schema, ID pattern, labels, unique IDs, normalized-duplicate rejection, few-shot demonstrations from the same-label pool); sidecar `total_count`/`counts_by_label` match the rows; no normalized held-out test overlap; no copied demonstrations; no pending journal.
  - Rule: per label, ascending SHA-256 hex of UTF-8 `scope30-v1|13|<task>|<setting>|<row-id>`, with row ID as tie-breaker, taking the first 30. The rule depends only on row IDs, never on text or test performance. A label with fewer than 30 valid rows fails as a shortfall.
  - Derived files next to the original: `<setting>_seed13.scope30-v1.selection.json`, a deterministic manifest with no timestamps. It records the rule, `selected_ids_by_label` in rank order, `selected_ids`, selected rows identity, the selected JSONL SHA-256, the original JSONL/sidecar SHA-256 and byte/row counts, `counts_by_label`/`excess_by_label`, the recorded `generation_fingerprint` and per-row fingerprints, model identity, temperature, recorded and per-session targets/statuses, total elapsed seconds, cumulative stats, and selected distinct demonstrations (C3 labeling cost). The second file is `<setting>_seed13.scope30-v1.selected.jsonl`: exactly 180 rows, byte-identical to their original lines, in original file order. Existing derived files with different content are not overwritten.
  - Handoff gate: `python -m ws1_generation.select_scope30 --seed 13 --verify` (read-only). It re-derives everything from the originals and requires exact manifest and selected-file equality, 30 per label, 180 per artifact, unique IDs and texts, and no test overlap. A changed original therefore invalidates the selection. `--status` reports counts and the shortfall to 30. `--dry-run` uses fixtures only (`real_run_evidence: false`). Passing these checks is structural evidence only; it does not establish quality acceptance (#11/#12) or freeze (#12). C3 consumes the selected few-shot corpus and C4 samples 15/label within it (consumer work: #13).
  - Tests: `tests/test_select_scope30.py` (fake model / mocked Ollama only).
- **AugGPT artifacts (Issue #9)**:
  - Current scope (ADR 0012): exactly two real artifacts are required, `trec/auggpt_seed13` and `emotion/auggpt_seed13`, each 360 paraphrases (six for each of the 60 `lowres_seed13` sources), 720 in total. AugGPT has no excess-selection step: every source must have exactly six, and the producer artifact is the C5b input. Run and verify with an explicit `--seed 13`. With the scope30 config, `--seed all` also resolves to `[13]`. Existing seed 42/7 output, if any, is historical and preserved. The fingerprint does not include `per_label`, `train_per_label` or `seeds`, so the scope30 config change keeps existing seed-13 AugGPT artifacts resumable.
  - Row IDs are `<source_id>.seed<seed>.aug<n>` with `n` in `1..auggpt_k` (6); `label` equals the source label; `source_ids` is exactly `[<source_id>]` from that repetition's own `lowres_seed<seed>.jsonl`. A source shared by several repetitions is paraphrased separately in each; rows are never copied between repetitions (wrong-repetition rows fail validation).
  - Sidecar uses `target_per_source`, `source_count`, `counts_by_source` (every exact source, including zeros), `incomplete_source_ids`, `total_count`, `is_complete` (true only when every source has exactly six), `model`/`model_identity`, `temperature`, `generation_identity`/`generation_fingerprint`, append-only `sessions` (`in_progress` checkpoints, then `complete`/`incomplete`/`no_op`/`interrupted`) and `cumulative_stats` (`calls`, `call_failures`, `parse_failures`, `candidates_received`, `rejected_duplicate`, `rejected_test_leak`, `rejected_source_copy`, `accepted_new`). These keys are additive; the row schema is unchanged.
  - Rejected before counting: unparseable/empty/one-word/over-length or assistant-chatter replies, normalized duplicates within the artifact, exact held-out test text, and copies of any source in the repetition's low-resource sample. Source-meaning preservation is **not** machine-checked; it remains the Checkpoint 2 content-audit requirement (ADR 0008/0009).
  - Resume counts each source's existing valid paraphrases and requests only the shortfall; a source with saved rows is incomplete until it has six. Each accepted batch uses the same identity-bound write-ahead journal and atomic sidecar checkpoint as zero/few-shot, and `KeyboardInterrupt` checkpoints an `interrupted` session, so reruns neither lose provenance nor inflate counts. Each invocation allows three calls per incomplete source; rerun the same command until complete.
  - Length guard (`auggpt-length-guard-v1`, operator/lead decision 31 October 2026): a source whose whitespace word count exceeds the task's `max_words` gets `prompt_auggpt_length_guard`, which is `prompt_auggpt` plus an instruction to keep each rephrasing within `max_words` by dropping minor details while keeping the meaning and the writer's feeling. Sources at or below the limit receive the unchanged `prompt_auggpt` text. Reason: the real `qwen3:4b` emotion seed-13 run stalled at 341/360 because the remaining sources (41–47 words) were paraphrased at their own length and every candidate was rejected by the existing 30-word filter. The `max_words` filter itself is unchanged.
  - The current AugGPT `generation_identity` is the earlier identity plus `prompt_extensions.auggpt_length_guard` (version, rule, `max_words`, and source hashes of the guard template and selector), so the fingerprint changes. Rows saved under the immediately preceding identity (identical apart from the guard) are **carried forward**: they keep their recorded `generation_fingerprint` and are never rewritten, new rows carry the current fingerprint, and the sidecar lists each carried identity with its row count under `carried_forward_identities`. Resume, journal recovery and `--verify` accept exactly the current identity and that one predecessor; any other model, source, prompt or option mismatch still fails. Resume and `--verify` also check the record against the rows: a current-identity sidecar must list every predecessor fingerprint present in the rows, with the recomputed predecessor identity and the exact row count, and nothing else (missing, emptied, unknown, altered, duplicated, wrong-count or self-referencing entries fail). A genuine predecessor-only sidecar (top-level fingerprint is the predecessor) must contain only its own rows and claim no carried identities; no guard provenance is invented for it. The `--verify` summary reports the recorded `generation_fingerprint` and per-fingerprint `row_fingerprints` counts, not the recomputed current fingerprint. Rows from long sources produced before the guard were made with the unguarded prompt and remain identifiable by their fingerprint. Verified against the real artifacts on 31 October 2026 (read-only): the predecessor fingerprints are reproduced exactly (emotion `9a13d809…`, trec `4f48db72…`), all 341 emotion and 360 trec rows are accepted, and trec still verifies complete.
  - **TREC answer-type correction (`auggpt-trec-answer-type-v2`, [ADR 0013](../adr/0013-trec-auggpt-answer-type.md) and its Amendment 1; direct lead approval of the code changes only, 31 October 2026, quoted in the ADR; generation/promotion/merge/freeze not authorized or performed):** applies to `trec` seed 13 only (`AUGGPT_CORRECTIONS`). Every source gets `prompt_auggpt_trec_answer_type` via `auggpt_prompt_for`, with its canonical label and `config.yaml` definition and the answer-type, fact, participant, negation and intent constraints. It never gets `prompt_auggpt` or the length guard. Emotion's prompt, identity and artifact are byte-for-byte unchanged. Model, temperature, parser, filters, row schema and ID format are unchanged. v2 adds a `human` rule (a "who is <named referent>" question stays a who-question with the human label) and bounds every corrected call with `AUGGPT_CORRECTION_REQUEST` (`"think": false`, `options.num_predict` 1024) via `ollama_payload`; all other calls send the unchanged default body. The incomplete v1 candidate (`candidates/auggpt-trec-answer-type-v1/`, 354/360) is preserved evidence only and is not readable under v2.
    - Identity: the length-guard identity with `prompt_extensions` replaced by `trec_answer_type` (version, decision, label definitions, `max_words`, template/selector hashes, `request`, `request_builder_sha256`, `carry_forward: none`). Nothing is carried forward: the old TREC fingerprints (`4f48db72…` pre-guard and the guard identity) are rejected by resume and by plain `--verify`, which reports "superseded". `--verify --historical` reads pre-correction evidence under the superseded identities and reports `corrected_output: false`.
    - Candidate: `--setting auggpt --task trec --seed 13 --candidate` writes `synthetic/trec/candidates/auggpt-trec-answer-type-v2/auggpt_seed13.jsonl` plus its own sidecar and journal. It resumes the same way and never touches the active artifact. Real in-place TREC seed-13 generation is refused. The sidecar adds `prompt_correction` (version, decision, `generated_as`, superseded artifact name). `--candidate --verify` applies every AugGPT check under the corrected identity.
    - Promotion: `python -m ws1_generation.promote_auggpt --task trec --seed 13` with `--plan` (read-only; prints `confirm_with`), then `--promote --candidate-sha256 … --original-sha256 …`, then `--verify-promotion` (read-only). Promotion refuses pending journals, hash mismatches, incomplete or uncorrected candidates and source-coverage differences. It copies the original JSONL and sidecar byte-for-byte to `synthetic/trec/superseded/auggpt-trec-answer-type-v2/` (never overwriting different bytes) and writes `auggpt_seed13.promotion.json` (`in_progress` → `complete`, with original, archive, candidate and promoted hashes and fingerprints). It then atomically replaces the active JSONL and sidecar with the candidate's exact bytes. An interrupted promotion is completed by rerunning `--promote`. The candidate stays in place.
    - Active-path enforcement: where the correction applies, plain `verify_auggpt` (the active path) accepts corrected output only if `generate.check_promotion_record` passes. That check is shared with `promote_auggpt.verify_promotion`; it compares hashes only and never re-verifies artifacts, so the two cannot recurse. It fails closed on:
      - a missing record or one that is not valid JSON;
      - a missing section, a `schema_version` other than 1, or a hash that is not 64-hex;
      - a `status` other than `complete`;
      - a task, repetition or mode mismatch;
      - a `correction` or `decision` other than ADR 0013;
      - promoted hashes that differ from the candidate's;
      - any `original`/`candidate`/`promoted` `generation_fingerprint` that is not a lowercase 64-hex string (validated before any comparison, so lists, objects or null give a controlled `FAIL`, not a traceback); candidate and promoted fingerprints that differ from each other or from the corrected identity; or an original recorded under the corrected identity;
      - any missing or hash-mismatched file: active JSONL/sidecar, preserved original JSONL/sidecar, or retained candidate JSONL/sidecar.

      A candidate copied to the active path by hand therefore fails. `--candidate --verify` needs no record, and `--verify --historical` reads the preserved original. `promote --promote` marks the record complete only after the active bytes equal the candidate's and the candidate re-verifies.
    - Consumers: `verify_auggpt` now also returns `artifact_sha256`, `artifact_bytes`, `provenance_sha256`, `prompt_correction`, `historical_identity`, `corrected_output` (null where no correction applies) and `promotion_record` (the checked record's hash and status, and the original/candidate hashes; additive keys). C5b (`train.py`) and the #11 review read the active path unchanged. Row IDs keep their format (lead direction), so review evidence binds rows by artifact hash, fingerprint, row ID and text hash. The #11 round-2 prior-ID exclusion is unchanged. A targeted re-check of the round-1 IDs, at least the two errors, sits outside the random-10 denominator. The round-1 `trec/auggpt` evidence stays bound to the preserved original (`4711ebe6…`); #11 needs a fresh round after promotion.
    - Tests: `tests/test_auggpt_trec_correction.py` (fake/scripted model). Existing `tests/test_auggpt.py` runs with `AUGGPT_CORRECTIONS` disabled to keep covering the uncorrected length-guard and carry-forward machinery.
  - Approved generator binding: `ws1_generation.generate.APPROVED_GENERATORS` pins the ADR 0011 digest for the configured tag (`qwen3:4b` → `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`; an optional `sha256:` prefix is equivalent). Real AugGPT generation refuses any other resolved identity before calling the model, and verification rejects a sidecar whose provider, configured tag, resolved name or digest differs, even if rows and sidecar are self-consistent. Fake artifacts must carry exactly `FIXTURE_MODEL_IDENTITY`, live only under `synthetic-fixtures/`, and are reported as `real_run_evidence: false` / "FAKE fixture" by `--verify`. Changing the approved generator requires a recorded lead decision and an update to this pin.
  - Held-out overlap: loaded/resumed rows and verified rows are checked against normalized held-out test text, not only new candidates. A contaminated artifact fails with every offending row ID and its matching test ID (first 10 listed) and instructions to move the artifact and sidecar aside and regenerate that task × repetition before the freeze. Neither path edits the rows.
  - Handoff gate: `python -m ws1_generation.generate --setting auggpt --seed all --verify` (read-only) checks the approved model identity, recomputes the fingerprint from it plus current sources/prompts, revalidates every row, requires the sample to be exactly `lowres_per_label` per label (60 sources), six paraphrases per source (360 rows per task × repetition) and a matching complete sidecar, and refuses pending journals. It prints `PASS`/`FAIL` per task × repetition and exits non-zero on any failure; it never repairs data. Consumer enforcement in `ws2_training/train.py` (still legacy unseeded selection) is Issue #13's scope.
- **Migration & backward compatibility**:
  - `common.synth_path(task, setting, seed=None, mode=None)` defaults to real storage. `mode="fake"` selects fixture-only storage; the offline pipeline test opts into it with `SYNTHTEXT_GENERATION_MODE=fake`.
  - Unseeded file naming remains available for the legacy smoke seam, but legacy rows without the required identity fields and matching sidecar are intentionally not resumable. Regenerate them in the selected mode instead of relabeling or silently counting them.
- **Verification**:
  - Unit tests: `tests/test_generation.py` verifies repetition isolation, fake-to-real isolation, model-call interruption, crash recovery across initial and resumed JSONL/provenance writes, partial first-row and split-UTF-8 recovery, stale temporary checkpoint handling, fail-closed corrupt/legacy/misplaced rows, model/prompt dependency incompatibility, no-op history preservation, demonstration source tracking, AugGPT fixture isolation, non-example/malformed reply filtering, and mock connection recovery. `tests/test_auggpt.py` (fake model only) verifies partial-source resume to exactly six, interruption checkpoints, journal crash recovery, separate generation of sources shared across repetitions, rejection of empty/duplicate/test-leak/source-copy candidates, `--verify` rejection of missing, inexact, tampered or incomplete artifacts, read-only rejection of held-out overlap in verified and resumed rows, rejection of self-consistent unapproved or altered model identities, refusal to start real generation with an unapproved model, and a passing approved-identity artifact. It also verifies that the length guard is sent only for sources over `max_words` (short sources get the byte-identical base prompt), that rows from the pre-guard identity resume and verify unchanged alongside new current-identity rows with the carried identity recorded, and that rows from any other identity are still rejected without model calls. Carry-forward record checks reject seven tamperings of a mixed artifact read-only, reject a missing record on resume before model calls, reject a legacy sidecar that claims carried identities or holds current-identity rows, and confirm the summary reports recorded fingerprints for legacy and mixed artifacts.
  - Full pipeline test: `tests/test_pipeline.py` uses fake-only storage and offline embeddings. Passing it is smoke validation, not Ollama or experimental evidence.

## Real-data baseline run records (Checkpoint 1)

- **Approving issue & lead decision**: Issue #7 ([Training and project lead] Publish four seed-13 real-data baselines at 30 per label), approved by @kausthubabelagodbdes24-oss under ADR 0012.
- **Producer**: `ws2_training/train.py` (Owner: @kausthubabelagodbdes24-oss).
- **Consumer**: `ws3_evaluation/analyze.py` (Owner: @hem4456).
- **File path**: `results/scope30-seed13/runs.csv` (and compatibility mirror at `results/runs.csv`); historical 100/60 target runs archived at `results/archive/runs_target100.csv`.
- **Serialization**: Canonical UTF-8 CSV with explicit LF (`\n`, byte `0x0A`, no OS-specific CRLF translation) using `csv.DictWriter(..., lineterminator="\n")`.
- **Schema**:
  - `task`: task identifier string (`trec` or `emotion`).
  - `condition`: training condition string (`C1` for full real data, `C5a` for low-resource 10-per-label real data).
  - `seed`: integer sampling seed (`13`).
  - `n_train`: integer number of training examples used.
  - `macro_f1`: float macro-F1 score on the held-out real test set, rounded to 4 decimal places.
  - `accuracy`: float accuracy score on the held-out real test set, rounded to 4 decimal places.
- **Planned training counts (scope30-v1 under ADR 0012)**:
  - `trec C1`: exactly 180 rows across all 6 labels (30 per label). The former 76-abbreviation constraint under the 100 target is resolved by the approved 30/label target.
  - `trec C5a`: exactly 60 rows across all 6 labels (10 per label from `trec/lowres_seed13.jsonl`).
  - `emotion C1`: exactly 180 rows across all 6 labels (30 per label for sadness, joy, love, anger, fear, surprise).
  - `emotion C5a`: exactly 60 rows across all 6 labels (10 per label from `emotion/lowres_seed13.jsonl`).
  - Total: exactly 4 unique baseline result keys for seed 13.
- **Observed Checkpoint 1 baseline values (scope30-v1 recorded in results/scope30-seed13/runs.csv)**:
  - `trec,C1,13`: n_train=180, macro_f1=0.6009, accuracy=0.6520
  - `trec,C5a,13`: n_train=60, macro_f1=0.5974, accuracy=0.6400
  - `emotion,C1,13`: n_train=180, macro_f1=0.4082, accuracy=0.4420
  - `emotion,C5a,13`: n_train=60, macro_f1=0.3381, accuracy=0.3730
- **Historical Archived Runs (ADR 0011 100/60 target in results/archive/runs_target100.csv)**:
  - Preserved 9 historical rows (Emotion C1=600; Emotion C5a=60; TREC C5a=60 across seeds 13, 42, 7).
- **Provenance Sidecar**:
  - `results/scope30-seed13/baseline_provenance.json` (and `results/baseline_provenance.json`) records `scope30-v1` identity, embedding model (`sentence-transformers/all-MiniLM-L6-v2`), classifier hyperparameters (`LogisticRegression(C=1.0, max_iter=2000, random_state=13)`), input SHA-256 hashes, exact counts, and baseline run metrics.
- **Resume & Replacement Policy**:
  - Target-independent resume validation: in real mode, resume verifies existing rows against `approved_target_count(task, cond)` independently of sampler output. Rows with obsolete counts (e.g. 600 or 576) are discarded when validating requested conditions.
  - Targeted rerun isolation: replacement is strictly restricted to conditions specified in `--conditions`. Unrequested condition rows are preserved untouched without deletion; any count discrepancies for unrequested conditions are reported as informational notices without deleting rows.
  - Fail-closed data validation: missing labels or incomplete class coverage fail closed with `ValueError` rather than producing partial-class runs. Exact per-label sampling (`exact=True`) prevents silent undersampling.
- **Verification**:
  - Unit tests in `tests/test_train.py` verify incomplete-class rejection, test-set presence validation, metric bounds [0, 1], resume skipping, `--force` targeted condition preservation, unrequested condition preservation despite discrepancies, fail-closed exact sampling, target-independent resume validation, and canonical LF CSV serialization.
  - End-to-end pipeline test in `tests/test_pipeline.py` passes offline.

## Optional evaluation and annotation response tooling (historical)

- **Authority & Status**: Historical optional response tooling preserved under ADR 0012. Independent human annotation, external distribution, and human-agreement completion gates are removed under sole-operator execution.
- **Scope Distinction**: These optional 60-item held-out test annotation utilities (`make`, `combine`, `score`) are strictly distinct from the required 60 generated-text quality reviews conducted in Workstream 1/3 (#11/#12).
- **Producer**: `ws3_evaluation/annotate.py` (Owner: @hem4456; operator: @kausthubabelagodbdes24-oss).
- **Consumers**: Optional in-repository evaluation workflows.
- **File format and schema**:
  - `results/annotation_key.csv`:
    - Headers: `item_id`, `task`, `text`, `gold_label`, `allowed_labels`
    - Items: exactly 5 items per label per task (30 per task, 60 total items), deterministically selected from `data/real/<task>/test.jsonl` and deterministically shuffled per task using `seed=0` so row position does not leak gold labels.
    - Private audit key; not exposed to evaluators during blind scoring.
  - `results/annotation_<name>.csv` / evaluation templates:
    - Headers: `item_id`, `task`, `text`, `choose_one_of`, `label`
    - Identical item order and content as `annotation_key.csv`, but without `gold_label`.
    - Initial `label` column is blank.
  - `results/annotation_filled.csv`:
    - Headers: `item_id`, `task`, `text`, `choose_one_of`, `kausthuba`, `harshita`, `hementh`
    - Contains combined responses in canonical order matching `annotation_key.csv`.
  - `results/subjectivity.csv`:
    - Headers: `task`, `items`, `annotators`, `mean_agreement`, `alpha`
- **Validation rules**:
  - `make`: Protected against accidental overwrite; aborts with `FileExistsError` listing conflicting files unless `--force` (or `force=True`) is provided.
  - `combine`: Strict validation against `results/annotation_key.csv`:
    - Validates exact headers (`['item_id', 'task', 'text', 'choose_one_of', 'label']`).
    - Validates no duplicate item IDs within any submission.
    - Validates no missing item IDs and no extra item IDs.
    - Validates that `task`, `text`, and `choose_one_of` metadata match canonical key exactly.
    - Validates labels with `strip()` only (exact case match; no lowercase normalization). Blank labels accepted as incomplete.
- **Verification**:
  - Unit & behavioral tests in `tests/test_annotate.py` (18 tests) verify balanced counts, deterministic shuffle, non-grouped ordering, overwrite protection, strict schema validation, active annotator reporting, and majority agreement proportion calculation.
