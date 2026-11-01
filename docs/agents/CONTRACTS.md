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
| Run records and paired analysis | Four seed-13 baseline result keys (C1=180, C5a=60 for trec and emotion) in results/scope30-seed13 (#7, preserved). Twelve-key freeze-bound grid: `ws2_training/train_scope30.py` (#13; see the #13 section below). The real grid ran once at `1354118` and was accepted by the lead on #13 (1 November 2026); `grid_complete.json` sha256 `3b12bbb8…`. Single-seed paired analysis: `ws3_evaluation/analyze_scope30.py` (#14; see the #14 section below). | Training (@kausthubabelagodbdes24-oss) + evaluation |
| Generation report sections and appendix | Generation deliverables for Issue #15 (Sections 2, 3a, 6, Appendix A) delivered to `docs/report/` for consumption by #17. Presentation slides separate under ADR 0014. | Sole operator and author: @kausthubabelagodbdes24-oss (historical: WS1 @harshitadjainbdes24-byte) |
| Reproducibility walkthrough | `notebooks/SynthText_scope30_walkthrough.ipynb` separates a fake-fixture pipeline demonstration from read-only accepted real analysis. It never reruns the accepted experiment. Final presentation slides are outside repository and issue scope under ADR0014. | Sole operator and lead: @kausthubabelagodbdes24-oss |
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

## Model-assisted generated-text quality review (Issue #11, ADR 0012)

- **Consumer of**: the verified `<setting>_seed13.scope30-v1.selected.jsonl` zero/few-shot corpora (not the larger originals) and the complete `auggpt_seed13.jsonl` artifacts. Producers' read-only verifiers (`select_scope30.verify_selection`, `generate.verify_auggpt`) must pass, and every input/manifest/sidecar SHA-256 is recorded with its generation fingerprint(s). Missing, changed or invalid inputs fail closed (`ws3_evaluation/quality_review.py`).
- **Sampling rule `issue11-quality-v1`**: per task × setting, ascending SHA-256 hex of UTF-8 `issue11-quality-v1|round<R>|13|<task>|<setting>|<row-id>` (row ID as tie-breaker), first 10; IDs sampled in earlier rounds of the same batch are excluded. IDs only; no filtering or replacement. Six batches, 60 reviews per round.
- **Evidence** (`docs/reviews/issue11/`, no corpus text): `round<R>_sample.json` (IDs in order, input hashes), `round<R>_judgments.json` (reviewer identity, rubric, one judgment and explanation per item, bound to the sample SHA-256), `round<R>_outcome.json` (per-batch counts/IDs). The text-bearing packet stays under the git-ignored data root (`<SYNTHTEXT_DATA>/reviews/issue11/`). Earlier rounds and outcomes are never overwritten.
- **Trigger**: `clear_error` = wrong intended label or (AugGPT) changed source meaning/label; at least 2 of 10 → `FLAGGED` (`check` exits 2). Ambiguity, copies, unnatural wording, style mismatch and repeated frames are recorded separately. A flagged batch needs a producer correction recorded in GitHub and a fresh round (`sample --round R+1`).
- **Supplementary targeted check**: `targeted` / `check-targeted` review every Emotion AugGPT paraphrase of a source over `max_words` (length guard), kept separate and marked `counts_toward_required_random_60: false`.
- **Round-1 status (31 October 2026)**: 60/60 recorded by Claude Opus 5.5 (model-assisted; exact snapshot unavailable). trec/zero_shot, trec/few_shot, emotion/zero_shot, emotion/few_shot and emotion/auggpt PASS; trec/auggpt FLAGGED (2 answer-type changes), blocking its acceptance pending #9 correction and round 2. See `docs/reviews/issue11/README.md`. Not human annotation, agreement, alpha or population accuracy; no freeze implied.
- **Later rounds (schema 2, 1 November 2026)**:
  - `sample --round R --batch <task/setting>` restricts a fresh audit to the batches that need it. The same hash rule applies, earlier-round IDs of those batches are excluded, and `batches_in_round` is recorded. Round 1 keeps schema 1, so its evidence re-derives unchanged.
  - Schema-2 input records add the AugGPT sidecar hash, `prompt_correction`, `corrected_output` and the checked ADR 0013 `promotion_record`.
  - Every item carries a `binding`: input artifact SHA-256, generation fingerprint, row ID, text SHA-256 and, for AugGPT, exact source-text SHA-256. Every judgment must repeat that binding exactly, because corrected rows reuse superseded row IDs.
- **Re-check** (`recheck` / `check-recheck --round R`):
  - Covers every ID sampled in round R-1's FLAGGED batches, taken from the preserved sample and outcome, as each row now reads in the current input.
  - Each item records both its current binding and its round R-1 binding, and the re-check refuses a row whose text is unchanged.
  - Marked `counts_toward_random_denominator: false` and `trigger_applies: false`. Files: `round<R>_recheck_{sample,judgments,outcome}.json`.
- **Adjudication** (`check-adjudication --adjudication <file>`):
  - A dated, append-only re-judgment of rows from the supplementary targeted check. It must bind the preserved targeted sample and judgments by hash, cover every row of each named source in order, and repeat each recorded judgment unchanged under `original` beside a full rubric judgment under `adjudicated`.
  - The original targeted files are never rewritten, and the adjudication never counts toward the random 60.
- **Round-2 status (1 November 2026)**: see `docs/reviews/issue11/README.md`.
  - trec/auggpt round 2 (promoted v2 artifact `cf526bb9…`, fingerprint `c4d5308a…`): 1/10 clear errors (`04163.aug4`, entity "Hot Wheels" changed; reclassified at lead review 5724372746), PASS below the trigger.
  - Re-check of the 10 round-1 IDs: `04257.aug5` resolved; `00845.aug3` still a clear answer-type change, outside the denominator and unresolved.
  - Emotion re-adjudication: all 12 paraphrases of `07790`/`15223` are `changed`, i.e. clear errors in the supplementary check. They are unresolved for #12.
  - Model-assisted only; no acceptance or freeze is implied.
- **Tests**: `tests/test_quality_review.py` (fake fixtures only) covers the hash rule, IDs-only evidence, demonstration/source packets, fresh later rounds, no overwrite, changed/missing input blocking (including edits producer verifiers tolerate), the two-error trigger, judgment validation and CLI exit codes, and separation of the targeted check. It also covers a promoted ADR 0013 fixture, flagged-batch round 2 with prior-ID exclusion and preserved round-1 files, exact-binding rejection per field, the re-check (outside the denominator, superseded text rejected), and adjudication binding and tamper rejection.

## Frozen seed-13 synthetic handoff (Issue #12, ADR 0012, lead Option A)

- **Approving decision**: lead Option A on issue #12 ([comment 6547818106](https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/12#issuecomment-6547818106), 31 Oct 2026). The current verified corpora are accepted without regeneration, relabelling or row deletion, and known problems are disclosed. The freeze takes effect when the lead accepts the PR that adds the manifest.
- **Producer**: `ws1_generation/freeze_scope30.py`.
  - `--audit`: read-only.
  - `--write`: writes `docs/freeze/scope30-v1-seed13.freeze.json`. It requires a clean tracked tree, records `git rev-parse HEAD`, and never overwrites different content.
  - `--verify`: read-only and fail-closed.
- **Consumers**:
  - `ws2_training/train_scope30.py` (#13): trains all twelve keys only on the frozen files after `verify_freeze` passes; the legacy `ws2_training/train.py` refuses real C2/C3/C4/C5b. See the #13 section below.
  - Evaluation and reporting (#14–#17).
- **Manifest** (deterministic, no timestamps, no corpus text):
  - For each of the six batches (`trec`/`emotion` × `zero_shot`/`few_shot`/`auggpt`), the consumed file's path, SHA-256, bytes, rows, IDs in file order and per-label counts.
  - For zero/few-shot, the selection manifest and original artifact.
  - The provenance sidecar's hash, `updated_at`, session count and last status.
  - The generation fingerprint and per-row fingerprints.
  - The model identity, plus prompt/options/prompt-extension/sources identity hashes, options and temperature.
  - For TREC AugGPT, the ADR 0013 correction and the promotion record, preserved original and retained candidate (all with hashes).
  - Real `pool`/`lowres_seed13`/`test` hashes.
  - Every `docs/reviews/issue11/` file (LF-normalized SHA-256), with each batch's latest round, outcome and clear-error IDs, the re-check, the Emotion adjudication and the curated scope-scan IDs.
  - The decision, the limitations and `repository_revision`.
- **Fail-closed rules**:
  - Every producer verifier must pass (`verify_selection` ×4, `verify_auggpt` ×2, `verify_promotion` for TREC).
  - Independent re-read checks: exact counts, 6-label coverage, unique IDs and normalized texts, selected IDs equal to the selection manifest, zero held-out test overlap, zero-shot without sources, few-shot with 3 same-label pool demonstrations, AugGPT sources equal to the exact 60 `lowres_seed13` rows with 6 distinct paraphrases each and inherited labels, and one model identity.
  - For each batch, the latest #11 round must be PASS and must have reviewed the exact frozen file hash. Each outcome must be bound to its sample and judgments, and the scope scan must name the frozen AugGPT hashes.
  - For a real freeze, all 16 #11 evidence files are required. The manifest may not contain any `text` key or any normalized real or generated corpus text.
  - `--verify` rebuilds the manifest and requires byte equality (CRLF tolerated), and requires the recorded revision to be an ancestor of HEAD. So a missing, replaced, reordered, reformatted, tampered or incompatible artifact, sidecar, reference or evidence file fails, as does an edited manifest.
- **Limitations (Option A)**: see [docs/freeze/README.md](../freeze/README.md). Retained TREC v2 and Emotion errors are disclosed, not corrected. All findings are model-assisted or heuristic and are not population error rates.
- **Tests**: `tests/test_freeze_scope30.py` (fake fixtures only). It covers deterministic reruns, identical-rewrite no-op, no overwrite, all-six binding without text, and 16 tamper cases. It also covers post-review reorder, a flagged latest round, unbound outcomes, missing required evidence, corpus-text rejection, revision checks, CRLF tolerance and CLI exit codes.

## Freeze-bound twelve-key training grid (Issue #13, ADR 0012)

- **Status**: implemented in PR #33. Review 5724732904 approved the C4 rule and the MiniLM revision. Merging does not run the experiment, and no real twelve-key result exists yet.
- **Producer**: `ws2_training/train_scope30.py`. Commands:
  - `--check-inputs`: read-only. Verifies the freeze and builds all twelve training sets, without fitting or writing anything.
  - `--results-dir DIR [--keys K] [--replace K]`: trains the grid.
  - `--validate`: read-only.
  - `--dry-run`: fake fixtures only.
- **Freeze enforcement**:
  - `freeze_scope30.verify_freeze` must pass before any set is built, any classifier is fitted or anything is written.
  - Real runs accept only `docs/freeze/scope30-v1-seed13.freeze.json` with canonical (LF) SHA-256 `dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1`, `generation_mode: real` and the committed #11 evidence.
  - Synthetic inputs are read only from the paths the freeze names. Those must be the `.scope30-v1.selected.jsonl` corpora and the `auggpt_seed13.jsonl` artifacts, with matching SHA-256 and IDs in file order. Original or legacy default paths are refused.
  - Real `train`/`test`/`pool`/`lowres_seed13` must match the canonical #5 hashes, and `test`/`pool`/`lowres` must also match the freeze.
  - Real runs refuse fixture environment variables, a `sample` data root, a dirty tracked checkout and any config drift from 30/10/6/[13]/MiniLM.
- **Grid**: `trec`, `emotion` × `C1 C2 C3 C4 C5a C5b` × seed 13. That is exactly twelve keys; any other seed or key is rejected.
  - C1 (180): `per_label_sample(train, 30, 13, exact)`, the same sampler as the #7 baselines.
  - C2 (180): the frozen selected zero-shot corpus.
  - C3 (180): the frozen selected few-shot corpus.
  - C4 (180): `per_label_sample(train, 15, 13, exact)` plus 15 per label from the C3 corpus. The few-shot half takes the first 15 per label in ascending `sha256("scope30-v1|13|<task>|few_shot|<row-id>")` (row ID tie-break), i.e. the first half of the same ADR 0012 rank order that selected the 30. No test data or performance is used. The lead approved this rule in PR #33 review 5724732904.
  - C5a (60): the exact frozen `lowres_seed13` rows, 10 per label, all from the pool.
  - C5b (420): those rows plus the frozen AugGPT artifact. Every source has exactly six paraphrases with distinct normalized texts, none equal to its source, all with the source's label.
- **Rejected inputs**: wrong counts, missing labels, duplicate IDs or normalized texts, real rows from the pool, reordered corpora, extra rows, and any test-ID overlap. Any normalized test-text overlap is also rejected, with one exception (next item).
- **Lead decision, issue #13, 1 November 2026 ("Pin + disclose")**:
  - The canonical seed-13 TREC sample contains one official-dataset train/test duplicate: `trec.train.05262` = `trec.test.00378` (abbreviation; normalized-text SHA-256 `ba6af17c366034f8dca2abff920ba01cc773fc77eda6188a1127b8190b324d2e`).
  - It is allowed only in TREC C1/C4 real rows, matched by both IDs and the text hash. It is recorded in `training.disclosed_test_text_overlap` of each affected result.
  - The #7 TREC C1 baseline was trained with it. Sampling is unchanged.
  - Every other overlap fails closed.
- **Classifier**: `sentence-transformers/all-MiniLM-L6-v2`, pinned to the lead-approved revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (PR #33 review 5724732904).
  - The approved snapshot is read directly, never through `refs/main`. Its identity (the revision plus the SHA-256 of every snapshot file) is recorded in every binding.
  - The loader receives that recorded identity. Immediately before and immediately after loading, it recomputes the canonical approved identity from the pinned snapshot: kind, name, revision, every file hash, `identity_sha256`, `real_model`, `normalize_embeddings`, `batch_size`, `device` and `fine_tuned`.
  - The recorded identity must equal that canonical identity exactly: the same key set and the same values with the same JSON types (canonical-JSON byte equality, so `True` is not `1` and `64` is not `64.0`).
  - It loads only `snapshots/<revision>` and encodes with exactly the verified device, batch size and normalization.
  - Any altered, missing or extra field, changed file, other revision or forged hash fails closed before training.
  - Record reuse compares every binding section the same way, and `metrics`/`fit` must have their exact key sets. Embeddings are normalized, batch size 64, never fine-tuned. The classifier is `LogisticRegression(C=1.0, max_iter=2000, random_state=13)`, fitted after `random.Random(13).shuffle`, as in #7.
- **Results directory**:
  - A fresh, scope-specific directory (default `results/scope30-v1-seed13-freeze-v1`). `results/`, `results/scope30-seed13` and `results/archive` are refused, as is any non-empty directory without its `grid.json` identity. Real and fake results never share a directory.
  - `grid.json`: the scope, seed, mode, freeze ID and hash, and the twelve keys.
  - `records/<task>.<cond>.seed13.json`: the result binding plus metrics and runtime, self-hashed in `record_sha256`.
  - The binding covers the freeze ID, hash, path and audit revision; the task, condition and seed; n_train, per-label counts and composition; the ordered training IDs and their SHA-256; the fit-order SHA-256; and the disclosed overlap. It also covers the consumed input roles, paths, hashes and rows (with fingerprints); the test path, hash, ID hash and counts; the embedder identity and classifier settings; the repository revision and clean flag; and Python, OS and package versions.
  - Metrics are unrounded floats in [0, 1]. The record also holds `n_iter`/`converged`.
  - `runs.csv`: the six legacy columns, derived from the valid records in canonical key order, with LF line endings.
  - `grid_complete.json`: written only when exactly twelve valid keys exist, with the record hashes, the `runs.csv` hash and the runtimes.
- **Resume and replace**:
  - A key is reused only if its whole binding equals the binding recomputed now.
  - A stale, partial, tampered, misplaced or wrong-flag record stops the run before any fitting, unless that key is named in `--replace`. A replaced record is moved to `superseded/` and never deleted.
  - Unrequested keys are never touched.
  - An unexpected file (for example a seed-42 record) stops the run.
  - Completion is reported only after `--validate`-equivalent checks pass for exactly twelve keys.
- **Legacy consumer**: `ws2_training/train.py` keeps C1/C5a (#7) and the fake pipeline, but refuses real C2/C3/C4/C5b before touching any results file.
- **Tests**: `tests/test_train_scope30.py` uses an exact-count fake freeze built with the real #10/#9/#11/#12 tooling and the hashing embedder. Its scores are never evidence. It covers:
  - exact counts, labels and composition; C1 matching the baseline sampler; deterministic C4 selection; and the exact C5a/C5b relationship;
  - incomplete, contaminated and wrong C5b/C1–C5a sets, and the pinned-duplicate scope;
  - missing or tampered freeze and inputs (no fitting, no directory created), real-mode accepted-freeze and fake-freeze refusal, and fixture-setting and config-drift refusal;
  - wrong seed, legacy/unsafe paths, and the legacy trainer's refusal;
  - fake end-to-end with full provenance, and refusal of fake results as real;
  - exact-match resume, partial records, stale provenance (six fields), tampered or misplaced records, targeted replacement, unrequested stale rows left untouched, missing keys, duplicate `runs.csv` rows, wrong-seed files, incomplete grids, and protected or unrelated directories;
  - key parsing, the approved model-snapshot pin, and the loader using only the recorded snapshot even if `refs/main` moves; fail-closed loading when weights or files change after recording or during loading, or when the identity is missing, unapproved or forged (19 metadata forgeries: altered, type-swapped, missing or extra fields); the loader encoding with exactly the verified settings; records with type or key-set drift not being reused; `run_grid` passing the recorded identity to the loader; and the CLI.

## Single-seed gap analysis of the accepted grid (Issue #14, ADR 0012)

- **Approving issue and decision**: issue #14. The lead accepted the real #13 grid on 1 November 2026 and directed analysis of those exact frozen records without retraining ([evidence](https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/13#issuecomment-6553500190)).
- **Producer**: `ws3_evaluation/analyze_scope30.py`. It only reads the #13 grid and the accepted freeze manifest.
- **Input**: the grid directory written by `ws2_training/train_scope30.py`, containing `grid.json`, `grid_complete.json`, `runs.csv` and the twelve `records/`.
  - Records bind the producing commit (`1354118`) and its environment, so `train_scope30 --validate` is run from that pinned checkout.
  - This module validates the frozen records on their own:
    - each record's canonical bytes and `record_sha256`, and its file hash against `grid_complete.json`;
    - the exact key and counts; the exact per-condition training composition (C1 180 real; C2 180 zero-shot; C3 180 few-shot; C4 90 real + 90 few-shot; C5a 60 sources; C5b 60 sources + 360 AugGPT), compared structurally so that wrong, extra, missing, negative, fractional, boolean or non-numeric counts are rejected cleanly; metrics in [0, 1]; and the generation-mode flag;
    - the freeze, classifier and repository bindings against `grid_complete.json`;
    - `runs.csv` re-derived byte for byte;
    - the `grid.json` identity; and exactly twelve keys, with no foreign files.
  - In real mode, `grid_complete.json` must have SHA-256 `3b12bbb878d8191e387e3abc0f77fc02d9f081b98f0216f9b64dbb144bf40032`, which transitively pins every record and `runs.csv`. It also requires revision `1354118b4b05bec31973c268057d8a713907f3a6` (clean), the accepted freeze and the approved MiniLM. Fake grids are never accepted as real.
- **Calculations**: values are kept at full precision until display.
  - C2/C3 relative gap = (synthetic macro-F1 − same-task seed-13 C1 macro-F1) / C1 macro-F1, with the absolute gap reported as well.
  - If C1 is 0, the relative gap is `null`, labelled "undefined", and the task is retained with its absolute difference.
  - A missing C1 blocks the analysis.
  - The four planned comparisons are within-task differences in macro-F1 and accuracy: the cross-task C2/C3 gap (Emotion − TREC), C3 − C2, C4 − C1, and C5b − C5a (the training-size difference is stated).
  - No SD, confidence interval, significance or stability figure is computed.
- **Costs**: for each condition, real training rows and synthetic training rows come from the record composition. The distinct real generation demonstrations (`selected_distinct_demonstrations`) come from the grid's own freeze manifest, whose LF SHA-256 must equal the grid's freeze hash. For C4 the demonstration count is an upper bound.
- **Outputs**: written once to a fresh directory (default `docs/analysis/scope30-v1-seed13/`):
  - `summary.md`;
  - `f1_by_condition.png`, six-condition grouped bars with no error bars;
  - `gap.png`, the relative C2/C3 gap, where an undefined case is a labelled slot and never a zero bar;
  - `analysis.json`, canonical, containing the grid/freeze/classifier identities and per-record hashes, the analysis checkout revision and clean flag (real mode requires clean), the full-precision results, gaps, comparisons, costs, limitations and the SHA-256 of the other three files.
  - Fake outputs are tagged `FAKE FIXTURE`. Outputs are not committed until the lead has reviewed them.
- **Committed real outputs**: `docs/analysis/scope30-v1-seed13/` holds the four files written once by
  `run-issue14-analysis.bat WRITE` at the PR #34 merge `c7761ec944d86e866ce1103ed2b90ecf57b3ac8c` (clean). They are
  committed byte for byte:
  - `analysis.json` `d0486efd…`;
  - `summary.md` `a95abe66…`;
  - `f1_by_condition.png` `fc6e2f6a…`;
  - `gap.png` `5ed188eb…`.

  `docs/analysis/.gitattributes` (`* -text`) prevents line-ending conversion so the recorded hashes hold on every
  platform. `tests/test_analysis_outputs_scope30.py` pins the four SHA-256s, checks the binding to the accepted grid,
  freeze, model and clean merge checkout, and regenerates `summary.md` and every gap, comparison and cost from
  `analysis.json`. The written reading is `docs/analysis/scope30-v1-seed13-interpretation.md`. It was
  model-assisted and reviewed by the lead in the PR.
- **Legacy consumer**: `ws3_evaluation/analyze.py` (mean ± SD over seeds) stays for the historical fake pipeline only.
- **Tests**: `tests/test_analyze_scope30.py` covers:
  - known-value gaps and comparisons, C1 = 0 (undefined but retained, labelled in the summary and chart), and a missing C1 blocking the analysis;
  - cost separation;
  - acceptance of the fake grid built end to end by the real #13 fixture, refusal of fake grids as real, and the real pin (a resealed metric change is rejected), revision, freeze and model;
  - tampered, resealed-inconsistent and invalid records (nine cases), 25 wrong or malformed compositions plus a non-object training section, foreign record files, `runs.csv` mismatch, `grid.json` identity and the freeze used for costs;
  - no error bars on single-seed charts, charted values equal to the computed values, write-once outputs whose hashes match `analysis.json`, the clean-checkout requirement, and the CLI guards.

## Colab reproducibility walkthrough and presentation boundary

- **Authority**: [ADR0014](../adr/0014-presentation-separation-colab.md), direct lead decision on 1 November 2026.
- **Artifact**: `notebooks/SynthText_scope30_walkthrough.ipynb`, notebook format 4 and Google Colab-compatible.
- **Fake demonstration lane**:
  - uses only committed `data/sample/real/` fixtures;
  - explicitly sets `SYNTHTEXT_GENERATION_MODE=fake`, `SYNTHTEXT_EMBEDDER=hashing`, fixture data/results paths and seed-13 configuration;
  - may run the three fake generation settings, legacy fixture trainer and fixture analysis;
  - labels every resulting row, score and chart as fake demonstration output, never experimental evidence.
- **Accepted-real lane**:
  - reads only committed `docs/analysis/scope30-v1-seed13/analysis.json`, `summary.md` and the two reviewed PNGs;
  - displays the twelve real result rows, C2/C3 gaps, comparison tables, costs, provenance and limitations without training or rewriting them.
- **Hard boundary**: the notebook does not run Ollama, download/substitute another generator, access uncommitted real data, generate/promote/freeze real corpora, train the accepted twelve-key grid or overwrite the accepted analysis directory.
- **Presentation separation**: final professor-facing slides, speaker notes and submission logistics are not repository artifacts or GitHub issue acceptance criteria. Reports, tables, charts and the notebook may be used as source material outside GitHub.
- **Verification**: `tests/test_colab_walkthrough.py` parses the notebook, checks the two evidence lanes, required pins/disclosures, safe fake environment and absence of commands that run real generation/training or write accepted analysis.

## Generation report sections and Appendix A (Issue #15, scope30-v1)

- **Authority & Status**: [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0013](../adr/0013-trec-auggpt-answer-type.md), [ADR 0014](../adr/0014-presentation-separation-colab.md), and [PROJECT_SPEC.md](../PROJECT_SPEC.md). Delivered in PR for Issue #15.
- **Producer**: Sole operator and author: @kausthubabelagodbdes24-oss (historical: Harshita @harshitadjainbdes24-byte, WS1).
- **Consumer**: Issue #16 (data/classifier methods), Issue #17 ([Evaluation] Assemble the revised exploratory report and reviewed PDF).
- **Deliverable Paths**:
  - `docs/report/section_2_related_work.md`: Related work on synthetic data utility across task subjectivity (Li et al., 2023), text augmentation via AugGPT (Dai et al., 2023), few-shot in-context learning (Brown et al., 2020), and Sentence-BERT dense representations (Reimers and Gurevych, 2019) from verified primary sources.
  - `docs/report/section_3a_generation_methods.md`: Accepted Qwen model identity (`qwen3:4b`, digest `359d7dd4…`), temperature T=0.9, thinking behavior, prompt strategies (zero-shot, few-shot with 3 pool demonstrations, AugGPT with length guard and ADR 0013 v2 answer-type constraints), filtering rules, write-ahead journals, seed-13 selection (720 zero/few-shot across four artifacts + 720 AugGPT paraphrases across two artifacts = 1,440 synthetic examples), demonstration costs (59 TREC / 54 Emotion), quality screening (multi-stage: initial gate of 10 rows in each of six batches, round 2, targeted rechecks, supplementary Emotion check, and heuristic scope scans; disclosed as model-assisted and heuristic, not independent human review or population error rate), and freeze under lead Option A.
  - `docs/report/section_6_limitations_and_conclusions.md`: Grounded strictly in observed numerical directions from the four accepted single-seed comparisons in `docs/analysis/scope30-v1-seed13/` (recording observed contrary directions between TREC and Emotion without inferential claims like 'outperformed', 'gain', or 'loss'; stating reliability is unknown with one seed and no equivalence threshold was defined; single-seed, small data, task confounding, demonstration cost, and model-assisted screening limitations).
  - `docs/report/appendix_a_prompts_and_examples.md`: Verbatim generation prompt templates from `ws1_generation/generate.py`, two good and two problematic examples per task with verified row ID and artifact SHA-256 provenance from accepted final artifacts, and the eight historical pilot examples from ADR 0011 / `PILOT_REPORT.md` explicitly distinguished.
  - `docs/report/README.md`: Directory structure and workstream index reconciling #15, #16, and #17 deliverables.
- **Report Scaffold Integrity**: The shared report assembly file (`docs/REPORT.md`) is preserved untouched for #17's final assembly to maintain modular file separation between workstreams.
- **Presentation Separation**: Presentation slides and submission logistics remain outside GitHub scope per ADR 0014.
- **Verification**: `tests/test_generation_report_sections.py` validates file presence, primary citations, model settings, exact hashes, prompt template parity with `ws1_generation/generate.py`, example provenance, and scientific boundary disclosures.

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
