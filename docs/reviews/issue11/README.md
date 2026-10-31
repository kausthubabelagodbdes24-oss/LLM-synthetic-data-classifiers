# Issue #11: model-assisted quality review of seed-13 generated text

Authority: [ADR 0012](../../adr/0012-eod-exploratory-30-seed13.md), [specification § Quality review](../../PROJECT_SPEC.md), [migration](../../SCOPE_MIGRATION.md). The historical ADR 0008/0009 two-clear-errors trigger is retained.

**Status, round 1 (31 October 2026):** 60 of 60 required reviews are recorded. Five batches pass the screen. **`trec/auggpt` is FLAGGED** with two clear errors. Acceptance of `trec/auggpt` is blocked until the producer corrects it (#9) and a fresh, recorded round-2 random audit of that batch is done. No data was relabelled, deleted, regenerated or frozen.

This is a **model-assisted** screen. It is not independent-human annotation, human consensus, Krippendorff's alpha or a population accuracy estimate. Ten-item counts are screening counts, not error rates.

## Reviewer

| Field | Value |
|---|---|
| Kind | model-assisted (no human judged these items) |
| Provider / model | Anthropic, Claude Opus 5.5 (`claude-opus-5-5`) |
| Interface | Claude Code CLI session operated by @kausthubabelagodbdes24-oss |
| Exact version | Unavailable. The session exposes no snapshot or version identifier beyond the model ID. |
| Method | Single pass over the local text packet: generated text, intended label, the row's own three few-shot demonstrations or its exact AugGPT source, and lexical aids. Judged against the unchanged `config.yaml` label definitions. No voting, no second reviewer. |

The same model also wrote the tooling and drew the sample. The judgments can share that model's biases, and nobody independently checked them.

## Inputs (verified read-only before sampling and again before outcomes)

`SYNTHTEXT_DATA` was set explicitly to the accepted local root `.ollama/pilot-qwen3-4b/data`. The root is not committed. Both `python -m ws1_generation.select_scope30 --seed 13 --verify` (4/4 PASS) and `python -m ws1_generation.generate --setting auggpt --seed 13 --verify` (2/2 PASS, 360/360) passed. `quality_review` re-runs both verifiers and also records file hashes. Its tests show that the hashes catch edits the producer verifiers tolerate, such as a still-valid paraphrase text or whitespace in a manifest.

| Batch | Input (relative to data root) | Rows | SHA-256 | Selection manifest / sidecar SHA-256 | Original artifact SHA-256 | Generation fingerprint(s) (row counts) |
|---|---|---|---|---|---|---|
| trec/zero_shot | `synthetic/trec/zero_shot_seed13.scope30-v1.selected.jsonl` | 180 | `de2ce33ae18b4dbc53f61e365758ef6a7e3e25d96cfcbcfc4884ac8c119fff79` | `279c8c6c6ed4274bc32c3000bb0be9752557ec170541dbed4a8bd9df7183a312` | `dc3aebb1b87899a7451454e7cab92febab7b030f053c79c3cfd90a3dea22e5c3` (244 rows) | `68a635f2a535ac66b7e4c78fce0e3e67b963b5a8e76ae5861b16504971ef0dda` |
| trec/few_shot | `synthetic/trec/few_shot_seed13.scope30-v1.selected.jsonl` | 180 | `e803ddfe533d3b8685afabf477d22dcff7e44a6cdd3eca5e0988699bf8d48767` | `99c9f2d83588d73996c61e9cb005966f2a685c8896b698b205faee91f7f79ae8` | `e803ddfe533d3b8685afabf477d22dcff7e44a6cdd3eca5e0988699bf8d48767` (180 rows) | `cce56ae121e55f4943553814460a1aaad00a0a372671d1618671bb7888d65e8e` |
| trec/auggpt | `synthetic/trec/auggpt_seed13.jsonl` | 360 | `4711ebe6e6918a99bb71a42d9987cfc37c9aa8596732746da95b35401336b76f` | `a8153679f03789ef0c5da74b14c795e9e91eb62a6752ee59c8dc31590f327f75` (sidecar) | — | `4f48db72c5345a00e9371c4734249a79afa49bb6339b3eef4b37df4d618687dd` ×360 |
| emotion/zero_shot | `synthetic/emotion/zero_shot_seed13.scope30-v1.selected.jsonl` | 180 | `a7458c204cc4317c0ea0d5b630174a008819107afa807546763871c8ae783285` | `839586defd790f288496beac67915479e8233cdab7f59152001496bfa2a6b90d` | `a7458c204cc4317c0ea0d5b630174a008819107afa807546763871c8ae783285` (180 rows) | `4d2ab0f440733aa9d98b1107a776f528534d68318770d4c13652758ec17388f7` |
| emotion/few_shot | `synthetic/emotion/few_shot_seed13.scope30-v1.selected.jsonl` | 180 | `78948f67bef46675ec01ea68ec02c52527875814b7aca3b94dc381b7cf3e08ad` | `c84fabc727213db5364b8f61605a8210eb8b659f7da6e723e4c0d8b81baa95ca` | `78948f67bef46675ec01ea68ec02c52527875814b7aca3b94dc381b7cf3e08ad` (180 rows) | `cec310b818092931f94e1fad8c19745e4ced1d59e21392cc6abc6a73901a73d4` |
| emotion/auggpt | `synthetic/emotion/auggpt_seed13.jsonl` | 360 | `59b2a6fb414c51c43ffb922b85735a81d45eafd513e95feec962eb5a8f46f67d` | `315ba9ef47be14d5d88afbc73207e67345beefdea8410e19b278d85db52f7899` (sidecar) | — | `629fb0fd6933303afa986594b883d5c39aa358d02d505461b58a871bd46f5982` ×19, `9a13d809fb04918be6754c6c80db6c3eaf12a5f15ccea8c6d226116cf5731b2f` ×341 |

The real `pool` and `lowres_seed13` reference hashes are recorded in `round1_sample.json` (`inputs.reference_inputs`) and match `CONTRACTS.md`. The emotion AugGPT fingerprints are `629fb0fd…` for the 19 length-guard rows and `9a13d809…` for the 341 carried-forward rows.

## Sampling (reproducible; no convenience selection)

Rule `issue11-quality-v1`: for each task × setting, rank row IDs by ascending SHA-256 of UTF-8 `issue11-quality-v1|round<R>|13|<task>|<setting>|<row-id>` (row ID breaks ties) and take the first 10. A later round excludes IDs sampled earlier in the same batch. The rule sees only IDs, never text, labels or quality. No row was filtered out or replaced. The draw is unstratified, so not every label appears in every batch: per-batch label counts are recorded, and TREC zero-shot drew 5 `human` rows. Selected IDs in order are in `round1_sample.json` → `batches.<key>.selected_ids_in_order`. Round-1 sample SHA-256: `cb0b933d0d8d108ba9d94101175beac13078bb7eace5fcee5ef0d11ab1f48359`.

## Rubric

Label fit (`fits`/`ambiguous`/`wrong`) is judged against the existing definitions in `config.yaml`, which keep annoyance under anger and worry under fear. The other judged fields are naturalness, task/dataset style fit, repeated frames, copying of the row's own demonstrations (few-shot only) and preservation of the exact source meaning and label (AugGPT only). `clear_error` = `wrong` label or `changed` source meaning. `ambiguous` = not a clear error, with an ambiguous label or partially changed meaning. Only clear errors count toward the trigger. The full rubric is in each judgments file.

## Round-1 outcomes

| Batch | Clear errors | Ambiguous | Other notes | Outcome |
|---|---|---|---|---|
| trec/zero_shot | 0 | 0 | Generic, easy trivia. One recurring "expansion of" frame (3/30 abbreviation rows). Two under-specified premises. | PASS |
| trec/few_shot | 0 | 0 | No copied demonstrations. `location.0021` is one of the 18 texts shared with selected zero-shot. | PASS |
| **trec/auggpt** | **2** | 1 | Clear errors: `trec.train.04257.seed13.aug5` and `trec.train.00845.seed13.aug3`. Ambiguous: `trec.train.04712.seed13.aug4`. | **FLAGGED** |
| emotion/zero_shot | 0 | 0 | 5/10 use the explicit "i feel <emotion word>" frame. All 30 love rows open with the same phrase, a strong lexical cue. | PASS |
| emotion/few_shot | 0 | 1 | `sadness.0027` could be sadness or fear. Outputs are cleaner and more literal than the noisy real demonstrations. No copies. | PASS |
| emotion/auggpt | 0 | 6 | Mostly label weakness inherited from noisy dair-ai/emotion source labels. 3 partial meaning changes (detail loss). 9/10 are capitalised and punctuated, unlike the lower-case tweet style. | PASS |

Individual judgments and explanations: `round1_judgments.json`. Computed outcome: `round1_outcome.json`.

### Flagged batch investigation: `trec/auggpt`

- **Errors.** In `04257.aug5`, an abbreviation source ("what is <acronym>") becomes a request for a *definition*, which the project definitions assign to description. In `00845.aug3`, a description source ("what is <person> known for") becomes "which *event*…", an entity answer type. In both, the general topic survives but the TREC label, which encodes the expected answer type, changes.
- **Likely cause.** The AugGPT prompt asks the model to keep the meaning, but it does not tell the model to keep the question's expected answer type. qwen3:4b freely rewords question frames. Bare "What is <ACRONYM>?" sources are especially exposed.
- **Read-only scope scan (heuristic, not judgments, not counted in the 60).** A regex scan of all 360 TREC paraphrases found 17 abbreviation-source rows reworded as meaning/define/definition questions, across 7 of the 10 abbreviation sources. It also found 1 description→event row and 1 entity→"who" row. Some of these, such as "what does X mean", are often treated as TREC abbreviation. The full candidate list and the decision on how to correct belong to the producer (#9).
- **Correction path (not performed here).** The producer must decide and record the correction in #9. Options needing a lead decision include regenerating the affected sources under a prompt that preserves answer type, with the new identity recorded, or another approved remedy. The original artifact stays preserved. After the corrected artifact passes `--verify`, run `python -m ws3_evaluation.quality_review sample --round 2` and judge the fresh round-2 `trec/auggpt` sample. The operator runs any generation.

## Supplementary targeted check (separate; not part of the random 60)

These are all 54 paraphrases of the nine Emotion sources longer than 30 words, which fall under the approved `auggpt-length-guard-v1`: 19 rows carry the guard fingerprint `629fb0fd…` and 35 the carried-forward `9a13d809…`. The review found 0 clear errors and 35 ambiguous or partially changed rows. Four rows overlap the random sample and carry identical judgments. Findings for #12:

- `emotion.train.07790` (all 6, guard rows): the self-doubt is kept, but a detail is inverted. The list of what *the writer* did becomes a list of what *the husband* did.
- `emotion.train.15223` (all 6): a third party's "he is strong" becomes the writer claiming strength.
- `emotion.train.13142.aug4` loses the "insulted" anger cue. `07494.aug2`/`aug3` weaken the bewildered feeling that carries the surprise label.
- Label fit for `07790`, `09194`, `14287` (surprise) and `07681` (love) is weak *in the source itself*. The paraphrases inherit that.
- Guard rows are often telegraphic (`14287`). No guard row changed a label clearly.

Files: `targeted_long_sources_{sample,judgments,outcome}.json`.

## Disclosed limitations

1. **Cross-condition overlap.** 18 normalized TREC texts occur in both selected zero-shot and few-shot corpora; Emotion has 0. This is allowed across separate conditions and was not removed. It limits TREC diversity and the C2-versus-C3 comparison. One sampled few-shot row is among them.
2. **Labelling cost.** The selected few-shot corpora use 59 distinct real demonstrations for TREC and 54 for Emotion. C3 therefore draws on real labelled examples beyond equal 180-row counts.
3. **AugGPT length guard.** Long Emotion sources were shortened within 30 words. The targeted check above documents detail loss and two systematic distortions, but no clear label change.
4. **Model-assisted review.** One model gave one judgment per item. That is no human agreement, and ten items per batch cannot estimate population accuracy. Borderline calls are reviewer judgments. In particular, the two TREC clear errors rest on the answer-type reading of the label definitions.
5. **Text-bearing material.** Corpus text is not committed. The local packet (`<data root>/reviews/issue11/round1_packet.{json,md}`, `targeted_long_sources_packet.json`) and the scripts that wrote the judgment files are kept under the ignored data root.

## Handoff to #12

- **Passing round-1 batches:** trec/zero_shot, trec/few_shot, emotion/zero_shot, emotion/few_shot, emotion/auggpt.
- **Blocked:** trec/auggpt (FLAGGED). Needs a producer correction in #9 and a fresh round-2 review that passes.
- **Unresolved flags to weigh before freeze:** the ambiguous IDs listed above; the emotion/auggpt style mismatch and inherited weak source labels; the targeted-check distortions in `07790` and `15223`; the explicit-emotion-word frames in emotion zero-shot; the 18-text TREC overlap.

## Round 2 and appended records (1 November 2026)

Everything above records round 1 and is unchanged. The round-1, targeted sample, judgment and outcome files are byte-identical. `check --round 1` now reports `inputs changed since sampling (batches ['trec/auggpt'])`. That is expected: the round stays bound to the superseded original `4711ebe6…`, which is preserved under `synthetic/trec/superseded/auggpt-trec-answer-type-v2/`.

The reviewer is the same as for round 1: model-assisted Claude Opus 5.5 (`claude-opus-5-5`), with no exact snapshot ID. It is the same model family that wrote the tooling and drew the samples, and there was one judgment per item and no human review. The judgments were written on 2026-11-01.

**Input (verified read-only).** The corrected and promoted `synthetic/trec/auggpt_seed13.jsonl` has SHA-256 `cf526bb948ef0e5227d54434ec897429bb0afe3f7a803579d34b374c5755ef07`. Its sidecar is `756b90b75c6b6deb3fc3183a3d8f69bc2e73631ac55f4b6f53192a34532bb0a4`, and all 360 rows carry fingerprint `c4d5308aa37a6ca8c8a820fc14e042376101bb2cddf352f25c5b933c2e3a4f56` (`auggpt-trec-answer-type-v2`, `corrected_output: true`). The promotion record is `89487383ddc3159d1b720140d1d7a1eb44160c8f81abccfb5446308115ec23ad`. `promote_auggpt --verify-promotion` and `generate --setting auggpt --seed 13 --verify` (2/2) pass. Emotion is unchanged at `59b2a6fb…`.

**Binding (sample schema 2).** Corrected rows reuse superseded row IDs (ADR 0013), so each round-2 and re-check item carries a `binding`: artifact SHA-256, generation fingerprint, row ID, text SHA-256 and exact source-text SHA-256. Every judgment must repeat that binding exactly. `check` and `check-recheck` reject any mismatch.

### Round 2: fresh random audit of `trec/auggpt` only (counts toward the trigger)

- The same `issue11-quality-v1` hash rule (`round2`) is restricted to the flagged batch (`sample --round 2 --batch trec/auggpt`).
- The 10 round-1 IDs are excluded, leaving 350 eligible rows. The 10 drawn IDs are in `round2_sample.json`.
- Result: **0 clear errors, 4 ambiguous (all `partially_changed`), outcome PASS.**
  - `02857.aug2` drops "special"/"name of".
  - `00607.aug1` adds "origins".
  - `04163.aug4` corrupts the product name to "Hot and Wheels". This is borderline. A stricter entity-preservation reading would count it as `changed`, giving 1 clear error, still below the trigger.
  - `02740.aug3` garbles the expansion request into "what abbreviation does A&W represent".
- Two rows are unnatural (`04163.aug4`, `02740.aug3`).
- Hashes:
  - sample `1e9fccf8af297db5b33f958d56992d663a24379025750739cf1ee9e688835ef7`
  - judgments `59a4a0d8488f11847299190f8966f07188dd94451fdc439b07dbab9adb8ec5f3`
  - outcome `d74c0a159bad1ad62555f5416dba693fc3172004e1f0970145b0afe318897b92`

### Targeted re-check of all 10 round-1 `trec/auggpt` IDs (outside the random denominator; no trigger)

- `recheck --round 2` derives the IDs from `round1_sample.json` and the FLAGGED batches in `round1_outcome.json`, not from a choice made now.
- Each item binds the corrected row and also records its round-1 binding: original artifact `4711ebe6…`, old fingerprint and old text hash.
- Results for the two known failures:
  - `trec.train.04257.seed13.aug5` is **resolved**. It now asks what SAP is expanded into, so it stays abbreviation.
  - `trec.train.00845.seed13.aug3` is **not resolved**. It now asks "what specific incident" D.B. Cooper's identity refers to. That expects an event (entity), while the source is description. It repeats the confirmed round-1 failure and is a clear error.
- The other 8 have no clear error. `04163.aug5` is partially changed: "history" narrows to how the cars were created.
- Hashes:
  - sample `2b807110f5b909a402039646114e89d6b2569cd0148115f3ea2185334607d3eb`
  - judgments `acb2ef588b67220fee813d3f35ff66b8cfec5fcb80e38c6e9310b660edffafb7`
  - outcome `aa7c20490673fb229af54509bb1c1277830fc764611b21bde07de50979e279c9`

### Emotion targeted re-adjudication (appended; supplementary 54, not the random 60)

- `targeted_long_sources_adjudication_2026-11-01.json` binds the preserved targeted sample (`8d61c9e8…`) and judgments by hash. It repeats each of the 12 original judgments unchanged under `original` and gives a complete rubric judgment under `adjudicated`. `check-adjudication` verifies all of this.
- **All 12 are re-adjudicated as `source_meaning: changed`, i.e. clear errors:**
  - `emotion.train.07790` ×6: the list of what the writer did becomes the husband's activities, so the agent is reversed.
  - `emotion.train.15223` ×6: the third party's "he is strong" becomes the writer's own strength, so the subject is changed.
- Retaining the Emotion label does not offset these propositional changes.
- These 12 rows are outside the random 60. They change neither the round-1 `emotion/auggpt` PASS nor any trigger count. Per the lead's direction, confirmed material changes must be investigated and remedied before the #12 freeze.
- Hashes:
  - adjudication `558d862b87159dc94bd06b8c59a12fbd4dafe82f6ef93e8da51bc029f89133aa`
  - outcome `8aa27fa5a5a3d957ab26176ef004c242b31fbb28484f86c2a1d0ec954dc8f4b9`

### Handoff to #12 (updated)

- **Random screen:**
  - `trec/auggpt` round 2 is PASS (0/10).
  - The other five batches keep their round-1 PASS.
  - Model-assisted screening only; no acceptance or freeze is implied.
- **Unresolved, for the lead to weigh before any freeze:**
  1. `trec.train.00845.seed13.aug3` still changes the answer type (description → event/entity) after the ADR 0013 v2 correction. The other five paraphrases of `trec.train.00845` and other description sources with "what is X known for" frames were not reviewed here. A read-only scope check is recommended before deciding a remedy.
  2. `emotion.train.07790` and `emotion.train.15223`: 12 confirmed clear errors in the supplementary check. They need an investigation and remedy decision for `emotion/auggpt` (e.g. regenerate or exclude these sources' paraphrases) before freeze. No change was made here.
  3. Round-2 borderline rows: `04163.aug4` ("Hot and Wheels" entity corruption) and `02740.aug3` (garbled expansion request).
  4. All round-1 unresolved flags listed above.

## Reproduce

```cmd
set SYNTHTEXT_DATA=C:\Users\Kausthuba Belagod\Documents\Synthtext\.ollama\pilot-qwen3-4b\data
python -m ws3_evaluation.quality_review check --round 1      & rem re-verifies inputs; exit 2 = a batch is FLAGGED
python -m ws3_evaluation.quality_review check-targeted
python -m ws3_evaluation.quality_review check --round 2      & rem round 2 (trec/auggpt only); exit 0 = PASS
python -m ws3_evaluation.quality_review check-recheck --round 2
python -m ws3_evaluation.quality_review check-adjudication --adjudication docs\reviews\issue11\targeted_long_sources_adjudication_2026-11-01.json
```

From 1 November, `check --round 1` exits 1 ("inputs changed", `trec/auggpt`) because round 1 is bound to the superseded original. The operator's local `run-issue11-round2-verify.bat` (kept beside the data root, not committed) runs every read-only check above.

`sample` refuses to overwrite a recorded round with different content. `check` refuses to overwrite a recorded outcome, and blocks the review if any input hash or verifier result has changed.
