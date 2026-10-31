# Issue #12: frozen seed-13 synthetic handoff (`scope30-v1-seed13-freeze-v1`)

Authorities: [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0013](../adr/0013-trec-auggpt-answer-type.md), and the lead's [Option A decision](https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/12#issuecomment-6547818106) (31 October 2026). The review evidence is in [docs/reviews/issue11](../reviews/issue11/README.md).

The manifest, `scope30-v1-seed13.freeze.json`, is produced and checked by `python -m ws1_generation.freeze_scope30` (`--audit` / `--write` / `--verify`). It holds paths, hashes, counts, IDs and identities, and no corpus text. The freeze takes effect when the lead accepts the PR that adds it. Any later corpus change needs explicit lead approval, plus retraining and reanalysis of the affected results.

## What is frozen

| Batch | Consumed file (relative to the data root) | Rows | Latest review |
|---|---|---|---|
| trec/zero_shot | `synthetic/trec/zero_shot_seed13.scope30-v1.selected.jsonl` | 180 | round 1 PASS (0/10) |
| trec/few_shot | `synthetic/trec/few_shot_seed13.scope30-v1.selected.jsonl` | 180 | round 1 PASS (0/10) |
| trec/auggpt | `synthetic/trec/auggpt_seed13.jsonl` (ADR 0013 v2, promoted) | 360 | round 2 PASS (1/10) |
| emotion/zero_shot | `synthetic/emotion/zero_shot_seed13.scope30-v1.selected.jsonl` | 180 | round 1 PASS (0/10) |
| emotion/few_shot | `synthetic/emotion/few_shot_seed13.scope30-v1.selected.jsonl` | 180 | round 1 PASS (0/10) |
| emotion/auggpt | `synthetic/emotion/auggpt_seed13.jsonl` | 360 | round 1 PASS (0/10) |

The totals are 720 selected zero/few-shot texts and 720 AugGPT paraphrases. For each batch the manifest binds:
- the consumed file's SHA-256, byte and row counts, every ID in file order, and per-label counts;
- the selection manifest and original artifact (zero/few-shot);
- the provenance sidecar hash, last update, session count and status;
- the generation fingerprint and per-row fingerprints;
- the model identity (`qwen3:4b` @ `359d7dd4…`);
- the prompt, options and sources identity hashes and the temperature;
- for TREC AugGPT, the promotion record, preserved original (`4711ebe6…`) and retained candidate.

It also binds the real `pool`, `lowres_seed13` and `test` hashes, every file in `docs/reviews/issue11/` (LF-normalized SHA-256), the decision, the limitations, and the repository revision the audit ran at.

## Audit performed (read-only)

The producer verifiers ran first: `select_scope30.verify_selection` for 4 artifacts, `generate.verify_auggpt` for 2, and `promote_auggpt.verify_promotion` for TREC. The freeze tool then independently re-read every consumed file and checked:
- exact counts and coverage: 30 per label × 6 labels, and AugGPT sources equal the exact 60 `lowres_seed13` rows (10 per label);
- uniqueness: unique IDs and normalized texts, and IDs equal to the selection manifest;
- held-out exclusion: zero normalized test-text overlap;
- source identity: zero-shot rows have no sources; each few-shot row has 3 same-label pool demonstrations; each AugGPT paraphrase has its source's label, with exactly 6 distinct paraphrases per source;
- model identity: one accepted identity across all six.

For every batch, the latest #11 review round must be PASS and must have reviewed this exact file hash. A flagged batch, or a review of a different file, blocks the freeze.

## Limitations (Option A: accepted and disclosed, not corrected)

- **TREC AugGPT (corrected, ADR 0013 v2):**
  - **Random review.** The formal round-2 screen passed with **1/10 clear errors and 3 ambiguous**, below the two-clear-errors trigger. The clear error is `trec.train.04163.seed13.aug4`, where "Hot Wheels" becomes "Hot and Wheels".
  - **Targeted re-check.** Outside the random denominator, `trec.train.00845.seed13.aug3` still changes the expected answer type (description → a named event).
  - **Heuristic scan.** A read-only scope scan found **about 13 likely problematic rows plus 6 borderline** of 360:
    - answer-type changes (`00845.aug1`–`aug3`);
    - corrupted entity or text names, seen only in the v2 output (9 rows);
    - one reversed relation (`04708.aug4`);
    - abbreviation questions drifting toward definitions.
  - **Label impact.** Around **3–4 rows may change the intended label or answer type**.
- **Emotion AugGPT.** All **12 paraphrases** of `emotion.train.07790` (agent reversed) and `emotion.train.15223` (third-party strength made self-attributed) **change the source meaning while keeping their labels**.
- **Status of these rows.** They remain in the frozen corpora. They are not relabelled, deleted, corrected or hidden. They are accepted only for the exploratory scope30-v1, seed-13 study, and are not described as harmless or scientifically negligible.
- **Nature of the findings.** All findings are model-assisted (Claude Opus 5.5, one judgment per item, same model family as the tooling) or heuristic. They are not independent-human judgments, human agreement or population error estimates.
- **What the freeze does not establish.** It does not establish corpus correctness, semantic quality, robustness, statistical stability or generalization.
- **Other #11 flags.** The other flags remain as recorded: ambiguous rows, Emotion style mismatch and inherited weak source labels, 18 TREC texts shared by the selected zero- and few-shot corpora, and the few-shot labelling cost.

The full ID lists are in `docs/reviews/issue11/scope_scan_2026-11-01_ids.json`, the `round2_*` files, `round2_recheck_*` and `targeted_long_sources_adjudication_2026-11-01*`.
