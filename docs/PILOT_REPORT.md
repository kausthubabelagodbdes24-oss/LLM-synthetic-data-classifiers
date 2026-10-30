# Generation pilot report and target decision

- **Operator and lead:** Kausthuba (@kausthubabelagodbdes24-oss), now executing all workstreams personally in separate workspaces.
- **Status:** Initial real pilot and 40-item human inspection completed. The first refined real pilot is also structurally verified; its human quality inspection is pending. Issue #6 is not accepted; dependent experiments remain blocked.
- **Last updated:** 30 October 2026.

## Verified real execution

The original pilot used generation code `ec632407d918afb3cc2ddab3ff63a592f295c85f`, Ollama 0.40.13, `llama3.2:3b`, temperature 0.9, and an NVIDIA RTX 4060 Laptop GPU (8 GiB). Actual model digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.

The post-reboot real smoke check passed in 4.613852 seconds. All 12 real-data input files matched the accepted count/SHA-256 manifest before generation. Commands were `python -m ws1_generation.generate --setting zero_shot --pilot --seed 13 --task all` and the equivalent `few_shot` command, without dry-run. Data root was the ignored local `.ollama/pilot/data` folder, selected through `SYNTHTEXT_DATA`.

| Task | Setting | Accepted rows | Rows per label | Session seconds | Model calls | Distinct real demonstrations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TREC | Zero-shot | 60 | 10 | 28.839273 | 8 | 0 |
| TREC | Few-shot | 60 | 10 | 37.034964 | 8 | 23 |
| Emotion | Zero-shot | 60 | 10 | 26.198785 | 7 | 0 |
| Emotion | Few-shot | 60 | 10 | 30.672938 | 7 | 21 |

Total: 240 accepted real outputs, 122.745960 session seconds, 30 calls. No recorded call or parse failures. The generator rejected four duplicate candidates and two normalized exact test-text matches; accepted outputs passed current exact-leak, source-ID, repetition, identity and completeness checks. These are structural checks, not proof of semantic quality. Unused surplus candidates are not rejection events.

Artifact SHA-256 values and run IDs are recorded in PR #22's verified local pilot evidence and in local `.ollama/pilot/verification-summary.json`. Raw input/output texts, sidecars and model binaries remain local and are not committed.

## Original human review, preserved unchanged

Kausthuba completed the 40-item inspection at `2026-10-30T17:23:51.885Z`. The submitted JSON matches all 40 selected row IDs, texts, tasks, labels and settings. Submission SHA-256: `73089a1a9b4540aa39017bf2ed8d5d3f76821d162f3cbada5f48dff5f8e8ea00`. An exact copy is retained locally in `.ollama/pilot/human-review/`.

Selection was ten items per task/setting: one deterministic random item from each label, then four randomly selected remaining items, using `issue6-pilot-review-13:<task>:<setting>` as the sampling seed. The initial packet covers every label.

| Group | Good | Ambiguous | Bad |
| --- | ---: | ---: | ---: |
| TREC zero-shot | 8 | 0 | 2 |
| TREC few-shot | 9 | 1 | 0 |
| Emotion zero-shot | 6 | 2 | 2 |
| Emotion few-shot | 2 | 2 | 6 |
| Total | 25 | 5 | 10 |

These are the operator's original screening judgments, not adjudicated population error estimates or independent annotator agreement. No original answers were changed.

The review contains clear label-fit concerns, including a painting-name question generated under human, and tiredness/routine statements generated under surprise. Some Emotion notes also use finer distinctions than the existing six-label rubric. Kausthuba explicitly confirmed keeping the existing definitions, including annoyance/irritation under anger and anxiety/worry/nervousness under fear. That clarification does not retroactively change submitted judgments. Ambiguous cases remain separate from clear errors.

No appendix selection boxes were checked. Two actual good and two actual bad examples per task still need final selection from preserved inspected outputs; do not invent bad examples or claim the user selected them.

## Approved prompt-refinement round 1

Kausthuba approved retaining the current label definitions for a clearer review guide and the next prompt-refinement round. The first refinement is implemented in `b082ecd3881bc302a09c9359f8665420d6621c49`:

- Include all configured label meanings and classify TREC by the requested answer type, avoiding yes/no questions and misleading mentions of people or places.
- Clarify broad Emotion categories and require the requested emotion to be the clear dominant feeling; discourage tiredness/routine under surprise and happiness alone under love.
- Encourage varied situations rather than repeated emotion-label words.
- Keep three same-label real demonstrations and their IDs, while making the label guide authoritative when noisy real examples are shown.

No model, temperature, real dataset labels, sampling seeds or experimental conditions were changed. The changed template source hashes bind a new generation identity. All 45 software tests passed before the fresh run; this does not establish improved real output quality.

The first refinement uses a separate ignored data root `.ollama/pilot-round1/data`. All original pilot rows, sidecars and the human submission remain preserved. No old rows are copied into the refined synthetic artifacts or counted toward their targets. Real input files were copied unchanged and their 12 checksums/counts reverified. Fresh seed-13 generation and a new 40-item human inspection are required. At most two prompt-refinement rounds are allowed; one has now completed generation, with human inspection pending.

## Timing and target decision

Linear extrapolation from the initial per-artifact timing estimates about 59 minutes of remaining zero/few-shot generation at 100 per label, or 35 minutes at 60, after accounting for three repetitions and the completed initial pilot. These initial estimates are superseded if those pilot rows are retired after refinement. AugGPT timing is unmeasured and additional (2,160 paraphrases at either target). Later rejections, warm-up, thermal throttling and contention can increase runtime. Do not report extrapolations as measured full-run durations.

The default remains 100 per label. The only approved fallback is uniform 60 across tasks/repetitions if timing requires it before the agreed freeze. Final target choice remains pending review of the refined pilot and explicit lead decision; no automatic 30-second threshold applies. The freeze is 1 November 2026 at 18:00 IST under the current specification.

## Acceptance and handoff

Keep PR #22 and issue #6 open until the refined real pilot is structurally verified and inspected, actual good/bad examples are retained, and the uniform target decision is recorded. Keep #7, #9 and #10 blocked until acceptance. Do not start full generation, freeze data, train dependent conditions or report fake scores as scientific results based on software tests alone.

## Round 1 measured execution

The first refined pilot generated and verified 240 new real rows at code commit `b082ecd3881bc302a09c9359f8665420d6621c49`. Every artifact has exactly ten rows per label, valid current fingerprints and matching source IDs. Original pilot bytes and the submitted human review were re-hashed after refinement and remain unchanged.

| Task | Setting | Rows | Session seconds | Calls | Run ID | JSONL SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| trec | zero_shot | 60 | 48.724219 | 13 | e45332d2-6a15-4c9b-b355-9a06750e9976 | f03f79558d6866112472a504718a74655ff0984f6138bb413e838828d5c8fa0d |
| trec | few_shot | 60 | 30.853399 | 8 | d8109509-16b6-4ac6-8356-7c347cd78408 | 1711035a6d4d9ead3a2002dcb552decc2660a2cb282f903f909eaa8f6f3185f5 |
| emotion | zero_shot | 60 | 23.593819 | 6 | 4f312e96-cf1f-4ae7-9acc-69fb054fe121 | 5d7227855e42db214b079cc16e1e1c846ea80db177f9655c6d4eb7a76c9e1d5c |
| emotion | few_shot | 60 | 26.96042 | 6 | c79a40c7-472f-456c-b2e9-ba68b7d95dec | b88e8aba4cfb4912688188ef2c31cf632771d8a0f5915d0c28030d30f2ee233d |

Total session time: 130.131857 seconds; 33 calls. No improvement in semantic quality is claimed before the new human inspection. The separate review page includes the approved broad-label guide, blank judgments and a different browser-autosave identity bound to the new generation fingerprints and texts.

The refined pilot's linear timing estimate is about 63 minutes of remaining zero/few-shot work at 100 per label, or 37 minutes at 60, assuming the refined pilot is accepted and retained. These replace the initial timing estimates for the refined protocol. AugGPT runtime remains unmeasured and additional. Final target approval remains pending.
