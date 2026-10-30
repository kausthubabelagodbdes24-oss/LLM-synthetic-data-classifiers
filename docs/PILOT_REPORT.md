# Generation pilot report and target decision

- **Operator and lead:** Kausthuba (@kausthubabelagodbdes24-oss), now executing all workstreams personally in separate workspaces.
- **Status:** Initial real pilot and 40-item human inspection completed. Round-one human inspection also found clear quality failures. The final allowed refined pilot and its human inspection are complete, with remaining quality failures; no full-generation acceptance has been given. A lead decision on the proposed fallback is required. Issue #6 is not accepted; dependent experiments remain blocked.
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

The first refinement uses a separate ignored data root `.ollama/pilot-round1/data`. All original pilot rows, sidecars and the human submission remain preserved. No old rows are copied into the refined synthetic artifacts or counted toward their targets. Real input files were copied unchanged and their 12 checksums/counts reverified. Fresh seed-13 generation and a new 40-item human inspection are required. At most two prompt-refinement rounds are allowed; round one has completed generation and human inspection; round two is now initiated as the final permitted refinement.

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

## Round 1 human outcome and appendix selections

Kausthuba Belagod completed the refined 40-item inspection at `2026-10-30T17:43:11.194Z`. All submitted IDs, texts, tasks, settings and labels matched the preserved round-one packet. The submission is archived unchanged locally; SHA-256: `e0d1780066154c8603ae1fa539b69d3690ba337ca49700f9a329f1372b40b3d1`.

| Round-one reviewed group | Good | Ambiguous | Bad |
| --- | ---: | ---: | ---: |
| TREC zero-shot | 1 | 1 | 8 |
| TREC few-shot | 6 | 2 | 2 |
| Emotion zero-shot | 5 | 2 | 3 |
| Emotion few-shot | 3 | 5 | 2 |
| Total | 15 | 10 | 15 |

The prompt refinement did not establish adequate quality. These small screening counts are not population error rates, causal estimates or independent human agreement. Some category boundaries remain ambiguous; those judgments are preserved separately from bad items. Original review files and pilot artifacts are retained, not relabeled or deleted.

The operator explicitly selected these actual outputs for the appendix: two good and two bad per task, with provenance and judgments unchanged.

| Task | Setting | Item ID | Intended label | Human judgment | Actual generated text |
| --- | --- | --- | --- | --- | --- |
| trec | zero_shot | trec.zero_shot.seed13.location.0009 | location | bad | What is the definition of sustainable energy? |
| trec | few_shot | trec.few_shot.seed13.location.0007 | location | good | Where is the Grand Canyon located? |
| trec | few_shot | trec.few_shot.seed13.description.0002 | description | good | How does a solar panel convert sunlight to electricity ? |
| trec | few_shot | trec.few_shot.seed13.entity.0003 | entity | bad | What is the meaning of the term 'sustainable energy'? |
| emotion | zero_shot | emotion.zero_shot.seed13.fear.0001 | fear | good | i feel like i'm going to fail my exam |
| emotion | zero_shot | emotion.zero_shot.seed13.surprise.0002 | surprise | bad | i feel angry with the traffic on my way to work today |
| emotion | few_shot | emotion.few_shot.seed13.sadness.0007 | sadness | good | i feel hopeless |
| emotion | few_shot | emotion.few_shot.seed13.anger.0005 | anger | bad | i feel like i'm walking on eggshells around my family |
## Final prompt-refinement round 2

A local diagnostic requested ten location questions using the round-one all-label guide. The model returned mixed types, including acronym, definition, person and number questions. A short prompt focused solely on location produced place questions in that small probe. This supports investigating instruction interference; one stochastic diagnostic is not a guarantee of general improvement. Diagnostic outputs are saved separately under `.ollama/pilot-round1/prompt-diagnostic.json` and are not accepted pilot rows or classifier training data.

The second refinement, code commit `49f071f89afbd0916537babd6b65a2c9f8661680`, replaces the all-label guide with concise requirements for only the requested label. Zero-shot still contains no real demonstrations; few-shot retains exactly three same-label real demonstration IDs. Model, temperature, label scheme, seeds and training conditions remain unchanged. All 45 software tests passed before generation.

Round two uses a fresh ignored `.ollama/pilot-round2/data` root and reverified real inputs. Earlier pilot outputs and both human submissions are retained unchanged. New generation fingerprints prevent mixing outputs across template versions. This is the second and final refinement permitted by the current specification. If its human quality inspection still fails, do not perform an unapproved third round, silently change models, relabel bad examples or begin full generation; record the failure and obtain a lead decision.

## Round 2 measured execution

The second and final refined pilot generated 240 new real rows at code commit `49f071f89afbd0916537babd6b65a2c9f8661680`. All four artifacts have exactly ten rows per label and passed current fingerprint, source-ID, normalized exact-leak, completeness and provenance checks. Both earlier pilot artifacts and both human-review submissions were re-hashed and remain unchanged.

| Task | Setting | Rows | Session seconds | Calls | Run ID | JSONL SHA-256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| trec | zero_shot | 60 | 25.377108 | 7 | 933a9276-b6f8-4c20-b30f-45eadffb3a75 | e2a7773996788fd1337f9e377ceb38db71b847182b7642abcc75dfd6d08a2819 |
| trec | few_shot | 60 | 22.362892 | 6 | 936fcbfe-f19c-4c04-a82c-1306474bef48 | 934aae698e874c902f84994bd545a9bdf36b40062e7a67498fdf9d96e6f4a033 |
| emotion | zero_shot | 60 | 26.320795 | 7 | 8d8899c8-aa41-4be0-a9b0-09fcac2ae743 | 4892a3dd1456c6378e79fd0a82ff0b864893b31026676cb735746af5fc8c3de4 |
| emotion | few_shot | 60 | 25.011854 | 6 | 64dffefa-44a2-4960-be6b-3d645f353dd4 | c969467db88419539940a190375bbb779fca3feed144872ef8012227a394b4b7 |

Total session time: 99.072649 seconds; 26 calls. Detailed counters, fingerprints and model identity are preserved locally in `.ollama/pilot-round2/verification-summary.json`. Human quality judgments for round two remain blank/pending. Approximate remaining zero/few-shot runtime is 48 minutes at 100 per label or 28 minutes at 60, using a linear extrapolation and assuming this refined pilot is accepted and retained; AugGPT is unmeasured and additional. Target approval remains pending.

## Round 2 final human outcome — decision required

Kausthuba Belagod completed the final 40-item review at `2026-10-30T17:54:54.977Z`. All 40 submitted IDs, texts, tasks, settings and labels exactly matched the final-round packet. The submission is archived unchanged locally; SHA-256: `cb540e95a2078948e419a2baec7728db073a10f63dc7dfab8a01109eaebaed91`. All three rounds' artifacts and all three submitted human reviews were re-hashed and remain unchanged.

| Final reviewed group | Good | Ambiguous | Bad |
| --- | ---: | ---: | ---: |
| TREC zero-shot | 6 | 2 | 2 |
| TREC few-shot | 6 | 1 | 3 |
| Emotion zero-shot | 7 | 3 | 0 |
| Emotion few-shot | 7 | 1 | 2 |
| Total | 26 | 7 | 7 |

These are small human quality-screening counts, not population error estimates or independent annotator agreement. Ambiguous judgments remain separate. Clear answer-type and emotion-label errors remain, including an award-name question under human and an explicit fear statement under surprise. TREC few-shot also contains underspecified or malformed questions. Counts/provenance and successful software tests do not establish semantic acceptance. Keep issue #6 and PR #22 open and dependent full-generation/training work blocked.

The operator selected the following final-round examples, exactly two Good and two Bad per task. Round-one selections remain available as historical examples; these are the current-round selections and have not been silently substituted into training data.

| Task | Setting | Item ID | Intended label | Human judgment | Actual generated text |
| --- | --- | --- | --- | --- | --- |
| trec | zero_shot | trec.zero_shot.seed13.description.0005 | description | good | How does a solar panel work? |
| trec | zero_shot | trec.zero_shot.seed13.location.0003 | location | bad | Is the Grand Canyon located in the US? |
| trec | few_shot | trec.few_shot.seed13.numeric.0002 | numeric | bad | What is the area of a triangle? |
| trec | few_shot | trec.few_shot.seed13.entity.0003 | entity | good | What is the largest planet in our solar system? |
| emotion | zero_shot | emotion.zero_shot.seed13.fear.0001 | fear | good | i feel so scared on my first day of work |
| emotion | few_shot | emotion.few_shot.seed13.anger.0003 | anger | good | i feel so frustrated with my job i just want to quit but i dont know how to pay the bills |
| emotion | few_shot | emotion.few_shot.seed13.love.0009 | love | bad | i feel so happy to have found a community that shares my values |
| emotion | few_shot | emotion.few_shot.seed13.surprise.0005 | surprise | bad | i feel really scared of that haunted house |
### Proposed next step, not approved or executed

The specification's two prompt-refinement rounds are exhausted. Section 193 of `docs/PROJECT_SPEC.md` already proposes `qwen3:4b` as the failed-quality fallback, explicitly requiring approval of the shared-setting change and coordinated regeneration. No third prompt refinement or silent model change is authorized.

Recommendation: obtain the lead's approval to test that named fallback, keeping the final focused prompts, task definitions, temperature and repetitions unchanged. First download and identify `qwen3:4b`, verify real JSON completion compatibility and measure its actual runtime, then produce a fresh isolated 240-row pilot and obtain human inspection before approving full generation. If accepted, use the same accepted generator for both tasks, all three synthetic settings and all three repetitions; retain old Llama pilots as failed-pilot evidence, never as Qwen-generated rows. Any further prompt changes, runtime option changes or acceptance-policy changes require an explicit recorded decision.

Ollama's official `qwen3:4b` listing identifies a 4.02B Q4_K_M model with a 2.5 GB download: https://ollama.com/library/qwen3:4b. A larger parameter count is not a quality guarantee. Qwen throughput, thinking behavior, JSON compatibility and final runtime must be measured rather than inferred from the Llama pilot. The final uniform 100-or-60 target remains pending. No paid cloud inference is proposed.