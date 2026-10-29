# SynthText project specification

Status: version 2, approved by Kausthuba for publication on 29 October 2026 following the grill-with-docs review.
Prepared: 29 October 2026. Submission deadline: 4 November 2026.

This specification consolidates the supplied master plan and all three instruction manuals dated 29 October 2026, reconciled with the orchestrator handoff, configuration, scripts, tests and subsequent corrections from Kausthuba. Requirements describe intended behavior; they do not certify that the implementation or experiment has passed them. Original documents are preserved with this specification for traceability. Later direct decisions from Kausthuba take precedence over older source text.

## Problem Statement

The team needs to investigate whether classifiers trained on LLM-generated text lose more performance relative to real training data on a subjective task than on an objective task. It must deliver a reproducible student-scale experiment, evidence of task subjectivity, a report and presentation material by the submission deadline. Three contributors need explicit ownership and verified handoffs so generation, training and evaluation do not block each other unnecessarily.

## Solution

Compare six training conditions on TREC question classification and Emotion classification, using three fixed sampling seeds per condition and task. Generate synthetic examples locally with Ollama and a small language model. Classify real held-out test examples using frozen sentence embeddings and logistic regression. Report macro-F1, variation across seeds and relative synthetic-versus-real performance gaps. Independently collect three human annotations on 30 test items per task and compute nominal Krippendorff's alpha.

The hypothesis is a larger negative synthetic-data performance gap for Emotion, with few-shot generation closing some of the gap. Treat that as a question to test, not an outcome to enforce. Annotation agreement supplies evidence for the subjectivity comparison; it does not guarantee the presumed ordering. Equal label counts do not control all other differences between the tasks: domain, style, size and class distributions can also affect performance. Kausthuba explicitly chose to retain the two-task comparison with careful comparative claims, without claiming that subjectivity caused an observed difference.

## User Stories

1. As project lead, I want one agreed experimental configuration, so that all contributors run the same experiment.
2. As a contributor, I want a reproducible setup and an offline smoke test, so that I can verify my installation before using real data.
3. As a contributor, I want deterministic data preparation, so that our local datasets correspond.
4. As a data owner, I want real data excluded from Git, so that repository contents follow the project policy.
5. As the generation owner, I want a ten-per-label pilot, so that I can inspect quality before committing to a full run.
6. As the generation owner, I want zero-shot prompts with label descriptions, so that examples can be generated without demonstrations.
7. As the generation owner, I want three same-label real demonstrations in few-shot prompts, so that generation follows the dataset style.
8. As the generation owner, I want paraphrases linked to their original low-resource examples, so that training uses the correct augmentations.
9. As the generation owner, I want generation to resume, so that interruption does not require starting over.
10. As the generation owner, I want counts and schema checks, so that incomplete data is caught before handoff.
11. As the training owner, I want real-data baselines before synthetic data arrives, so that independent work can proceed.
12. As the training owner, I want six explicitly defined conditions, so that comparisons have a stable interpretation.
13. As the training owner, I want three recorded repetitions with separately generated data for every generation setting, so that variation between generation runs is included.
14. As the training owner, I want the same classifier and held-out test set across conditions, so that comparisons isolate training-data differences.
15. As the training owner, I want training to resume and selected conditions to rerun, so that recoveries do not duplicate result rows.
16. As the evaluation owner, I want a balanced 60-item annotation sheet without gold answers, so that annotators can label independently.
17. As an annotator, I want clear allowed labels and a separate response file, so that I can submit judgments without seeing anyone else's answers.
18. As the evaluation owner, I want agreement scores and annotator coverage reported, so that missing responses are visible.
19. As a reader, I want mean macro-F1 and standard deviation, so that I can compare performance and variation.
20. As a reader, I want clearly defined relative gaps, so that the main hypothesis can be assessed.
21. As the evaluation owner, I want reproducible summary tables and charts, so that the report matches recorded runs.
22. As project lead, I want objective checkpoint evidence, so that a completion message alone cannot advance the project.
23. As a contributor, I want explicit ownership and review rules, so that shared changes are coordinated.
24. As project lead, I want synthetic data frozen after acceptance, so that later results retain a stable provenance.
25. As a contributor, I want agreed fallbacks for slow generation and failed downloads, so that the main dependency chain can recover.
26. As the report assembler, I want assigned sections and a completeness gate, so that the final draft covers every required part.
27. As project lead, I want final submission and release tagging handled separately, so that a release requires my explicit approval.
28. As the generation owner, I want to retain both good and bad pilot examples, so that the appendix represents observed quality honestly.
29. As a reader, I want model identity, accepted counts and deviations recorded, so that I can understand what was actually run.
30. As a presenter, I want assigned slides and a shared deck order, so that the presentation covers the whole study.
31. As project lead, I want dated checkpoints and clear handover signals, so that dependencies can be managed before the deadline.
32. As a reader, I want all four planned comparisons discussed, so that the conclusions address both synthetic-only training and augmentation.

## Implementation Decisions

### Experimental design

- Objective task: TREC, using SetFit/TREC-QC, with abbreviation, entity, description, human, location and numeric labels.
- Subjective task: dair-ai/emotion, split configuration, with sadness, joy, love, anger, fear and surprise labels.
- Preparation shuffles training data with a fixed seed of zero, reserves ten percent as a demonstration and low-resource pool, and retains the dataset's held-out test split.
- Experimental repetitions are identified by seeds 13, 42 and 7. Each repetition has separately generated zero-shot, few-shot and AugGPT data for both tasks. Low-resource samples contain ten examples per label per repetition; insufficient availability blocks the affected planned condition.
- Default generation model is llama3.2:3b through local Ollama; temperature is 0.9, requests ask for ten examples, and zero-shot and few-shot targets are 100 accepted examples per label each.
- Few-shot generation uses three real examples of the requested label from the reserved pool.
- AugGPT requires six valid, distinct, meaning-preserving paraphrases per low-resource source example within each repetition. Even when a source occurs in multiple repetitions, generate that repetition's paraphrases separately rather than sharing generated outputs across repetitions. C5b uses its repetition's exact C5a sources and their corresponding paraphrases.
- TREC prompts target short natural English questions, at most 20 words. Emotion prompts target short first-person lower-case English tweets, at most 30 words, without hashtags or emojis. Prompt constraints require pilot inspection; requesting them does not establish compliance.
- Classifier: frozen all-MiniLM-L6-v2 sentence embeddings with L2 normalization, followed by logistic regression with C=1.0 and a 2,000-iteration limit.
- Generation requires no paid API. First-time dataset, generation-model and embedding-model downloads need connectivity; an offline generation model does not make initial setup offline.
- The original master plan's text counts, call counts and runtime estimates predate independent generation repetitions and must not be used as the revised budget. At 100 per label, zero-shot and few-shot total 7,200 accepted texts, with a further 2,160 AugGPT paraphrases at six per source: 9,360 total. The 60-per-label fallback reduces the first total to 4,320 and the overall total to 6,480. Rejected outputs and retries add generation work; use pilot timing and record observed counts and runtime.

| Condition | Training data per task and seed |
| --- | --- |
| C1 | Real only, 100 examples per label |
| C2 | Zero-shot synthetic only, 100 examples per label |
| C3 | Few-shot synthetic only, 100 examples per label |
| C4 | 50 real and 50 few-shot synthetic examples per label |
| C5a | Real only, ten examples per label from that seed's pool sample |
| C5b | C5a plus AugGPT paraphrases of those exact source examples |

The complete result grid remains two tasks × six conditions × three repetitions = 36 unique results. C1–C4 normally contain 600 training examples; C5a contains 60 and C5b contains 420. If pilot timing requires the approved fallback, choose and document it before full generation: C1–C4 then contain 360 training examples, C4 uses 30 real plus 30 synthetic per label, and C5a/C5b remain at 60/420. Apply the chosen size uniformly across both tasks and all repetitions. Incomplete data blocks acceptance and affected training; any further reduction requires an explicit decision.

C2–C5b use synthetic data belonging to their own repetition. All conditions and repetitions share the fixed held-out real test set. Separately generated datasets include generation variation but do not alone establish statistical independence.

The equal-size comparison applies to classifier inputs for C1–C4. Report distinct real examples used as generation demonstrations separately; do not describe the experiment as controlling human-labeling cost.

### Contracts and evaluation

- Real examples have an identifier, text and configured label. Synthetic examples additionally record the generation setting and a list of source identifiers.
- Zero-shot source lists are empty; few-shot source lists identify the three demonstrations; AugGPT source lists identify one original example.
- Generated artifacts must unambiguously identify their repetition so that resume, source provenance, training selection and audits cannot mix repetitions.
- Generation must be checked for valid labels, identifiers, nonempty text, completeness, source provenance and leakage. Existing zero-shot/few-shot generation rejects normalized exact matches to test text; that does not establish semantic independence or cover every generation mode.
- The pilot covers both zero-shot and few-shot settings, ten examples per label per task. Inspect about 40 texts across tasks and settings for label fit, dataset style, variation and copied demonstrations. Allow at most two prompt-refinement rounds before the full run. Retain two good and two bad examples per task for the appendix. Accepted pilot rows count toward the full generation target unless a coordinated pre-freeze reset is required.
- Record the actual model tag and model ID, temperature, prompt templates, accepted counts and generation problems. Do not claim a measured parse-failure rate unless attempts and failures were recorded.
- Run records identify task, condition, seed, training count, macro-F1 and accuracy. Validate uniqueness and full grid coverage, not only total row count.
- Summaries report macro-F1 mean and standard deviation over repetitions. For each task and C2/C3, first compute (synthetic macro-F1 − C1 macro-F1) / C1 macro-F1 within each matching repetition, then report the mean and standard deviation of the three relative gaps. Do not calculate the headline gap only after averaging F1 scores. A zero denominator or missing baseline remains an analysis exception requiring an explicit handling rule before final analysis.
- Annotation selects five test examples per label, giving 30 items per task and the same 60 items for all three annotators. Each person completes a separate response file without seeing other responses or gold answers. Hementh completes his own judgments before combining submissions into the scorer's expected combined format. Individual collection replaces the manuals' shared answer-sheet editing process.
- Agreement uses nominal Krippendorff's alpha. Missing responses require coverage checks. A two-annotator fallback is permitted and must be disclosed. The accepted count is 30 sufficiently annotated items per task; do not interpret unavailable or undefined agreement as a measured result.
- Interpret four comparisons explicitly: synthetic-versus-real gaps across tasks; few-shot versus zero-shot; C4 versus C1; and C5b versus C5a. Include both task agreement scores and mean agreement alongside the performance table and two charts.
- Before Checkpoint 2, audit ten randomly selected generated examples per task × setting × repetition: 180 in total. Record label fit, naturalness, errors and ambiguous cases; compare AugGPT output against its source meaning. A batch is the set for one task, setting and repetition. At least two clear wrong-label or changed-meaning examples among its ten audited rows trigger investigation, correction of the underlying problem and a fresh random audit before acceptance. Record ambiguity separately and treat this as a quality screen rather than a statistical accuracy guarantee.

### Ownership and coordination

| Person | Workstream | Deliverables and report sections |
| --- | --- | --- |
| Harshita (@harshitadjainbdes24-byte) | Generation | Synthetic data; related work, model and generation, limitations and conclusion, Appendix A |
| Kausthuba (@kausthubabelagodbdes24) | Data, training and project lead | Real data preparation, run records; tasks and data, classifier and conditions; combined slides and submission |
| Hementh (@hem4456) | Evaluation | Annotation artifacts, agreement, summaries and charts; introduction and results; report assembly and final edit |

The repository is kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers. This replaces the manuals' placeholder repository name and owner. Kausthuba supplied kausthubabelagodbdes24 as his personal username; the authenticated repository-owner account is kausthubabelagodbdes24-oss. Verify which account should be used for automated review ownership before enforcing it.

The dependency chain is synthetic data → remaining training → tables and charts → assembled report → submission. Contributors waiting on a dependency draft their report sections.

Review changes through pull requests, verify CI and ownership, reject secrets and real-data additions, and merge with merge commits. Shared configuration or helper changes require Kausthuba's approval and a group heads-up. Force-pushes, branch deletion, post-freeze regeneration and release tags require explicit approval. Messages to teammates require direct authorization.

The three workstream branches are ws1-generation, ws2-training and ws3-evaluation, in addition to main. Each contributor brings merged main changes into their workstream at the start of a session. Harshita owns generation and synthetic artifacts; Kausthuba owns preparation, training and run records; Hementh owns evaluation, annotation and analysis artifacts. Preserve the manuals' exclusive artifact ownership. Code-review and coordination documentation are shared project-lead responsibilities.

### Delivery schedule and handoffs

All checkpoint times below use Asia/Calcutta (IST). The master plan supplies dates absent from the stage-only handoff. Kausthuba leads a 15-minute checkpoint call with each person's done, next and blocked status, after verifying merged deliverables.

| Date | Work or checkpoint |
| --- | --- |
| Thu 29 October | Repository, invitations, local setup and generation model download |
| Thu 29–Fri 30 October | Real data, baseline runs, annotation sheet and shared report document |
| Fri 30–Sat 31 October | Pilot followed by zero-shot and few-shot generation |
| Fri 30 October, 9 pm | Checkpoint 1: baselines, inspected pilot, common annotation items distributed through separate response files, and shared report document |
| Sat 31 October–Sun 1 November | Independent annotation, agreement scoring, AugGPT and synthetic-data handoff |
| Sun 1 November, 6 pm | Checkpoint 2: accepted synthetic data frozen and agreement known |
| Sun 1–Mon 2 November | Remaining training, report sections, summaries and charts |
| Mon 2 November, 9 pm | Checkpoint 3: final results and complete assembled report draft |
| Tue 3 November | Slides, full-team read-through and final edit |
| Wed 4 November | Kausthuba submits report and slides; no submission time was provided |

At synthetic-data acceptance, Kausthuba starts the remaining 24 runs immediately. Once all results are merged, Hementh produces analysis and supplies gap values and C4-versus-C1 findings to Harshita for the conclusion. Team members post handover or blocker updates themselves; this specification does not authorize the assistant to send group messages.

The source manuals budget approximately six hands-on hours per person, but independent generation repetitions and the 180-example audit add work. Reassess that estimate using the pilot; the original generation duration is not a promise for the revised scope. Actual availability and runtime determine escalation.

### Report and presentation

The working report target is 6–8 pages, pending professor requirements; whether Appendix A counts toward that total is unspecified. Hementh creates the shared report document with all headings, assembles sections, standardizes references, performs the final edit after everyone reads it, and exports the final PDF for Kausthuba.

| Report part | Owner | Required content / planning length |
| --- | --- | --- |
| 1. Introduction | Hementh | Data-labeling motivation, research question and hypothesis, study overview; about half a page |
| 2. Related work | Harshita | One concise paragraph each on Li et al., AugGPT, GPT-3 few-shot prompting and Sentence-BERT; about half a page |
| 3a. Model and generation | Harshita | Actual model ID, temperature, three settings, accepted counts and filtering; about three quarters of a page |
| 3b. Tasks and data | Kausthuba | Dataset identities, label definitions, observed split sizes, reserved pool and untouched official test sets; about half a page |
| 4. Classifier and conditions | Kausthuba | Frozen embeddings, logistic regression, six-condition table, sizes, seeds and metric aggregation; about three quarters of a page |
| 5. Results | Hementh | Agreement table, macro-F1 table, both charts, paired-gap mean and standard deviation, additional prompting-example counts and the four comparisons; one to one and a half pages |
| 6. Limitations and conclusion | Harshita | Two tasks, small model and training sets, three annotators and limited sample, simple classifier, confounds and evidence-based answer; about half a page |
| Appendix A | Harshita | All three actual prompt templates and two good/two bad pilot examples per task; about one page |

Presentation: Harshita prepares two to three slides on generation, examples and conclusion; Kausthuba prepares two on tasks and conditions; Hementh prepares three on subjectivity and results. Kausthuba combines contributions into approximately nine slides: title; question/hypothesis; related work; data/conditions; generation/examples; subjectivity; two results slides; limitations/conclusion. Consolidate overlapping material to fit. All three present their own part. These are planning targets, not confirmed professor rules.

## Testing Decisions

Use the existing end-to-end fake-data pipeline test as the primary automated seam, complemented by real-artifact checkpoint validation. This approach was confirmed by Kausthuba on 29 October 2026.

- Test observable outcomes: valid generated records, resumability, complete unique run keys, and produced summaries and charts. Avoid tests coupled to internal function layout.
- Reuse the repository's temporary-directory pipeline test with a fake model and hashing embedder. These scores are fixtures and must never be reported as experimental findings.
- Retain the existing malformed-model-reply parser checks as supporting coverage. A passing fake-data test does not validate Ollama, dataset compatibility, model quality or actual sentence embeddings.
- Verify real preparation counts, all six labels, stable sampling, separation of pool/training/test roles, and exact-source augmentation selection before full runs.
- Inspect the real-model pilot manually for label correctness, naturalness, repetitions, prompt compliance and unacceptable output. Perform and record the accepted 180-example content audit before freezing, with the two-clear-errors investigation trigger.
- Verify every generated setting exists separately for all three repetitions and that each condition selects only its matching artifacts. Shared test data is intentional; shared generated outputs across repetitions are not.
- Verify C5b has six valid distinct paraphrases for each of its own 60 source examples. Incomplete targets block Checkpoint 2 and affected training runs rather than silently reducing counts.
- Verify independent annotation collection and that Hementh completes his response before viewing others. Test combining submissions without changing their judgments.
- Verify relative gaps are calculated per matched repetition before aggregation, using a case where that result differs from a ratio of aggregate means.
- Verify result-grid completeness, training counts, metric ranges, annotation coverage and reproducibility of summary calculations at their handoffs.

| Gate | Required evidence |
| --- | --- |
| Setup | Four branches on GitHub; each contributor confirms Python 3.11 setup and the two existing tests pass; generation owner verifies Ollama/model availability |
| Checkpoint 1 | Merged baseline records with 12 unique task/condition/repetition rows for C1 and C5a; plausible C1 performance; inspected pilot and documented timing/target choice; merged 60-item annotation template; separate response files distributed and report document shared |
| Checkpoint 2 | Separate zero-shot, few-shot and AugGPT data for all three repetitions and both tasks; exact chosen counts for all labels and six valid distinct paraphrases per source; valid schema, labels and provenance; completed 180-example audit with flagged batches resolved and re-audited; independently collected agreement results with 30 scored items per task; synthetic data frozen |
| Results | Merged 36-row complete result grid, summary and both report charts |
| Checkpoint 3 | Report sections 1–6 and Appendix A all present in a complete assembled draft; results and charts final |
| Finish | Final report and presentation material checked; Kausthuba submits; release tag only after explicit approval |

The handoff expects C5a to underperform C1 and describes approximately 0.17 as a six-label chance reference. Verify and report actual metrics; an unexpected ordering triggers investigation, not deletion or adjustment of results. Random-guess macro-F1 depends on the guessing rule and class distribution, so 0.17 is a rough reference rather than a universal acceptance threshold.

## Out of Scope

- Fine-tuning the embedding model, large-scale benchmarking, additional tasks or additional classifiers.
- The earlier plan's additional two datasets, classmate survey, diversity/fidelity metrics, ceiling run and 8B model. The retained 60-item annotation exercise is separate from the removed classmate survey.
- Replacing human annotations with model-generated judgments.
- Reporting fake-data test scores as research results.
- Committing real datasets, credentials or secrets.
- Treating the hypothesis as proven before real experiments and annotations finish.
- Changing agreed shared settings, frozen synthetic data or release state without required approval.
- Inventing professor requirements or experimental results.

## Further Notes

- Slow generation: overnight execution first; transfer to Kausthuba's GPU laptop only after coordination prevents duplicate runs. Kausthuba resolved the original 50-versus-60 conflict by approving the uniform 60-per-label fallback when pilot timing requires it, chosen before full generation. Further reductions require an explicit decision; partial datasets cannot pass Checkpoint 2 or feed affected training runs. Any approved post-freeze change requires retraining affected conditions and rebuilding analysis.
- Failed model quality: the proposed fallback is qwen3:4b, requiring approval of the shared setting change and coordinated regeneration.
- Failed dataset downloads: retry using an alternative connection such as a phone hotspot.
- After an approved post-freeze rerun, replace the affected synthetic-dependent training results before rebuilding analysis.
- Use the more restrictive orchestrator-handoff rules for post-freeze regeneration and final tagging: no automatic reruns or tags, even where the manuals show commands directly.
- Professor report length/format, presentation rules and spelling of Hementh's name remain unconfirmed. Working targets are recorded above.

### Implementation gaps to verify before claiming compliance

These are findings from reading the supplied bundle, not changes made by this documentation task:

- Generation and training currently share one synthetic artifact per task and setting; the accepted three-repetition design requires separate artifacts, generation/resume boundaries and matching training selection.
- The manuals describe dropping duplicates and test-identical text generally. The AugGPT implementation does not perform that filtering, and a partially populated source is treated as complete on resume. Validate and resolve before accepting the augmentation handoff.
- Reply parsing can accept multiword non-example text through its fallback; nonempty output is not proof of correct JSON or label quality. The prose description of retry/fallback order is more optimistic than the actual behavior. Connection errors also require operational recovery.
- Training can proceed with fewer than all six classes, and missing augmentation can leave C5b equivalent to C5a. Gate counts and class coverage before treating any row as a completed planned condition.
- Preparation uses a shuffled, unstratified pool and sampling takes up to the requested amount. Verify ten available examples of every label before assuming all low-resource sets contain 60 rows.
- The annotation scorer records the number of named columns, which need not equal the number of active annotators. Confirm coverage manually for the two-annotator fallback and report the actual active count.
- Summary gaps are computed from rounded aggregated means in the supplied implementation. The accepted revision requires matched repetition-level gaps before aggregation, retaining precision until presentation.
- The original annotation workflow exposes shared response columns. Collection and combination need to follow the accepted independent-submission process.
- Exact-count blocking and the 180-example audit are accepted requirements, not guarantees provided by the current scripts.
- Prompt word limits and style rules are requested rather than mechanically enforced. The report must distinguish intended filtering from checks actually performed.

These checks use the accepted testing approach. They do not authorize unrelated code changes, configuration edits or data regeneration.

### Sources and precedence

- [Master plan](source-plans/SynthText-Master-Plan.md)
- [Generation manual — Harshita](source-plans/SynthText-Manual-1-Generation-Harshita.md)
- [Data, training and lead manual — Kausthuba](source-plans/SynthText-Manual-2-Data-Training-Lead-Kausthuba.md)
- [Evaluation manual — Hementh](source-plans/SynthText-Manual-3-Evaluation-Hementh.md)

The four supplied Markdown exports are the primary planning sources. Current code informs implementation details and gaps. The orchestrator handoff adds approval and verification safeguards. Later confirmed interview decisions supersede conflicting original planning text: see the decision records in [adr](adr/) and agreed vocabulary in [CONTEXT.md](../CONTEXT.md). Conversation corrections supply the actual repository identity, teammate usernames and confirmed testing approach. Historical repository placeholders and elapsed setup instructions in the sources remain archived as originally supplied; use this consolidated specification for the current plan.

Source papers, retained as source-document references rather than independently reviewed for this conversion: [Li et al.](https://arxiv.org/abs/2310.07849), [Dai et al., AugGPT](https://arxiv.org/abs/2302.13007), [Brown et al., GPT-3](https://arxiv.org/abs/2005.14165), and [Reimers and Gurevych, Sentence-BERT](https://arxiv.org/abs/1908.10084).
