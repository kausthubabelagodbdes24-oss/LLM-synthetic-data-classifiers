# ADR 0012: EOD exploratory comparison, 30 per label, seed 13 only

Status: explicitly approved by Kausthuba, training owner and project lead, 31 October 2026 (IST). Workstream ownership remains Harshita for generation and Hementh for evaluation.

The lead approved “30 per label, seed 13 only, both tasks and all six conditions” and requested repository documentation and closed issues be aligned. The lead also removed independent-human annotation and distribution/report-sharing requirements and requested agent-performed evaluation. This record persists both decisions.

## Accepted design

- Retain TREC and Emotion, six labels each, official fixed test sets, disjoint training/pool splits, frozen MiniLM embeddings and logistic regression settings.
- Seed 13 only. C1/C2/C3 use exactly 30 examples/label (180/task). C4 uses 15 real plus 15 selected few-shot/label (180/task).
- C5a retains ten canonical real sources/label (60/task). C5b retains those exact sources plus six valid distinct paraphrases/source (420/task). Their unequal training sizes are intentional and must be disclosed.
- Four baseline result rows; twelve unique task/condition/seed result keys overall. Four zero/few-shot selected artifacts total 720 texts; two AugGPT artifacts total 720 paraphrases. Total selected generated texts: 1,440.
- Accepted generator remains qwen3:4b, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, temperature 0.9, unchanged accepted focused prompts/default thinking, three few-shot demonstrations, six paraphrases/source.
- Quality screen: ten reproducibly randomly selected examples/task/setting at seed 13: six batches, 60 reviews. At least two clear wrong-label/changed-meaning errors in a batch trigger investigation, correction and a fresh recorded review. Record ambiguities separately and compare AugGPT to its exact source. Preserve every audit round.
- Model-assisted review is allowed. Disclose reviewer/provider/version if available, rubric, selected IDs and limitations. No independent-human annotation, distribution, external shared document, human alpha or three-person read-through gate. Never represent model judgments as independent humans or measured human agreement.
- Report single-run macro-F1/accuracy and C2/C3 relative gaps against same-task seed-13 C1. No repetition SD, zero uncertainty bars, significance or stability claims. Missing C1 blocks analysis; C1=0 gives undefined relative gap plus absolute F1 difference, retaining the task.
- This is an exploratory two-task comparison, not a causal test of subjectivity, reliable superiority claim or equal-labeling-budget experiment.

## Supersession

Conflicting three-repetition, 100/60-target, 36-row, 180-audit and independent-human requirements in earlier plans, ADRs and issue handoffs are superseded. Exact completeness, source identity, test exclusion, model/prompt integrity and quality-investigation safeguards remain. Historical pilot reviews are not the new full-corpus audit.

## Safe artifact and result migration

Set shared config `per_label=30`, `train_per_label=30`, `seeds=[13]`; retain `lowres_per_label=10`, `auggpt_k=6`. Preserve model/prompt/options and source identity. Running processes and other workspaces do not automatically load a main-branch config change. Active implementers must read this decision at a safe artifact boundary; explicitly use seed 13. Do not kill an active job, rewrite another checkout or start competing GPU work. No new seed42/7 work is required; preserve any existing work.

Preserve original JSONL, provenance sidecars, journals, reviews and historical results. Do not truncate oversized files, rewrite original targets/counts or fabricate new generation fingerprints. For compatible excess zero/few-shot output, reject missing/duplicate IDs, validate all candidate rows, then select exactly 30/label by ascending SHA-256 of UTF-8 `scope30-v1|13|<task>|<setting>|<row-id>`, with row ID as tie-breaker. Record selected IDs, source artifact hashes and original generation fingerprints in a derived selection manifest. Never select by test performance. A historical producer target of 100 is not the selected training target; validate exact selected counts independently. If any label has fewer than 30 valid rows, resume compatible generation for its shortfall. C3/C4 consume the same selected few-shot corpus; C4 deterministically samples 15/label within it.

Real C1/C4 use exact seed-13 sampling from the unchanged training split. C5a uses canonical lowres_seed13 unchanged. AugGPT still requires six valid distinct paraphrases of every exact C5a source.

Use a fresh scope-specific results directory, e.g. `results/scope30-seed13`, because existing six-column result keys do not encode target revision. Archive old 100/60-target runs. Rerun affected C1–C4. Reuse C5 scores only with verified identical selected IDs/hashes, model/settings, test hashes and quality acceptance. Preserve unrelated results. Documentation/configuration changes do not establish executed migration or completed experiments; #7/#10/#13/#14 must implement and verify their consumer behavior.

## Closed issues and delivery

Closed #1/#4/#5/#6 retain status and historical evidence. #1 points to the revised spec; #4 supplies setup/history; #5 supplies unchanged real splits and seed-13 sources; #6 supplies the accepted model/prompts/pilot, with its 100 target superseded. None proves the revised experiment ran. Open #7–#18 use revised acceptance criteria.

Working target: reviewable report/slides by 31 October 2026 EOD (23:59 IST), not a guaranteed or professor-confirmed deadline. Formal submission remains planned for 4 November unless confirmed otherwise. Freeze selected manifests after quality acceptance and before training. Approved post-freeze corrections require affected reruns/rebuilt analysis. No automatic release tags or claims of actual submission.
