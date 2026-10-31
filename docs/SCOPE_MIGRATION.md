# Migration to scope30-v1

Authority: [ADR0012](adr/0012-eod-exploratory-30-seed13.md), explicitly approved31 October2026. Both tasks, six conditions, seed13, C1–C4=30/label(180/task), C5a/C5b=60/420, twelve result keys. Docs/config changes do not establish executed experiments.

|Issue|Revised delivery/compatibility|
|---|---|
|#1 closed|Revised spec/ADR supersede historical body; no new execution implied.|
|#4 closed|Setup/history/CI remain useful; sole laptop replaces teammate setup gates.|
|#5 closed|Reuse canonical splits/hashes and lowres_seed13 unchanged; preserve extra seeds.|
|#6 closed|Reuse accepted Qwen identity/prompts/pilot;100 target superseded, full quality gate still needed.|
|#7|Four baseline keys; C1=180,C5a=60; rerun affected old baselines in fresh results.|
|#8|Repository report skeleton/optional tools; no distribution/external-document gate.|
|#9|Two seed13 AugGPT artifacts;360 paraphrases/task, six per each of60 sources. TREC is regenerated under [ADR0013](adr/0013-trec-auggpt-answer-type.md) (candidate, then reviewed promotion; original preserved).|
|#10|Four selected seed13 zero/few-shot artifacts;30/label, derived selection manifests.|
|#11|Disclosed model-assisted content review:ten/task/setting,60 total; no alpha/humans.|
|#12|Resolve review flags, validate exact selected counts/provenance and freeze manifests.|
|#13|Twelve-key grid;180/180/180/180/60/420 per task, frozen accepted inputs.|
|#14|Single-seed values/gaps,two charts,no SD/alpha. Zero C1:undefined gap+absolute difference.|
|#15–#16|Methods/limitations/slides describe actual revised design, review and costs.|
|#17|Report/PDF based on twelve results; lead review replaces three-person review.|
|#18|Slides,lead review,actual submission evidence; tag still needs separate approval.|

## Safe adoption order

1. Active agents fetch/merge main normally and reread ADR0012/live issue. Preserve checkouts/jobs. Main updates do not reconfigure another workspace or running process.
2. Coordinate GPU use; adopt seed13-only work at a safe artifact boundary. Do not kill jobs or delete extra-seed output. No new seed42/7 work is required.
3. Preserve original JSONL/sidecars/journals. Validate rows, apply ADR0012's exact scope30-v1 ID-hash rule and write derived manifests. Complete missing labels; original producer targets are not selected training targets.
4. Freeze selected IDs/hashes after #11/#12 quality gates, before synthetic training.
5. Use fresh results,e.g.`results/scope30-seed13`. Preserve old runs; rerun affected C1–C4. Validate identical inputs/settings/quality before C5 reuse. Existing keys alone cannot establish compatibility.
6. #7 retains exact-count/unrelated-result preservation fixes. #13/#14 implement seed-specific exact selection,current-grid validation and single-seed analysis; config alone does not fix legacy consumers.
7. Publish selected IDs/hashes,commands,environment,quality and result evidence without committing raw real/generated corpora. Historical plans/ADRs/closed acceptance evidence remain intact.
