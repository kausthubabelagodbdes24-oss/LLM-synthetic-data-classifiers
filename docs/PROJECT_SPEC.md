# SynthText specification — scope30-v1

Approved 31 October 2026 by Kausthuba. [ADR 0012](adr/0012-eod-exploratory-30-seed13.md) supersedes conflicting earlier sizes, repetitions, annotation gates and schedules. Earlier manuals/ADRs remain historical.

## Goal and boundaries

Compare real and locally LLM-generated classifier training on TREC and Emotion. Interpret four comparisons: synthetic-only versus real across tasks; few-shot versus zero-shot; mixed versus real; AugGPT versus low-resource real. This is a small single-repetition exploratory comparison, not a causal test of subjectivity or evidence of statistical superiority/stability. Results may contradict the hypothesis.

## Data and models

TREC: SetFit/TREC-QC; abbreviation, entity, description, human, location, numeric. Prepared train/pool/test counts:4,907/545/500. Emotion: dair-ai/emotion split configuration; sadness, joy, love, anger, fear, surprise; counts14,400/1,600/2,000. Annoyance belongs under anger; anxiety under fear.

Preserve deterministic disjoint training/pool splits and untouched official test sets. Do not borrow pool/test rows for C1. Known TREC abbreviation training count76 suffices for30. Use canonical lowres_seed13:ten examples/label,60/task. Keep seed42/7 files archived. Canonical hashes remain in [contracts](agents/CONTRACTS.md).

Generator:qwen3:4b, digest `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`, accepted focused prompts, temperature0.9, default thinking unchanged, three demonstrations and six paraphrases/source. Classifier:frozen sentence-transformers/all-MiniLM-L6-v2; logistic regression C=1.0,max_iter=2000,random_state=13. Record actual environment/identity/commands.

## Exact experiment grid

| Condition | Construction per label | Total/task |
| --- | --- | --- |
| C1 | 30 real training examples |180|
| C2 | 30 zero-shot synthetic |180|
| C3 | 30 few-shot synthetic |180|
| C4 | 15 real +15 selected few-shot |180|
| C5a | 10 canonical low-resource sources |60|
| C5b | Same10 sources +six distinct valid paraphrases/source |420|

Two tasks ×six conditions ×seed13 =**12 unique result keys**. #7 supplies four baselines; #13 validates all twelve. C1–C4 match classifier input counts, not labeling budgets. C5a/C5b intentionally differ in size. Disclose labeled demonstration/source costs separately.

Four selected zero/few-shot artifacts total720 texts; two AugGPT artifacts total720 paraphrases. Total selected generated texts=1,440; preserved excess output is outside this target.

## Artifact migration and freeze

Use seed-specific `data/synthetic/<task>/<setting>_seed13.jsonl` plus provenance, separating fake and real storage. Enforce six labels, exact selected counts, unique IDs, schema, normalized duplicate/test-text exclusion, accepted model/prompt identity and source matching. C5b requires six distinct valid paraphrases for every exact C5a source. Invalid/incomplete inputs block training.

Preserve raw files, sidecars/journals, reviews and old results. Apply ADR0012's exact `scope30-v1` ID-hash rule to select30/label from compatible excess output and write a derived manifest with selected IDs, source hashes/fingerprints and scope ID. Never truncate originals, falsify metadata or select by test performance. C3/C4 share the selected few-shot corpus; C4 takes15/label within it. Resume compatible shortfalls only.

Use fresh scope-specific result storage; old keys cannot distinguish target revisions. Rerun affected C1–C4. Reuse C5 only after verifying identical inputs/settings, test hashes and quality acceptance. See [migration](SCOPE_MIGRATION.md) and ADR0012. Config/docs express intent; main's legacy consumers still need #7/#10/#13/#14 integration.

## Quality review

Reproducibly randomly sample ten examples/task/setting at seed13:six batches,**60 reviews**. Record IDs, method, label fit, naturalness, variation, copied demonstrations, clear errors and ambiguities separately. Compare AugGPT to exact source meaning. At least two clear wrong-label/changed-meaning errors among a batch's ten trigger investigation, correction and a fresh recorded random audit. Preserve all rounds; no cherry-picking.

Model-assisted review is allowed with reviewer/version if available, method and limitations disclosed. It is not independent-human annotation, human agreement or population accuracy. No distribution, external shared-document, three-person read-through or Krippendorff alpha gate. Preserve historical optional tools/responses without fabricating human completion.

Freeze selected manifest/hash revisions after structural and quality acceptance, before synthetic training. Approved post-freeze correction requires affected retraining/rebuilt analysis. Counts/tests/CI alone do not establish quality or freeze acceptance.

## Evaluation and reporting

Record task,condition,seed13,exact n_train,macro-F1,accuracy on unchanged official tests and frozen-input provenance. Retain precision until display. Require exactly12 unique keys; reject missing/duplicate/incompatible or out-of-range results.

C2/C3 relative gap=(synthetic F1−matching C1 F1)/C1 F1 within task/seed. Missing C1 blocks analysis. C1=0:relative gap undefined; retain task and report absolute difference. Report single-run values without repetition SD, significance, confidence intervals or misleading zero uncertainty bars. No human-agreement metric required.

Produce a readable six-condition macro-F1 chart and synthetic-versus-real gap chart; label undefined cases. Interpret all four comparisons, unequal C5 sizes and extra demonstration costs. Inspect rendered charts. Model-assisted quality judgments do not establish human task subjectivity.

Retain sections1–6 and Appendix A:introduction; related work(Li et al.,AugGPT,GPT-3 few-shot,Sentence-BERT); generator/prompts and tasks/data; classifier/conditions; actual results and two charts; limitations/conclusion; actual templates and two good/two bad preserved examples/task. Verify citations against sources. The working report target is6–8 pages pending course requirements. Replace the former alpha/subjectivity section with design, quality method and limitations. Kausthuba reviews repository deliverables.

The final professor-facing presentation deck and submission logistics are deliberately separate from development: they are not repository artifacts and are not tracked in GitHub issues. The repository instead provides reusable reports, comparison tables, charts and a Google Colab-compatible walkthrough. Local files or repository completion do not prove presentation or submission; release tags still need separate approval. See [ADR0014](adr/0014-presentation-separation-colab.md).

## Ownership, checkpoints and verification

Kausthuba(@kausthubabelagodbdes24-oss) is sole operator. Keep separate workspaces, one active implementer/branch per issue, preserve dirty checkouts/history and normally merge main. No unauthorized reset, force-push, branch deletion or tags. Coordinate GPU use; do not stop active jobs or alter their checkout. Running processes adopt scope explicitly at safe boundaries.

Working target:EOD31 October2026,23:59IST; not guaranteed or professor-confirmed. Formal submission remains planned4 November but is owner-managed outside GitHub. Earlier checkpoint dates are superseded by evidence gates:#4/#5 setup/canonical inputs; #6 accepted pilot/model; #7 four baselines; #8 repository report scaffold; #9/#10 selected corpora; #11 disclosed quality review; #12 resolved flags/frozen manifests; #13 twelve results; #14 tables/charts; #15/#16 methods; #17 report/PDF; #18 reproducibility/Colab walkthrough. Closed #1/#4/#5/#6 retain historical evidence, not proof of revised execution.

Use Python3.11, meaningful behavior checks and the previously approved fake-data end-to-end seam. Verify exact sampling,resume identity,deterministic selection,contamination rejection,matching C5 sources,unrelated-result preservation,twelve-key coverage and single-seed analysis. Record actual commands/outcomes/unrun checks. Fake/hashing scores are fixtures only.

Out of scope:embedding fine-tuning,extra tasks/classifiers,paid inference substitution,fabricated human judgments/results,independent-human agreement,unsupported statistical claims,committed raw real/generated corpora or silent frozen-data changes. Original [source plans](source-plans/) and earlier [ADRs](adr/) preserve history; ADR0012,this specification,[glossary](../CONTEXT.md) and contracts define current intent.
