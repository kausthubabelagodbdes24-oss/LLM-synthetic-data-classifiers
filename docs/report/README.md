# Modular Report Structure (scope30-v1)

This directory contains modular section deliverables for the final SynthText exploratory research report.

Authorities: [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0013](../adr/0013-trec-auggpt-answer-type.md), [ADR 0014](../adr/0014-presentation-separation-colab.md), [PROJECT_SPEC.md](../PROJECT_SPEC.md), and [SCOPE_MIGRATION.md](../SCOPE_MIGRATION.md).

## Module Structure and Workstream Ownership

| File | Content / Focus | Author / Workstream | Target Scaffold Section | Status / Issue |
|---|---|---|---|---|
| [`section_2_related_work.md`](section_2_related_work.md) | Related work from verified primary sources (Li et al. 2023, Dai et al. 2023, Brown et al. 2020, Reimers & Gurevych 2019) mapped to conditions C1–C5. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS1 Harshita) | Section 2 | Delivered (Issue #15) |
| [`section_3a_generation_methods.md`](section_3a_generation_methods.md) | Generation architecture, Qwen model identity (`qwen3:4b`), prompt templates, filtering rules, write-ahead journals, seed-13 selection, demonstration costs, quality screening, and Option A freeze. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS1 Harshita) | Section 3a | Delivered (Issue #15) |
| [`methods_data_classifiers.md`](methods_data_classifiers.md) | Tasks, real data preparation, data partitioning, untouched tests, duplicate disclosure, frozen MiniLM embedder, LogisticRegression classification head, scope30-v1 six conditions, training harness, cost separation, metrics, and accepted results reference. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS2) | Sections 3b and 4 | Delivered (Issue #16) |
| [`section_6_limitations_and_conclusions.md`](section_6_limitations_and_conclusions.md) | Limitations (single-seed, small data, task confounding, demonstration costs, model-assisted screening) and conclusions grounded strictly in observed numerical directions from the four accepted comparisons. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS1 Harshita) | Section 6 | Delivered (Issue #15) |
| [`appendix_a_prompts_and_examples.md`](appendix_a_prompts_and_examples.md) | Verbatim prompt templates from `generate.py` and two good / two problematic preserved examples per task with verified row ID and artifact SHA-256 provenance; historical pilot evidence explicitly segregated. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS1 Harshita) | Appendix A | Delivered (Issue #15) |
| [`../REPORT.md`](../REPORT.md) | Canonical assembled report: Sections 1–6, References and Appendix A, reconciled to the accepted analysis, freeze and split hashes by `ws3_evaluation/report_scope30.py`. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS3 Hementh) | Sections 1–6 and Appendix A | Assembled for lead review (Issue #17) |
| [`../REPORT.pdf`](../REPORT.pdf) + `../REPORT.pdf.json` | PDF of `REPORT.md`; the sidecar binds the PDF SHA-256, page count and the source Markdown hash. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator | Whole report | Built for lead review (Issue #17); not yet reviewed |

## Workstream Separation and Shared Assembly Target

To prevent concurrent editing collisions across workstreams and issues:
- `docs/REPORT.md` is preserved untouched for Issue #17 final assembly.
- Individual workstream deliverables are stored as separate modular files in this directory (`docs/report/`).
- Issue #15 delivers Sections 2, 3a, 6, and Appendix A.
- Issue #16 delivers Sections 3b and 4 (`methods_data_classifiers.md`).
- Issue #17 combined these section files into `docs/REPORT.md` (condensed for the 6–8 page target) and built `docs/REPORT.pdf`. `REPORT.md` is the canonical report. The modules stay as the accepted #15/#16 inputs.

## Issue #17 reconciliation notes on the modules

`REPORT.md` reports only what the committed evidence supports. Where a module disagreed with that evidence, the report follows the evidence. The modules were left unchanged for the lead to decide:

- `methods_data_classifiers.md` §1.2/§1.5 gives TREC `train`/`pool`/`test`/`lowres_seed13` SHA-256 values (`00bfe51d…`, `08b68832…`, `cbf804ec…`, `50f24e93…`). These match neither `prepare_data.CHECKSUMS` nor the freeze manifest (`b587c84d…`, `0a78fbe6…`, `415f5262…`, `901c0596…`). The report uses the canonical values.
- `methods_data_classifiers.md` §1.2 gives a TREC pool label distribution (10/118/125/123/92/77) that conflicts with the #5 contract (10/126/128/124/78/79). Neither can be checked against committed data, so the report omits it.
- `methods_data_classifiers.md` §1.5 says `lowres_seed13` was drawn from `train.jsonl`, but `prepare_data.py` samples it from `pool.jsonl`. §2.3 says C4's real half is the first 15 per label of the C1 sample, but `train_scope30.py` draws an independent seed-13 `per_label_sample(train, 15)`. §2.2 lists solver, tolerance, loss and a convergence guarantee, none of which is recorded in the grid binding.
- `section_3a_generation_methods.md` describes normalization as punctuation-stripping, but `common.norm` only lower-cases and collapses whitespace. It also attributes the Ollama 0.40.13 / RTX 4060 environment to all generation, whereas the repository records it for the pilot (`PILOT_REPORT.md`).
- `section_2_related_work.md` attributes claims to Li et al. (2023) and Dai et al. (2023) beyond their verified abstracts. The report keeps only the verified claims.
- The superseded scaffold's surnames ("Harshita Jain", "Hementh M") have no authoritative record and are not used.

## Presentation Boundaries (ADR 0014)

Under project lead decision [ADR 0014](../adr/0014-presentation-separation-colab.md):
- Repository deliverables are strictly limited to technical methods, reproducibility assets, analysis artifacts, and the written report/PDF.
- The final professor-facing presentation deck, speaker notes, and submission logistics are managed privately outside GitHub and are not tracked as repository issues or deliverables.
