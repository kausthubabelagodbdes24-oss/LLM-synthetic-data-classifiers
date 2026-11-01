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
| `../REPORT.md` | Full integrated report assembled across all workstream deliverables. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / Sole operator (historical: WS3 Hementh) | Sections 1–6 and Appendix A | Assembly (Issue #17) |

## Workstream Separation and Shared Assembly Target

To prevent concurrent editing collisions across workstreams and issues:
- `docs/REPORT.md` is preserved untouched for Issue #17 final assembly.
- Individual workstream deliverables are stored as separate modular files in this directory (`docs/report/`).
- Issue #15 delivers Sections 2, 3a, 6, and Appendix A.
- Issue #16 delivers Sections 3b and 4 (`methods_data_classifiers.md`).
- Issue #17 combines these verified section files into `docs/REPORT.md` and compiles the reviewed PDF.

## Presentation Boundaries (ADR 0014)

Under project lead decision [ADR 0014](../adr/0014-presentation-separation-colab.md):
- Repository deliverables are strictly limited to technical methods, reproducibility assets, analysis artifacts, and the written report/PDF.
- The final professor-facing presentation deck, speaker notes, and submission logistics are managed privately outside GitHub and are not tracked as repository issues or deliverables.
