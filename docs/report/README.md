# Modular Report Structure (scope30-v1)

Authorities: [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0014](../adr/0014-presentation-separation-colab.md), [PROJECT_SPEC.md](../PROJECT_SPEC.md), and [SCOPE_MIGRATION.md](../SCOPE_MIGRATION.md).

## Workstream Separation and Report Assembly

To preserve workstream boundaries, prevent concurrent edit conflicts, and maintain clean modular provenance:
- **Issue #15 ([Generation])**: Delivers generation-related report sections and Appendix A:
  - Section 2: Related Work (`section_2_related_work.md`)
  - Section 3a: Model and Generation Method (`section_3a_generation_methods.md`)
  - Section 6: Limitations and Conclusions (`section_6_limitations_and_conclusions.md`)
  - Appendix A: Prompt Templates and Generation Examples (`appendix_a_prompts_and_examples.md`)
  - Consolidated generation deliverable: `generation_report_sections.md`
- **Issue #16 ([Training & Lead])**: Delivers data and classifier methods:
  - Section 3b: Tasks and Real Data Preparation
  - Section 4: Classifier and Experimental Conditions
  - Consolidated data/classifier deliverable: `methods_data_classifiers.md`
- **Issue #17 ([Evaluation & Lead])**: Assembles the complete report into the central scaffold (`docs/REPORT.md`) and compiles the reviewed PDF.

`docs/REPORT.md` is preserved as the shared assembly target; individual workstreams deliver verified modular section components to this directory (`docs/report/`) to avoid pre-assembly collisions.

## Delivered Section Modules

Per [AGENTS.md](../../AGENTS.md), Kausthuba Belagod (@kausthubabelagodbdes24-oss) is the sole operator across all workstreams, with historical workstream role attribution preserved for modularity.

| File | Content | Workstream / Owner | Status |
|---|---|---|---|
| [`section_2_related_work.md`](section_2_related_work.md) | Related work on synthetic text generation, task subjectivity, AugGPT paraphrasing, few-shot prompting, and Sentence-BERT representations from verified primary sources. | Harshita (WS1) / Kausthuba (sole operator) | Delivered (Issue #15) |
| [`section_3a_generation_methods.md`](section_3a_generation_methods.md) | Generation architecture, Qwen model identity, prompt templates, filtering, write-ahead journals, seed-13 selection, demonstration costs, quality screening, and freeze. | Harshita (WS1) / Kausthuba (sole operator) | Delivered (Issue #15) |
| [`section_6_limitations_and_conclusions.md`](section_6_limitations_and_conclusions.md) | Limitations (single-seed, small data, task confounding, model-assisted screening) and conclusions grounded in the four accepted single-seed comparisons, highlighting contrary outcomes. | Harshita (WS1) / Kausthuba (sole operator) | Delivered (Issue #15) |
| [`appendix_a_prompts_and_examples.md`](appendix_a_prompts_and_examples.md) | Full text of all three generation prompt templates and two good / two problematic examples per task with verified provenance, distinguishing final accepted artifacts from historical pilot evidence. | Harshita (WS1) / Kausthuba (sole operator) | Delivered (Issue #15) |
| [`generation_report_sections.md`](generation_report_sections.md) | Consolidated, sequential assembly of Sections 2, 3a, 6, and Appendix A. | Harshita (WS1) / Kausthuba (sole operator) | Delivered (Issue #15) |
| [`methods_data_classifiers.md`](methods_data_classifiers.md) | Tasks, real data preparation, data partitioning, untouched tests, duplicate disclosure, frozen MiniLM embedder, LogisticRegression classification head, scope30-v1 six conditions, training harness, cost separation, metrics, and accepted results reference. | Kausthuba (WS2 / sole operator) | Delivered (Issue #16) |

## Development and Presentation Boundaries (ADR 0014)

Under project lead decision [ADR 0014](../adr/0014-presentation-separation-colab.md):
- Repository deliverables are strictly limited to technical methods, reproducibility assets, analysis artifacts, and the written report/PDF.
- The final professor-facing presentation deck, speaker notes, and submission logistics are managed privately outside GitHub and are not tracked as repository issues or deliverables.
