# Modular Report Structure (scope30-v1)

This directory contains modular section deliverables for the final SynthText exploratory research report.

Authorities: [ADR 0012](../adr/0012-eod-exploratory-30-seed13.md), [ADR 0013](../adr/0013-trec-auggpt-answer-type.md), [ADR 0014](../adr/0014-presentation-separation-colab.md), [PROJECT_SPEC.md](../PROJECT_SPEC.md), and [SCOPE_MIGRATION.md](../SCOPE_MIGRATION.md).

## Module Structure and Workstream Ownership

| File | Content / Focus | Author / Workstream | Target Scaffold Section | Status / Issue |
|---|---|---|---|---|
| [`section_2_related_work.md`](section_2_related_work.md) | Related work from verified primary sources (Li et al. 2023, Dai et al. 2023, Brown et al. 2020, Reimers & Gurevych 2019) mapped to conditions C1–C5. | Generation owner/author: Harshita (@harshitadjainbdes24-byte); implemented/integrated by Kausthuba on her behalf | Section 2 | Delivered (Issue #15) |
| [`section_3a_generation_methods.md`](section_3a_generation_methods.md) | Generation architecture, Qwen model identity (`qwen3:4b`), prompt templates, filtering rules, write-ahead journals, seed-13 selection, demonstration costs, quality screening, and Option A freeze. | Generation owner/author: Harshita (@harshitadjainbdes24-byte); implemented/integrated by Kausthuba on her behalf | Section 3a | Delivered (Issue #15) |
| [`methods_data_classifiers.md`](methods_data_classifiers.md) | Tasks, real data preparation, data partitioning, untouched tests, duplicate disclosure, frozen MiniLM embedder, LogisticRegression classification head, scope30-v1 six conditions, training harness, cost separation, metrics, and accepted results reference. | Training owner and project lead: Kausthuba Belagod (@kausthubabelagodbdes24-oss) | Sections 3b and 4 | Delivered (Issue #16) |
| [`section_6_limitations_and_conclusions.md`](section_6_limitations_and_conclusions.md) | Limitations (single-seed, small data, task confounding, demonstration costs, model-assisted screening) and conclusions grounded strictly in observed numerical directions from the four accepted comparisons. | Generation owner/author: Harshita (@harshitadjainbdes24-byte); implemented/integrated by Kausthuba on her behalf | Section 6 | Delivered (Issue #15) |
| [`appendix_a_prompts_and_examples.md`](appendix_a_prompts_and_examples.md) | Verbatim prompt templates from `generate.py` and two good / two problematic preserved examples per task with verified row ID and artifact SHA-256 provenance; historical pilot evidence explicitly segregated. | Generation owner/author: Harshita (@harshitadjainbdes24-byte); implemented/integrated by Kausthuba on her behalf | Appendix A | Delivered (Issue #15) |
| [`../REPORT.md`](../REPORT.md) | Canonical assembled report: Sections 1–6, References and Appendix A, reconciled to the accepted analysis, freeze and split hashes by `ws3_evaluation/report_scope30.py`. | Evaluation/report owner: Hementh (@hem4456); implemented/integrated by Kausthuba on his behalf | Sections 1–6 and Appendix A | Reviewed and accepted by the project lead on 1 November 2026 (Issue #17) |
| [`../REPORT.pdf`](../REPORT.pdf) + `../REPORT.pdf.json` | PDF of `REPORT.md`; the sidecar binds the PDF SHA-256, page count and the source Markdown hash. | Evaluation/report owner: Hementh (@hem4456); project lead/reviewer: Kausthuba | Whole report | Reviewed and accepted by the project lead on 1 November 2026 (Issue #17) |

## Workstream Separation and Shared Assembly Target

To prevent concurrent editing collisions across workstreams and issues:
- `docs/REPORT.md` is preserved untouched for Issue #17 final assembly.
- Individual workstream deliverables are stored as separate modular files in this directory (`docs/report/`).
- Issue #15 delivers Sections 2, 3a, 6, and Appendix A.
- Issue #16 delivers Sections 3b and 4 (`methods_data_classifiers.md`).
- Issue #17 combined these section files into `docs/REPORT.md` (condensed for the 6–8 page target) and built `docs/REPORT.pdf`. `REPORT.md` is the canonical report. The modules remain the #15/#16 inputs, corrected as listed below.

## Issue #17 corrections to the modules (PR #39 review)

The lead's review of PR #39 required the modules to agree with the committed code, freeze and training evidence. Issue #17 corrected them, and the module tests now assert the corrected values:

- `methods_data_classifiers.md`:
  - The TREC `train`/`pool`/`test`/`lowres_seed13` SHA-256 values are now the `prepare_data.CHECKSUMS` values (`b587c84d…`, `0a78fbe6…`, `415f5262…`, `901c0596…`), which the freeze also records. The earlier values (`00bfe51d…`, `08b68832…`, `cbf804ec…`, `50f24e93…`) matched no committed record.
  - The pool label counts are taken from the Issue #5 contract and attributed to it.
  - `lowres_seed13` is drawn from `pool.jsonl`.
  - C4's real half is described as an independent seed-13 draw of 15 per label.
  - Only the recorded classifier settings are stated: the solver, tolerance, loss and convergence claims were removed.
  - C4's demonstration cost is shown as at most 59 / 54.
  - The unverified count of ten official duplicates was removed.
- `section_3a_generation_methods.md`:
  - Normalization is lower-casing plus whitespace collapsing (`common.norm`).
  - The Ollama 0.40.13 / RTX 4060 environment is attributed to the pilot record only, and "Windows 11" was removed.
  - "Significant label drift" became the operator's recorded pilot label-fit problems.
- `section_2_related_work.md`: the Li et al. (2023) and Dai et al. (2023) paragraphs keep only claims supported by the papers' abstracts.
- `section_6_limitations_and_conclusions.md`:
  - The 20/30-word limits are described as generation limits, not real-text lengths.
  - The abbreviation-class sensitivity follows the #14 interpretation (roughly 0.01–0.02).
  - "Future research should evaluate replications" became a statement of what was not run.
- Author names: no authoritative record gives the teammates' surnames, so the report uses given names plus GitHub handles.
- Not modified: `docs/analysis/scope30-v1-seed13-interpretation.md` is an accepted #14 output and still calls C4 − C1 on TREC "the largest within-task difference in the grid". The report narrows this to the largest absolute macro-F1 difference among the four comparisons in Table 2 of the report.

## Presentation Boundaries (ADR 0014)

Under project lead decision [ADR 0014](../adr/0014-presentation-separation-colab.md):
- Repository deliverables are strictly limited to technical methods, reproducibility assets, analysis artifacts, and the written report/PDF.
- The final professor-facing presentation deck, speaker notes, and submission logistics are managed privately outside GitHub and are not tracked as repository issues or deliverables.
