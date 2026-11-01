# SynthText Report Modules

This directory contains modular section deliverables for the final SynthText exploratory research report.

## Module Structure and Workstream Ownership

| File | Content / Focus | Author / Workstream | Target Scaffold Section | Status / Issue |
|---|---|---|---|---|
| `methods_data_classifiers.md` | Tasks, real data preparation, data partitioning, untouched tests, duplicate disclosure, frozen MiniLM embedder, LogisticRegression classification head, scope30-v1 six conditions, training harness, cost separation, metrics, and accepted results reference. | Kausthuba Belagod (@kausthubabelagodbdes24-oss) / WS2 | Sections 3b and 4 | Delivered (Issue #16) |
| `generation_methods_appendix.md` (planned) | Prompting templates, Qwen generator identity, sampling parameters, AugGPT workflow, model-assisted quality review findings, limitations, and Appendix A. | Harshita Jain (@harshitadjainbdes24-byte) / WS1 | Sections 2, 3a, 6, and Appendix A | Planned (Issue #15) |
| `../REPORT.md` | Full integrated report assembled across all workstream deliverables. | Hementh M (@hem4456) / WS3, reviewed by lead | Sections 1–6 and Appendix A | Assembly (Issue #17) |

## Development and Presentation Boundaries (ADR 0014)

Under project lead decision [ADR 0014](../adr/0014-presentation-separation-colab.md):
- Repository deliverables are strictly limited to technical methods, reproducibility assets, analysis artifacts, and the written report/PDF.
- The final professor-facing presentation deck, speaker notes, and submission logistics are managed privately outside GitHub and are not tracked as repository issues or deliverables.
