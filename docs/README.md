# Documentation guide

This page separates the finished study materials from technical reproduction and development history.

## Recommended reading path

For a professor or reviewer:

1. [Final report (PDF)](REPORT.pdf)
2. [Final experiment specification](PROJECT_SPEC.md)
3. [Results summary](analysis/scope30-v1-seed13/summary.md)
4. [Written interpretation](analysis/scope30-v1-seed13-interpretation.md)
5. [Google Colab walkthrough](../notebooks/SynthText_scope30_walkthrough.ipynb)

The report contains the research question, related work, methods, complete results table, both charts, limitations, conclusions, prompts, and reviewed examples.

## Results and figures

| Artifact | Purpose |
| --- | --- |
| [analysis.json](analysis/scope30-v1-seed13/analysis.json) | Machine-readable accepted results and comparisons |
| [summary.md](analysis/scope30-v1-seed13/summary.md) | Human-readable tables, comparisons, costs, and limitations |
| [f1_by_condition.png](analysis/scope30-v1-seed13/f1_by_condition.png) | Macro-F1 across all six conditions |
| [gap.png](analysis/scope30-v1-seed13/gap.png) | Synthetic-only relative gap against real-only training |

## Methods and reproducibility

- [Reproducibility guide](REPRODUCIBILITY.md)
- [Data and classifier methods](report/methods_data_classifiers.md)
- [Generation methods](report/section_3a_generation_methods.md)
- [Related work](report/section_2_related_work.md)
- [Limitations and conclusions](report/section_6_limitations_and_conclusions.md)
- [Prompt templates and reviewed examples](report/appendix_a_prompts_and_examples.md)
- [Accepted freeze overview](freeze/README.md)

## Development archive

The following material preserves auditability but is not needed for the final project narrative.

<details>
<summary>Show internal development and provenance records</summary>

- [Architecture decision records](adr/README.md)
- [Scope migration record](SCOPE_MIGRATION.md)
- [Generator pilot history](PILOT_REPORT.md)
- [Quality-review evidence](reviews/issue11/README.md)
- [Original planning documents](source-plans/README.md)
- [Internal interface contracts](agents/CONTRACTS.md)
- [Implementation protocol](agents/IMPLEMENTATION_PROTOCOL.md)
- [Reusable implementation prompt](agents/ISSUE_PROMPT.md)
- [Historical optional annotation tooling](ANNOTATION_GUIDE.md)

</details>

Historical documents are retained to keep the repository honest and reproducible. They do not override the final specification or the accepted report.
