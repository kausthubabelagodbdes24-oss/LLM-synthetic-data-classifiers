# Match classifier training size and disclose prompting examples separately

Historical record: [ADR0012](0012-eod-exploratory-30-seed13.md) supersedes conflicting size, repetition, human-annotation, audit-count and analysis requirements below. Unchanged model/source/integrity safeguards remain.


On 29 October 2026, Kausthuba confirmed that C1–C4 compare equal classifier training-set sizes rather than equal human-labeling budgets. Report the distinct real examples used for prompting separately, because few-shot generation receives labeled demonstrations in addition to its downstream synthetic training examples. This retains the planned condition comparison while ruling out claims of equal labeling cost.
