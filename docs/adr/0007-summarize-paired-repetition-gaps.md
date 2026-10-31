# Calculate relative gaps within repetitions before aggregating

Historical record: [ADR0012](0012-eod-exploratory-30-seed13.md) supersedes conflicting size, repetition, human-annotation, audit-count and analysis requirements below. Unchanged model/source/integrity safeguards remain.


On 29 October 2026, Kausthuba approved calculating each C2 and C3 relative gap against C1 within the same task and repetition, then reporting the mean and standard deviation of the three gaps. This replaces the supplied analysis behavior of calculating a gap from already averaged scores, preserving the pairing between each generated dataset and its real-data baseline. A zero C1 denominator still requires an explicit handling rule before analysis can be finalized.
