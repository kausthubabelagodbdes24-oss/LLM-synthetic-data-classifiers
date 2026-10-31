# Block incomplete synthetic-data handoffs and affected training runs

Historical record: [ADR0012](0012-eod-exploratory-30-seed13.md) supersedes conflicting size, repetition, human-annotation, audit-count and analysis requirements below. Unchanged model/source/integrity safeguards remain.


On 29 October 2026, Kausthuba accepted a strict completion gate: both tasks and all three repetitions must contain the chosen zero-shot and few-shot count for every one of the six labels, and six valid, distinct AugGPT paraphrases per low-resource source example. Shortfalls require retries before the freeze; incomplete datasets block Checkpoint 2 and affected training runs instead of silently becoming smaller experimental conditions. Any further target reduction requires an explicit decision, accepting a possible delay to preserve comparability and a truthful completion status.
