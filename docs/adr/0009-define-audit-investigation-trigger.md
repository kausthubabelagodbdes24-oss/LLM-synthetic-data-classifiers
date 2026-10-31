# Flag batches with two clear errors in a ten-example audit

Historical record: [ADR0012](0012-eod-exploratory-30-seed13.md) supersedes conflicting size, repetition, human-annotation, audit-count and analysis requirements below. Unchanged model/source/integrity safeguards remain.


On 29 October 2026, Kausthuba approved investigating a batch when at least two of its ten audited examples clearly have the wrong label or, for AugGPT, change the source meaning. Record ambiguous cases separately; address the underlying problem and perform a fresh random audit before accepting a flagged batch. This is a practical review trigger rather than a statistical guarantee of dataset accuracy.
