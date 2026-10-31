# Use a uniform 60-per-label fallback when pilot timing requires it

Historical record: [ADR0012](0012-eod-exploratory-30-seed13.md) supersedes conflicting size, repetition, human-annotation, audit-count and analysis requirements below. Unchanged model/source/integrity safeguards remain.


On 29 October 2026, Kausthuba approved retaining 100 examples per label as the target, with a fallback to 60 if pilot timing shows the full design cannot finish before the synthetic-data freeze. Choose and document the fallback before full generation, applying it to both tasks, all three repetitions and C1–C4; C4 then uses 30 real plus 30 synthetic examples per label, while C5a remains ten real examples per label and C5b retains six requested paraphrases per C5a example. This resolves the original plans' conflicting 50-versus-60 fallback and prioritizes complete, comparable repetitions over uneven partial datasets; no shared configuration is changed by this note.
