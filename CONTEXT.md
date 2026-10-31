# SynthText current vocabulary

Authority: [ADR0012](docs/adr/0012-eod-exploratory-30-seed13.md) and [specification](docs/PROJECT_SPEC.md).

- **Two-task comparison:** exploratory comparison on TREC and Emotion. Task differences do not establish subjectivity as cause.
- **scope30-v1:** seed13 only; C1–C4=30/label(180/task), C4=15+15/label, C5a=60,C5b=420; twelve task/condition/seed keys.
- **Repetition:** one seed-identified six-condition comparison. Current scope has one; seeds42/7 are historical/future work, not completed replications.
- **Selected corpus:** exact30/label view derived by ADR0012's stable ID-hash rule from compatible original output; records selected IDs/source hashes. Originals/sidecars remain untouched.
- **Equal classifier training size:** C1–C4 only, excludes labeled generation demonstrations, does not mean equal labeling budgets. C5 sizes differ intentionally.
- **Model-assisted quality screen:** sixty disclosed reviews, ten/task/setting, source comparison for AugGPT. Not human annotation, independent-human agreement or population accuracy.
- **Paired relative gap:** (synthetic F1−same-task seed13 C1 F1)/C1 F1. No cross-seed mean/SD. Zero C1:undefined relative gap plus absolute difference; missing C1 blocks analysis.
- **Frozen handoff:** selected IDs/hashes/provenance accepted after structural and quality checks. Counts, CI or historical closure alone do not establish acceptance.
- **Independent annotation:** historical optional tooling concept, not a current requirement. Never recast model judgments as independent-human responses.
