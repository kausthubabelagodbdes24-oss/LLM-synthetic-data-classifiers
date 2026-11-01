# SynthText glossary

This glossary defines the terms used in the final report, code, and accepted artifacts.

- **Two-task comparison:** Exploratory comparison of TREC question classification and Emotion classification. A difference between tasks does not establish subjectivity as its cause.
- **`scope30-v1`:** Final study identifier: seed 13, six conditions per task, and twelve task/condition result keys.
- **Selected corpus:** The exact accepted synthetic rows identified by stable IDs and hashes. Source artifacts and provenance remain preserved.
- **Equal classifier training size:** C1–C4 each contain 180 rows per task. This does not imply equal real-labelling cost because few-shot generation uses labelled demonstrations.
- **Low-resource sources:** Ten real examples per label, or 60 per task, used by C5a and as the sources for C5b.
- **AugGPT augmentation:** Six accepted paraphrases for each low-resource source, producing 360 synthetic rows plus the 60 sources in C5b.
- **Model-assisted quality screen:** A disclosed content review performed by a model. It is not independent-human annotation, human agreement, or a population accuracy estimate.
- **Frozen handoff:** The accepted set of row IDs, paths, hashes, model identities, provenance, and review evidence consumed by training.
- **Paired relative gap:** `(synthetic macro-F1 - same-task real-only macro-F1) / same-task real-only macro-F1`.
- **Single-seed result:** One observed run for seed 13. It does not provide a standard deviation, significance test, or stability estimate.
- **Fake demonstration:** Fixture-only execution used to show that the pipeline works. Fake rows and scores are never experimental evidence.
- **Accepted real results:** The frozen twelve-condition grid and its reviewed, read-only analysis artifacts.
