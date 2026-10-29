# Shared contract register

The specification defines experimental meaning. Record exact accepted interfaces here or link to their authoritative definitions. Proposed formats are not accepted contracts.

| Interface | Current state / agreement required | Owners |
| --- | --- | --- |
| Repetition-specific generation | Legacy artifacts are shared across seeds. Agree naming/layout, repetition identity, IDs, source provenance, generation provenance, resume boundaries and legacy handling before the generation producer and training consumer migrate. | Generation + training; lead approves shared changes |
| Completion, audit and freeze evidence | Exact targets and the audit rule are agreed. Define how accepted artifacts identify the frozen revision and how training verifies approval/completeness. | Generation + training; evaluation consumes provenance |
| Run records and paired analysis | Preserve unique task/condition/repetition keys and exact matching baselines. Agree precision, provenance and schema changes before changing writer/reader. Zero-baseline policy remains unresolved. | Training + evaluation |
| Annotation exchange | Independent responses must combine without changed judgments. Agree stable item matching, label validation, missing responses and actual active-annotator reporting; preserve original responses. | Evaluation + all human annotators |

For each accepted change record: exact contract, approving issue/PR and lead decision, affected producers/consumers, migration/compatibility behavior and verification evidence. Do not mark a format agreed without recording details. Keep consumers working during migration or coordinate a bounded integration change explicitly. Update this register as implementation lands; do not freeze an invented schema just to make another model's task easier.
