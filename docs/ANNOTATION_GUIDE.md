# Evaluation and Annotation Operational Guide

*Owner: Hementh (@hem4456); Operator: Kausthuba (@kausthubabelagodbdes24-oss)*

**Status (31 October 2026):** Per [ADR 0012](adr/0012-in-repo-evaluation-without-human-annotation.md), independent human annotation and external file distribution (email/shared drive/Google Docs) are no longer required and are superseded by in-repository evaluation under the sole-operator execution model. This guide documents the preserved evaluation utilities (`make`, `combine`, `score`), data formats, and in-repository evaluation procedures.

## 1. Evaluation Item Selection and Shuffling

Run:
```bash
python -m ws3_evaluation.annotate make [--force]
```

This selects 60 balanced evaluation items from `data/real/<task>/test.jsonl`:
- Exactly 5 items per label for TREC (30 items total across abbreviation, description, entity, human, location, numeric).
- Exactly 5 items per label for Emotion (30 items total across sadness, joy, love, anger, fear, surprise).
- Deterministically sampled using `seed=0` with `exact=True` coverage checks.
- **Deterministic shuffling**: After stratified selection, items within each task are deterministically shuffled using `ANNOTATION_SEED=0` so that row order does not leak gold-label groupings.

The command writes the following files to `results/`:

| File | Purpose | Contains gold labels? |
|------|---------|----------------------|
| `results/annotation_key.csv` | Internal audit record; canonical reference key | **Yes** |
| `results/annotation_<name>.csv` | Evaluation templates | No |

## 2. Overwrite Protection

`python -m ws3_evaluation.annotate make` protects existing files from accidental erasure:
- If candidate output files already exist, `make` aborts with `FileExistsError` listing the conflicting paths.
- To intentionally regenerate templates, pass `--force` (CLI) or `force=True` (API) after verifying that no completed judgments will be overwritten.

## 3. Combining and Validating Submissions

To merge evaluation response files into `results/annotation_filled.csv`, run:
```bash
python -m ws3_evaluation.annotate combine
```

This merges the response files while validating every submission against the canonical `results/annotation_key.csv`:
- Exact header match (`item_id`, `task`, `text`, `choose_one_of`, `label`).
- No duplicate item IDs within any submission.
- Complete canonical item ID coverage (no missing IDs, no foreign/extra IDs).
- Exact metadata match (`task`, `text`, `choose_one_of`).
- Exact case-sensitive label matching (`strip()` only, no `.lower()`). Blank labels are preserved as empty strings and treated as incomplete.

## 4. Subjectivity & Agreement Scoring

Run:
```bash
python -m ws3_evaluation.annotate score
```

This computes:
- Active evaluator count per task (evaluators providing at least one valid label for that task).
- Number of items with at least 2 valid judgments.
- Mean majority agreement proportion across valid items.
- Nominal Krippendorff's alpha ($\alpha$) with fixed label domain.
- Outputs summary to `results/subjectivity.csv`.

**Policy on model-assisted evaluation:** If model-assisted judgments are scored, the model identity and method must be explicitly recorded and described as model-assisted evaluation. They must never be reported as human annotation or human agreement.
