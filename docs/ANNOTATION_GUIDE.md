# Annotation File Distribution and Operational Guide

*Owner: Hementh (@hem4456) — operational reference for distributing annotation response files.*

This guide details the procedure for generating, distributing, collecting, and scoring the independent human annotations for Checkpoint 1 and Checkpoint 2 (Workstream 3).

## 1. File Generation and Distribution

Run `python -m ws3_evaluation.annotate make` to produce the 60 balanced annotation items (5 per label per task, 30 per task, deterministically sampled from `data/real/<task>/test.jsonl` with seed 0 and deterministically shuffled per task to prevent order-based gold label leakage).

The command writes the following files to `results/`:

| File | Recipient | Contains gold labels? | Safe to share? |
|------|-----------|----------------------|----------------|
| `results/annotation_key.csv` | Internal only — do NOT share | **Yes** | **No — private audit record** |
| `results/annotation_kausthuba.csv` | @kausthubabelagodbdes24-oss only | No | Yes — only to this annotator |
| `results/annotation_harshita.csv` | @harshitadjainbdes24-byte only | No | Yes — only to this annotator |
| `results/annotation_hementh.csv` | @hem4456 only | No | Yes — only to this annotator |

**Distribution rules:**
- `results/annotation_key.csv` is strictly private. It contains the `gold_label` column and must never be distributed or visible to any annotator.
- Each annotator receives only their own response file (`annotation_<name>.csv`), which has columns `[item_id, task, text, choose_one_of, label]`. Initial `label` entries are blank.
- Annotators label independently without consulting gold answers, external sources, or other annotators.

## 2. Human Process Requirement — Hementh Completes First

Per ADR-0005, the independent annotation rule requires that Hementh (@hem4456) completes and submits his judgments before viewing any peer submissions and before `combine` is run:
- This is an operational human process requirement, not enforced in code.
- The team lead (@kausthubabelagodbdes24-oss) must confirm Hementh's completed submission (e.g., via issue comment on GitHub) prior to running `combine`.

## 3. Overwrite Protection

`python -m ws3_evaluation.annotate make` protects existing files from accidental erasure:
- If candidate output files already exist, `make` aborts with `FileExistsError` listing the conflicting paths.
- To intentionally regenerate templates, pass `--force` (CLI) or `force=True` (API) after verifying that no completed judgments will be overwritten.

## 4. Combining and Validating Submissions

Once completed response files are returned, run:
```bash
python -m ws3_evaluation.annotate combine
```

This merges the response files into `results/annotation_filled.csv` while validating every submission against the canonical `results/annotation_key.csv`:
- Exact header match (`item_id`, `task`, `text`, `choose_one_of`, `label`).
- No duplicate item IDs within any submission.
- Complete canonical item ID coverage (no missing IDs, no foreign/extra IDs).
- Exact metadata match (`task`, `text`, `choose_one_of`).
- Exact case-sensitive label matching (`strip()` only, no `.lower()`). Blank labels are preserved as empty strings and treated as incomplete.

## 5. Subjectivity Scoring

Run:
```bash
python -m ws3_evaluation.annotate score
```

This computes:
- Active annotator count per task (annotators providing at least one valid label for that task).
- Number of items with at least 2 valid annotator judgments.
- Mean majority agreement proportion across valid items.
- Nominal Krippendorff's alpha ($\alpha$) with fixed label domain.
- Outputs summary to `results/subjectivity.csv`.
