# WS3 — Evaluation and subjectivity (Member C)

You own the subjectivity study, the data-quality metrics, the analysis and the figures,
and you lead the report. Branch: `ws3-evaluation`.

| File | What it does |
| --- | --- |
| `subjectivity.py` | Survey questions (`form`), survey scoring (`survey`), annotation sheet (`make-sheet`), agreement and Krippendorff's α (`score`) |
| `metrics.py` | Diversity (Remote-Clique, Chamfer), faithfulness, label fidelity, duplicates → `results/quality.csv` |
| `analyze.py` | Summary tables, gaps, Spearman ρ for H1, all report figures |

## Checklist

**By Oct 29**
- [ ] `subjectivity form` → build the Google Form (6 questions) and send it to the team + at least 7 classmates
- [ ] `subjectivity make-sheet` (after WS2's `prepare_data`) → upload `reports/annotation/annotation_sheet.csv` to a Google Sheet; one `ann_<name>` column each

**By Nov 1**
- [ ] All 160 items annotated by all three of us (don't look at `annotation_key.csv` before that)
- [ ] Download the sheet as `reports/annotation/annotation_filled.csv`; run `score`; download the Form responses as `reports/annotation/survey.csv`; run `survey`

**By Nov 5**
- [ ] `metrics --task all` on the frozen synthetic data
- [ ] `analyze` → figures and tables drafted

**By Nov 8**
- [ ] Report lead: introduction, subjectivity method, results; assemble the full report

## Commands

```bash
python -m workstreams.ws3_evaluation.subjectivity form
python -m workstreams.ws3_evaluation.subjectivity make-sheet --annotators kausthuba,member_b,member_c
python -m workstreams.ws3_evaluation.subjectivity score --sheet reports/annotation/annotation_filled.csv
python -m workstreams.ws3_evaluation.subjectivity survey --csv reports/annotation/survey.csv
python -m workstreams.ws3_evaluation.metrics --task all
python -m workstreams.ws3_evaluation.analyze
```
Until real results exist, run these on the sample data (see the main README, "Working on samples").
