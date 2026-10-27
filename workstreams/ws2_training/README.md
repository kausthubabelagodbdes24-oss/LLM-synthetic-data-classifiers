# WS2 — Data and training (Member B)

You own the real data and every classifier run. Branch: `ws2-training`.

| File | What it does |
| --- | --- |
| `prepare_data.py` | Downloads the 4 datasets, maps labels, writes train / pool / test / lowres files |
| `sampling.py` | Deterministic stratified sampling helpers |
| `conditions.py` | Builds the training set for each run-matrix condition (C1–C5c, CEIL) |
| `classifier.py` | Frozen SBERT embeddings + logistic regression; macro-F1 and accuracy |
| `run_matrix.py` | Runs every (task, condition, seed) cell → `results/runs.csv` (resumable) |

The embedding cache lives in shared code (`src/synthtext/embeddings.py`); you maintain it.

## Checklist

**By Oct 29** (WS1 is waiting on the example pool)
- [ ] `prepare_data --task all` runs; label counts in each `manifest.json` look right
- [ ] Push the code (the data itself is git-ignored; everyone rebuilds it with the same seed)
- [ ] `run_matrix` works on the sample data

**By Nov 1**
- [ ] C1 and CEIL done on real data for all 4 tasks (they need no synthetic data)
- [ ] Embedding cache warm for every real text

**By Nov 5**
- [ ] Full run matrix on the frozen synthetic data: 4 tasks × 10 conditions × 3 seeds (CEIL once)
- [ ] `results/runs.csv`, `predictions/`, `fidelity/` committed

**By Nov 8**
- [ ] Report: tasks and data; classifier and training conditions

## Commands

```bash
python -m workstreams.ws2_training.prepare_data --task all
python -m workstreams.ws2_training.run_matrix --task all --conditions C1,CEIL     # before synthetic exists
python -m workstreams.ws2_training.run_matrix --task all                          # everything
python -m workstreams.ws2_training.run_matrix --task sst5 --conditions C3 --seeds 13 --force
```
