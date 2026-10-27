# SynthText

**Does LLM-generated training data work worse the more subjective the task is?**
We re-test the finding of Li et al. (EMNLP 2023) on four datasets they did not use, with an
open LLM (Qwen3 8B via Ollama) generating zero-shot, few-shot and AugGPT-style synthetic
data, and a frozen Sentence-BERT + logistic-regression classifier trained on real,
synthetic and mixed data.

**Master plan (design, run matrix, timeline, decisions):** [SynthText — Master Plan](https://claude.ai/artifact/QPhCrDYnEJNbwqiL3KBNBK)
— ask Kausthuba to share it with you if the link doesn't open.

## Team

| Workstream | Owner | Folder | Branch |
| --- | --- | --- | --- |
| WS1 Generation | Kausthuba | [`workstreams/ws1_generation/`](workstreams/ws1_generation/) | `ws1-generation` |
| WS2 Data and training | Member B | [`workstreams/ws2_training/`](workstreams/ws2_training/) | `ws2-training` |
| WS3 Evaluation and subjectivity | Member C | [`workstreams/ws3_evaluation/`](workstreams/ws3_evaluation/) | `ws3-evaluation` |

Each folder has its own README with that person's checklist per sync point.
How we use branches and pull requests: [CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

```bash
git clone https://github.com/<owner>/<repo>.git && cd <repo>
git switch ws2-training                 # your own branch
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                  # 14 tests, all should pass
bash scripts/smoke_test.sh              # whole pipeline on fake sample data, < 1 minute
```

Python 3.10+ works; we use 3.11. WS1 also needs [Ollama](https://ollama.com) and `ollama pull qwen3:8b`.

## The pipeline, in order

All commands run from the repo root.

| Step | Who | Command | Writes |
| --- | --- | --- | --- |
| 1. Real data | WS2 | `python -m workstreams.ws2_training.prepare_data --task all` | `data/real/<task>/` |
| 2. Generate | WS1 | `python -m workstreams.ws1_generation.generate --task all --setting few_shot` (then `zero_shot`, `auggpt`) | `data/synthetic/<task>/raw/` |
| 3. Filter | WS1 | `python -m workstreams.ws1_generation.filter --task all --setting few_shot` (each setting) | `data/synthetic/<task>/<setting>.jsonl` |
| 4. Train + evaluate | WS2 | `python -m workstreams.ws2_training.run_matrix --task all` | `results/runs.csv`, `results/predictions/`, `results/fidelity/` |
| 5. Subjectivity | WS3 | `python -m workstreams.ws3_evaluation.subjectivity make-sheet` → annotate → `score`; `form` → survey → `survey` | `results/subjectivity.csv`, `results/item_agreement.csv` |
| 6. Quality metrics | WS3 | `python -m workstreams.ws3_evaluation.metrics --task all` | `results/quality.csv` |
| 7. Analysis | WS3 | `python -m workstreams.ws3_evaluation.analyze` | `results/summary.md`, `results/gaps.csv`, `results/correlation.json`, `reports/figures/` |

Every command has `--help`. Steps 2 and 4 are resumable: stop them and run the same command again.

## Working on samples

Until the real files exist, build and test your stage on the small **fake** fixtures in
`data/sample/` (contract-valid, not real text — never report numbers from them):

```bash
export SYNTHTEXT_DATA=data/sample SYNTHTEXT_RESULTS=results/sample SYNTHTEXT_FIGURES=results/sample/figures
export SYNTHTEXT_EMBEDDER=hashing     # offline embedder; skip it to test the real SBERT model
python -m workstreams.ws2_training.run_matrix --task trec --n-train 30
```

`--dry-run` on `generate` swaps the LLM for a fake one, so the generation code can be tested without a GPU.

## Layout

```
configs/            tasks.yaml · generation.yaml · experiments.yaml        (shared)
src/synthtext/      config, paths, JSONL I/O, data contracts, embeddings     (shared)
workstreams/        ws1_generation/ · ws2_training/ · ws3_evaluation/       (one per person)
data/sample/        fake fixtures (committed)
data/real/          rebuilt by prepare_data (git-ignored)
data/synthetic/     generated text (committed: it is the expensive part)
results/            runs.csv, quality.csv, subjectivity.csv, summaries (committed)
reports/            figures/, annotation/, report and slides drafts
scripts/            smoke_test.sh, make_sample_data.py
tests/              run in CI on every pull request
```

## Data contracts

| File | One row |
| --- | --- |
| `data/real/<task>/*.jsonl` | `{"id", "text", "label"}` — label as the string name from `configs/tasks.yaml` |
| `data/synthetic/<task>/<setting>.jsonl` | `{"id", "text", "label", "setting", "source_ids", "model", "prompt_version"}` |
| `results/runs.csv` | `task, condition, seed, n_train, macro_f1, accuracy, git_sha, timestamp` |

Defined in [`src/synthtext/contracts.py`](src/synthtext/contracts.py); `pytest` checks the sample files against them.

## Papers

1. Li, Zhu, Lu, Yin — *Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations*, EMNLP 2023 — [arXiv 2310.07849](https://arxiv.org/abs/2310.07849)
2. Dai et al. — *AugGPT: Leveraging ChatGPT for Text Data Augmentation*, 2023 — [arXiv 2302.13007](https://arxiv.org/abs/2302.13007)
3. Brown et al. — *Language Models are Few-Shot Learners*, NeurIPS 2020 — [arXiv 2005.14165](https://arxiv.org/abs/2005.14165)
4. Reimers & Gurevych — *Sentence-BERT*, EMNLP 2019 — [arXiv 1908.10084](https://arxiv.org/abs/1908.10084)

## Data licences

TREC (via `SetFit/TREC-QC`), Rotten Tomatoes, SST-5 (`SetFit/sst5`) and `dair-ai/emotion` are
downloaded from Hugging Face by `prepare_data` and are not stored in this repo.
`dair-ai/emotion` is for educational and research use only; TREC's licence is listed as unknown.
