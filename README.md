# SynthText

Does a classifier trained on LLM-written data fall further behind real data on a **subjective**
task (emotion) than on an **objective** one (question type)? A small re-test of Li et al. (EMNLP 2023)
using a 3B model that runs **offline** on a laptop. Due **4 Nov 2026**.

Plan, timeline and decisions: [SynthText — Master Plan](https://claude.ai/artifact/QPhCrDYnEJNbwqiL3KBNBK)

| Person | Folder | Branch | Runs |
| --- | --- | --- | --- |
| Kausthuba | `ws1_generation/` | `ws1-generation` | `generate.py` |
| Member B | `ws2_training/` | `ws2-training` | `prepare_data.py`, `train.py` |
| Member C | `ws3_evaluation/` | `ws3-evaluation` | `annotate.py`, `analyze.py` |

Work on your own branch and open a pull request into `main` at each sync point (Fri 30 Oct 9 pm, Sun 1 Nov 6 pm, Mon 2 Nov 9 pm).
Each person has a step-by-step manual; the links are in the master plan.

## Setup

```bash
git clone https://github.com/<owner>/<repo>.git && cd <repo>
git switch ws2-training                      # your branch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                       # 2 tests, should pass
```

Kausthuba only: install [Ollama](https://ollama.com) and run `ollama pull llama3.2:3b` (2.0 GB). After that, no internet is needed.

## Run it, in order (from the repo root)

```bash
python -m ws2_training.prepare_data                      # B: download data      -> data/real/
python -m ws1_generation.generate --setting few_shot     # K: then zero_shot, then auggpt -> data/synthetic/
python -m ws3_evaluation.annotate make                   # C: 60-item sheet; all three label it
python -m ws3_evaluation.annotate score                  # C: -> results/subjectivity.csv
python -m ws2_training.train                             # B: 36 runs            -> results/runs.csv
python -m ws3_evaluation.analyze                         # C: table + 2 charts   -> results/
```

Generation and training both resume where they stopped. Add `--pilot` to `generate` for a quick 10-per-label check.

## Before the real data exists

Test your part on the fake sample data (never report these numbers):

```bash
export SYNTHTEXT_DATA=data/sample SYNTHTEXT_RESULTS=results/sample SYNTHTEXT_EMBEDDER=hashing
python -m ws1_generation.generate --setting few_shot --pilot --dry-run   # fake model
python -m ws2_training.train
python -m ws3_evaluation.analyze
```

## Files

```
config.yaml       every setting: model, amounts, tasks, conditions
common.py         config, JSONL read/write, sampling, embeddings
ws1_generation/   generate.py   zero-shot, few-shot, AugGPT
ws2_training/     prepare_data.py · train.py
ws3_evaluation/   annotate.py · analyze.py
data/sample/      fake fixtures      data/real/ rebuilt (ignored)      data/synthetic/ committed
results/          runs.csv, summary.md, charts
```

Data rows are `{"id", "text", "label"}`; synthetic rows add `"setting"` and `"source_ids"`.

## Papers

1. Li et al., *Synthetic Data Generation with LLMs for Text Classification*, EMNLP 2023 — [arXiv 2310.07849](https://arxiv.org/abs/2310.07849)
2. Dai et al., *AugGPT*, 2023 — [arXiv 2302.13007](https://arxiv.org/abs/2302.13007)
3. Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020 — [arXiv 2005.14165](https://arxiv.org/abs/2005.14165)
4. Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019 — [arXiv 1908.10084](https://arxiv.org/abs/1908.10084)

Data: [SetFit/TREC-QC](https://huggingface.co/datasets/SetFit/TREC-QC), [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (research and educational use only).
