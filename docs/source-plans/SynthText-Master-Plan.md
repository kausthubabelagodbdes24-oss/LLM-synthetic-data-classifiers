# SynthText — Master Plan

LLM-generated synthetic data for training text classifiers · AI & ML minor project

As of 29 Oct 2026 · Project lead: Kausthuba Belagod

## At a glance

We submit on Wed 4 Nov 2026. To make that with little time, this is a bare-bones version: two tasks, a 3B model running offline on a laptop, six training conditions, and about three short work sessions each.

**Step-by-step manuals, one each:** SynthText Manual 1 — Generation (Harshita) · SynthText Manual 2 — Data, Training and Project Lead (Kausthuba) · SynthText Manual 3 — Evaluation (Hementh). All three share the same five stages, three checkpoints and workload table. If any of that changes, change it in all three.

**Question.** Does a classifier trained on LLM-written data fall further behind real data on a subjective task (emotion) than on an objective one (question type)?

**Hypothesis (from Li et al. 2023).** Training on synthetic data costs more macro-F1 on Emotion than on TREC. Few-shot generation closes some of that gap.

**Done means**

- [ ] 2 tasks × 6 conditions × 3 seeds = 36 results in `results/runs.csv`
- [ ] One table and two charts in the report
- [ ] Report and slides submitted on 4 Nov

**Cut from the first plan:** the other two datasets, the classmate survey, the diversity and fidelity metrics, the ceiling run, and the 8B model.

## Design

### Tasks

Both tasks have 6 labels, so the only thing that differs between them is how subjective they are.

| Task | Hugging Face ID | Labels | Subjectivity |
| --- | --- | --- | --- |
| TREC question type | [SetFit/TREC-QC](https://huggingface.co/datasets/SetFit/TREC-QC) | abbreviation, entity, description, human, location, numeric | Low |
| Emotion in tweets | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | sadness, joy, love, anger, fear, surprise | High |

### Model: small and offline

[Llama 3.2 3B](https://ollama.com/library/llama3.2) in Ollama (`llama3.2:3b`, a 2.0 GB download). After that one download it runs with no internet, no API and no cost. It is fast on a laptop GPU and still workable on a CPU (slower, so start it before bed). Backup: [`qwen3:4b`](https://ollama.com/library/qwen3) (2.5 GB).

### Generation

| Setting | Prompt | Amount |
| --- | --- | --- |
| Zero-shot | Task, label and its meaning, style, word limit | 100 per label |
| Few-shot | Same, plus 3 real examples of that label (Brown et al.) | 100 per label |
| AugGPT | "Rephrase this sentence 6 ways" on each low-resource real example (Dai et al.) | 6 per example |

About 4,500 texts from roughly 600 model calls. AugGPT runs on the low-resource examples of all three seeds. As a rough guess, that is an evening on a laptop GPU, or overnight on a CPU. Exact duplicates and unparseable replies are dropped.

### Training conditions

Each run has 100 examples per label (600 in total), except the low-resource pair.

| ID | Training data |
| --- | --- |
| C1 | Real only |
| C2 | Zero-shot synthetic only |
| C3 | Few-shot synthetic only |
| C4 | 50% real + 50% few-shot synthetic |
| C5a | Real, 10 per label only |
| C5b | C5a + 6 AugGPT paraphrases of each example |

**Classifier.** Frozen [`all-MiniLM-L6-v2`](https://arxiv.org/pdf/1908.10084) sentence embeddings, then logistic regression. It trains in seconds on a CPU. Seeds 13, 42 and 7.

**Measure.** Macro-F1 on the real test set. Gap = (F1 synthetic − F1 real) / F1 real, compared between the two tasks.

**Subjectivity check.** All three of us label the same 30 test items per task (60 items, about 15 minutes each) without seeing the answers. Krippendorff's α per task confirms that Emotion really is the more subjective one.

## Who does what

| Person | Owns | Runs | Writes in the report |
| --- | --- | --- | --- |
| Harshita | Generation (`ws1_generation/`) | `generate.py` for all three settings | Related work, generation method, limitations and conclusion, appendix |
| Kausthuba | Project lead and repo owner; data and training (`ws2_training/`) | `prepare_data.py`, then `train.py` | Tasks, classifier, conditions |
| Hementh | Evaluation (`ws3_evaluation/`) | `annotate.py`, then `analyze.py` | Introduction, subjectivity check, results; puts the report together |

All three label the 60 items and present their own part.

### Repo

One repo, one folder and one branch per person (`ws1-generation`, `ws2-training`, `ws3-evaluation`). Open a pull request into `main` at each checkpoint. The whole pipeline is 6 short Python files plus one `config.yaml`.

```
synthtext/
├── config.yaml          tasks, model, amounts, conditions
├── common.py            load config, read/write JSONL, embed
├── ws1_generation/      generate.py
├── ws2_training/        prepare_data.py · train.py
├── ws3_evaluation/      annotate.py · analyze.py
├── data/                real/ (rebuilt) · synthetic/ (committed) · sample/ (fake fixtures)
└── results/             runs.csv · summary.md · figures
```

Each file reads and writes simple JSONL rows: `{"id", "text", "label"}`, plus `"setting"` and `"source_ids"` on synthetic rows. Until the real files exist, everyone tests on `data/sample/`.

## Timeline

Setup tonight, experiments Fri to Mon, the report on Mon and Tue, and submission on Wed 4 Nov.

| Who | What | When |
| --- | --- | --- |
| Harshita | Setup: invite, Ollama, model | Thu 29 Oct |
| Harshita | Pilot, then full generation | Fri 30 – Sat 31 Oct |
| Harshita | Merge synthetic data, write sections | Sun 1 – Mon 2 Nov |
| Kausthuba | Create repo, set up, data, C1 and C5a | Thu 29 – Fri 30 Oct |
| Kausthuba | Label items, draft sections | Sat 31 Oct |
| Kausthuba | All 36 results, write sections | Sun 1 – Mon 2 Nov |
| Hementh | Setup, annotation sheet, report doc | Thu 29 – Fri 30 Oct |
| Hementh | Label, score agreement | Sat 31 Oct – Sun 1 Nov |
| Hementh | Table, charts, assemble report | Mon 2 Nov |
| All three | Slides and final edit | Tue 3 Nov |

| Checkpoint | When | Gate |
| --- | --- | --- |
| 1 | Fri 30 Oct, 9 pm | Data and pilot ready |
| 2 | Sun 1 Nov, 6 pm | Synthetic data frozen |
| 3 | Mon 2 Nov, 9 pm | Results and full draft |
| Submit | Wed 4 Nov | Report and slides in |

At each checkpoint: a 15-minute call run by Kausthuba, after everyone merges their branch into `main`. After Sun 1 Nov, 6 pm, nobody regenerates data.

## Report, risks and open questions

### Report outline (6–8 pages)

1. Introduction and question (Hementh)
2. The four papers (Harshita)
3. Tasks, model and generation (Harshita, Kausthuba)
4. Classifier and conditions (Kausthuba)
5. Results: F1 table, gap chart, α per task (Hementh)
6. Limitations and conclusion (Harshita)

### Risks

| Risk | Fallback |
| --- | --- |
| Generation too slow on a CPU | Run on the laptop GPU, or drop to 50 per label |
| The 3B model returns broken JSON | The script retries 3 times, then reads it line by line; report the failure rate |
| Someone falls behind | The sample data lets the others keep going; the owner pairs with a teammate for one evening |

### Open questions

- [ ] Harshita's and Hementh's GitHub usernames
- [ ] Report length and format the professor expects

## Sources

- Papers: [Li et al. 2023](https://arxiv.org/pdf/2310.07849) · [Dai et al. 2023 (AugGPT)](https://arxiv.org/pdf/2302.13007) · [Brown et al. 2020](https://arxiv.org/pdf/2005.14165) · [Reimers & Gurevych 2019](https://arxiv.org/pdf/1908.10084)
- Data: [SetFit/TREC-QC](https://huggingface.co/datasets/SetFit/TREC-QC) · [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (research and educational use only)
- Models: [Ollama llama3.2](https://ollama.com/library/llama3.2) · [Ollama qwen3](https://ollama.com/library/qwen3), checked 29 Oct 2026
