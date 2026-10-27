# WS1 — Generation (Kausthuba)

You own every synthetic text in the project: the prompts, the generation runs and the
filter. Branch: `ws1-generation`.

| File | What it does |
| --- | --- |
| `llm_client.py` | Ollama client (default), OpenAI-compatible client (vLLM / Kaggle / hosted), fake client for tests |
| `prompts.py` | Zero-shot, few-shot and AugGPT prompt templates + a forgiving JSON parser |
| `generate.py` | Resumable generation → `data/synthetic/<task>/raw/<setting>.jsonl` (+ `.stats.json`) |
| `filter.py` | Cleans raw files → `data/synthetic/<task>/<setting>.jsonl` (+ `.filter.json`) |

## Setup (once)

```bash
ollama pull qwen3:8b          # ~5.2 GB; fits an 8 GB GPU at 4-bit
ollama serve                  # if it isn't already running
python -m workstreams.ws1_generation.generate --task trec --setting few_shot --pilot --labels human
```
Record the model digest that `generate` prints: it goes in the report.

## Checklist

**By Oct 29**
- [ ] Ollama running; one pilot call per task works
- [ ] Read 20 pilot texts per label for each task and setting; tweak `prompts.py`, then bump `prompt_version`
- [ ] Commit the pilot raw + filtered files; post the per-label counts and failure rate in the group

**By Nov 1**
- [ ] Few-shot generation complete for all 4 tasks (500 per label) — run it overnight
- [ ] Zero-shot started

**By Nov 3 (freeze)**
- [ ] Zero-shot complete; AugGPT complete (needs WS2's `lowres_seed*.jsonl`)
- [ ] All three settings filtered; `*.filter.json` reviewed; files committed
- [ ] No more regenerating after this

**By Nov 8**
- [ ] Report: related work, generation method, prompts and example outputs (appendix)

## Commands

```bash
python -m workstreams.ws1_generation.generate --task all --setting few_shot    # then zero_shot, auggpt
python -m workstreams.ws1_generation.filter   --task all --setting few_shot
python -m workstreams.ws1_generation.generate --task all --setting few_shot --model llama3.1:8b  # backup model
```
No GPU handy? Set `backend: openai_compat` and `openai_base_url` in `configs/generation.yaml`
to use a vLLM server on Kaggle/Colab with the same model.
