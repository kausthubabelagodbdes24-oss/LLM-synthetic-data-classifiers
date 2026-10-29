# LLM-synthetic-data-classifiers
 An open LLM is used to generate synthetic training examples in zero-shot and few-shot (example-guided) settings. Classifiers are then trained on (a) real data only, (b) synthetic data only, and (c) a mix of real and synthetic data at different ratios, and evaluated on a held-out real test set.

## Working with AI models

Every model must read [AGENTS.md](AGENTS.md) and the [shared implementation protocol](docs/agents/IMPLEMENTATION_PROTOCOL.md). Copy the [reusable issue prompt](docs/agents/ISSUE_PROMPT.md), replace ISSUE_NUMBER, and provide repository access. Claude and Gemini entry files route to the same instructions. Consult the [shared contract register](docs/agents/CONTRACTS.md) before cross-workstream interface changes. Models must use isolated issue branches and PRs; human owners retain accountability.

## Approved plan

Read the [SynthText project specification](docs/PROJECT_SPEC.md) for the experiment design, responsibilities, schedule, checkpoint checks and deliverables. The original master plan and three manuals are preserved in [source plans](docs/source-plans/).

The published version 2 specification, [glossary](CONTEXT.md) and [decision records](docs/adr/) are the current plan. This restoration supplies the **legacy pipeline**, not a completed version 2 experiment. Three separate generation repetitions, strict dataset gates, independent annotation collection and paired repetition-level gap analysis still require later workstream issues. The legacy scripts reuse synthetic files across seeds, accept undersized inputs, expose shared annotation columns and calculate gaps from aggregate scores. Do not use those behaviors as evidence of compliance or report fake-data scores as research results.

## Contributors and branches

| Owner | Workstream | Folder | Branch |
| --- | --- | --- | --- |
| @harshitadjainbdes24-byte | Generation | `ws1_generation/` | `ws1-generation` |
| @kausthubabelagodbdes24-oss | Training and project lead | `ws2_training/` | `ws2-training` |
| @hem4456 | Evaluation | `ws3_evaluation/` | `ws3-evaluation` |

At the start of a session, commit or safely preserve local work, then run `git fetch origin` and `git merge origin/main` on your workstream branch. Open pull requests into `main`; merge with a merge commit after CI passes. Coordinate changes to `config.yaml` and `common.py` with the project lead and group. Do not force-push, delete branches or tag releases without explicit approval.

## Setup: Python 3.11

Install Git and Python **3.11** first. Clone using your authenticated GitHub account (this repository requires access). Run commands from the repository root. Replace `ws2-training` below with your workstream branch.

Windows PowerShell (activation is unnecessary, so no execution-policy change is needed):

```powershell
git clone https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers.git
cd LLM-synthetic-data-classifiers
git switch ws2-training
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest
```

If `py` is unavailable, use the full path to an installed Python 3.11 executable to create the environment. The Windows Store `python` alias alone is not evidence that Python 3.11 is installed.

macOS / Linux:

```bash
git clone https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers.git
cd LLM-synthetic-data-classifiers
git switch ws2-training
python3.11 -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest
```

Expected: **2 passed** (fake-data pipeline and model-reply parser). The pipeline test uses temporary directories, fake generation and hashing embeddings; it needs no Ollama service, model download or real dataset. CI runs these same tests on Python 3.11 on Windows and Linux. Installing dependencies initially requires internet access.

`requirements-test.txt` contains the lightweight subset needed by these tests. For real-data preparation and actual sentence embeddings, install the supplied full dependencies in the same environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
```

The full installation includes `datasets`, `sentence-transformers` and their dependencies. Dependency files specify minimum versions, not a reproducible lock of all packages. Initial dataset and embedding-model downloads also require internet access. The generation owner separately installs [Ollama](https://ollama.com) and downloads `llama3.2:3b`; model availability and quality must be confirmed on that computer. Passing tests does not establish real-data or real-model readiness.

### Real-data preparation and verification

Once full dependencies are installed, prepare the reproducible real datasets and low-resource source samples:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m ws2_training.prepare_data
```

```bash
# macOS / Linux
.venv/bin/python -m ws2_training.prepare_data
```

To verify existing real data files against the recorded SHA-256 checksums and exact line counts without redownloading:

```bash
python -m ws2_training.prepare_data --verify
```

## Optional manual fake-data smoke run

The committed fixtures in `data/sample/real/` are deliberately fake. Run **all three** settings before training. The commands below keep generated fixtures and outputs in ignored sample directories and leave `config.yaml` unchanged.

Windows PowerShell:

```powershell
$env:SYNTHTEXT_DATA = 'data/sample'
$env:SYNTHTEXT_RESULTS = 'results/sample'
$env:SYNTHTEXT_EMBEDDER = 'hashing'
$env:PYTHONUTF8 = '1'
try {
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting zero_shot --dry-run
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting few_shot --dry-run
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting auggpt --dry-run
    .\.venv\Scripts\python.exe -m ws2_training.train
    .\.venv\Scripts\python.exe -m ws3_evaluation.analyze
} finally {
    Remove-Item Env:SYNTHTEXT_DATA, Env:SYNTHTEXT_RESULTS, Env:SYNTHTEXT_EMBEDDER, Env:PYTHONUTF8 -ErrorAction SilentlyContinue
}
```

macOS / Linux (a subshell keeps these variables out of later real runs):

```bash
(
export SYNTHTEXT_DATA=data/sample
export SYNTHTEXT_RESULTS=results/sample
export SYNTHTEXT_EMBEDDER=hashing
export PYTHONUTF8=1
.venv/bin/python -m ws1_generation.generate --setting zero_shot --dry-run
.venv/bin/python -m ws1_generation.generate --setting few_shot --dry-run
.venv/bin/python -m ws1_generation.generate --setting auggpt --dry-run
.venv/bin/python -m ws2_training.train
.venv/bin/python -m ws3_evaluation.analyze
)
```

Expected outputs: synthetic fixture JSONL files in `data/sample/synthetic/`, plus `runs.csv` (36 legacy task/condition/seed rows), `summary.md`, `f1_by_condition.png` and `gap.png` in `results/sample/`. Resume skips existing generated rows and run keys. Fake scores have no scientific meaning; sample sets intentionally do not meet the planned experimental sizes.

## Files and remaining setup confirmations

- `config.yaml`, `common.py`: supplied settings and shared helpers, restored unchanged.
- `ws1_generation/`, `ws2_training/`, `ws3_evaluation/`: generation, preparation/training and annotation/analysis scripts. Their module help describes legacy commands; use the current specification and follow-up issues before real experiments.
- `tests/`, `data/sample/real/`: existing tests and fake fixtures.
- `data/real/`, local environments, credentials, generated sample files and results are ignored. Later result handoffs require explicit review and addition of the accepted owned files. Synthetic experiment artifacts are subject to the later acceptance/freeze process; do not add unreviewed artifacts.

Each contributor must separately confirm Python 3.11 and the two tests on their own computer; the generation owner must also confirm Ollama/model setup. Record actual confirmations on [issue #4](https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/4). CI is automated evidence only. Check collaborator write access separately, and leave the issue open while required human confirmations are missing.

The supplied bundle history is retained through a merge with the existing GitHub history. Published planning documents and the license are preserved. Real datasets, secrets, environments and experimental results must not be committed.
