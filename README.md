# LLM-synthetic-data-classifiers
 An open LLM is used to generate synthetic training examples in zero-shot and few-shot (example-guided) settings. Classifiers are then trained on (a) real data only, (b) synthetic data only, and (c) a mix of real and synthetic data at different ratios, and evaluated on a held-out real test set.

## Current approved scope

**31 October 2026:** [ADR0012](docs/adr/0012-eod-exploratory-30-seed13.md) approves 30/label, seed13 only, both tasks and all six conditions. C1–C4=180/task; C4=15 real+15 few-shot/label; C5a/C5b=60/420. Twelve result keys. Independent-human annotation, distribution and external report sharing are removed.

Read [scope migration](docs/SCOPE_MIGRATION.md) before resuming old jobs or reusing results. Preserve original output/sidecars and use derived selections plus fresh result storage. Closed setup/preparation/pilot issues retain historical evidence, not proof the new experiment ran.

## Working with AI models

Every model must read [AGENTS.md](AGENTS.md) and the [shared implementation protocol](docs/agents/IMPLEMENTATION_PROTOCOL.md). Copy the [reusable issue prompt](docs/agents/ISSUE_PROMPT.md), replace ISSUE_NUMBER, and provide repository access. Claude and Gemini entry files route to the same instructions. Consult the [shared contract register](docs/agents/CONTRACTS.md) before cross-workstream interface changes. Models must use isolated issue branches and PRs; human owners retain accountability.

## Approved plan

Read the [SynthText project specification](docs/PROJECT_SPEC.md) for the experiment design, responsibilities, schedule, checkpoint checks and deliverables. The original master plan and three manuals are preserved in [source plans](docs/source-plans/).

Current specification/ADR0012 define intent. Main includes accepted preparation and Qwen pilot code, but downstream exact selection, training gates and single-seed analysis still need issue-specific integration. CI and fake-data scores are not experiment results.

## Ownership and separate workspaces

Kausthuba (@kausthubabelagodbdes24-oss) owns generation (`ws1_generation/`), training (`ws2_training/`) and evaluation (`ws3_evaluation/`). Keep workspaces and issue branches separate; one active implementer/issue. Fetch and normally merge main, preserving local changes and active jobs. This revision's config changes are lead-approved. Do not force-push, delete branches, stop another job or tag releases without authorization.

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

Expected: all current tests pass (the suite has expanded beyond the original two tests). The pipeline test uses temporary directories, fake generation and hashing embeddings; it needs no Ollama service, model download or real dataset. CI runs these same tests on Python 3.11 on Windows and Linux. Installing dependencies initially requires internet access.

`requirements-test.txt` contains the lightweight subset needed by these tests. For real-data preparation and actual sentence embeddings, install the supplied full dependencies in the same environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
```

The full installation includes `datasets`, `sentence-transformers` and their dependencies. Dependency files specify minimum versions, not a reproducible lock of all packages. Initial dataset and embedding-model downloads also require internet access. The generation owner separately installs [Ollama](https://ollama.com) and uses the accepted `qwen3:4b` digest in ADR0012; model availability and quality must be confirmed on that computer. Passing tests does not establish real-data or real-model readiness.

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
$env:SYNTHTEXT_GENERATION_MODE = 'fake'
$env:PYTHONUTF8 = '1'
try {
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting zero_shot --dry-run
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting few_shot --dry-run
    .\.venv\Scripts\python.exe -m ws1_generation.generate --setting auggpt --dry-run
    .\.venv\Scripts\python.exe -m ws2_training.train
    .\.venv\Scripts\python.exe -m ws3_evaluation.analyze
} finally {
    Remove-Item Env:SYNTHTEXT_DATA, Env:SYNTHTEXT_RESULTS, Env:SYNTHTEXT_EMBEDDER, Env:SYNTHTEXT_GENERATION_MODE, Env:PYTHONUTF8 -ErrorAction SilentlyContinue
}
```

macOS / Linux (a subshell keeps these variables out of later real runs):

```bash
(
export SYNTHTEXT_DATA=data/sample
export SYNTHTEXT_RESULTS=results/sample
export SYNTHTEXT_EMBEDDER=hashing
export SYNTHTEXT_GENERATION_MODE=fake
export PYTHONUTF8=1
.venv/bin/python -m ws1_generation.generate --setting zero_shot --dry-run
.venv/bin/python -m ws1_generation.generate --setting few_shot --dry-run
.venv/bin/python -m ws1_generation.generate --setting auggpt --dry-run
.venv/bin/python -m ws2_training.train
.venv/bin/python -m ws3_evaluation.analyze
unset SYNTHTEXT_DATA SYNTHTEXT_RESULTS SYNTHTEXT_EMBEDDER SYNTHTEXT_GENERATION_MODE PYTHONUTF8
)
```

Expected outputs: synthetic fixture JSONL files in `data/sample/synthetic-fixtures/`, plus `runs.csv` (12 configured task/condition/seed rows; fixture-only smoke evidence), `summary.md`, `f1_by_condition.png` and `gap.png` in `results/sample/`. The fake pipeline explicitly opts into fixture storage; real consumers default to `data/synthetic/`. Resume skips only compatible existing rows and run keys. Fake scores have no scientific meaning; sample sets intentionally do not meet the planned experimental sizes.

### Freeze-bound twelve-key grid (issue #13)

The real scope30-v1 seed-13 grid is trained only by `ws2_training.train_scope30`. Before it builds or fits anything, the accepted freeze (`docs/freeze/scope30-v1-seed13.freeze.json`) must verify. The legacy `ws2_training.train` refuses real C2/C3/C4/C5b. Point `SYNTHTEXT_DATA` at the accepted real data root, then run:

```bash
python -m ws2_training.train_scope30 --check-inputs                     # read-only; fits/writes nothing
python -m ws2_training.train_scope30 --results-dir <fresh dir>          # trains; resumes exact matches only
python -m ws2_training.train_scope30 --results-dir <fresh dir> --validate
```

Real runs need a clean checkout, `sentence-transformers` + `torch` and a locally cached MiniLM snapshot. `--dry-run` uses fake fixtures and is never evidence. See the #13 section of [contracts](docs/agents/CONTRACTS.md).

### Single-seed gap analysis (issue #14)

`ws3_evaluation.analyze_scope30` analyses the frozen grid that the lead accepted on #13. In real mode the grid's `grid_complete.json` SHA-256 is pinned. The legacy `ws3_evaluation.analyze` reports a mean ± SD over seeds and must not be used for scope30-v1.

```bash
python -m ws3_evaluation.analyze_scope30 --check --results-dir <grid dir>   # read-only; writes nothing
python -m ws3_evaluation.analyze_scope30 --results-dir <grid dir> [--out <fresh dir>]
```

It writes `summary.md`, `analysis.json`, `f1_by_condition.png` and `gap.png` once, to a fresh directory (default `docs/analysis/scope30-v1-seed13/`). Real analysis needs a clean checkout. Inspect both charts before reporting. See the #14 section of [contracts](docs/agents/CONTRACTS.md).

## Files and remaining setup confirmations

- `config.yaml`, `common.py`: shared settings/helpers; config records ADR0012 scope.
- `ws1_generation/`, `ws2_training/`, `ws3_evaluation/`: generation, preparation/training and annotation/analysis scripts. Their module help describes legacy commands; use the current specification and follow-up issues before real experiments.
- `tests/`, `data/sample/real/`: existing tests and fake fixtures.
- `data/real/`, local environments, credentials, generated sample files and results are ignored. Later result handoffs require explicit review and addition of the accepted owned files. Synthetic experiment artifacts are subject to the later acceptance/freeze process; do not add unreviewed artifacts.

Kausthuba verifies Python3.11, current tests and actual Ollama/model readiness on the working laptop. Historical setup evidence remains on issue #4. Teammate-computer confirmations are no longer gates; CI is automated evidence only.

The supplied bundle history is retained through a merge with the existing GitHub history. Published planning documents and the license are preserved. Real datasets, secrets, environments and experimental results must not be committed.
