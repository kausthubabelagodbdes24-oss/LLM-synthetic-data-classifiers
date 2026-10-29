# SynthText Manual 3 — Evaluation (Hementh)

As of 29 Oct 2026 · Project lead: Kausthuba Belagod

## Your job

You turn the results into findings, and you assemble and finish the report. There are two pieces of analysis:

1. **Subjectivity check.** The three of you label the same 60 test items (30 per task) without seeing the answers. Krippendorff's α measures how much you agree. Low agreement means a subjective task; we expect Emotion to be lower than TREC.
2. **Results.** From Kausthuba's 36 results you make one table and two charts that answer the question: does training on synthetic data cost more on Emotion than on TREC?

You're the last link in the main chain: make the charts as soon as Kausthuba's results are in `main`. Kausthuba owns the repo and leads the project; ask him if you're stuck.

| You hand over | To | At |
| --- | --- | --- |
| Annotation sheet shared as a Google Sheet | Kausthuba, Harshita | Stage 2 (they label it in stage 3) |
| Shared report doc with the section headings | Kausthuba, Harshita | Stage 2 |
| `results/subjectivity.csv` (α per task) merged into `main` | Team | Checkpoint 2 |
| Table and charts merged into `main`; key numbers posted in the group chat | Team (Harshita needs them for section 6) | Stage 4 |
| Full report assembled | Team | Checkpoint 3 |
| Final report PDF, plus your slides | Kausthuba (submits) | Stage 5 |

| You need | From | At |
| --- | --- | --- |
| The repo invite | Kausthuba | Stage 1 |
| Everyone's 60 labels in the sheet | Kausthuba, Harshita | Stage 3 |
| All 36 results in `main` | Kausthuba | Stage 4 |
| Everyone's report sections | Kausthuba, Harshita | Checkpoint 3 |

## How the three manuals fit together

This section is the same in all three manuals. The project runs in five stages, and each person's steps below are grouped the same way. At each of the three **checkpoints**, everyone merges their work into `main`, then Kausthuba runs a 15-minute call.

| Stage | Harshita (generation) | Kausthuba (data, training, project lead) | Hementh (evaluation) |
| --- | --- | --- | --- |
| **1. Setup** | Accept the invite; set up Python; install Ollama and the model | Create the GitHub repo and invite Harshita and Hementh (do this first, because they're waiting on it); set up Python | Accept the invite; set up Python |
| **2. Independent work** | Download the data; pilot run; start full zero-shot and few-shot generation (runs unattended) | Download the data; run the real-data results (C1 and C5a) | Download the data; make and share the 60-item annotation sheet; create the shared report doc |
| **Checkpoint 1** | Pilot texts look sensible | C1 and C5a numbers exist | Sheet and report doc shared |
| **3. Labels and synthetic data** | Label the 60 items; run AugGPT; merge all synthetic data | Label the 60 items; review Harshita's data; draft sections 3b and 4 | Label the 60 items; score the agreement; draft section 1 |
| **Checkpoint 2** | Synthetic data is in `main` and frozen | Starts the full run straight after | Agreement and α per task known |
| **4. Results** | Write sections 2, 3a and 6 and the appendix | Run all 36 results and merge them | Make the table and charts as soon as Kausthuba's results land; write section 5; assemble the report |
| **Checkpoint 3** | Sections in | Sections in | Results final; full report assembled |
| **5. Finish** | Your slides; read the report | Combine everyone's slides into one deck; read the report; submit; tag `v1.0` | Your slides; final edit of the report |

### Workload

| Person | Hands-on work | Computer running unattended |
| --- | --- | --- |
| Harshita | about 6 hours | roughly 2 hours of generation on a laptop GPU, or overnight on a CPU |
| Kausthuba | about 6 hours | under 1 hour of training |
| Hementh | about 6 hours | — |

That fits before the deadline with room to spare, as long as one chain doesn't stall: Harshita's synthetic data, then Kausthuba's full run, then Hementh's charts. Everyone posts in the group chat the moment they hand something over or get stuck; whoever is waiting drafts their report sections in the meantime.

## Stage 1: setup (about 45 minutes)

Most of it is waiting for downloads.

### 1. Install the tools (skip any you already have)

| Tool | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Git | `winget install Git.Git` | `xcode-select --install` | `sudo apt install git` (Fedora: `sudo dnf install git`) |
| GitHub CLI | `winget install GitHub.cli` | `brew install gh` | `sudo apt install gh` (Fedora: `sudo dnf install gh`) |
| Python **3.11** | [python.org](https://www.python.org/downloads/) installer; tick **Add to PATH** | `brew install python@3.11` | `sudo apt install python3.11 python3.11-venv` |

Use 3.11, not the newest Python; the machine-learning libraries lag behind new releases.

### 2. Get the repo

1. Accept Kausthuba's invite: check your email, or github.com → Notifications.
2. Log in and clone. Replace `<owner>` with Kausthuba's GitHub username:

```bash
gh auth login                      # GitHub.com → HTTPS → log in with a browser
gh repo clone <owner>/synthtext
cd synthtext
git switch ws3-evaluation          # your branch, for the whole project
```

### 3. Set up Python

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # every time you open a new terminal
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1         # every time; if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then, on any system:

```bash
pip install -r requirements.txt    # a few minutes; it downloads PyTorch
pytest
```

**Check:** the last line says `2 passed`. Tell the group chat you're set up.

## Every session starts the same way

From the `synthtext` folder:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
git switch ws3-evaluation
git pull origin main               # brings in everyone's merged work
```

## Stage 2: the annotation sheet and the report doc (about 1 hour)

- [ ] **Download the real data** (a few minutes; needs internet):

```bash
python -m ws2_training.prepare_data
```

- [ ] **Make the annotation sheet:**

```bash
python -m ws3_evaluation.annotate make
```

Check: it prints `wrote 60 items to …/results/annotation_sheet.csv`.

- [ ] **Turn it into a Google Sheet.** At sheets.google.com: **Blank → File → Import → Upload** `annotation_sheet.csv` → **Replace spreadsheet**. Name it "SynthText annotation".
    - Don't rename or reorder the columns. The three empty ones (`kausthuba`, `harshita`, `hementh`) are where each person types.
    - Each row's `choose_one_of` column lists the allowed labels. Tell everyone to copy one exactly as written.
    - The correct answers are deliberately not in the sheet.
- [ ] **Share it** with Kausthuba and Harshita (Share → Editor) and post the link in the group chat. This is the first thing they need from you, so do it early.
- [ ] **Make the shared report doc:** a Google Doc called "SynthText report" with these headings, shared with Kausthuba and Harshita as editors:
    1. Introduction (Hementh)
    2. Related work (Harshita)
    3. Tasks, model and generation: 3a model and generation (Harshita), 3b tasks and data (Kausthuba)
    4. Classifier and conditions (Kausthuba)
    5. Results (Hementh)
    6. Limitations and conclusion (Harshita)
    7. Appendix A: prompts and examples (Harshita)
- [ ] Commit the sheet so the repo records exactly which items were labelled:

```bash
git add results/annotation_sheet.csv
git commit -m "ws3: annotation sheet"
git push origin ws3-evaluation
gh pr create --base main --head ws3-evaluation --title "Annotation sheet" --body "60 items for the subjectivity check."
```

Once the CI check is green, merge it: `gh pr merge --merge`, or the green button on GitHub.

- [ ] **Checkpoint 1:** confirm the sheet and the report doc are shared.

## Stage 3: label, score and draft (about 1½ hours)

- [ ] **Label your 60 items** in the `hementh` column. Go by your own judgement and don't look anything up. About 15 minutes.
- [ ] Draft section 1 (see "Your report sections" below).
- [ ] When all three columns are full (nudge anyone who hasn't finished): **File → Download → Comma-separated values (.csv)**. Rename the file to exactly `annotation_filled.csv` and move it into the repo's `results/` folder.
- [ ] Score it:

```bash
python -m ws3_evaluation.annotate score
```

Check that it prints one line per task, for example `{'task': 'trec', 'items': 30, 'annotators': 3, 'mean_agreement': 0.9, 'alpha': 0.8}`.

- `items` should be 30. If it's lower, some labels were mistyped: fix them in the sheet, download again, rerun.
- We expect Emotion's α to be clearly lower than TREC's. If it isn't, that's a finding to report, not a bug.

- [ ] Commit `results/annotation_filled.csv` and `results/subjectivity.csv`, push, open a pull request and merge it (same commands as stage 2, with the message `ws3: subjectivity scores`).
- [ ] **Checkpoint 2:** share the two α values.

## Stage 4: results and assembly (about 2½ hours)

- [ ] As soon as Kausthuba posts **"results are in main"**:

```bash
git pull origin main
python -m ws3_evaluation.analyze
```

This writes `results/summary.md` (the macro-F1 table and the gaps), `results/f1_by_condition.png` and `results/gap.png`. Open the two PNGs and check they're readable.

- [ ] Commit those three files, then push, open a pull request and merge it.
- [ ] **Post the key numbers in the group chat** (the two gaps, and C4 vs C1), so Harshita can finish the conclusion.
- [ ] **Write section 5** (see below).
- [ ] Paste everyone's sections into one flow, make the headings and numbering consistent, and fix the reference list. **Checkpoint 3:** full draft ready for everyone to read.

## Stage 5: slides and final edit (about 1 hour)

- [ ] Make your 3 slides (subjectivity check; results × 2) and send them to Kausthuba, who builds the combined deck.
- [ ] Final edit of the report once Kausthuba and Harshita have read it. Export it as a PDF and send it to Kausthuba to submit.

## Git rules

1. Work only on `ws3-evaluation`, and only change `ws3_evaluation/` and your files in `results/`.
2. Start each session with `git pull origin main`.
3. To get work into `main`: push, open a pull request, and merge it yourself once CI is green.
4. Changing `config.yaml` or `common.py`? Post it in the group chat first, because everyone's code reads them.

| Files | Written only by |
| --- | --- |
| `data/synthetic/` | Harshita |
| `results/runs.csv` | Kausthuba |
| `results/annotation_*.csv`, `results/subjectivity.csv`, `results/summary.md`, `results/*.png` | Hementh (you) |
| `data/real/` | Nobody commits it (each person rebuilds it) |

## Your report sections

| Section | Length | What to put in it |
| --- | --- | --- |
| 1. Introduction | about ½ page | Labelled data is slow and costly to collect. LLMs can write training examples instead, but Li et al. (2023) found this works worse on subjective tasks. Our question and hypothesis, word for word from the master plan. What we did, in two sentences. |
| 5. Results | 1–1½ pages | (a) The subjectivity check: α and mean agreement per task from `subjectivity.csv`, in a small table. (b) The macro-F1 table from `summary.md`, plus the bar chart. (c) The gap chart and the gap numbers. Then answer four questions in plain sentences: Is the gap more negative for Emotion than for TREC? Does few-shot shrink it compared with zero-shot? How close does the 50/50 mix (C4) get to real-only (C1)? Does AugGPT help when real data is scarce (C5b vs C5a)? Report what the numbers say even if the hypothesis fails. |

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: common` | Run commands from the `synthtext` folder, always with `python -m …` |
| `score` fails with a file error | The file must be exactly `results/annotation_filled.csv`; Google adds " - Sheet1" to the name, so rename it |
| `score` fails with a column error | A column was renamed in the sheet. Restore the headers `item_id, task, text, choose_one_of, kausthuba, harshita, hementh` |
| `items` below 30 | Labels mistyped or missing. Fix them in the sheet and download again |
| Someone can't finish labelling | Score with the columns that are full (it works with 2), and say so in the report |
| `analyze`: no `runs.csv`, or conditions missing | Kausthuba's results aren't in your branch: `git pull origin main`. If they aren't in `main` either, ask him |
| Kausthuba's results are late | Write section 5's structure and the subjectivity part first; drop the numbers in when they arrive |
| `git push` is rejected | `git pull origin ws3-evaluation`, then push again |
