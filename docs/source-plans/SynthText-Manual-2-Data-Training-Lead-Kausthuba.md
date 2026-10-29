# SynthText Manual 2 — Data, Training and Project Lead (Kausthuba)

As of 29 Oct 2026 · Project lead: Kausthuba Belagod

## Your job

You have two roles. **As project lead and repo owner**, you set up the repo, run the three checkpoint calls, keep the main chain moving, and combine the slides and submit. **As the data and training owner**, you prepare the real data and run every classifier: six kinds of training data, each tested on the same real test set.

| ID | Training data (per task) |
| --- | --- |
| C1 | Real only: 100 per label, 600 in total |
| C2 | Zero-shot synthetic only: 600 |
| C3 | Few-shot synthetic only: 600 |
| C4 | 50 real + 50 few-shot per label: 600 |
| C5a | Real, only 10 per label: 60 |
| C5b | C5a + 6 AugGPT rephrasings of each: about 420 |

That's 2 tasks (TREC and Emotion) × 6 conditions × 3 seeds = **36 results**. The code already does all of it; you run it at the right moment, check the output and explain it in the report. You're the middle link in the main chain: start the full run as soon as Harshita's synthetic data is in `main`, because Hementh makes the charts from your results.

| You hand over | To | At |
| --- | --- | --- |
| GitHub repo with all four branches; Harshita and Hementh invited | Harshita, Hementh | Start of stage 1 (they wait on this) |
| C1 and C5a results (real data only) in `main` | Team | Checkpoint 1 |
| All 36 results in `results/runs.csv`, merged into `main` | Hementh | Stage 4, as soon as you can |
| Report sections 3b and 4 | Hementh | Checkpoint 3 |
| One combined slide deck, and the submission | Professor | Stage 5 |

| You need | From | At |
| --- | --- | --- |
| The annotation sheet link | Hementh | Stage 3 |
| Synthetic data merged into `main` | Harshita | Checkpoint 2 |
| Everyone's slides | Harshita, Hementh | Stage 5 |
| The final report PDF | Hementh | Stage 5 |

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

## Your project-lead duties

- **Checkpoint calls.** Run each one in 15 minutes: one round of "done, next, blocked" per person, then confirm everyone's work is merged into `main` (the repo's **Pull requests** tab shows anything still open).
- **Watch the main chain.** If Harshita's generation looks slow after the pilot, act early: you can run the same commands on your laptop's GPU (see "If something goes wrong"). If your full run is delayed, tell Hementh so he can draft section 5's text first.
- **Guard the shared files.** `config.yaml` and `common.py` change only after a message in the group chat. After checkpoint 2, nobody regenerates synthetic data.
- **Finish.** Combine the slides, submit, and tag the final code (stage 5).

## Stage 1: set up the repo for everyone (about 1¼ hours)

Do steps 1 and 2 first, because Harshita and Hementh can't start until they have the invite.

### 1. Put the repo on GitHub

1. Download `synthtext.bundle` from the chat. It's the whole repo, history and branches included, in one file.
2. On github.com, create a **new private repository** called `synthtext`. Leave README, .gitignore and licence **unticked**, so it starts empty.
3. Install the GitHub CLI and log in, choosing GitHub.com, HTTPS, then log in with a browser:

```bash
sudo dnf install -y gh
gh auth login
```

4. Unpack the bundle and push all four branches. Replace `<you>` with your GitHub username:

```bash
cd ~/Downloads                      # wherever the bundle landed
git clone -b main synthtext.bundle ~/synthtext
cd ~/synthtext
for b in ws1-generation ws2-training ws3-evaluation; do git branch $b origin/$b; done
git remote set-url origin https://github.com/<you>/synthtext.git
git push -u origin main ws1-generation ws2-training ws3-evaluation
```

**Check:** the repo page on GitHub shows 4 branches and folders `ws1_generation`, `ws2_training`, `ws3_evaluation`.

### 2. Invite the team

1. Go to the repo's **Settings → Collaborators → Add people** and add Harshita and Hementh by GitHub username.
2. Post the repo link and their manuals in the group chat.
3. Optional, 2 minutes: once you know their GitHub usernames, replace the placeholders in `.github/CODEOWNERS` so GitHub suggests the right reviewer on each pull request.

### 3. Set up Python

Use Python 3.11. The newest Python that Fedora ships may be too new for the machine-learning libraries.

```bash
sudo dnf install -y python3.11
cd ~/synthtext
python3.11 -m venv .venv
source .venv/bin/activate          # do this every time you open a new terminal
pip install -r requirements.txt    # a few minutes; it downloads PyTorch
pytest
```

**Check:** the last line says `2 passed`.

### 4. Switch to your branch

```bash
git switch ws2-training
```

## Every session starts the same way

```bash
cd ~/synthtext && source .venv/bin/activate
git switch ws2-training && git pull origin main
```

## Stage 2: real data and the first results (about 30 minutes)

- [ ] **Download the real data** (a few minutes; needs internet):

```bash
python -m ws2_training.prepare_data
```

Check that it prints two lines, about `trec: train 4907, pool 545, test 500` and `emotion: train 14400, pool 1600, test 2000`. The files go to `data/real/`, which Git ignores; Harshita and Hementh run the same command and get identical files.

- [ ] **Run the two real-data conditions.** The first run also downloads the small embedding model (about 90 MB):

```bash
python -m ws2_training.train --conditions C1,C5a
```

Check that it prints 12 lines like `trec C1 seed 13: macro-F1 0.xxx (n=600)`, then sanity-check them:

- C1 should be well above chance. With 6 labels, chance is about 0.17.
- C5a (only 60 examples) should be lower than C1.
- The three seeds should give similar numbers.

- [ ] Save and merge it:

```bash
git add results/runs.csv
git commit -m "ws2: C1 and C5a results on real data"
git push origin ws2-training
gh pr create --base main --head ws2-training --title "C1 and C5a results" --body "Real-data baselines."
```

Once the CI check is green, merge it: `gh pr merge --merge`, or the green button on GitHub.

- [ ] **Run Checkpoint 1:** share the C1 and C5a macro-F1 for each task; confirm Harshita's pilot is fine and Hementh's sheet is shared.

## Stage 3: label, review and draft (about 2 hours)

- [ ] **Label the 60 items** in Hementh's Google Sheet. Fill only the `kausthuba` column, using one label from `choose_one_of` exactly as written. Go by your own judgement and don't look anything up. About 15 minutes.
- [ ] Draft report sections 3b and 4 (see "Your report and slides" below). Everything they need is already known; fill in numbers later.
- [ ] When Harshita opens the "Synthetic data" pull request, open **Files changed** and skim 10 or so lines of `data/synthetic/emotion/few_shot.jsonl`. Comment if anything looks broken.
- [ ] **Run Checkpoint 2** once her data is merged.

## Stage 4: the full run (about 30 minutes)

- [ ] When Harshita posts **"synthetic data is in main"**, start straight away:

```bash
git pull origin main
python -m ws2_training.train       # runs the 24 missing results; skips the 12 you have
```

This takes a few minutes; the Emotion test set (2,000 tweets) is the slow part.

- [ ] Check you have all 36:

```bash
python -c "import csv; print(len(list(csv.DictReader(open('results/runs.csv')))))"
```

This should print `36`. If a line said `skipped (no data yet)`, the synthetic files aren't in your branch yet: run `git pull origin main` and try again.

- [ ] Commit, push, open a pull request and merge it (same commands as stage 2, with the message `ws2: all 36 results`). **Post "results are in main" in the group chat.** That's Hementh's signal.
- [ ] Finish sections 3b and 4 with the real numbers. **Run Checkpoint 3:** everyone's sections are in and Hementh has the full draft.

## Stage 5: slides, submission, tag (about 1½ hours)

- [ ] Make your 2 slides (see below).
- [ ] Collect Harshita's and Hementh's slides and combine them into one deck of about 9 slides, in this order: title; question and hypothesis; the four papers (Harshita); data and conditions (you); generation and examples (Harshita); subjectivity check (Hementh); results, 2 slides (Hementh); limitations and conclusion (Harshita).
- [ ] Read the whole report once and fix anything in your sections. Hementh does the final edit and sends you the PDF.
- [ ] Submit the report and slides.
- [ ] Tag the final version of the code:

```bash
git switch main && git pull origin main
git tag v1.0 && git push origin v1.0
```

## Git rules

1. Work only on `ws2-training`, and only change `ws2_training/` and `results/runs.csv`.
2. Start each session with `git pull origin main`.
3. To get work into `main`: push, open a pull request, and merge it yourself once CI is green.
4. Changing `config.yaml` or `common.py`? Post it in the group chat first, because everyone's code reads them.

| Files | Written only by |
| --- | --- |
| `data/synthetic/` | Harshita |
| `results/runs.csv` | Kausthuba (you) |
| `results/annotation_*.csv`, `results/subjectivity.csv`, `results/summary.md`, `results/*.png` | Hementh |
| `data/real/` | Nobody commits it (each person rebuilds it) |

## Your report and slides

Hementh sets up the shared report doc in stage 2. Write straight into it.

| Section | Length | What to put in it |
| --- | --- | --- |
| 3b. Tasks and data | about ½ page | The two datasets, with links: TREC question type (`SetFit/TREC-QC`, 6 labels) and Emotion (`dair-ai/emotion`, 6 labels). Why these two: same number of labels, but one objective and one subjective. Sizes from your `prepare_data` output. 10% of train kept apart as a pool for few-shot examples and the low-resource sets; the official test sets are used untouched. |
| 4. Classifier and conditions | about ¾ page | Frozen `all-MiniLM-L6-v2` sentence embeddings (Sentence-BERT) plus logistic regression (C = 1.0), same settings for every run. The six-condition table from "Your job". Training size of 100 per label. Seeds 13, 42 and 7. The metric is macro-F1 on the real test set, mean ± std over seeds. |

**Your slides (2):** the two tasks with one example each; the six conditions.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError: common` | Run commands from the `synthtext` folder, always with `python -m …` |
| `prepare_data` fails to download | Retry; university Wi-Fi sometimes blocks Hugging Face, so try a phone hotspot; `pip install -U datasets` |
| `pip install` fails on PyTorch | You're on the wrong Python. Delete `.venv` and redo setup with 3.11 |
| `skipped (no data yet)` for C2–C4 or C5b | Expected before Checkpoint 2. After it: `git pull origin main` and run again |
| Need to redo some results (e.g. the synthetic data was re-merged) | `python -m ws2_training.train --conditions C2,C3,C4,C5b --force` replaces just those rows |
| Harshita's generation is too slow (no GPU) | Run it on your laptop instead: `curl -fsSL https://ollama.com/install.sh \| sh`, `ollama pull llama3.2:3b`, then the generate commands from her manual on the `ws1-generation` branch. Agree with her first so you don't both run it |
| Harshita's synthetic data is late | Draft your sections in the meantime. If she merges a partial dataset, run with it and rerun with `--force` when the rest arrives |
| `git push` is rejected | `git pull origin ws2-training`, then push again |
