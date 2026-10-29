# SynthText Manual 1 — Generation (Harshita)

As of 29 Oct 2026 · Project lead: Kausthuba Belagod

## Your job

You produce every piece of synthetic text in the project: the prompts, the generation runs and the final synthetic dataset. You're the first link in the main chain of work. Kausthuba can't run the experiment until your synthetic data is in `main`, so start generating as early as you can.

Kausthuba owns the repo and leads the project; ask him if you're stuck on anything outside your part.

| You hand over | To | At |
| --- | --- | --- |
| Pilot check: 10 texts per label look sensible | Team | Checkpoint 1 |
| `data/synthetic/<task>/{zero_shot,few_shot,auggpt}.jsonl` merged into `main` | Kausthuba | Checkpoint 2 |
| Report sections 2, 3a and 6, plus Appendix A | Hementh (assembles the report) | Checkpoint 3 |
| Your slides | Kausthuba (builds the deck) | Stage 5 |

| You need | From | At |
| --- | --- | --- |
| The repo invite | Kausthuba | Stage 1 |
| The annotation sheet link | Hementh | Stage 3 |
| The key results numbers, for section 6 | Hementh | Stage 4 |

You don't wait on anyone's data. You run `prepare_data` yourself, so you have the real examples for few-shot prompts straight away.

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

## Stage 1: setup (about 1 hour)

Most of it is waiting for downloads.

### 1. Install the tools (skip any you already have)

| Tool | Windows | macOS | Linux |
| --- | --- | --- | --- |
| Git | `winget install Git.Git` | `xcode-select --install` | `sudo apt install git` (Fedora: `sudo dnf install git`) |
| GitHub CLI | `winget install GitHub.cli` | `brew install gh` | `sudo apt install gh` (Fedora: `sudo dnf install gh`) |
| Python **3.11** | [python.org](https://www.python.org/downloads/) installer; tick **Add to PATH** | `brew install python@3.11` | `sudo apt install python3.11 python3.11-venv` |
| Ollama | Installer from [ollama.com/download](https://ollama.com/download) | Installer from [ollama.com/download](https://ollama.com/download) | `curl -fsSL https://ollama.com/install.sh \| sh` |

Use Python 3.11, not the newest version; the machine-learning libraries lag behind new releases.

### 2. Get the repo

1. Accept Kausthuba's invite: check your email, or github.com → Notifications.
2. Log in and clone. Replace `<owner>` with Kausthuba's GitHub username:

```bash
gh auth login                      # GitHub.com → HTTPS → log in with a browser
gh repo clone <owner>/synthtext
cd synthtext
git switch ws1-generation          # your branch, for the whole project
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

**Check:** the last line says `2 passed`.

### 4. Download the model

```bash
ollama pull llama3.2:3b            # 2.0 GB, once; no internet needed after this
ollama run llama3.2:3b "Write one short question about rivers."
```

**Check:** it answers with a question within a few seconds. Then, in a second terminal while it runs, `ollama ps` shows whether it's using the GPU (`100% GPU`) or the CPU. On a CPU it still works, just slower: leave long runs going overnight. Tell the group chat which one you got.

## Every session starts the same way

From the `synthtext` folder:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
git switch ws1-generation
git pull origin main               # brings in everyone's merged work
```

To count what you've generated so far (use it after every run; it works on any system):

```bash
python -c "import json,glob,collections as c; [print(f, dict(c.Counter(json.loads(l)['label'] for l in open(f)))) for f in sorted(glob.glob('data/synthetic/*/*.jsonl'))]"
```

## Stage 2: data, pilot and generation (about 1½ hours, then unattended)

- [ ] **Download the real data** (about 2 minutes). It writes `data/real/`, which Git ignores; Kausthuba and Hementh run the same command and get identical files.

```bash
python -m ws2_training.prepare_data
```

Check: it prints `trec: train 4907, pool 545, test 500` (or close) and a similar line for emotion.

- [ ] **Pilot run:** 10 texts per label, for each setting:

```bash
python -m ws1_generation.generate --setting zero_shot --pilot
python -m ws1_generation.generate --setting few_shot --pilot
```

- [ ] **Read about 40 pilot texts.** Open `data/synthetic/emotion/few_shot.jsonl` and `data/synthetic/trec/zero_shot.jsonl` in any text editor and check four things:
    1. Does each text fit its label? (A `fear` tweet sounds afraid; a `numeric` question asks for a number.)
    2. Does it look like the real data? (TREC: short questions. Emotion: lower-case "i feel…" tweets.)
    3. Are the texts varied, not the same sentence over and over?
    4. Few-shot only: are they new, not copies of the examples?
- [ ] If something is clearly off, edit the prompt wording in `prompt_zero_shot` in `ws1_generation/generate.py`. Then delete the `.jsonl` files inside `data/synthetic/` (keep the folders) and rerun the pilot. Do this at most twice; "good enough" is fine.
- [ ] Save two good and two bad pilot examples per task into a note for the report appendix.
- [ ] **Start the full run.** Set your laptop to never sleep while plugged in, then:

macOS / Linux (keeps running if you close the terminal):

```bash
nohup sh -c "python -m ws1_generation.generate --setting zero_shot && python -m ws1_generation.generate --setting few_shot" > gen.log 2>&1 &
tail -f gen.log                    # Ctrl-C stops watching; the run keeps going
```

Windows (leave the window open until it finishes):

```powershell
python -m ws1_generation.generate --setting zero_shot; python -m ws1_generation.generate --setting few_shot
```

The pilot texts count toward the 100 per label; the run picks up from them.

- [ ] **Checkpoint 1:** tell the team the pilot looks fine, or what you changed.

## Stage 3: AugGPT, labels, freeze the data (about 1 hour, plus AugGPT running)

- [ ] Check the full run with the count command. You want 100 per label for both settings and both tasks. If any are short, rerun that setting; it continues from where it stopped.
- [ ] **Run AugGPT:** about 360 calls, roughly as long as one of the other settings.

```bash
python -m ws1_generation.generate --setting auggpt
```

- [ ] **Label the 60 items** in Hementh's Google Sheet. Fill only the `harshita` column, using one label from `choose_one_of` exactly as written. Go by your own judgement and don't look anything up. About 15 minutes.
- [ ] When AugGPT finishes, run the count command once more. Expected: `zero_shot` and `few_shot` at 100 per label each; `auggpt` with roughly 6 rows per seed example.
- [ ] Record the model version for the report: `ollama list` shows `llama3.2:3b` and its ID.
- [ ] Commit, push and open a pull request:

```bash
git add data/synthetic
git commit -m "ws1: synthetic data (zero-shot, few-shot, AugGPT), llama3.2:3b"
git push origin ws1-generation
gh pr create --base main --head ws1-generation --title "Synthetic data" --body "Frozen synthetic data for the experiment."
```

- [ ] When CI shows a green tick, merge it: `gh pr merge --merge`, or the green button on GitHub. **Post "synthetic data is in main" in the group chat.** That message is Kausthuba's signal to start the full run.
- [ ] **Checkpoint 2.** From now on, nobody regenerates data.

## Stage 4: write (about 2½ hours)

- [ ] Write sections 2 and 3a and Appendix A in the shared report (details under "Your report and slides" below).
- [ ] Draft section 6 (limitations). When Hementh posts the key results, finish it with the conclusion paragraph.
- [ ] **Checkpoint 3:** your sections are in.

## Stage 5: slides and read-through (about 45 minutes)

- [ ] Make your slides and send them to Kausthuba, who builds the combined deck.
- [ ] Read the whole report once and fix anything in your sections. Hementh does the final edit.

## Git rules

1. Work only on your own branch (`ws1-generation`), and only change files in `ws1_generation/` and `data/synthetic/`.
2. Start each session with `git pull origin main`, so you have everyone else's merged work.
3. Save often:

```bash
git add ws1_generation data/synthetic
git commit -m "ws1: what you changed"
git push origin ws1-generation
```

4. To get work into `main`, open a pull request (`gh pr create …` as in stage 3, or **Compare & pull request** on GitHub). Merge it yourself once the CI check is green.
5. Changing `config.yaml` or `common.py`? Post it in the group chat first, because everyone's code reads them.

Who writes which files, so nobody overwrites anyone:

| Files | Written only by |
| --- | --- |
| `data/synthetic/` | Harshita (you) |
| `results/runs.csv` | Kausthuba |
| `results/annotation_*.csv`, `results/subjectivity.csv`, `results/summary.md`, `results/*.png` | Hementh |
| `data/real/` | Nobody commits it (each person rebuilds it) |

## Your report and slides

Hementh sets up the shared report doc in stage 2. Write straight into it.

| Section | Length | What to put in it |
| --- | --- | --- |
| 2. Related work | about ½ page | One short paragraph per paper. **Li et al.**: synthetic data helps less on subjective tasks; we re-test that. **AugGPT**: rephrasing real sentences as augmentation; our C5b. **Brown et al.**: few-shot prompting; our few-shot setting. **Sentence-BERT**: the embeddings our classifier uses. |
| 3a. Model and generation | about ¾ page | Model `llama3.2:3b` via Ollama, offline, with the ID from `ollama list`; temperature 0.9. The three settings (zero-shot, few-shot with 3 real examples, AugGPT with 6 rephrasings). How many texts were kept per setting, from the count command. That we drop duplicates and any text identical to a test item. |
| 6. Limitations and conclusion | about ½ page | Only 2 tasks, one small 3B model, 3 annotators on 30 items each, small training sets, one simple classifier. One sentence each on what you'd do with more time (more tasks, a bigger model). End with a one-paragraph answer to the research question, using Hementh's results. |
| Appendix A | 1 page | The three prompt templates (copy them from `generate.py`) and 2 good and 2 bad examples per task from the pilot. |

**Your slides (2–3):** how the model generates data, with the prompts in brief; real vs synthetic examples side by side; and the final "limitations and conclusion" slide.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| `Cannot connect` or `Connection refused` from generate | Ollama isn't running. Open the Ollama app, or run `ollama serve` in another terminal |
| Very slow, `ollama ps` shows CPU | It still works: run it overnight. If it would take too long, tell Kausthuba; his laptop has an NVIDIA GPU and he can run the same commands. Or set `per_label: 60` in `config.yaml` and tell the team |
| The model won't download, or gives poor output | Switch to the backup: `ollama pull qwen3:4b`, set `model: qwen3:4b` in `config.yaml`, tell the team, delete the old files and rerun |
| A label stops short of 100 | Rerun the same command; it continues where it stopped |
| `No example pool` error | Run `python -m ws2_training.prepare_data` first |
| `ModuleNotFoundError: common` | Run commands from the `synthtext` folder, always with `python -m …` |
| `git push` is rejected | `git pull origin ws1-generation`, then push again |
| Generation is taking far longer than expected | Merge whatever you have (even 60 per label) and tell Kausthuba. A smaller, finished dataset beats a late one |
