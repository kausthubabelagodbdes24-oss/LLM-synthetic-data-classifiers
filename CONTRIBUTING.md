# How we work in this repo

One repo, one folder and one long-lived branch per person. Everything reaches `main`
through a reviewed pull request.

## Branches

| Branch | Who pushes | How it reaches `main` |
| --- | --- | --- |
| `main` | nobody directly (protected) | — |
| `ws1-generation` | Kausthuba | pull request, at least at every sync point |
| `ws2-training` | Member B | same |
| `ws3-evaluation` | Member C | same |
| `shared/<topic>` | anyone | short-lived; for `src/`, `configs/`, `tests/` or contract changes; all three approve |

## Rules

1. Work only inside your own `workstreams/` folder, on your own branch.
2. Start each day by bringing `main` into your branch:
   ```bash
   git switch ws2-training
   git pull origin main          # merge, don't rebase a shared branch
   ```
3. Commit small and often, prefixed with your workstream: `ws2: add embedding cache`.
4. Open a pull request into `main` when a piece works, and at least at every sync point.
   It needs green CI and one approval from someone else. CODEOWNERS requests the right reviewer.
5. Changing a data contract, a config value or shared code? Make a `shared/<topic>` branch
   from `main`, change it there, update the sample files and the master plan, and get all
   three approvals.
6. Never commit API keys, `.env`, `data/real/` or the embedding cache.
7. After Nov 3 nobody regenerates synthetic data; every run uses the frozen files.

## Sync points

A 20-minute call at each one; everyone merges first.

| Date | Gate |
| --- | --- |
| Oct 29 | Contracts fixed; real data + example pool merged; 20-per-label pilot from each setting; survey sent |
| Nov 1 | Few-shot generation done; runner works end to end on samples; all 160 items annotated |
| Nov 3 | Synthetic data frozen and committed |
| Nov 5 | `runs.csv`, `quality.csv`, `subjectivity.csv` complete; figures drafted |
| Nov 8 | Report and slides done; tag `v1.0` |

## Making a pull request

```bash
git add workstreams/ws2_training/
git commit -m "ws2: run matrix handles missing synthetic files"
git push origin ws2-training
```
Then on GitHub: **Compare & pull request** → base `main` ← compare `ws2-training`, fill in the template.

## One-time GitHub settings (repo owner)

- Settings → Branches → add a rule for `main`: require a pull request, 1 approval, and the `CI` check.
- Settings → Collaborators → add both teammates with write access.
- Replace the placeholders in `.github/CODEOWNERS` with real GitHub usernames.
