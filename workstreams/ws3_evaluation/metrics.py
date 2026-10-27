"""Quality of each synthetic set, compared with real data.

    python -m workstreams.ws3_evaluation.metrics --task all

Per (task, setting), writes one row of results/quality.csv:
- remote_clique   mean cosine distance of a text to all others (higher = more diverse)  [Li et al.]
- chamfer         mean cosine distance of a text to its nearest neighbour (higher = more diverse) [Li et al.]
- faithfulness    mean of each synthetic text's best cosine to a real train text (Dai et al.)
- label_fidelity  share of synthetic texts the CEIL model labels as intended (needs run_matrix CEIL)
- dup_rate        share of exact duplicates (case/spacing ignored) — should be 0 after filter
- mean_words
A `real` row per task gives the reference values. Up to `--max-n` texts per set are sampled.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from synthtext import config
from synthtext.contracts import SETTINGS
from synthtext.embeddings import embed
from synthtext.io import normalise, read_jsonl, write_csv
from synthtext.paths import real_dir, results_root, synthetic_dir

COLUMNS = ["task", "setting", "n", "remote_clique", "chamfer", "faithfulness",
           "label_fidelity", "dup_rate", "mean_words"]


def remote_clique(X: np.ndarray) -> float:
    n = len(X)
    if n < 2:
        return float("nan")
    sims = X @ X.T
    dist = 1.0 - sims
    return float((dist.sum(axis=1) / (n - 1)).mean())  # diagonal distance is 0


def chamfer(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("nan")
    sims = X @ X.T
    np.fill_diagonal(sims, -np.inf)
    return float((1.0 - sims.max(axis=1)).mean())


def faithfulness(X: np.ndarray, R: np.ndarray) -> float:
    if len(X) == 0 or len(R) == 0:
        return float("nan")
    return float((X @ R.T).max(axis=1).mean())


def dup_rate(texts: list[str]) -> float:
    keys = [normalise(t) for t in texts]
    return 0.0 if not keys else 1.0 - len(set(keys)) / len(keys)


def label_fidelity(task_id: str, setting: str) -> float | None:
    rows = read_jsonl(results_root() / "fidelity" / task_id / f"{setting}.jsonl")
    if not rows:
        return None
    return float(np.mean([r["label"] == r["pred"] for r in rows]))


def _sample(records: list[dict], n: int) -> list[dict]:
    records = sorted(records, key=lambda r: r["id"])
    return random.Random(0).sample(records, n) if len(records) > n else records


def _r(x: float | None) -> str | float:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    ap.add_argument("--max-n", type=int, default=1000)
    args = ap.parse_args(argv)
    model = config.experiments()["diversity_model"]

    rows = []
    for task_id in config.task_ids(args.task):
        real_train = _sample(read_jsonl(real_dir(task_id) / "train.jsonl"), args.max_n)
        if not real_train:
            print(f"[{task_id}] no real data; skipping")
            continue
        R = embed([r["text"] for r in real_train], model)
        sets = {"real": real_train}
        for s in SETTINGS:
            recs = read_jsonl(synthetic_dir(task_id) / f"{s}.jsonl")
            if recs:
                sets[s] = recs
        for setting, recs in sets.items():
            sample = _sample(recs, args.max_n)
            X = R if setting == "real" else embed([r["text"] for r in sample], model)
            rows.append({
                "task": task_id, "setting": setting, "n": len(recs),
                "remote_clique": _r(remote_clique(X)),
                "chamfer": _r(chamfer(X)),
                "faithfulness": "" if setting == "real" else _r(faithfulness(X, R)),
                "label_fidelity": "" if setting == "real" else _r(label_fidelity(task_id, setting)),
                "dup_rate": _r(dup_rate([r["text"] for r in recs])),
                "mean_words": _r(np.mean([len(r["text"].split()) for r in recs])),
            })
            print(f"[{task_id}] {setting}: " + ", ".join(f"{k}={rows[-1][k]}" for k in COLUMNS[2:]))
    write_csv(results_root() / "quality.csv", rows, COLUMNS)
    print(f"wrote {results_root() / 'quality.csv'}")


if __name__ == "__main__":
    main()
