"""Train and evaluate every (task, condition, seed) cell; append to results/runs.csv.

    python -m workstreams.ws2_training.run_matrix --task all
    python -m workstreams.ws2_training.run_matrix --task trec --conditions C1,C2,C3 --seeds 13

Resumable: cells already in runs.csv are skipped unless you pass --force.
Also writes, for WS3:
    results/predictions/<task>/<condition>_seed<S>.jsonl   test predictions {id, pred}
    results/fidelity/<task>/<setting>.jsonl                CEIL model's label for each synthetic text
CEIL uses the whole train split, so it runs once (first seed) rather than per seed.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from synthtext import config
from synthtext.contracts import RUN_COLUMNS, SETTINGS
from synthtext.io import append_csv_row, git_sha, read_csv, read_jsonl, write_jsonl
from synthtext.paths import real_dir, results_root, synthetic_dir
from workstreams.ws2_training import classifier
from workstreams.ws2_training.conditions import build


def _list_arg(value: str | None, default: list) -> list:
    if value in (None, "", "all"):
        return list(default)
    return [type(default[0])(v.strip()) for v in value.split(",") if v.strip()]


def run_cell(task_id: str, condition: str, seed: int, n_train: int | None, exp: dict) -> dict | None:
    ts = build(task_id, condition, seed, n_train)
    for note in ts.notes:
        print(f"    note: {note}")
    if not ts.records:
        print(f"    skipped: empty training set")
        return None
    test = read_jsonl(real_dir(task_id) / "test.jsonl")
    labels = config.labels(task_id)
    fitted = classifier.fit(
        [r["text"] for r in ts.records], [r["label"] for r in ts.records],
        exp["embedding_model"], exp["classifier"], seed,
    )
    preds = fitted.predict([r["text"] for r in test]).tolist()
    scores = classifier.score([r["label"] for r in test], preds, labels)

    out = results_root() / "predictions" / task_id / f"{condition}_seed{seed}.jsonl"
    write_jsonl(out, ({"id": r["id"], "pred": p} for r, p in zip(test, preds)))

    if condition == "CEIL":
        for setting in SETTINGS:
            synth = read_jsonl(synthetic_dir(task_id) / f"{setting}.jsonl")
            if synth:
                sp = fitted.predict([r["text"] for r in synth]).tolist()
                write_jsonl(results_root() / "fidelity" / task_id / f"{setting}.jsonl",
                            ({"id": r["id"], "label": r["label"], "pred": p} for r, p in zip(synth, sp)))
    return {
        "task": task_id, "condition": condition, "seed": seed, "n_train": len(ts.records),
        **scores, "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--n-train", type=int, default=None, help="override n_train (sample runs only)")
    ap.add_argument("--force", action="store_true", help="re-run cells already in runs.csv")
    args = ap.parse_args(argv)

    exp = config.experiments()
    conditions = _list_arg(args.conditions, list(exp["conditions"]))
    seeds = _list_arg(args.seeds, exp["seeds"])
    runs_csv = results_root() / "runs.csv"
    done = {(r["task"], r["condition"], int(r["seed"])) for r in read_csv(runs_csv)}

    for task_id in config.task_ids(args.task):
        for condition in conditions:
            for seed in (seeds[:1] if condition == "CEIL" else seeds):
                key = (task_id, condition, seed)
                if key in done and not args.force:
                    continue
                start = time.time()
                print(f"[{task_id}] {condition} seed {seed}")
                try:
                    row = run_cell(task_id, condition, seed, args.n_train, exp)
                except ValueError as exc:
                    print(f"    skipped: {exc}")
                    continue
                if row:
                    append_csv_row(runs_csv, row, RUN_COLUMNS)
                    print(f"    macro-F1 {row['macro_f1']:.3f}  acc {row['accuracy']:.3f}  "
                          f"n={row['n_train']}  ({time.time() - start:.1f}s)")
    print(f"results: {runs_csv}")


if __name__ == "__main__":
    main()
