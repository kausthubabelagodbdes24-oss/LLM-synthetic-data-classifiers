"""WS2 (Member B): train and test a classifier for every task x condition x seed.

    python -m ws2_training.train                     # everything (skips results already in runs.csv)
    python -m ws2_training.train --conditions C1,C5a # before synthetic data exists

Classifier: frozen Sentence-BERT embeddings (all-MiniLM-L6-v2) + logistic regression.
Appends to results/runs.csv: task, condition, seed, n_train, macro_f1, accuracy.
"""

import argparse
import csv
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import CFG, TASKS, embed, labels, per_label_sample, read_jsonl, real_path, results_dir, synth_path

COLUMNS = ["task", "condition", "seed", "n_train", "macro_f1", "accuracy"]


def training_set(task, cond, seed):
    n = CFG["train_per_label"]
    task_labels = labels(task)
    real_rows = read_jsonl(real_path(task, "train"))
    real = per_label_sample(real_rows, n, seed, required_labels=task_labels) if real_rows else []

    if cond == "C1":
        return real
    if cond == "C2":
        s_path = synth_path(task, "zero_shot", seed=seed)
        rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "zero_shot"))
        return per_label_sample(rows, n, seed, required_labels=task_labels) if rows else []
    if cond == "C3":
        s_path = synth_path(task, "few_shot", seed=seed)
        rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "few_shot"))
        return per_label_sample(rows, n, seed, required_labels=task_labels) if rows else []
    if cond == "C4":
        half = n // 2
        s_path = synth_path(task, "few_shot", seed=seed)
        s_rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "few_shot"))
        if not real or not s_rows:
            return []
        return (per_label_sample(real, half, seed, required_labels=task_labels)
                + per_label_sample(s_rows, n - half, seed, required_labels=task_labels))
    low = read_jsonl(real_path(task, f"lowres_seed{seed}"))
    if cond == "C5a":
        return low
    if cond == "C5b":
        s_path = synth_path(task, "auggpt", seed=seed)
        s_rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "auggpt"))
        if not low or not s_rows:
            return []
        ids = {r["id"] for r in low}
        return low + [r for r in s_rows if r.get("source_ids") and r["source_ids"][0] in ids]
    raise ValueError(cond)


def run(task, cond, seed):
    train = training_set(task, cond, seed)
    if not train:
        print(f"  {task} {cond} {seed}: skipped (no data yet)")
        return None

    expected_labels = set(labels(task))
    present_labels = {r["label"] for r in train}
    if present_labels != expected_labels:
        missing = sorted(expected_labels - present_labels)
        raise ValueError(
            f"{task} {cond} seed {seed}: incomplete class coverage, missing label(s): {missing}"
        )

    test = read_jsonl(real_path(task, "test"))
    if not test:
        raise ValueError(f"{task}: held-out real test set is missing or empty at {real_path(task, 'test')}")

    test_present = {r["label"] for r in test}
    if test_present != expected_labels:
        missing = sorted(expected_labels - test_present)
        raise ValueError(f"{task}: held-out test set missing label(s): {missing}")

    shuffled_train = list(train)
    random.Random(seed).shuffle(shuffled_train)

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(embed([r["text"] for r in shuffled_train]), [r["label"] for r in shuffled_train])
    pred = clf.predict(embed([r["text"] for r in test]))
    gold = [r["label"] for r in test]

    macro_f1 = round(f1_score(gold, pred, labels=labels(task), average="macro", zero_division=0), 4)
    acc = round(accuracy_score(gold, pred), 4)

    if not (0.0 <= macro_f1 <= 1.0) or not (0.0 <= acc <= 1.0):
        raise ValueError(f"Invalid metric range for {task} {cond} seed {seed}: macro_f1={macro_f1}, accuracy={acc}")

    return {
        "task": task,
        "condition": cond,
        "seed": seed,
        "n_train": len(train),
        "macro_f1": macro_f1,
        "accuracy": acc,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=",".join(CFG["conditions"]))
    ap.add_argument("--force", action="store_true", help="re-run the chosen conditions, replacing their old rows")
    args = ap.parse_args(argv)
    path = results_dir() / "runs.csv"
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    old = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if not r or not r.get("task") or not r.get("condition") or not r.get("seed"):
                    continue
                # Validate that existing row matches current expected n_train
                t_set = training_set(r["task"], r["condition"], int(r["seed"]))
                if t_set and int(r["n_train"]) != len(t_set):
                    # Incompatible target or data count - discard and replace
                    continue
                old.append(r)

    if args.force:  # drop the old rows of the chosen conditions, keep the rest
        old = [r for r in old if r["condition"] not in conds]

    # De-duplicate existing rows while preserving order
    unique_old = []
    seen = set()
    for r in old:
        key = (r["task"], r["condition"], int(r["seed"]))
        if key not in seen:
            seen.add(key)
            unique_old.append(r)
    old = unique_old
    done = set(seen)

    # Write cleaned baseline/existing rows first
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(old)

    for task in TASKS:
        for cond in conds:
            for seed in CFG["seeds"]:
                if (task, cond, seed) in done:
                    print(f"  {task} {cond} seed {seed}: already recorded (skipping)")
                    continue
                row = run(task, cond, seed)
                if row:
                    with open(path, "a", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
                        writer.writerow(row)
                    done.add((task, cond, seed))
                    print(f"  {task} {cond} seed {seed}: macro-F1 {row['macro_f1']:.3f} (n={row['n_train']})")


if __name__ == "__main__":
    main()
