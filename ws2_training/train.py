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
    real = per_label_sample(read_jsonl(real_path(task, "train")), n, seed)
    if cond == "C1":
        return real
    if cond == "C2":
        return per_label_sample(read_jsonl(synth_path(task, "zero_shot")), n, seed)
    if cond == "C3":
        return per_label_sample(read_jsonl(synth_path(task, "few_shot")), n, seed)
    if cond == "C4":
        half = n // 2
        return (per_label_sample(real, half, seed)
                + per_label_sample(read_jsonl(synth_path(task, "few_shot")), n - half, seed))
    low = read_jsonl(real_path(task, f"lowres_seed{seed}"))
    if cond == "C5a":
        return low
    if cond == "C5b":
        ids = {r["id"] for r in low}
        return low + [r for r in read_jsonl(synth_path(task, "auggpt")) if r["source_ids"][0] in ids]
    raise ValueError(cond)


def run(task, cond, seed):
    train = training_set(task, cond, seed)
    if len({r["label"] for r in train}) < 2:
        print(f"  {task} {cond} {seed}: skipped (no data yet)")
        return None
    random.Random(seed).shuffle(train)
    test = read_jsonl(real_path(task, "test"))
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(embed([r["text"] for r in train]), [r["label"] for r in train])
    pred = clf.predict(embed([r["text"] for r in test]))
    gold = [r["label"] for r in test]
    return {"task": task, "condition": cond, "seed": seed, "n_train": len(train),
            "macro_f1": round(f1_score(gold, pred, labels=labels(task), average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(gold, pred), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=",".join(CFG["conditions"]))
    ap.add_argument("--force", action="store_true", help="re-run the chosen conditions, replacing their old rows")
    args = ap.parse_args(argv)
    path = results_dir() / "runs.csv"
    conds = args.conditions.split(",")
    old = list(csv.DictReader(open(path))) if path.exists() else []
    if args.force:  # drop the old rows of the chosen conditions, keep the rest
        old = [r for r in old if r["condition"] not in conds]
    done = {(r["task"], r["condition"], int(r["seed"])) for r in old}
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(old)
        for task in TASKS:
            for cond in conds:
                for seed in CFG["seeds"]:
                    if (task, cond, seed) in done:
                        continue
                    row = run(task, cond, seed)
                    if row:
                        writer.writerow(row)
                        fh.flush()
                        print(f"  {task} {cond} seed {seed}: macro-F1 {row['macro_f1']:.3f} (n={row['n_train']})")


if __name__ == "__main__":
    main()
