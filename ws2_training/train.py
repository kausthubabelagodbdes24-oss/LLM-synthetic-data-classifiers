"""WS2 (Member B): train and test a classifier for every task x condition x seed.

    python -m ws2_training.train                     # everything (skips results already in runs.csv)
    python -m ws2_training.train --conditions C1,C5a # before synthetic data exists

Classifier: frozen Sentence-BERT embeddings (all-MiniLM-L6-v2) + logistic regression.
Appends to results/runs.csv: task, condition, seed, n_train, macro_f1, accuracy.
"""

import argparse
import csv
import os
import random
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import CFG, TASKS, embed, labels, per_label_sample, read_jsonl, real_path, results_dir, synth_path

COLUMNS = ["task", "condition", "seed", "n_train", "macro_f1", "accuracy"]


def is_sample_mode() -> bool:
    """Return True if executing in offline smoke or sample-data mode."""
    mode = os.environ.get("SYNTHTEXT_GENERATION_MODE")
    embedder = os.environ.get("SYNTHTEXT_EMBEDDER")
    data_env = os.environ.get("SYNTHTEXT_DATA", "")
    return mode == "fake" or embedder == "hashing" or "sample" in data_env.replace("\\", "/").split("/")


def approved_target_count(task: str, cond: str) -> int:
    """Return the approved total training count for a task and condition under CFG."""
    n_labels = len(labels(task))
    n = CFG["train_per_label"]
    low = CFG["lowres_per_label"]
    if cond in ("C1", "C2", "C3", "C4"):
        return n_labels * n
    if cond == "C5a":
        return n_labels * low
    if cond == "C5b":
        return n_labels * low + n_labels * low * CFG.get("auggpt_k", 6)
    raise ValueError(f"Unknown condition: {cond!r}")


def training_set(task, cond, seed, exact=None):
    n = CFG["train_per_label"]
    task_labels = labels(task)
    if exact is None:
        exact = not is_sample_mode()

    real_rows = read_jsonl(real_path(task, "train"))
    if not real_rows:
        real = []
    else:
        try:
            real = per_label_sample(real_rows, n, seed, required_labels=task_labels, exact=exact)
        except ValueError as exc:
            raise ValueError(
                f"Cannot sample approved target for {task} {cond} (seed {seed}): {exc}. "
                f"The approved target requires {n} examples per label ({approved_target_count(task, cond)} total), "
                f"but real training split {real_path(task, 'train')} has insufficient examples. "
                f"Silently undersampling, borrowing from demonstration-pool or test splits, duplicating examples, "
                f"or changing the target without explicit project lead approval is prohibited."
            ) from exc

    if cond == "C1":
        return real
    if cond == "C2":
        s_path = synth_path(task, "zero_shot", seed=seed)
        rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "zero_shot"))
        return per_label_sample(rows, n, seed, required_labels=task_labels, exact=exact) if rows else []
    if cond == "C3":
        s_path = synth_path(task, "few_shot", seed=seed)
        rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "few_shot"))
        return per_label_sample(rows, n, seed, required_labels=task_labels, exact=exact) if rows else []
    if cond == "C4":
        half = n // 2
        s_path = synth_path(task, "few_shot", seed=seed)
        s_rows = read_jsonl(s_path if s_path.exists() else synth_path(task, "few_shot"))
        if not real_rows or not s_rows:
            return []
        try:
            real_half = per_label_sample(real_rows, half, seed, required_labels=task_labels, exact=exact)
            synth_half = per_label_sample(s_rows, n - half, seed, required_labels=task_labels, exact=exact)
        except ValueError as exc:
            raise ValueError(
                f"Cannot sample approved target for {task} {cond} (seed {seed}): {exc}."
            ) from exc
        return real_half + synth_half

    low = read_jsonl(real_path(task, f"lowres_seed{seed}"))
    if cond == "C5a":
        if not is_sample_mode() and low:
            expected_c5a = approved_target_count(task, "C5a")
            if len(low) != expected_c5a:
                raise ValueError(
                    f"Low-resource sample {real_path(task, f'lowres_seed{seed}')} has {len(low)} rows, "
                    f"expected approved target of {expected_c5a}."
                )
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

    if not is_sample_mode():
        target = approved_target_count(task, cond)
        if len(train) != target:
            raise ValueError(
                f"{task} {cond} seed {seed}: training count {len(train)} does not match approved target count {target}."
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
                cond = r["condition"]

                # P2: Restrict replacement strictly to requested conditions.
                # Unrelated rows must be preserved. If validity is uncertain, report it without deleting.
                if cond not in conds:
                    if not is_sample_mode():
                        try:
                            target = approved_target_count(r["task"], cond)
                            if int(r["n_train"]) != target:
                                print(
                                    f"  Notice: Preserving unrequested {r['task']} {cond} seed {r['seed']} row "
                                    f"(n_train={r['n_train']} != approved target {target}); not replacing because condition was not requested."
                                )
                        except Exception:
                            pass
                    old.append(r)
                    continue

                # The row's condition was requested.
                # If --force is specified, discard it so it will be rerun.
                if args.force:
                    continue

                # P1: Validate existing row against approved target count independently of sampler.
                if not is_sample_mode():
                    target = approved_target_count(r["task"], cond)
                    if int(r["n_train"]) != target:
                        print(
                            f"  Discarding obsolete {r['task']} {cond} seed {r['seed']} result: "
                            f"n_train={r['n_train']} does not match approved target count {target}."
                        )
                        continue
                else:
                    t_set = training_set(r["task"], cond, int(r["seed"]), exact=False)
                    if t_set and int(r["n_train"]) != len(t_set):
                        continue

                old.append(r)

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
