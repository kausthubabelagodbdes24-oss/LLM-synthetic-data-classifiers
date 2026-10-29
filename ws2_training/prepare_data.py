"""WS2 (Member B): download the two datasets and write the real-data files.

    python -m ws2_training.prepare_data
    python -m ws2_training.prepare_data --verify

Writes data/real/<task>/:
    train.jsonl           real training data (90% of train split)
    pool.jsonl            10% of train, kept apart: few-shot examples + low-resource seeds
    test.jsonl            held-out real test set (official test split)
    lowres_seed<S>.jsonl  10 per label from the pool, one file per seed
Same seed on every machine, so everyone gets identical files.
"""

import argparse
import hashlib
from collections import Counter
import random
from pathlib import Path

from common import CFG, TASKS, labels, per_label_sample, real_path, write_jsonl

# Reproducible SHA-256 checksums and line counts for the real datasets.
CHECKSUMS = {
    "trec/train.jsonl": {
        "lines": 4907,
        "sha256": "26bd132486c3991c1ebf5be9e6b784b6c9badb00f8b01ee88a4679bc4a9456cc",
    },
    "trec/pool.jsonl": {
        "lines": 545,
        "sha256": "b30a1a65677f0bbd13340b39d5c59fa07665b93d7defd535a8a5cffac779b22e",
    },
    "trec/test.jsonl": {
        "lines": 500,
        "sha256": "3247e50dae08c989a5199ac3eb6cf5fe2f1f9046fabe4fb8a800f67d85fdb50f",
    },
    "trec/lowres_seed13.jsonl": {
        "lines": 60,
        "sha256": "2016e4e6f41932e136cdd16ab323418d6383f51c5a6ea3052a9ec77e3d0992ae",
    },
    "trec/lowres_seed42.jsonl": {
        "lines": 60,
        "sha256": "0332b255b32d99761a8220a7de0e901a732b85aa57969fd2dd43a1bad6b5b1ff",
    },
    "trec/lowres_seed7.jsonl": {
        "lines": 60,
        "sha256": "86dd7cbcc0568c1bd6436316121b2f38f388d2e99520ba8397ad7dac08caf58d",
    },
    "emotion/train.jsonl": {
        "lines": 14400,
        "sha256": "4847106f2b3f7cfb18ca42e41050af9106e2d28630a4b14e52d4dbec270085cd",
    },
    "emotion/pool.jsonl": {
        "lines": 1600,
        "sha256": "b375ea8344b8e12621520a71cf2a81bd2a25f77d72e66cffe9b71b832e57c42c",
    },
    "emotion/test.jsonl": {
        "lines": 2000,
        "sha256": "fafbd19376757905f6bdc4308919b2351d065c61d0c62e29baa55c7a26ff79d5",
    },
    "emotion/lowres_seed13.jsonl": {
        "lines": 60,
        "sha256": "b79e10b6f9f0ad1f717d4a88b8a58013f5dc502b56bf16320b43d85d5385cfce",
    },
    "emotion/lowres_seed42.jsonl": {
        "lines": 60,
        "sha256": "28b3ce40a0cb4a5e90ef5d875ddb9c4062c068af07fefb129a8a0c3403391692",
    },
    "emotion/lowres_seed7.jsonl": {
        "lines": 60,
        "sha256": "bdfa0d51679f9410f61000a90e755d5b071f1a08558952a13fd4bca22eab6349",
    },
}


def to_label(raw, t):
    if "label_map" in t:
        return t["label_map"].get(raw)
    return next((name for key, name in t["label_contains"].items() if key in str(raw).lower()), None)


def load(task, split):
    from datasets import load_dataset
    t = CFG["tasks"][task]
    rows = []
    for i, ex in enumerate(load_dataset(t["hf_id"], t["hf_config"], split=split)):
        label = to_label(ex[t["label_field"]], t)
        if label is None:
            raise ValueError(f"{task}: no label mapping for {ex[t['label_field']]!r}")
        rows.append({"id": f"{task}.{split}.{i:05d}", "text": " ".join(str(ex[t["text_field"]]).split()), "label": label})
    return rows


def split_data(train_rows, test_rows, task, seeds=None, lowres_per_label=None):
    """Deterministically partition train into pool (10%) and train (90%),
    verify coverage, and extract exact low-resource samples for all seeds.

    Args:
        train_rows: list of dicts with 'id', 'text', 'label'
        test_rows: list of dicts with 'id', 'text', 'label'
        task: task name ('trec' or 'emotion')
        seeds: list of seeds (defaults to CFG['seeds'])
        lowres_per_label: int (defaults to CFG['lowres_per_label'])

    Returns:
        dict with keys: 'train', 'pool', 'test', and 'lowres_seed{s}' for each seed.
    """
    if seeds is None:
        seeds = CFG["seeds"]
    if lowres_per_label is None:
        lowres_per_label = CFG["lowres_per_label"]

    expected_labels = labels(task)

    # Validate test set
    test_labels = {r["label"] for r in test_rows}
    missing_test = set(expected_labels) - test_labels
    if missing_test:
        raise ValueError(f"{task}: official test set missing expected label(s): {sorted(missing_test)}")

    # 10% demonstration pool cut with seed 0
    train_copy = list(train_rows)
    random.Random(0).shuffle(train_copy)
    cut = len(train_copy) // 10
    pool = train_copy[:cut]
    train_split = train_copy[cut:]

    # Disjointness check
    overlap = set(r["id"] for r in pool).intersection(r["id"] for r in train_split)
    if overlap:
        raise ValueError(f"{task}: pool and train overlap with {len(overlap)} IDs")

    # Pool class coverage check
    pool_counts = Counter(r["label"] for r in pool)
    for lab in expected_labels:
        count = pool_counts.get(lab, 0)
        if count < lowres_per_label:
            raise ValueError(
                f"{task}: demonstration pool has insufficient examples for label {lab!r}: "
                f"found {count}, required at least {lowres_per_label}"
            )

    splits = {
        "train": train_split,
        "pool": pool,
        "test": test_rows,
    }

    for seed in seeds:
        lowres = per_label_sample(
            pool,
            lowres_per_label,
            seed,
            required_labels=expected_labels,
            exact=True,
        )
        expected_count = len(expected_labels) * lowres_per_label
        if len(lowres) != expected_count:
            raise ValueError(
                f"{task}: seed {seed} produced {len(lowres)} lowres examples, expected {expected_count}"
            )
        splits[f"lowres_seed{seed}"] = lowres

    return splits


def prepare_task(task):
    train_raw, test_raw = load(task, "train"), load(task, "test")
    splits = split_data(train_raw, test_raw, task)

    write_jsonl(real_path(task, "train"), splits["train"])
    write_jsonl(real_path(task, "pool"), splits["pool"])
    write_jsonl(real_path(task, "test"), splits["test"])

    for seed in CFG["seeds"]:
        write_jsonl(real_path(task, f"lowres_seed{seed}"), splits[f"lowres_seed{seed}"])

    print(f"{task}: train {len(splits['train'])}, pool {len(splits['pool'])}, test {len(splits['test'])}")
    for seed in CFG["seeds"]:
        print(f"  {task} lowres_seed{seed}: {len(splits[f'lowres_seed{seed}'])} examples ({CFG['lowres_per_label']} per label)")
    return splits


def verify_real_data(task=None):
    """Verify that generated files match line counts and SHA-256 checksums."""
    target_tasks = [task] if task else TASKS
    errors = []
    checked = 0
    for t in target_tasks:
        file_keys = ["train", "pool", "test"] + [f"lowres_seed{s}" for s in CFG["seeds"]]
        for key in file_keys:
            path = real_path(t, key)
            manifest_key = f"{t}/{key}.jsonl"
            expected = CHECKSUMS.get(manifest_key)
            if not expected:
                errors.append(f"No manifest entry for {manifest_key}")
                continue
            if not path.exists():
                errors.append(f"Missing file: {path}")
                continue
            content = path.read_bytes()
            actual_sha = hashlib.sha256(content).hexdigest()
            actual_lines = len([line for line in content.splitlines() if line.strip()])
            if actual_lines != expected["lines"]:
                errors.append(
                    f"{manifest_key}: line count mismatch (got {actual_lines}, expected {expected['lines']})"
                )
            if actual_sha != expected["sha256"]:
                errors.append(
                    f"{manifest_key}: sha256 mismatch (got {actual_sha}, expected {expected['sha256']})"
                )
            checked += 1

    if errors:
        msg = f"Data verification failed ({len(errors)} errors):\n  " + "\n  ".join(errors)
        raise ValueError(msg)
    print(f"Verified {checked} real data files: all SHA-256 hashes and line counts match.")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="verify SHA-256 checksums and line counts of existing data")
    args = parser.parse_args(argv)

    if args.verify:
        verify_real_data()
        return

    for task in TASKS:
        prepare_task(task)

    verify_real_data()


if __name__ == "__main__":
    main()
