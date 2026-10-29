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

# Reproducible SHA-256 checksums and line counts for the real datasets (canonical UTF-8 JSONL with LF newlines).
CHECKSUMS = {
    "trec/train.jsonl": {
        "lines": 4907,
        "sha256": "b587c84d26b361bbe6f4126aff136e9d737c6c349059e89d2a3ce79e226f4331",
    },
    "trec/pool.jsonl": {
        "lines": 545,
        "sha256": "0a78fbe6e38b8b9eb4f7d9e24e619d63f855e073bffb1e00edb23508490d3a6e",
    },
    "trec/test.jsonl": {
        "lines": 500,
        "sha256": "415f526277d0f23f9bfe903a2a6d994529ec183820b6d9d0d55d47c3195aa0c0",
    },
    "trec/lowres_seed13.jsonl": {
        "lines": 60,
        "sha256": "901c0596a42b6fd0fa128cb1dd05b26bb1dbdcbd0c1a0a3dca9011beb7a03a29",
    },
    "trec/lowres_seed42.jsonl": {
        "lines": 60,
        "sha256": "fce20620829492d339fea73ea89eca3f00df2bb4d84c93d6c4b399bbc1eb5ae3",
    },
    "trec/lowres_seed7.jsonl": {
        "lines": 60,
        "sha256": "cc790390509218422206fa1a1fb8c3afffce93297c12a62b66a32b047a7a3fee",
    },
    "emotion/train.jsonl": {
        "lines": 14400,
        "sha256": "4c6e92df9649661ab1c6fefe06f83206382322e6997fcb70ca53d22d302fac57",
    },
    "emotion/pool.jsonl": {
        "lines": 1600,
        "sha256": "dde4702b5f5168f5fad977ba6b659f9304ae02a3cb37c18281e208271f847d40",
    },
    "emotion/test.jsonl": {
        "lines": 2000,
        "sha256": "d3b7176e8ab6be052151511a3e9843fa633c5747d0dae73980a2e52a4fb6cd88",
    },
    "emotion/lowres_seed13.jsonl": {
        "lines": 60,
        "sha256": "7bec4826ca96fa056ec5defc13a051f1c2ac25ef0270164219af13e1471af27f",
    },
    "emotion/lowres_seed42.jsonl": {
        "lines": 60,
        "sha256": "4af40bceb9cb819ceb76ac3a6d99bc6495aec7501fc541d579adfb2649c5f192",
    },
    "emotion/lowres_seed7.jsonl": {
        "lines": 60,
        "sha256": "4134701559a61d2a288119a1bca4d2a84227ee48fca0962821959e59b30b10d3",
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
