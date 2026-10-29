"""WS2 (Member B): download the two datasets and write the real-data files.

    python -m ws2_training.prepare_data

Writes data/real/<task>/:
    train.jsonl         real training data
    pool.jsonl          10% of train, kept apart: few-shot examples + low-resource seeds
    test.jsonl          held-out real test set
    lowres_seed<S>.jsonl  10 per label from the pool, one file per seed
Same seed on every machine, so everyone gets identical files.
"""

import random

from common import CFG, TASKS, per_label_sample, real_path, write_jsonl


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


def main():
    for task in TASKS:
        train, test = load(task, "train"), load(task, "test")
        random.Random(0).shuffle(train)
        cut = len(train) // 10
        pool, train = train[:cut], train[cut:]
        write_jsonl(real_path(task, "train"), train)
        write_jsonl(real_path(task, "pool"), pool)
        write_jsonl(real_path(task, "test"), test)
        for seed in CFG["seeds"]:
            write_jsonl(real_path(task, f"lowres_seed{seed}"), per_label_sample(pool, CFG["lowres_per_label"], seed))
        print(f"{task}: train {len(train)}, pool {len(pool)}, test {len(test)}")


if __name__ == "__main__":
    main()
