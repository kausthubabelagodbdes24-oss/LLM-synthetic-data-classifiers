"""Download the four datasets and write the real-data files everyone uses.

    python -m workstreams.ws2_training.prepare_data --task all

For each task, writes to data/real/<task>/:
    train.jsonl           train split minus the example pool (real training data)
    pool.jsonl            10% of train, stratified (few-shot examples + low-resource seeds)
    test.jsonl            the held-out real test split
    lowres_seed<S>.jsonl  10 examples per class drawn from the pool, one file per seed
    manifest.json         counts per label, for the report
The pool split uses a fixed seed (0), so every machine gets identical files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from synthtext import config
from synthtext.contracts import check_file
from synthtext.io import write_jsonl
from synthtext.paths import real_dir
from workstreams.ws2_training.sampling import per_label_sample, split_off

POOL_SEED = 0


def map_label(raw, task_cfg: dict) -> str | None:
    if "label_map" in task_cfg:
        mapping = task_cfg["label_map"]
        if raw in mapping:
            return mapping[raw]
        return mapping.get(str(raw)) if not isinstance(raw, str) else None
    for needle, name in task_cfg["label_contains"].items():
        if needle.lower() in str(raw).lower():
            return name
    return None


def load_split(task_id: str, split: str) -> list[dict]:
    from datasets import load_dataset  # heavy import: only when downloading

    cfg = config.task(task_id)
    ds = load_dataset(cfg["hf_id"], cfg.get("hf_config"), split=split)
    records, unmapped = [], Counter()
    for i, row in enumerate(ds):
        label = map_label(row[cfg["label_field"]], cfg)
        if label is None:
            unmapped[row[cfg["label_field"]]] += 1
            continue
        text = " ".join(str(row[cfg["text_field"]]).split())
        if text:
            records.append({"id": f"{task_id}.{split}.{i:05d}", "text": text, "label": label})
    if unmapped:
        raise ValueError(f"{task_id}/{split}: labels with no mapping in tasks.yaml: {dict(unmapped)}")
    return records


def prepare(task_id: str) -> dict:
    cfg = config.task(task_id)
    exp = config.experiments()
    train_all = load_split(task_id, cfg["train_split"])
    test = load_split(task_id, cfg["test_split"])
    pool, train = split_off(train_all, exp["example_pool_frac"], seed=POOL_SEED)

    out = real_dir(task_id)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "pool.jsonl", pool)
    write_jsonl(out / "test.jsonl", test)
    for seed in exp["seeds"]:
        write_jsonl(out / f"lowres_seed{seed}.jsonl",
                    per_label_sample(pool, exp["low_resource_per_class"], seed))

    problems = []
    for name in ("train", "pool", "test"):
        problems += check_file(out / f"{name}.jsonl", "real", task_id)
    if problems:
        raise ValueError("Contract check failed:\n" + "\n".join(problems[:20]))

    manifest = {
        "task": task_id,
        "hf_id": cfg["hf_id"],
        "counts": {name: dict(Counter(r["label"] for r in recs))
                   for name, recs in (("train", train), ("pool", pool), ("test", test))},
        "sizes": {"train": len(train), "pool": len(pool), "test": len(test)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    args = ap.parse_args(argv)
    for task_id in config.task_ids(args.task):
        m = prepare(task_id)
        print(f"[{task_id}] train {m['sizes']['train']}, pool {m['sizes']['pool']}, test {m['sizes']['test']}")


if __name__ == "__main__":
    main()
