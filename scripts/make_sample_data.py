"""Write small FAKE real-data fixtures to data/sample/real/ (contract-valid, not real text).

    python scripts/make_sample_data.py

They let everyone build and test their stage before the real data and the
synthetic data exist. Never report numbers computed on them.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from synthtext import config  # noqa: E402
from synthtext.io import write_jsonl  # noqa: E402

SIZES = {"train": 40, "pool": 20, "test": 20}  # rows per split (spread over labels)
NOUNS = ["the eiffel tower", "a koala", "the nile", "jazz", "the moon", "bitcoin", "chess", "monsoon rain",
         "the printing press", "a violin", "mount everest", "the olympics", "photosynthesis", "a lighthouse"]
TEMPLATES = {
    "trec": {
        "abbreviation": ["what does {n} stand for ?", "what is the abbreviation for {n} ?"],
        "entity": ["what animal is linked to {n} ?", "what colour is {n} ?"],
        "description": ["why is {n} important ?", "how does {n} work ?"],
        "human": ["who invented {n} ?", "who first described {n} ?"],
        "location": ["where is {n} found ?", "in which country is {n} ?"],
        "numeric": ["how many years old is {n} ?", "how far away is {n} ?"],
    },
    "rotten_tomatoes": {
        "negative": ["a dull , lifeless film about {n} that drags on", "tedious and badly acted , like {n} on a bad day"],
        "positive": ["a warm , funny film about {n} that soars", "beautifully made , as moving as {n}"],
    },
    "sst5": {
        "very_negative": ["an utter disaster , the worst film about {n} ever", "unwatchable garbage about {n}"],
        "negative": ["a weak and forgettable take on {n}", "it never quite works , despite {n}"],
        "neutral": ["a film about {n} , with some good and some bad", "it is a movie about {n}"],
        "positive": ["a pleasant , solid film about {n}", "mostly charming , thanks to {n}"],
        "very_positive": ["a masterpiece , the best film about {n} in years", "brilliant and dazzling , like {n}"],
    },
    "emotion": {
        "sadness": ["i feel so lonely thinking about {n}", "i feel hopeless and hurt about {n}"],
        "joy": ["i feel really happy about {n} today", "i feel so glad and content with {n}"],
        "love": ["i feel such tender love for {n}", "i feel affectionate whenever i see {n}"],
        "anger": ["i feel so angry about {n} right now", "i feel irritated and annoyed by {n}"],
        "fear": ["i feel scared and anxious about {n}", "i feel nervous when i think of {n}"],
        "surprise": ["i feel amazed and shocked by {n}", "i feel so surprised by {n}"],
    },
}


def make(task_id: str, split: str, n: int, rng: random.Random) -> list[dict]:
    labels = config.labels(task_id)
    rows = []
    for i in range(n):
        label = labels[i % len(labels)]
        text = rng.choice(TEMPLATES[task_id][label]).format(n=rng.choice(NOUNS))
        rows.append({"id": f"{task_id}.sample-{split}.{i:05d}", "text": text, "label": label})
    return rows


def main() -> None:
    exp = config.experiments()
    for task_id in config.task_ids("all"):
        rng = random.Random(task_id)
        out = ROOT / "data" / "sample" / "real" / task_id
        splits = {s: make(task_id, s, n, rng) for s, n in SIZES.items()}
        for s, rows in splits.items():
            write_jsonl(out / f"{s}.jsonl", rows)
        for seed in exp["seeds"]:
            # 2 per class from the pool (the real run uses 10 per class)
            r = random.Random(seed)
            by_label: dict[str, list[dict]] = {}
            for row in splits["pool"]:
                by_label.setdefault(row["label"], []).append(row)
            low = [x for lab in sorted(by_label) for x in r.sample(by_label[lab], min(2, len(by_label[lab])))]
            write_jsonl(out / f"lowres_seed{seed}.jsonl", low)
        print(f"[{task_id}] sample files in {out}")


if __name__ == "__main__":
    main()
