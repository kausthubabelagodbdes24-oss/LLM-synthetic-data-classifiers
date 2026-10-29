"""Write small FAKE real-data files to data/sample/ so everyone can test before the real data exists.
Never report numbers computed on them."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CFG, ROOT, TASKS, labels, per_label_sample, write_jsonl  # noqa: E402

WORDS = ["the eiffel tower", "a koala", "the nile", "jazz", "the moon", "chess", "monsoon rain", "a violin"]
TEMPLATES = {
    "abbreviation": "what does {w} stand for ?", "entity": "what animal is linked to {w} ?",
    "description": "why is {w} important ?", "human": "who first described {w} ?",
    "location": "where is {w} found ?", "numeric": "how many years old is {w} ?",
    "sadness": "i feel so lonely thinking about {w}", "joy": "i feel really happy about {w} today",
    "love": "i feel such tender love for {w}", "anger": "i feel so angry about {w} right now",
    "fear": "i feel scared and anxious about {w}", "surprise": "i feel amazed and shocked by {w}",
}


def main():
    for task in TASKS:
        rng = random.Random(task)
        out = ROOT / "data" / "sample" / "real" / task
        splits = {}
        for split, n in (("train", 60), ("pool", 30), ("test", 30)):
            splits[split] = [{"id": f"{task}.{split}.{i:05d}", "label": (lab := labels(task)[i % 6]),
                              "text": TEMPLATES[lab].format(w=rng.choice(WORDS))} for i in range(n)]
            write_jsonl(out / f"{split}.jsonl", splits[split])
        for seed in CFG["seeds"]:
            write_jsonl(out / f"lowres_seed{seed}.jsonl", per_label_sample(splits["pool"], 2, seed))
    print("sample data written to data/sample/real/")


if __name__ == "__main__":
    main()
