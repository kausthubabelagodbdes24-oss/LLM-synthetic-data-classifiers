"""Build the training set for one (task, condition, seed) cell of the run matrix.

Conditions are defined in configs/experiments.yaml:
- mix     C1-C4: start from a stratified real sample of n_train (the "C1 sample");
          per label keep `real_share` of it and fill the rest of that label's
          count from a synthetic setting. Label counts are therefore identical
          across C1-C4 for a given seed.
- lowres  C5a-c: 10 real examples per class (from the pool), optionally plus
          AugGPT paraphrases of exactly those examples, or plus the same number
          of few-shot synthetic examples.
- full    CEIL: the whole real train split (train + pool).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from synthtext import config
from synthtext.io import read_jsonl
from synthtext.paths import real_dir, synthetic_dir
from workstreams.ws2_training.sampling import by_label, stratified_sample


@dataclass
class TrainingSet:
    records: list[dict]
    notes: list[str] = field(default_factory=list)

    @property
    def label_counts(self) -> Counter:
        return Counter(r["label"] for r in self.records)


def _real(task_id: str, name: str) -> list[dict]:
    return read_jsonl(real_dir(task_id) / f"{name}.jsonl")


def _synthetic(task_id: str, setting: str) -> list[dict]:
    return read_jsonl(synthetic_dir(task_id) / f"{setting}.jsonl")


def c1_sample(task_id: str, seed: int, n_train: int) -> list[dict]:
    return stratified_sample(_real(task_id, "train"), n_train, seed)


def _take(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    pool = sorted(pool, key=lambda r: r["id"])
    return rng.sample(pool, n) if n <= len(pool) else list(pool)


def build_mix(task_id: str, seed: int, n_train: int, real_share: float, setting: str | None) -> TrainingSet:
    base = c1_sample(task_id, seed, n_train)
    if real_share >= 1.0 or setting is None:
        return TrainingSet(base)
    synth = by_label(_synthetic(task_id, setting))
    if not synth:
        return TrainingSet([], [f"no synthetic file for '{setting}'"])
    rng = random.Random(seed)
    records, notes = [], []
    for label, items in sorted(by_label(base).items()):
        n_label = len(items)
        n_real = round(n_label * real_share)
        records.extend(sorted(items, key=lambda r: r["id"])[:n_real])
        want = n_label - n_real
        got = _take(synth.get(label, []), want, rng)
        if len(got) < want:
            notes.append(f"{label}: wanted {want} synthetic, only {len(got)} available")
        records.extend(got)
    rng.shuffle(records)
    return TrainingSet(records, notes)


def build_lowres(task_id: str, seed: int, add: str | None) -> TrainingSet:
    base = read_jsonl(real_dir(task_id) / f"lowres_seed{seed}.jsonl")
    if not base:
        return TrainingSet([], [f"no lowres_seed{seed}.jsonl"])
    if add is None:
        return TrainingSet(base)
    base_ids = {r["id"] for r in base}
    aug = [r for r in _synthetic(task_id, "auggpt") if r["source_ids"] and r["source_ids"][0] in base_ids]
    if add == "auggpt":
        notes = [] if aug else ["no AugGPT rows for these seeds"]
        return TrainingSet(base + aug, notes)
    if add == "few_shot_matched":
        k = config.generation()["auggpt_paraphrases"]
        aug_counts = Counter(r["label"] for r in aug)
        synth = by_label(_synthetic(task_id, "few_shot"))
        if not synth:
            return TrainingSet([], ["no few_shot synthetic file"])
        rng = random.Random(seed)
        extra, notes = [], []
        for label, items in sorted(by_label(base).items()):
            want = aug_counts.get(label, k * len(items))  # match AugGPT; fall back to k per seed
            got = _take(synth.get(label, []), want, rng)
            if len(got) < want:
                notes.append(f"{label}: wanted {want} few-shot, only {len(got)}")
            extra.extend(got)
        return TrainingSet(base + extra, notes)
    raise ValueError(f"unknown lowres add '{add}'")


def build(task_id: str, condition: str, seed: int, n_train: int | None = None) -> TrainingSet:
    exp = config.experiments()
    spec = exp["conditions"][condition]
    n_train = n_train or exp["n_train"]
    kind = spec["kind"]
    if kind == "mix":
        return build_mix(task_id, seed, n_train, spec["real_share"], spec.get("synthetic"))
    if kind == "lowres":
        return build_lowres(task_id, seed, spec.get("add"))
    if kind == "full":
        return TrainingSet(_real(task_id, "train") + _real(task_id, "pool"))
    raise ValueError(f"unknown condition kind '{kind}'")
