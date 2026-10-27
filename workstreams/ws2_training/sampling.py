"""Deterministic, label-aware sampling helpers."""

from __future__ import annotations

import random
from collections import defaultdict


def by_label(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["label"]].append(r)
    return groups


def proportional_counts(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across labels in proportion to `sizes` (largest-remainder rounding)."""
    total = sum(sizes.values())
    if total == 0:
        return {k: 0 for k in sizes}
    if n >= total:
        return dict(sizes)
    raw = {k: n * v / total for k, v in sizes.items()}
    counts = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: (raw[k] - counts[k], k), reverse=True)[:leftover]:
        counts[k] += 1
    return counts


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """n records keeping each label's share; deterministic for a given seed."""
    groups = by_label(records)
    counts = proportional_counts({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    out: list[dict] = []
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda r: r["id"])
        out.extend(rng.sample(items, counts[label]))
    rng.shuffle(out)
    return out


def split_off(records: list[dict], frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified split: (held_out with ~frac of each label, rest)."""
    held = stratified_sample(records, round(len(records) * frac), seed)
    held_ids = {r["id"] for r in held}
    rest = [r for r in records if r["id"] not in held_ids]
    return held, rest


def per_label_sample(records: list[dict], per_label: int, seed: int) -> list[dict]:
    groups = by_label(records)
    rng = random.Random(seed)
    out: list[dict] = []
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda r: r["id"])
        out.extend(rng.sample(items, min(per_label, len(items))))
    return out
