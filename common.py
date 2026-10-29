"""Shared helpers: config, file paths, JSONL, sampling, embeddings.

Environment variables (for testing on the fake sample data):
    SYNTHTEXT_DATA=data/sample      read/write data here instead of data/
    SYNTHTEXT_RESULTS=results/sample
    SYNTHTEXT_EMBEDDER=hashing      offline word-hashing embedder (tests only; never report its numbers)
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
TASKS = list(CFG["tasks"])


def data_dir() -> Path:
    return ROOT / os.environ.get("SYNTHTEXT_DATA", "data")


def results_dir() -> Path:
    path = ROOT / os.environ.get("SYNTHTEXT_RESULTS", "results")
    path.mkdir(parents=True, exist_ok=True)
    return path


def real_path(task, name):            # name: train | test | pool | lowres_seed13 ...
    return data_dir() / "real" / task / f"{name}.jsonl"


def synth_path(task, setting):        # setting: zero_shot | few_shot | auggpt
    return data_dir() / "synthetic" / task / f"{setting}.jsonl"


def labels(task):
    return list(CFG["tasks"][task]["labels"])


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text):
    return " ".join(text.lower().split())


def per_label_sample(rows, n, seed, required_labels=None, exact=False):
    """Sample n rows of each label reproducibly.

    If required_labels is given, checks that all required labels exist in rows.
    If exact=True, requires every label to have at least n rows, raising ValueError
    rather than silently undersampling.
    """
    rng = random.Random(seed)
    out = []
    expected_labels = sorted(required_labels) if required_labels is not None else sorted({r["label"] for r in rows})
    if required_labels is not None:
        present_labels = {r["label"] for r in rows}
        missing = [lab for lab in expected_labels if lab not in present_labels]
        if missing:
            raise ValueError(f"Missing required label(s): {missing}")
    for lab in expected_labels:
        group = sorted((r for r in rows if r["label"] == lab), key=lambda r: r["id"])
        if exact and len(group) < n:
            raise ValueError(f"Insufficient examples for label {lab!r}: requested {n}, found {len(group)}")
        out += rng.sample(group, min(n, len(group)))
    return out


_models = {}


def embed(texts):
    """L2-normalised sentence embeddings (Sentence-BERT)."""
    if os.environ.get("SYNTHTEXT_EMBEDDER") == "hashing":
        from sklearn.feature_extraction.text import HashingVectorizer
        vec = HashingVectorizer(n_features=1024, ngram_range=(1, 2), alternate_sign=False)
        return vec.transform(texts).toarray()
    name = CFG["embedding_model"]
    if name not in _models:
        from sentence_transformers import SentenceTransformer
        _models[name] = SentenceTransformer(name)
    return np.asarray(_models[name].encode(texts, batch_size=64, normalize_embeddings=True))
