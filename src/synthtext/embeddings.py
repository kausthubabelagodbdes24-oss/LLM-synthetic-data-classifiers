"""Sentence embeddings with an on-disk cache (shared; maintained by WS2).

    from synthtext.embeddings import embed
    X = embed(["a text", "another"], model="sentence-transformers/all-mpnet-base-v2")

Vectors are L2-normalised, so a dot product is a cosine similarity.
Each (model, text) pair is encoded once and stored under data/cache/emb/.

Set SYNTHTEXT_EMBEDDER=hashing (or pass model="hashing") to use a fast,
offline bag-of-words embedder. It is only for tests and smoke runs: never
report numbers produced with it.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from synthtext.io import text_hash
from synthtext.paths import cache_dir

HASHING = "hashing"


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


class _HashingEmbedder:
    dim = 1024

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=self.dim, ngram_range=(1, 2), alternate_sign=False, norm="l2"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._vec.transform(texts).toarray().astype(np.float32)


class _SentenceTransformerEmbedder:
    def __init__(self, model: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ImportError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`, "
                "or set SYNTHTEXT_EMBEDDER=hashing for an offline smoke test."
            ) from exc
        self._model = SentenceTransformer(model)
        self._batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 500,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


@lru_cache(maxsize=None)
def _backend(model: str):
    if model == HASHING:
        return _HashingEmbedder()
    return _SentenceTransformerEmbedder(model)


class _Cache:
    """keys.json (list of hashes) + vectors.npy, one pair per model."""

    def __init__(self, model: str) -> None:
        self.dir: Path = cache_dir() / "emb" / _slug(model)
        self.keys: list[str] = []
        self.index: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        keys_path, vec_path = self.dir / "keys.json", self.dir / "vectors.npy"
        if keys_path.exists() and vec_path.exists():
            self.keys = json.loads(keys_path.read_text())
            self.vectors = np.load(vec_path)
            self.index = {k: i for i, k in enumerate(self.keys)}

    def get(self, key: str) -> np.ndarray | None:
        i = self.index.get(key)
        return None if i is None or self.vectors is None else self.vectors[i]

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        start = len(self.keys)
        self.keys.extend(keys)
        self.index.update({k: start + i for i, k in enumerate(keys)})
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self.vectors)
        (self.dir / "keys.json").write_text(json.dumps(self.keys))


_caches: dict[str, _Cache] = {}


def resolve_model(model: str) -> str:
    """SYNTHTEXT_EMBEDDER=hashing overrides every model (smoke tests only)."""
    return HASHING if os.environ.get("SYNTHTEXT_EMBEDDER") == HASHING else model


def embed(texts: list[str], model: str, use_cache: bool = True) -> np.ndarray:
    model = resolve_model(model)
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    backend = _backend(model)
    if model == HASHING or not use_cache:
        return backend.encode(list(texts))

    cache = _caches.setdefault(model, _Cache(model))
    keys = [text_hash(t, salt=model) for t in texts]
    missing = {}
    for k, t in zip(keys, texts):
        if cache.get(k) is None and k not in missing:
            missing[k] = t
    if missing:
        new_vectors = backend.encode(list(missing.values()))
        cache.add(list(missing.keys()), new_vectors)
    return np.stack([cache.get(k) for k in keys])
