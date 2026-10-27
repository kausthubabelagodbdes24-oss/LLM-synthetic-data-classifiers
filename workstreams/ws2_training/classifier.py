"""Frozen sentence embeddings + logistic regression (Reimers & Gurevych 2019 for the encoder)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from synthtext.embeddings import embed


@dataclass
class Fitted:
    model: LogisticRegression
    embedding_model: str

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.model.predict(embed(texts, self.embedding_model))


def fit(texts: list[str], labels: list[str], embedding_model: str, clf_cfg: dict, seed: int = 0) -> Fitted:
    if len(set(labels)) < 2:
        raise ValueError("training set has fewer than 2 labels")
    X = embed(texts, embedding_model)
    model = LogisticRegression(C=clf_cfg["C"], max_iter=clf_cfg["max_iter"], random_state=seed)
    model.fit(X, labels)
    return Fitted(model, embedding_model)


def score(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, float]:
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
