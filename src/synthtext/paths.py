"""Where files live.

Three environment variables let you point the whole pipeline at the small
fixture data instead of the real data (see README, "Working on samples"):

    SYNTHTEXT_DATA=data/sample          # default: data
    SYNTHTEXT_RESULTS=results/sample    # default: results
    SYNTHTEXT_FIGURES=results/sample/figures   # default: reports/figures

All are relative to the repo root unless absolute.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: str) -> Path:
    value = os.environ.get(env_var, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def data_root() -> Path:
    return _resolve("SYNTHTEXT_DATA", "data")


def results_root() -> Path:
    return _resolve("SYNTHTEXT_RESULTS", "results")


def configs_dir() -> Path:
    return ROOT / "configs"


def real_dir(task: str) -> Path:
    return data_root() / "real" / task


def synthetic_dir(task: str) -> Path:
    return data_root() / "synthetic" / task


def cache_dir() -> Path:
    return data_root() / "cache"


def figures_dir() -> Path:
    return _resolve("SYNTHTEXT_FIGURES", "reports/figures")


def annotation_dir() -> Path:
    """reports/annotation normally; follows SYNTHTEXT_FIGURES for sample runs."""
    return figures_dir().parent / "annotation"
