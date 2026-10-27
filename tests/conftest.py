"""Every test runs against a temporary copy of the sample data, with the offline
hashing embedder, so tests never touch data/real or download anything."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REAL = ROOT / "data" / "sample" / "real"


@pytest.fixture()
def sample_env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    shutil.copytree(SAMPLE_REAL, data / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(data))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("SYNTHTEXT_FIGURES", str(tmp_path / "results" / "figures"))
    monkeypatch.setenv("SYNTHTEXT_EMBEDDER", "hashing")
    return tmp_path
