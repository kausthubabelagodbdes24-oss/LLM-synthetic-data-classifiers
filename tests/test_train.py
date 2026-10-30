"""Behavior tests for classifier training, validation, resume, and targeted reruns."""

import csv
import pytest
from pathlib import Path

from common import CFG, labels, read_jsonl, write_jsonl, real_path, results_dir
from ws2_training import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    import shutil
    shutil.copytree(ROOT / "data" / "sample" / "real", tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("SYNTHTEXT_EMBEDDER", "hashing")
    monkeypatch.setenv("SYNTHTEXT_GENERATION_MODE", "fake")


def test_missing_class_coverage_raises(tmp_path):
    # Remove one label from train.jsonl
    p = tmp_path / "data" / "real" / "trec" / "train.jsonl"
    rows = [r for r in read_jsonl(p) if r["label"] != "abbreviation"]
    write_jsonl(p, rows)

    with pytest.raises(ValueError, match="incomplete class coverage|Missing required label"):
        train.run("trec", "C1", 13)


def test_missing_test_set_raises(tmp_path):
    p = tmp_path / "data" / "real" / "trec" / "test.jsonl"
    p.unlink()

    with pytest.raises(ValueError, match="held-out real test set is missing"):
        train.run("trec", "C1", 13)


def test_missing_test_label_raises(tmp_path):
    p = tmp_path / "data" / "real" / "trec" / "test.jsonl"
    rows = [r for r in read_jsonl(p) if r["label"] != "numeric"]
    write_jsonl(p, rows)

    with pytest.raises(ValueError, match="held-out test set missing label"):
        train.run("trec", "C1", 13)


def test_metric_ranges():
    row = train.run("trec", "C1", 13)
    assert row is not None
    assert 0.0 <= row["macro_f1"] <= 1.0
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["n_train"] > 0
    assert row["task"] == "trec"
    assert row["condition"] == "C1"
    assert row["seed"] == 13


def test_resume_skips_existing_rows(tmp_path, monkeypatch):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1,C5a"])
    assert path.exists()
    rows1 = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows1) == 12

    # Monkeypatch run to fail if called again
    def fail_run(*args, **kwargs):
        raise RuntimeError("run should not be called on resume")

    monkeypatch.setattr(train, "run", fail_run)
    # Running without --force should skip all 12 rows
    train.main(["--conditions", "C1,C5a"])
    rows2 = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows2) == 12
    assert rows1 == rows2


def test_force_rerun_preserves_unaffected_conditions(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1,C5a"])
    rows_initial = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows_initial) == 12

    # Modify macro_f1 of C5a rows in runs.csv to verify they are preserved
    modified_rows = []
    for r in rows_initial:
        if r["condition"] == "C5a":
            r["macro_f1"] = "0.9999"
        modified_rows.append(r)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=train.COLUMNS)
        writer.writeheader()
        writer.writerows(modified_rows)

    # Rerun C1 with --force
    train.main(["--conditions", "C1", "--force"])
    rows_after = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows_after) == 12

    # C5a rows must still have the preserved value 0.9999
    c5a_rows = [r for r in rows_after if r["condition"] == "C5a"]
    assert len(c5a_rows) == 6
    assert all(r["macro_f1"] == "0.9999" for r in c5a_rows)

    # C1 rows must have been recalculated
    c1_rows = [r for r in rows_after if r["condition"] == "C1"]
    assert len(c1_rows) == 6
    assert all(r["macro_f1"] != "0.9999" for r in c1_rows)


def test_incompatible_n_train_row_is_replaced(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1"])
    rows_initial = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows_initial) == 6

    # Tamper with one row's n_train to simulate an old target
    rows_initial[0]["n_train"] = "999"
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=train.COLUMNS)
        writer.writeheader()
        writer.writerows(rows_initial)

    # Run without --force: the row with 999 must be replaced with the actual n_train
    train.main(["--conditions", "C1"])
    rows_after = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows_after) == 6
    assert all(r["n_train"] != "999" for r in rows_after)


def test_runs_csv_canonical_lf_newlines(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1,C5a"])
    raw_bytes = path.read_bytes()
    assert b"\r" not in raw_bytes, "Found CRLF in runs.csv output; canonical LF required"
    lines = [line for line in raw_bytes.split(b"\n") if line.strip()]
    assert len(lines) == 13, f"Expected header + 12 rows, got {len(lines)}"
