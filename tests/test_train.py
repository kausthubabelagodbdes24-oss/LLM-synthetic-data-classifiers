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
    expected_rows = len(train.TASKS) * 2 * len(CFG["seeds"])
    assert len(rows1) == expected_rows

    # Monkeypatch run to fail if called again
    def fail_run(*args, **kwargs):
        raise RuntimeError("run should not be called on resume")

    monkeypatch.setattr(train, "run", fail_run)
    # Running without --force should skip all rows
    train.main(["--conditions", "C1,C5a"])
    rows2 = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows2) == expected_rows
    assert rows1 == rows2


def test_force_rerun_preserves_unaffected_conditions(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1,C5a"])
    rows_initial = list(csv.DictReader(open(path, encoding="utf-8")))
    expected_rows = len(train.TASKS) * 2 * len(CFG["seeds"])
    assert len(rows_initial) == expected_rows

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
    assert len(rows_after) == expected_rows

    # C5a rows must still have the preserved value 0.9999
    c5a_rows = [r for r in rows_after if r["condition"] == "C5a"]
    expected_cond_rows = len(train.TASKS) * len(CFG["seeds"])
    assert len(c5a_rows) == expected_cond_rows
    assert all(r["macro_f1"] == "0.9999" for r in c5a_rows)

    # C1 rows must have been recalculated
    c1_rows = [r for r in rows_after if r["condition"] == "C1"]
    assert len(c1_rows) == expected_cond_rows
    assert all(r["macro_f1"] != "0.9999" for r in c1_rows)


def test_incompatible_n_train_row_is_replaced(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1"])
    rows_initial = list(csv.DictReader(open(path, encoding="utf-8")))
    expected_rows = len(train.TASKS) * len(CFG["seeds"])
    assert len(rows_initial) == expected_rows

    # Tamper with one row's n_train to simulate an old target
    rows_initial[0]["n_train"] = "999"
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=train.COLUMNS)
        writer.writeheader()
        writer.writerows(rows_initial)

    # Run without --force: the row with 999 must be replaced with the actual n_train
    train.main(["--conditions", "C1"])
    rows_after = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows_after) == expected_rows
    assert all(r["n_train"] != "999" for r in rows_after)


def test_runs_csv_canonical_lf_newlines(tmp_path):
    path = tmp_path / "results" / "runs.csv"
    train.main(["--conditions", "C1,C5a"])
    raw_bytes = path.read_bytes()
    assert b"\r" not in raw_bytes, "Found CRLF in runs.csv output; canonical LF required"
    lines = [line for line in raw_bytes.split(b"\n") if line.strip()]
    expected_lines = 1 + len(train.TASKS) * 2 * len(CFG["seeds"])
    assert len(lines) == expected_lines, f"Expected header + {expected_lines - 1} rows, got {len(lines)}"


def test_exact_real_sampling_fails_closed_when_insufficient_examples(tmp_path, monkeypatch):
    """P1 defect regression test: exact sampling must fail with clear explanation when target impossible."""
    # Simulate real mode (not fake/sample mode)
    monkeypatch.delenv("SYNTHTEXT_GENERATION_MODE", raising=False)
    monkeypatch.delenv("SYNTHTEXT_EMBEDDER", raising=False)
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "real_data"))

    target_per_label = CFG["train_per_label"]
    found_abbr = target_per_label - 10
    train_rows = []
    for i in range(found_abbr):
        train_rows.append({"id": f"trec.train.abbr.{i}", "text": f"abbr {i}", "label": "abbreviation"})
    for lab in ["description", "entity", "human", "location", "numeric"]:
        for i in range(target_per_label):
            train_rows.append({"id": f"trec.train.{lab}.{i}", "text": f"{lab} {i}", "label": lab})
    p = tmp_path / "real_data" / "real" / "trec" / "train.jsonl"
    write_jsonl(p, train_rows)

    with pytest.raises(ValueError, match="Cannot sample approved target for trec C1") as excinfo:
        train.training_set("trec", "C1", 13, exact=True)
    msg = str(excinfo.value)
    assert f"Insufficient examples for label 'abbreviation': requested {target_per_label}, found {found_abbr}" in msg
    assert f"approved target requires {target_per_label} examples per label" in msg
    assert "Borrowing from demonstration-pool or test splits" in msg or "Silently undersampling" in msg


def test_resume_validation_enforces_approved_target_independently(tmp_path, monkeypatch):
    """P1 defect regression test: resume must discard undersized rows (e.g. 576) in real mode."""
    # Simulate real mode
    monkeypatch.delenv("SYNTHTEXT_GENERATION_MODE", raising=False)
    monkeypatch.delenv("SYNTHTEXT_EMBEDDER", raising=False)
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "real_data"))

    path = tmp_path / "results" / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seed runs.csv with an undersized C1 row (576) and a valid C5a row (60)
    rows = [
        {"task": "trec", "condition": "C1", "seed": "13", "n_train": "576", "macro_f1": "0.7471", "accuracy": "0.768"},
        {"task": "trec", "condition": "C5a", "seed": "13", "n_train": "60", "macro_f1": "0.5974", "accuracy": "0.640"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=train.COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # When running with --conditions C1, the undersized row must be discarded
    # Monkeypatch run to avoid real training and verify it was triggered
    run_called = []
    def mock_run(task, cond, seed):
        run_called.append((task, cond, seed))
        return None

    monkeypatch.setattr(train, "run", mock_run)
    train.main(["--conditions", "C1"])

    # Verify mock_run was called for trec C1 13 because 576 was discarded
    assert ("trec", "C1", 13) in run_called

    # Verify that the unselected condition C5a was preserved untouched
    after_rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert any(r["condition"] == "C5a" and r["n_train"] == "60" for r in after_rows)


def test_targeted_rerun_preserves_unrequested_conditions_despite_count_discrepancy(tmp_path, monkeypatch):
    """P2 defect regression test: --conditions C1 --force must not delete C5a rows."""
    path = tmp_path / "results" / "runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create runs.csv with C1 and C5a rows
    initial_rows = [
        {"task": "emotion", "condition": "C1", "seed": "13", "n_train": "600", "macro_f1": "0.4629", "accuracy": "0.506"},
        {"task": "trec", "condition": "C5a", "seed": "13", "n_train": "60", "macro_f1": "0.5974", "accuracy": "0.640"},
        {"task": "emotion", "condition": "C5a", "seed": "13", "n_train": "60", "macro_f1": "0.3381", "accuracy": "0.373"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=train.COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(initial_rows)

    # In sample mode, C5a would normally produce 12 rows (not 60).
    # Previous code evaluated C5a, saw 60 != 12, and deleted C5a during --conditions C1 --force!
    # Run with --conditions C1 --force:
    train.main(["--conditions", "C1", "--force"])

    after_rows = list(csv.DictReader(open(path, encoding="utf-8")))
    # Verify C5a rows are completely preserved!
    c5a_rows = [r for r in after_rows if r["condition"] == "C5a"]
    assert len(c5a_rows) == 2, f"Expected 2 C5a rows preserved, got {len(c5a_rows)}: {after_rows}"
    assert any(r["task"] == "trec" and r["seed"] == "13" and r["n_train"] == "60" for r in c5a_rows)
    assert any(r["task"] == "emotion" and r["seed"] == "13" and r["n_train"] == "60" for r in c5a_rows)

