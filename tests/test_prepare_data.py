"""Behavioral tests for data preparation, label mapping, split disjointness,
strict coverage validation, and cross-platform canonical JSONL verification.
Runs offline without network access.
"""

import hashlib
import json
import pytest

from common import CFG, labels, per_label_sample, real_path, write_jsonl
from ws2_training.prepare_data import CHECKSUMS, split_data, to_label, verify_real_data


def test_to_label_trec():
    t = CFG["tasks"]["trec"]
    assert to_label("description and abstract concepts", t) == "description"
    assert to_label("numeric values", t) == "numeric"
    assert to_label("entities", t) == "entity"
    assert to_label("human beings", t) == "human"
    assert to_label("locations", t) == "location"
    assert to_label("abbreviations", t) == "abbreviation"
    assert to_label("UNKNOWN:xyz", t) is None


def test_to_label_emotion():
    t = CFG["tasks"]["emotion"]
    assert to_label(0, t) == "sadness"
    assert to_label(1, t) == "joy"
    assert to_label(2, t) == "love"
    assert to_label(3, t) == "anger"
    assert to_label(4, t) == "fear"
    assert to_label(5, t) == "surprise"
    assert to_label(999, t) is None


def test_write_jsonl_lf_newline_convention(tmp_path):
    """Verify write_jsonl produces canonical UTF-8 with LF only (no CRLF) on all platforms."""
    target = tmp_path / "test.jsonl"
    rows = [{"id": "1", "text": "line one"}, {"id": "2", "text": "line two"}]
    write_jsonl(target, rows)

    raw_bytes = target.read_bytes()
    assert b"\r" not in raw_bytes, "write_jsonl emitted carriage return byte (CRLF) on write"
    assert raw_bytes.endswith(b"\n")
    assert raw_bytes == b'{"id": "1", "text": "line one"}\n{"id": "2", "text": "line two"}\n'

    # Verify append mode preserves canonical LF
    write_jsonl(target, [{"id": "3", "text": "line three"}], append=True)
    raw_bytes_appended = target.read_bytes()
    assert b"\r" not in raw_bytes_appended, "write_jsonl emitted carriage return byte in append mode"
    assert raw_bytes_appended.endswith(b"\n")


def test_per_label_sample_backward_compat():
    rows = [
        {"id": "1", "label": "joy"},
        {"id": "2", "label": "joy"},
        {"id": "3", "label": "sadness"},
    ]
    # Default exact=False, required_labels=None takes min(n, len(group))
    sample = per_label_sample(rows, 5, seed=13)
    assert len(sample) == 3
    assert {r["id"] for r in sample} == {"1", "2", "3"}


def test_per_label_sample_rejects_missing_required_label():
    rows = [
        {"id": "1", "label": "joy"},
        {"id": "2", "label": "sadness"},
    ]
    with pytest.raises(ValueError, match="Missing required label"):
        per_label_sample(rows, 1, seed=13, required_labels=["joy", "sadness", "anger"])


def test_per_label_sample_rejects_insufficient_class_coverage():
    rows = [
        {"id": f"joy_{i}", "label": "joy"} for i in range(10)
    ] + [
        {"id": f"anger_{i}", "label": "anger"} for i in range(5)  # only 5, but 10 requested
    ]
    with pytest.raises(ValueError, match="Insufficient examples for label 'anger'"):
        per_label_sample(rows, 10, seed=13, exact=True)


def test_per_label_sample_reproducible_and_seed_distinct():
    task = "trec"
    labs = labels(task)
    rows = []
    for lab in labs:
        for i in range(20):
            rows.append({"id": f"{task}.{lab}.{i:03d}", "label": lab, "text": f"sample {lab} {i}"})

    sample_13_a = per_label_sample(rows, 10, seed=13, required_labels=labs, exact=True)
    sample_13_b = per_label_sample(rows, 10, seed=13, required_labels=labs, exact=True)
    assert [r["id"] for r in sample_13_a] == [r["id"] for r in sample_13_b]

    sample_42 = per_label_sample(rows, 10, seed=42, required_labels=labs, exact=True)
    assert [r["id"] for r in sample_13_a] != [r["id"] for r in sample_42]


def test_split_data_partition_disjointness_and_counts():
    task = "trec"
    labs = labels(task)
    train_rows = []
    for lab in labs:
        for i in range(300):
            train_rows.append({"id": f"trec.train.{lab}.{i:04d}", "label": lab, "text": f"text {lab} {i}"})
    test_rows = []
    for lab in labs:
        for i in range(10):
            test_rows.append({"id": f"trec.test.{lab}.{i:04d}", "label": lab, "text": f"test {lab} {i}"})

    splits = split_data(train_rows, test_rows, task)

    # 10% pool cut: 1800 // 10 = 180
    assert len(splits["pool"]) == 180
    assert len(splits["train"]) == 1620
    assert len(splits["test"]) == 60

    # Pool and train must be completely disjoint
    pool_ids = {r["id"] for r in splits["pool"]}
    train_ids = {r["id"] for r in splits["train"]}
    test_ids = {r["id"] for r in splits["test"]}
    assert len(pool_ids.intersection(train_ids)) == 0
    assert len(pool_ids.intersection(test_ids)) == 0

    # Low-resource samples for all seeds must have exactly 10 per label = 60
    for seed in CFG["seeds"]:
        lowres = splits[f"lowres_seed{seed}"]
        assert len(lowres) == 60
        counts = {lab: sum(1 for r in lowres if r["label"] == lab) for lab in labs}
        assert all(c == 10 for c in counts.values())


def test_split_data_rejects_insufficient_pool_class_coverage():
    task = "trec"
    labs = labels(task)
    train_rows = []
    for i in range(98):
        lab = labs[1 + (i % (len(labs) - 1))]  # omit abbreviation
        train_rows.append({"id": f"t.{i}", "label": lab, "text": f"text {i}"})
    train_rows.append({"id": "t.abbr.1", "label": "abbreviation", "text": "abbr 1"})
    train_rows.append({"id": "t.abbr.2", "label": "abbreviation", "text": "abbr 2"})

    test_rows = [{"id": f"test.{lab}", "label": lab, "text": "test"} for lab in labs]

    with pytest.raises(ValueError, match="insufficient examples for label 'abbreviation'"):
        split_data(train_rows, test_rows, task)


def test_split_data_rejects_missing_test_label():
    task = "trec"
    labs = labels(task)
    train_rows = [{"id": f"tr.{lab}.{i}", "label": lab, "text": "tr"} for lab in labs for i in range(20)]
    test_rows = [{"id": f"te.{lab}", "label": lab, "text": "te"} for lab in labs if lab != "numeric"]

    with pytest.raises(ValueError, match="official test set missing expected label"):
        split_data(train_rows, test_rows, task)


def test_verify_real_data_positive_round_trip(tmp_path, monkeypatch):
    """Write files via production write_jsonl, verify canonical LF bytes and positive verification."""
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))

    task = "trec"
    file_keys = ["train", "pool", "test"] + [f"lowres_seed{s}" for s in CFG["seeds"]]
    manifest = {}

    for key in file_keys:
        path = real_path(task, key)
        rows = [{"id": f"{task}.{key}.{i:03d}", "text": f"example {i}", "label": "numeric"} for i in range(5)]
        write_jsonl(path, rows)
        raw = path.read_bytes()
        # Assert canonical LF byte-level newline convention
        assert b"\r" not in raw, f"Found CRLF in production written file: {path}"
        assert raw.endswith(b"\n")
        manifest[f"{task}/{key}.jsonl"] = {
            "lines": len(rows),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    monkeypatch.setattr("ws2_training.prepare_data.CHECKSUMS", manifest)

    # Positive round trip must succeed
    assert verify_real_data(task) is True


def test_verify_real_data_rejects_crlf_format(tmp_path, monkeypatch):
    """Verify that CRLF newline formatting triggers sha256 mismatch failure."""
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))

    task = "trec"
    file_keys = ["train", "pool", "test"] + [f"lowres_seed{s}" for s in CFG["seeds"]]
    manifest = {}

    for key in file_keys:
        path = real_path(task, key)
        rows = [{"id": f"{task}.{key}.{i:03d}", "text": f"example {i}", "label": "numeric"} for i in range(5)]
        write_jsonl(path, rows)
        manifest[f"{task}/{key}.jsonl"] = {
            "lines": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr("ws2_training.prepare_data.CHECKSUMS", manifest)

    # Convert one file to CRLF
    train_path = real_path(task, "train")
    train_path.write_bytes(train_path.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_real_data(task)


def test_verify_real_data_detects_missing_and_tampered_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))

    # When files are missing
    with pytest.raises(ValueError, match="Data verification failed"):
        verify_real_data("trec")

    # Create dummy files with wrong content
    trec_dir = tmp_path / "data" / "real" / "trec"
    trec_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["train.jsonl", "pool.jsonl", "test.jsonl", "lowres_seed13.jsonl", "lowres_seed42.jsonl", "lowres_seed7.jsonl"]:
        (trec_dir / fname).write_text('{"id": "dummy"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_real_data("trec")
