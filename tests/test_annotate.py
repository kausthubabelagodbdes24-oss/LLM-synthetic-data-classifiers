"""Behavioral tests for independent annotation item distribution, separate response templates,
response combination validation, and subjectivity scoring. Runs offline without network access.
"""

import csv
from pathlib import Path
import pytest

from common import CFG, labels, read_jsonl
from ws3_evaluation.annotate import (
    ANNOTATORS,
    combine_responses,
    make,
    score,
    select_annotation_items,
)


def _mock_test_dataset():
    """Create mock test sets with all valid classes for TREC and Emotion."""
    trec_labs = labels("trec")
    emotion_labs = labels("emotion")
    data = {"trec": [], "emotion": []}
    for lab in trec_labs:
        for i in range(10):
            data["trec"].append({
                "id": f"trec.test.{lab}.{i:03d}",
                "text": f"Question about {lab} number {i}?",
                "label": lab,
            })
    for lab in emotion_labs:
        for i in range(10):
            data["emotion"].append({
                "id": f"emotion.test.{lab}.{i:03d}",
                "text": f"I feel {lab} today {i}",
                "label": lab,
            })
    return data


def test_select_annotation_items_counts_and_balance():
    """Verify selection produces exactly 5 items per label per task (30 per task, 60 total)."""
    mock_data = _mock_test_dataset()
    items = select_annotation_items(test_rows_by_task=mock_data)

    assert len(items) == 60
    trec_items = [it for it in items if it["task"] == "trec"]
    emotion_items = [it for it in items if it["task"] == "emotion"]
    assert len(trec_items) == 30
    assert len(emotion_items) == 30

    # Verify exactly 5 items per label
    trec_labs = labels("trec")
    emotion_labs = labels("emotion")
    for lab in trec_labs:
        count = sum(1 for it in trec_items if it["gold_label"] == lab)
        assert count == 5, f"TREC label {lab} count is {count}, expected 5"
    for lab in emotion_labs:
        count = sum(1 for it in emotion_items if it["gold_label"] == lab)
        assert count == 5, f"Emotion label {lab} count is {count}, expected 5"

    # Verify stable IDs
    assert len(set(it["item_id"] for it in items)) == 60


def test_make_generates_isolated_response_files_without_gold(tmp_path):
    """Verify make creates separate response files for annotators with no gold answers."""
    mock_data = _mock_test_dataset()
    items, created_files = make(out_dir=tmp_path, test_rows_by_task=mock_data)

    assert len(created_files) == len(ANNOTATORS)

    # Verify annotation key exists and has gold
    key_path = tmp_path / "annotation_key.csv"
    assert key_path.exists()
    with open(key_path, "r", encoding="utf-8") as fh:
        key_rows = list(csv.DictReader(fh))
    assert len(key_rows) == 60
    assert "gold_label" in key_rows[0]

    # Verify each annotator's response file
    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        assert rf.exists()
        with open(rf, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)

        # Fieldnames must NOT include gold_label, other annotators, or answers
        assert fieldnames == ["item_id", "task", "text", "choose_one_of", "label"]
        assert "gold_label" not in fieldnames
        assert "gold" not in fieldnames
        for other in ANNOTATORS:
            if other != name:
                assert other not in fieldnames

        assert len(rows) == 60
        # Responses start blank
        assert all(r["label"] == "" for r in rows)


def test_combine_responses_validates_and_merges_independently(tmp_path):
    """Verify combine merges independent response files and enforces label validation."""
    mock_data = _mock_test_dataset()
    items, created_files = make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Simulate annotators filling responses
    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        with open(rf, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            # Pick a valid label from choose_one_of
            valid_opts = [opt.strip() for opt in r["choose_one_of"].split("|")]
            r["label"] = valid_opts[0]
        with open(rf, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            writer.writerows(rows)

    combined = combine_responses(out_dir=tmp_path)
    assert len(combined) == 60
    assert (tmp_path / "annotation_filled.csv").exists()

    with open(tmp_path / "annotation_filled.csv", "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames == ["item_id", "task", "text", "choose_one_of"] + ANNOTATORS
    assert len(rows) == 60


def test_combine_responses_rejects_invalid_label(tmp_path):
    """Verify combine raises ValueError if an annotator provides an invalid label."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Set an invalid label in harshita's file
    rf = tmp_path / "annotation_harshita.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label"] = "not_a_valid_label"
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Invalid label 'not_a_valid_label'"):
        combine_responses(out_dir=tmp_path)


def test_score_subjectivity_with_known_agreement(tmp_path):
    """Verify score calculates agreement and Krippendorff's alpha."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Annotators agree on the label of each item, distributing across all labels
    key_rows = list(csv.DictReader(open(tmp_path / "annotation_key.csv", encoding="utf-8")))
    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        with open(rf, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r, k in zip(rows, key_rows):
            r["label"] = k["gold_label"]
        with open(rf, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            writer.writerows(rows)

    combine_responses(out_dir=tmp_path)
    res = score(filled_path=tmp_path / "annotation_filled.csv", out_dir=tmp_path)

    assert len(res) == 2
    for r in res:
        assert r["items"] == 30
        assert r["annotators"] == 3
        assert r["mean_agreement"] == 1.0
        assert r["alpha"] == 1.0
