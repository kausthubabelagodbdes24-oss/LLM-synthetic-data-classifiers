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


def _fill_responses(tmp_path, mock_data, label_fn=None):
    """Helper: run make then fill every annotator's file using label_fn(row) -> label."""
    make(out_dir=tmp_path, test_rows_by_task=mock_data)
    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        with open(rf, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            valid_opts = [opt.strip() for opt in r["choose_one_of"].split("|")]
            r["label"] = label_fn(r) if label_fn else valid_opts[0]
        with open(rf, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Original tests (kept intact)
# ---------------------------------------------------------------------------

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
    _fill_responses(tmp_path, mock_data)

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
    _fill_responses(tmp_path, mock_data)

    # Inject an invalid label into harshita's file (after it was filled)
    rf = tmp_path / "annotation_harshita.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label"] = "not_a_valid_label"
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not_a_valid_label"):
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


# ---------------------------------------------------------------------------
# New tests: overwrite protection
# ---------------------------------------------------------------------------

def test_make_refuses_overwrite_if_files_exist(tmp_path):
    """A second make() call without force=True must raise FileExistsError and leave files unchanged."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Record original content of one annotator file
    rf = tmp_path / "annotation_kausthuba.csv"
    original_text = rf.read_text(encoding="utf-8")

    # Simulate annotator filling a judgment (modifying the file)
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label"] = labels("trec")[0]
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)
    modified_text = rf.read_text(encoding="utf-8")
    assert modified_text != original_text, "Setup: file should differ after simulated annotation"

    # Attempt to regenerate without --force — must raise
    with pytest.raises(FileExistsError, match="already exist"):
        make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # The completed judgment must survive untouched
    assert rf.read_text(encoding="utf-8") == modified_text, (
        "make() overwrote a completed response file without force=True"
    )


def test_make_with_force_overwrites_successfully(tmp_path):
    """make(force=True) must succeed even when output files already exist."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Should not raise
    items, created_files = make(out_dir=tmp_path, test_rows_by_task=mock_data, force=True)
    assert len(items) == 60
    # Key and all annotator files re-created blank
    for rf in created_files:
        with open(rf, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["label"] == "" for r in rows)


# ---------------------------------------------------------------------------
# New tests: combine_responses validation hardening
# ---------------------------------------------------------------------------

def test_combine_rejects_duplicate_item_ids(tmp_path):
    """combine_responses must raise ValueError when an annotator file has duplicate item_id rows."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Duplicate the first row in hementh's file
    rf = tmp_path / "annotation_hementh.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.append(rows[0].copy())  # duplicate first row
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="duplicate item_id"):
        combine_responses(out_dir=tmp_path)


def test_combine_rejects_missing_item_ids(tmp_path):
    """combine_responses must raise ValueError when an annotator file is missing canonical items."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Remove first 3 rows from kausthuba's file
    rf = tmp_path / "annotation_kausthuba.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    truncated = rows[3:]
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(truncated)

    with pytest.raises(ValueError, match="missing"):
        combine_responses(out_dir=tmp_path)


def test_combine_rejects_extra_item_ids(tmp_path):
    """combine_responses must raise ValueError when an annotator file contains extra item IDs."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Add a row with a foreign item_id to harshita's file
    rf = tmp_path / "annotation_harshita.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    extra_row = rows[0].copy()
    extra_row["item_id"] = "PHANTOM_ID_9999"
    rows.append(extra_row)
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not in the canonical set"):
        combine_responses(out_dir=tmp_path)


def test_combine_rejects_wrong_headers(tmp_path):
    """combine_responses must raise ValueError when a submission has wrong column headers."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Replace hementh's file with one that has wrong headers
    rf = tmp_path / "annotation_hementh.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        # Extra column "gold_label" injected — wrong
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "gold_label", "label"])
        writer.writeheader()
        for r in rows:
            r["gold_label"] = ""
            writer.writerow(r)

    with pytest.raises(ValueError, match="Wrong headers"):
        combine_responses(out_dir=tmp_path)


def test_combine_rejects_mismatched_metadata(tmp_path):
    """combine_responses must raise ValueError when task or text metadata is altered."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Tamper with the task field in the first row of kausthuba's file
    rf = tmp_path / "annotation_kausthuba.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    original_task = rows[0]["task"]
    other_task = [t for t in ["trec", "emotion"] if t != original_task][0]
    rows[0]["task"] = other_task
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="task.*mismatch"):
        combine_responses(out_dir=tmp_path)


def test_combine_handles_blank_labels_as_incomplete(tmp_path):
    """Blank labels in a submission must be preserved as empty strings, not treated as complete."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Leave first 5 rows blank in hementh's file
    rf = tmp_path / "annotation_hementh.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows[:5]:
        r["label"] = ""
    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # combine should succeed (blank is allowed)
    combined = combine_responses(out_dir=tmp_path)
    assert len(combined) == 60

    # The blank entries must appear as empty string in the output, not any filled value
    blank_item_ids = {rows[i]["item_id"] for i in range(5)}
    with open(tmp_path / "annotation_filled.csv", "r", encoding="utf-8") as fh:
        out_rows = list(csv.DictReader(fh))
    for r in out_rows:
        if r["item_id"] in blank_item_ids:
            assert r["hementh"] == "", (
                f"Blank label for item {r['item_id']} was changed to {r['hementh']!r}"
            )

    # score() should handle blanks gracefully (not crash, items with blanks excluded from agreement)
    res = score(filled_path=tmp_path / "annotation_filled.csv", out_dir=tmp_path)
    assert len(res) == 2
    for r in res:
        assert r["items"] <= 30  # some items may be excluded due to blanks


def test_combine_rejects_invalid_label_without_normalization(tmp_path):
    """combine_responses must reject labels differing only in case — no case normalisation applied."""
    mock_data = _mock_test_dataset()
    _fill_responses(tmp_path, mock_data)

    # Find a valid emotion label (lowercase) and inject the Title-cased version
    emotion_labels = labels("emotion")
    valid_lower = emotion_labels[0]           # e.g. "anger"
    invalid_upper = valid_lower.capitalize()   # e.g. "Anger"

    rf = tmp_path / "annotation_kausthuba.csv"
    with open(rf, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Find the first emotion row and replace its label with the wrongly-cased version
    for r in rows:
        if r["task"] == "emotion":
            r["label"] = invalid_upper
            break

    with open(rf, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=invalid_upper):
        combine_responses(out_dir=tmp_path)


# ---------------------------------------------------------------------------
# New tests: deterministic shuffle, active annotator coverage, and metric accuracy
# ---------------------------------------------------------------------------

def test_select_annotation_items_deterministic_shuffle_and_non_grouped():
    """Selection must deterministically shuffle items per task so row position does not leak gold."""
    mock_data = _mock_test_dataset()
    items1 = select_annotation_items(test_rows_by_task=mock_data)
    items2 = select_annotation_items(test_rows_by_task=mock_data)

    # 1. Stable reproducibility across repeated calls
    assert [it["item_id"] for it in items1] == [it["item_id"] for it in items2]
    assert [it["gold_label"] for it in items1] == [it["gold_label"] for it in items2]

    # 2. Exact balance preserved: 30 per task, exactly 5 per label
    trec_items = [it for it in items1 if it["task"] == "trec"]
    emotion_items = [it for it in items1 if it["task"] == "emotion"]
    assert len(trec_items) == 30
    assert len(emotion_items) == 30
    for lab in labels("trec"):
        assert sum(1 for it in trec_items if it["gold_label"] == lab) == 5
    for lab in labels("emotion"):
        assert sum(1 for it in emotion_items if it["gold_label"] == lab) == 5

    # 3. Non-grouped ordering: items must NOT appear in 5-item contiguous blocks of the same label
    for task_name, task_items in [("trec", trec_items), ("emotion", emotion_items)]:
        task_labs = [it["gold_label"] for it in task_items]
        # If it were grouped, every chunk of 5 would have the identical label
        grouped = all(len(set(task_labs[i:i + 5])) == 1 for i in range(0, 30, 5))
        assert not grouped, f"{task_name} annotation items are still grouped in 5-item blocks by gold label"
        # Verify frequent transitions between different labels across the 30 items
        transitions = sum(1 for i in range(len(task_labs) - 1) if task_labs[i] != task_labs[i + 1])
        assert transitions >= 15, f"{task_name} has only {transitions} label transitions across 30 items"


def test_make_shuffled_order_matches_across_key_and_all_response_files(tmp_path):
    """make() must write identical shuffled item order to annotation_key.csv and all response files."""
    mock_data = _mock_test_dataset()
    items, created_files = make(out_dir=tmp_path, test_rows_by_task=mock_data)

    key_rows = list(csv.DictReader(open(tmp_path / "annotation_key.csv", encoding="utf-8")))
    key_ids = [r["item_id"] for r in key_rows]
    assert len(key_ids) == 60

    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        rows = list(csv.DictReader(open(rf, encoding="utf-8")))
        resp_ids = [r["item_id"] for r in rows]
        assert resp_ids == key_ids, f"Item ID order mismatch in {rf.name} compared to annotation_key.csv"


def test_score_reports_actual_active_annotator_coverage(tmp_path):
    """score() must report actual active annotators, not len(names), when an annotator is blank."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    # Kausthuba and Harshita provide valid labels; Hementh leaves all rows blank
    key_rows = list(csv.DictReader(open(tmp_path / "annotation_key.csv", encoding="utf-8")))
    for name in ["kausthuba", "harshita"]:
        rf = tmp_path / f"annotation_{name}.csv"
        rows = list(csv.DictReader(open(rf, encoding="utf-8")))
        for r, k in zip(rows, key_rows):
            r["label"] = k["gold_label"]
        with open(rf, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            writer.writerows(rows)

    # hementh's file remains completely blank
    combine_responses(out_dir=tmp_path)
    res = score(filled_path=tmp_path / "annotation_filled.csv", out_dir=tmp_path)

    assert len(res) == 2
    for r in res:
        assert r["items"] == 30
        # Active annotators must be 2, NOT 3
        assert r["annotators"] == 2, f"Expected 2 active annotators, got {r['annotators']}"
        assert r["mean_agreement"] == 1.0


def test_score_majority_agreement_calculation(tmp_path):
    """score() must correctly calculate mean majority agreement proportion across items."""
    mock_data = _mock_test_dataset()
    make(out_dir=tmp_path, test_rows_by_task=mock_data)

    key_rows = list(csv.DictReader(open(tmp_path / "annotation_key.csv", encoding="utf-8")))
    trec_labs = labels("trec")
    emotion_labs = labels("emotion")

    # On every item: Kausthuba and Harshita pick label 0, Hementh picks label 1
    # Majority agreement proportion is 2/3 = 0.667 for every item
    for name in ANNOTATORS:
        rf = tmp_path / f"annotation_{name}.csv"
        rows = list(csv.DictReader(open(rf, encoding="utf-8")))
        for r in rows:
            task_labs = trec_labs if r["task"] == "trec" else emotion_labs
            if name in ("kausthuba", "harshita"):
                r["label"] = task_labs[0]
            else:
                r["label"] = task_labs[1]
        with open(rf, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            writer.writerows(rows)

    combine_responses(out_dir=tmp_path)
    res = score(filled_path=tmp_path / "annotation_filled.csv", out_dir=tmp_path)

    for r in res:
        assert r["annotators"] == 3
        assert r["items"] == 30
        assert r["mean_agreement"] == round(2 / 3, 3)
