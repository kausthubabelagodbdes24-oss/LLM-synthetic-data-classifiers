"""Tests for WS1 generation module: repetition isolation, demonstration tracking,
robust reply parsing, connection error handling, and pilot workflow.
"""

import json
import re
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from common import CFG, labels, read_jsonl, synth_path
from ws1_generation import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Set up temporary data environment with sample real datasets."""
    sample_real = ROOT / "data" / "sample" / "real"
    shutil.copytree(sample_real, tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("SYNTHTEXT_EMBEDDER", "hashing")
    return tmp_path


def test_repetition_isolation(fake_env):
    """Verify that generation produces separate files per seed with correct repetition metadata."""
    # Generate for seed 13
    generate.main(["--setting", "few_shot", "--seed", "13", "--target", "3", "--dry-run", "--task", "trec"])
    path_13 = synth_path("trec", "few_shot", seed=13, mode="fake")
    assert path_13.exists()

    rows_13 = read_jsonl(path_13)
    assert len(rows_13) == 3 * len(labels("trec"))
    assert all(r["repetition"] == 13 for r in rows_13)
    assert all("seed13" in r["id"] for r in rows_13)

    # Generate for seed 42
    generate.main(["--setting", "few_shot", "--seed", "42", "--target", "3", "--dry-run", "--task", "trec"])
    path_42 = synth_path("trec", "few_shot", seed=42, mode="fake")
    assert path_42.exists()
    assert path_13 != path_42

    rows_42 = read_jsonl(path_42)
    assert len(rows_42) == 3 * len(labels("trec"))
    assert all(r["repetition"] == 42 for r in rows_42)
    assert all("seed42" in r["id"] for r in rows_42)

    # Verify seed 13 file was not touched or mixed
    assert read_jsonl(path_13) == rows_13


def test_resume_isolated_to_repetition(fake_env):
    """Verify resume skips already generated rows within the repetition and adds only the shortfall."""
    # Step 1: generate target 2
    generate.main(["--setting", "zero_shot", "--seed", "13", "--target", "2", "--dry-run", "--task", "emotion"])
    path = synth_path("emotion", "zero_shot", seed=13, mode="fake")
    rows_initial = read_jsonl(path)
    assert len(rows_initial) == 2 * len(labels("emotion"))

    # Step 2: rerun with same target -> no new rows added
    generate.main(["--setting", "zero_shot", "--seed", "13", "--target", "2", "--dry-run", "--task", "emotion"])
    rows_after_noop = read_jsonl(path)
    assert len(rows_after_noop) == len(rows_initial)

    # Step 3: resume with higher target 4 -> exactly 2 new rows per label added
    generate.main(["--setting", "zero_shot", "--seed", "13", "--target", "4", "--dry-run", "--task", "emotion"])
    rows_expanded = read_jsonl(path)
    assert len(rows_expanded) == 4 * len(labels("emotion"))
    # Verify initial rows are preserved as prefix
    assert rows_expanded[:len(rows_initial)] == rows_initial


def test_dry_run_rows_cannot_satisfy_a_real_generation_target(fake_env, monkeypatch):
    """Fixture output is isolated, so a real run still calls the configured model."""
    generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)

    fake_path = fake_env / "data" / "synthetic-fixtures" / "trec" / "zero_shot_seed13.jsonl"
    real_output = synth_path("trec", "zero_shot", seed=13)
    assert fake_path.exists()
    assert not real_output.exists()

    call_count = 0

    def real_ask(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return [f"what is real model output number {call_count} ?"]

    monkeypatch.setattr(generate, "ask", real_ask)
    monkeypatch.setattr(
        generate,
        "resolve_model_identity",
        lambda: {
            "provider": "ollama",
            "configured_tag": "llama3.2:3b",
            "resolved_name": "llama3.2:3b",
            "digest": "sha256:real-model-digest",
        },
    )

    rows, _ = generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)

    assert call_count == len(labels("trec"))
    assert len(rows) == len(labels("trec"))
    assert all(row["generation_mode"] == "real" for row in rows)
    assert len(read_jsonl(fake_path)) == len(labels("trec"))


@pytest.mark.parametrize(
    "corrupt_row",
    [
        lambda row: row.pop("repetition"),
        lambda row: row.__setitem__("repetition", None),
        lambda row: row.__setitem__("task", "emotion"),
        lambda row: row.__setitem__("setting", "few_shot"),
        lambda row: row.__setitem__("source_ids", ["trec.pool.00001"]),
        lambda row: row.pop("schema_version"),
        lambda row: row.pop("generation_fingerprint"),
        lambda row: row.__setitem__("id", "trec.zero_shot.seed42.abbreviation.0001"),
    ],
    ids=[
        "missing-repetition",
        "null-repetition",
        "wrong-task",
        "wrong-setting",
        "wrong-sources",
        "missing-schema",
        "missing-generation-identity",
        "misplaced-row-id",
    ],
)
def test_resume_rejects_missing_or_incompatible_row_identity(fake_env, monkeypatch, corrupt_row):
    """Existing rows count only when every persisted identity field matches the invocation."""
    model_identity = {
        "provider": "ollama",
        "configured_tag": "llama3.2:3b",
        "resolved_name": "llama3.2:3b",
        "digest": "sha256:model-a",
    }
    call_number = 0

    def real_ask(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        return [f"what is validated real output number {call_number} ?"]

    monkeypatch.setattr(generate, "resolve_model_identity", lambda: model_identity)
    monkeypatch.setattr(generate, "ask", real_ask)
    generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)

    path = synth_path("trec", "zero_shot", seed=13)
    rows = read_jsonl(path)
    corrupt_row(rows[0])
    from common import write_jsonl
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match="incompatible existing row"):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)


def test_noop_resume_preserves_prior_session_history(fake_env, monkeypatch):
    """A completed artifact retains its first run record when resumed without new work."""
    model_identity = {
        "provider": "ollama",
        "configured_tag": "llama3.2:3b",
        "resolved_name": "llama3.2:3b",
        "digest": "sha256:model-a",
    }
    call_number = 0

    def real_ask(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        return [f"what is history preserving output number {call_number} ?"]

    monkeypatch.setattr(generate, "resolve_model_identity", lambda: model_identity)
    monkeypatch.setattr(generate, "ask", real_ask)
    _, first = generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)
    _, second = generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)

    assert len(first["sessions"]) == 1
    assert len(second["sessions"]) == 2
    assert second["sessions"][0] == first["sessions"][0]
    assert second["created_at"] == first["created_at"]
    assert second["sessions"][1]["status"] == "no_op"
    assert second["sessions"][1]["rows_before"] == second["sessions"][1]["rows_after"]
    assert second["cumulative_stats"] == first["cumulative_stats"]


def test_interrupted_generation_checkpoints_and_resumes(fake_env, monkeypatch):
    """Accepted rows and their provenance survive an interruption between model calls."""
    calls = 0

    def interrupt_after_first_batch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return ["what is the first checkpointed output ?"]

    monkeypatch.setattr(generate, "ask", interrupt_after_first_batch)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)

    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    accepted_before_resume = read_jsonl(path)
    checkpoint = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert len(accepted_before_resume) == 1
    assert checkpoint["total_count"] == 1
    assert checkpoint["sessions"][-1]["status"] == "interrupted"
    assert checkpoint["sessions"][-1]["accepted_row_ids"] == [accepted_before_resume[0]["id"]]

    resumed_calls = 0

    def finish_generation(*args, **kwargs):
        nonlocal resumed_calls
        resumed_calls += 1
        return [f"what is resumed output number {resumed_calls} ?"]

    monkeypatch.setattr(generate, "ask", finish_generation)
    rows, provenance = generate.generate_labelled(
        "trec", "zero_shot", target=1, fake=True, seed=13
    )

    assert rows[0] == accepted_before_resume[0]
    assert len(rows) == len(labels("trec"))
    assert [session["status"] for session in provenance["sessions"]] == ["interrupted", "complete"]


def test_resume_recovers_crash_after_data_append_before_provenance(fake_env, monkeypatch):
    """A write-ahead checkpoint reconciles a batch saved before its sidecar."""
    original_write_jsonl = generate.write_jsonl
    crashed = False

    def crash_after_append(path, rows, append=False):
        nonlocal crashed
        original_write_jsonl(path, rows, append=append)
        if append and not crashed:
            crashed = True
            raise KeyboardInterrupt

    monkeypatch.setattr(generate, "write_jsonl", crash_after_append)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)

    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    rows_after_crash = read_jsonl(path)
    assert len(rows_after_crash) == 1
    assert not path.with_suffix(".provenance.json").exists()
    assert path.with_suffix(".generation-journal.json").exists()

    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)
    rows, provenance = generate.generate_labelled(
        "trec", "zero_shot", target=1, fake=True, seed=13
    )

    assert rows[0] == rows_after_crash[0]
    assert len(rows) == len(labels("trec"))
    assert len({row["id"] for row in rows}) == len(rows)
    assert [session["status"] for session in provenance["sessions"]] == ["interrupted", "complete"]
    assert not path.with_suffix(".generation-journal.json").exists()


def test_resume_reconciles_later_append_with_stale_sidecar_temp(fake_env, monkeypatch):
    """Journal recovery supersedes a stale sidecar and incomplete sidecar temp safely."""
    generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)
    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    original_rows = read_jsonl(path)
    original_write_jsonl = generate.write_jsonl
    crashed = False

    def crash_after_resumed_append(path, rows, append=False):
        nonlocal crashed
        original_write_jsonl(path, rows, append=append)
        if append and not crashed:
            crashed = True
            raise KeyboardInterrupt

    monkeypatch.setattr(generate, "write_jsonl", crash_after_resumed_append)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=2, fake=True, seed=13)

    rows_after_crash = read_jsonl(path)
    assert rows_after_crash[:len(original_rows)] == original_rows
    assert len(rows_after_crash) == len(original_rows) + 1
    sidecar = path.with_suffix(".provenance.json")
    stale_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stale_sidecar["total_count"] == len(original_rows)
    sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temp.write_text('{"truncated":', encoding="utf-8")

    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)
    rows, provenance = generate.generate_labelled(
        "trec", "zero_shot", target=2, fake=True, seed=13
    )

    assert len(rows) == 2 * len(labels("trec"))
    assert len({row["id"] for row in rows}) == len(rows)
    assert rows[:len(rows_after_crash)] == rows_after_crash
    assert [session["status"] for session in provenance["sessions"]] == [
        "complete", "interrupted", "complete"
    ]
    assert not sidecar_temp.exists()
    assert not path.with_suffix(".generation-journal.json").exists()


def test_recovery_rejects_rows_that_do_not_match_the_journal(fake_env, monkeypatch):
    """A journal cannot authorize a changed or unrelated row at the persistence boundary."""
    original_write_jsonl = generate.write_jsonl
    crashed = False

    def crash_after_append(path, rows, append=False):
        nonlocal crashed
        original_write_jsonl(path, rows, append=append)
        if append and not crashed:
            crashed = True
            raise KeyboardInterrupt

    monkeypatch.setattr(generate, "write_jsonl", crash_after_append)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)

    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    incompatible = read_jsonl(path)
    incompatible[0]["repetition"] = 42
    original_write_jsonl(path, incompatible)
    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)

    with pytest.raises(ValueError, match="persisted batch does not match"):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)


def test_resume_recovers_partial_first_json_row(fake_env, monkeypatch):
    """A valid journal repairs an interrupted byte prefix before strict JSONL parsing."""
    original_write_jsonl = generate.write_jsonl
    crashed = False

    def write_partial_row_then_crash(path, rows, append=False):
        nonlocal crashed
        if append and not crashed:
            crashed = True
            path.parent.mkdir(parents=True, exist_ok=True)
            intended = (json.dumps(rows[0], ensure_ascii=False) + "\n").encode("utf-8")
            with path.open("ab") as handle:
                handle.write(intended[:20])
            raise KeyboardInterrupt
        original_write_jsonl(path, rows, append=append)

    monkeypatch.setattr(generate, "write_jsonl", write_partial_row_then_crash)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)

    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    assert len(path.read_bytes()) == 20
    assert path.with_suffix(".generation-journal.json").exists()

    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)
    rows, provenance = generate.generate_labelled(
        "trec", "zero_shot", target=1, fake=True, seed=13
    )

    assert len(rows) == len(labels("trec"))
    assert len({row["id"] for row in rows}) == len(rows)
    assert provenance["sessions"][0]["status"] == "interrupted"
    assert not path.with_suffix(".generation-journal.json").exists()


def test_resume_recovers_partial_utf8_row_during_later_batch(fake_env, monkeypatch):
    """A resumed batch recovers even when interruption splits a UTF-8 code point."""
    generate.generate_labelled("trec", "zero_shot", target=1, fake=True, seed=13)
    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    original_rows = read_jsonl(path)
    original_write_jsonl = generate.write_jsonl
    answer_number = 0
    crashed = False

    def non_ascii_answer(*args, **kwargs):
        nonlocal answer_number
        answer_number += 1
        return [f"what is café resumed output number {answer_number} ?"]

    def split_utf8_then_crash(path, rows, append=False):
        nonlocal crashed
        if append and not crashed:
            crashed = True
            path.parent.mkdir(parents=True, exist_ok=True)
            intended = (json.dumps(rows[0], ensure_ascii=False) + "\n").encode("utf-8")
            split_at = intended.index("é".encode("utf-8")) + 1
            with path.open("ab") as handle:
                handle.write(intended[:split_at])
            raise KeyboardInterrupt
        original_write_jsonl(path, rows, append=append)

    monkeypatch.setattr(generate, "ask", non_ascii_answer)
    monkeypatch.setattr(generate, "write_jsonl", split_utf8_then_crash)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_labelled("trec", "zero_shot", target=2, fake=True, seed=13)

    assert path.read_bytes().endswith(b"\xc3")
    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)
    rows, provenance = generate.generate_labelled(
        "trec", "zero_shot", target=2, fake=True, seed=13
    )

    assert rows[:len(original_rows)] == original_rows
    assert len(rows) == 2 * len(labels("trec"))
    assert len({row["id"] for row in rows}) == len(rows)
    assert [session["status"] for session in provenance["sessions"]] == [
        "complete", "interrupted", "complete"
    ]


@pytest.mark.parametrize("changed_identity", ["model", "prompt"])
def test_resume_rejects_changed_model_or_prompt_identity(fake_env, monkeypatch, changed_identity):
    """A resume cannot append to rows produced by a different model digest or prompt protocol."""
    model_identity = {
        "provider": "ollama",
        "configured_tag": "llama3.2:3b",
        "resolved_name": "llama3.2:3b",
        "digest": "sha256:model-a",
    }
    call_number = 0

    def real_ask(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        return [f"what is stable identity output number {call_number} ?"]

    monkeypatch.setattr(generate, "resolve_model_identity", lambda: model_identity)
    monkeypatch.setattr(generate, "ask", real_ask)
    generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)

    if changed_identity == "model":
        monkeypatch.setattr(
            generate,
            "resolve_model_identity",
            lambda: {**model_identity, "digest": "sha256:model-b"},
        )
    else:
        monkeypatch.setattr(generate, "PROMPT_PROTOCOL_VERSION", "issue-6-v2-revised")

    with pytest.raises(ValueError, match="generation identity"):
        generate.generate_labelled("trec", "zero_shot", target=2, fake=False, seed=13)


def test_few_shot_resume_rejects_changed_inherited_zero_shot_template(fake_env, monkeypatch):
    """Few-shot identity includes the base zero-shot template it renders through."""
    generate.generate_labelled("trec", "few_shot", target=1, fake=True, seed=13)

    def revised_zero_shot(task, label, n, i):
        return (
            f"Revised inherited template for {task} label {label}. "
            f"Write {n} different examples. Reply only with JSON."
        )

    monkeypatch.setattr(generate, "prompt_zero_shot", revised_zero_shot)

    with pytest.raises(ValueError, match="generation identity"):
        generate.generate_labelled("trec", "few_shot", target=2, fake=True, seed=13)


def test_resume_rejects_corrupt_jsonl_before_counting(fake_env, monkeypatch):
    """Malformed persisted rows are reported as an incompatible artifact, never ignored or counted."""
    path = synth_path("trec", "zero_shot", seed=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"id": "truncated"\n', encoding="utf-8")
    monkeypatch.setattr(
        generate,
        "resolve_model_identity",
        lambda: {
            "provider": "ollama",
            "configured_tag": "llama3.2:3b",
            "resolved_name": "llama3.2:3b",
            "digest": "sha256:model-a",
        },
    )

    with pytest.raises(ValueError, match="invalid JSONL"):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)


def test_auggpt_dry_run_uses_identified_fixture_storage(fake_env):
    """Fake AugGPT rows are visibly fake and cannot occupy the real artifact path."""
    generate.generate_auggpt("trec", fake=True, seed=13)

    fixture_path = synth_path("trec", "auggpt", seed=13, mode="fake")
    real_output = synth_path("trec", "auggpt", seed=13, mode="real")
    rows = read_jsonl(fixture_path)
    assert rows
    assert not real_output.exists()
    assert all(row["generation_mode"] == "fake" for row in rows)
    assert fixture_path.with_suffix(".provenance.json").exists()


def test_resume_rejects_sidecar_identity_that_disagrees_with_rows(fake_env, monkeypatch):
    """A matching fingerprint cannot mask a misplaced or relabelled provenance sidecar."""
    model_identity = {
        "provider": "ollama",
        "configured_tag": "llama3.2:3b",
        "resolved_name": "llama3.2:3b",
        "digest": "sha256:model-a",
    }
    call_number = 0

    def real_ask(*args, **kwargs):
        nonlocal call_number
        call_number += 1
        return [f"what is sidecar validation output number {call_number} ?"]

    monkeypatch.setattr(generate, "resolve_model_identity", lambda: model_identity)
    monkeypatch.setattr(generate, "ask", real_ask)
    generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)

    sidecar = synth_path("trec", "zero_shot", seed=13).with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["task"] = "emotion"
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance identity does not match"):
        generate.generate_labelled("trec", "zero_shot", target=1, fake=False, seed=13)


def test_demonstration_sampling_and_provenance(fake_env):
    """Verify few-shot uses 3 demonstrations from pool.jsonl, tracks source_ids, and creates provenance."""
    generate.main(["--setting", "few_shot", "--seed", "7", "--target", "2", "--dry-run", "--task", "trec"])
    path = synth_path("trec", "few_shot", seed=7, mode="fake")
    rows = read_jsonl(path)

    # Every few-shot row must have exactly few_shot_k source demonstration IDs
    assert all(len(r["source_ids"]) == CFG["few_shot_k"] for r in rows)
    assert all(all(sid.startswith("trec.pool.") for sid in r["source_ids"]) for r in rows)

    # Check provenance sidecar file
    prov_path = path.with_suffix(".provenance.json")
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["task"] == "trec"
    assert prov["setting"] == "few_shot"
    assert prov["repetition"] == 7
    assert prov["is_complete"] is True
    assert prov["distinct_demonstrations_count"] > 0
    assert len(prov["distinct_demonstration_ids"]) == prov["distinct_demonstrations_count"]
    assert prov["total_count"] == 2 * len(labels("trec"))


def test_reject_copied_demonstrations(fake_env, monkeypatch):
    """Ensure that if the model regurgitates a demonstration text, it is rejected."""
    demo_text = None
    calls = 0

    def mock_ask(prompt, *args, **kwargs):
        nonlocal calls, demo_text
        calls += 1
        m = re.search(r'- "(.*?)"', prompt)
        if m:
            demo_text = m.group(1)
            return [demo_text, "what is a valid fresh question ?"]
        return ["what is another valid fresh question ?"]

    monkeypatch.setattr(generate, "ask", mock_ask)
    rows, prov = generate.generate_labelled("trec", "few_shot", target=1, fake=True, seed=13)
    # The demonstration text must not appear in accepted rows
    assert demo_text is not None
    assert not any(r["text"] == demo_text for r in rows)
    assert prov["stats"]["rejected_demo_leak"] >= 1


def test_parse_robustness():
    """Verify reply parser handles code fences, conversational boilerplate, line formats and length constraints."""
    # Fenced JSON block
    fenced_json = "```json\n{\"examples\": [\"how hot is the sun ?\", \"who won the 1994 world cup ?\"]}\n```"
    assert generate.parse(fenced_json) == ["how hot is the sun ?", "who won the 1994 world cup ?"]

    # Conversational lead-in and closing remarks
    conversational = (
        "Sure, here are some examples for your dataset:\n"
        "1. how many planets are in the solar system ?\n"
        "2. what causes aurora borealis ?\n"
        "I hope this helps! Let me know if you need anything else."
    )
    parsed = generate.parse(conversational)
    assert parsed == ["how many planets are in the solar system ?", "what causes aurora borealis ?"]

    # Apology / refusal reply
    assert generate.parse("I apologize, but I cannot fulfill this request.") == []
    assert generate.parse("Sorry, as an AI language model...") == []

    # Single word strings rejected (< 2 words)
    assert generate.parse('{"examples": ["hello", "valid sentence"]}') == ["valid sentence"]

    # Max words filtering
    long_q = "this is a question with way too many words that exceeds the allowed maximum length for trec questions"
    assert generate.parse(f'{{"examples": ["{long_q}", "short question ?"]}}', max_words=10) == ["short question ?"]


@pytest.mark.parametrize("unusable_reply", [
    '{"examples": [], "message": "generation failed please retry"}',
    '{"examples": "generation failed please retry"}',
    '{"error": ["model unavailable please retry"]}',
    'Result: {"error": ["model unavailable please retry"]}',
    '{"examples": ["unfinished example"',
])
def test_structured_non_examples_retry_before_counting_candidates(monkeypatch, unusable_reply):
    replies = iter([unusable_reply, '{"examples": ["what causes the tides ?"]}'])
    monkeypatch.setattr(generate, "ask_ollama", lambda *args, **kwargs: next(replies))
    stats = {"calls": 0, "call_failures": 0, "parse_failures": 0, "candidates_received": 0}

    result = generate.ask("test prompt", fake=False, retries=2, stats=stats)

    assert result == ["what causes the tides ?"]
    assert stats == {"calls": 2, "call_failures": 0, "parse_failures": 1, "candidates_received": 1}


def test_network_connection_failure_handling(monkeypatch):
    """Verify that network/Ollama connection failures retry and fail cleanly without corrupting datasets."""
    mock_post = MagicMock(side_effect=requests.exceptions.ConnectionError("Connection refused"))
    monkeypatch.setattr(requests, "post", mock_post)

    stats = {
        "calls": 0, "call_failures": 0, "parse_failures": 0,
        "candidates_received": 0, "rejected_duplicate": 0,
        "rejected_test_leak": 0, "rejected_demo_leak": 0,
        "rejected_too_long": 0, "rejected_too_short": 0,
        "accepted_new": 0
    }
    out = generate.ask("test prompt", fake=False, retries=2, stats=stats)
    assert out == []
    assert stats["call_failures"] == 2
    assert mock_post.call_count == 2


def test_smoke_ollama_status_diagnostics(monkeypatch):
    """Verify smoke check returns diagnostic info when Ollama is offline."""
    mock_get = MagicMock(side_effect=requests.exceptions.ConnectionError("Connection refused"))
    monkeypatch.setattr(requests, "get", mock_get)

    res = generate.check_ollama_status()
    assert res["ollama_running"] is False
    assert res["smoke_call_success"] is False
    assert "Could not connect to Ollama" in res["message"]


def test_pilot_workflow_and_uniform_target(fake_env):
    """Run pilot mode for both tasks and both settings, verifying exact counts and completeness."""
    for setting in ["zero_shot", "few_shot"]:
        generate.main(["--setting", setting, "--pilot", "--seed", "13", "--dry-run", "--task", "all"])

    for task in ["trec", "emotion"]:
        for setting in ["zero_shot", "few_shot"]:
            p = synth_path(task, setting, seed=13, mode="fake")
            rows = read_jsonl(p)
            # Pilot target is 10 per label across 6 labels = 60
            assert len(rows) == 10 * len(labels(task))
            for lab in labels(task):
                assert sum(r["label"] == lab for r in rows) == 10
            prov = json.loads(p.with_suffix(".provenance.json").read_text(encoding="utf-8"))
            assert prov["is_complete"] is True
            assert prov["target_per_label"] == 10
