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
    path_13 = synth_path("trec", "few_shot", seed=13)
    assert path_13.exists()

    rows_13 = read_jsonl(path_13)
    assert len(rows_13) == 3 * len(labels("trec"))
    assert all(r["repetition"] == 13 for r in rows_13)
    assert all("seed13" in r["id"] for r in rows_13)

    # Generate for seed 42
    generate.main(["--setting", "few_shot", "--seed", "42", "--target", "3", "--dry-run", "--task", "trec"])
    path_42 = synth_path("trec", "few_shot", seed=42)
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
    path = synth_path("emotion", "zero_shot", seed=13)
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


def test_demonstration_sampling_and_provenance(fake_env):
    """Verify few-shot uses 3 demonstrations from pool.jsonl, tracks source_ids, and creates provenance."""
    generate.main(["--setting", "few_shot", "--seed", "7", "--target", "2", "--dry-run", "--task", "trec"])
    path = synth_path("trec", "few_shot", seed=7)
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
            p = synth_path(task, setting, seed=13)
            rows = read_jsonl(p)
            # Pilot target is 10 per label across 6 labels = 60
            assert len(rows) == 10 * len(labels(task))
            for lab in labels(task):
                assert sum(r["label"] == lab for r in rows) == 10
            prov = json.loads(p.with_suffix(".provenance.json").read_text(encoding="utf-8"))
            assert prov["is_complete"] is True
            assert prov["target_per_label"] == 10
