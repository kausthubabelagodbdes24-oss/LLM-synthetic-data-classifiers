"""Fake-model behavior checks for issue #9: source-complete, resumable, repetition-isolated AugGPT.

These use fixture storage and a scripted stand-in model; they say nothing about real Qwen output.
"""

import json
import shutil
from pathlib import Path

import pytest

from common import CFG, labels, read_jsonl, real_path, synth_path, write_jsonl
from ws1_generation import generate

ROOT = Path(__file__).resolve().parents[1]
K = CFG["auggpt_k"]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data" / "sample" / "real", tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    # The committed fixtures hold two sources per label; verification checks that exact size.
    monkeypatch.setitem(CFG, "lowres_per_label", 2)
    return tmp_path


def scripted_model(monkeypatch, per_call=None, prefix="paraphrase"):
    """Return unique candidates; per_call limits how many each call yields."""
    state = {"calls": 0}

    def fake_ask(prompt, fake, max_words=None, retries=3, stats=None):
        state["calls"] += 1
        if stats is not None:
            stats["calls"] += 1
        n = int(prompt.split("sentence ")[1].split(" ")[0])
        n = min(n, per_call) if per_call else n
        out = [f"{prefix} number {state['calls']} variant {i}" for i in range(n)]
        if stats is not None:
            stats["candidates_received"] += len(out)
        return out

    monkeypatch.setattr(generate, "ask", fake_ask)
    return state


def sources(seed):
    return {row["id"]: row for row in read_jsonl(real_path("trec", f"lowres_seed{seed}"))}


def per_source_counts(rows):
    counts = {}
    for row in rows:
        counts[row["source_ids"][0]] = counts.get(row["source_ids"][0], 0) + 1
    return counts


def test_partial_sources_resume_until_six_valid_paraphrases(fake_env, monkeypatch):
    """A source with saved rows is not complete until it has six; reruns fill only the shortfall."""
    scripted_model(monkeypatch, per_call=1)  # three attempts x one candidate = three per source
    first_rows, first = generate.generate_auggpt("trec", fake=True, seed=13)

    assert set(per_source_counts(first_rows).values()) == {3}
    assert first["is_complete"] is False
    assert sorted(first["incomplete_source_ids"]) == sorted(sources(13))
    assert first["sessions"][-1]["status"] == "incomplete"
    with pytest.raises(ValueError, match="sources lack"):
        generate.verify_auggpt("trec", 13, mode="fake")

    scripted_model(monkeypatch, prefix="resumed")
    rows, provenance = generate.generate_auggpt("trec", fake=True, seed=13)

    assert rows[:len(first_rows)] == first_rows
    assert per_source_counts(rows) == {source_id: K for source_id in sources(13)}
    assert len(rows) == K * len(sources(13)) == provenance["total_count"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["id"].rsplit(".aug", 1)[1] in {str(i) for i in range(1, K + 1)} for row in rows)
    assert [s["status"] for s in provenance["sessions"]] == ["incomplete", "complete"]
    assert provenance["is_complete"] is True and provenance["incomplete_source_ids"] == []
    assert generate.verify_auggpt("trec", 13, mode="fake")["total_count"] == len(rows)

    calls = scripted_model(monkeypatch)
    _, noop = generate.generate_auggpt("trec", fake=True, seed=13)
    assert calls["calls"] == 0
    assert noop["sessions"][-1]["status"] == "no_op" and noop["total_count"] == len(rows)


def test_interrupted_auggpt_checkpoints_without_inflating_counts(fake_env, monkeypatch):
    scripted_model(monkeypatch, per_call=2)
    real_ask = generate.ask
    calls = 0

    def interrupt_on_fifth(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt
        return real_ask(*args, **kwargs)

    monkeypatch.setattr(generate, "ask", interrupt_on_fifth)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_auggpt("trec", fake=True, seed=13)

    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    saved = read_jsonl(path)
    checkpoint = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert len(saved) == 8 == checkpoint["total_count"]
    assert checkpoint["sessions"][-1]["status"] == "interrupted"
    assert checkpoint["is_complete"] is False
    assert sum(checkpoint["counts_by_source"].values()) == len(saved)

    scripted_model(monkeypatch, prefix="after interrupt")
    rows, provenance = generate.generate_auggpt("trec", fake=True, seed=13)
    assert rows[:len(saved)] == saved
    assert per_source_counts(rows) == {source_id: K for source_id in sources(13)}
    assert [s["status"] for s in provenance["sessions"]] == ["interrupted", "complete"]


def test_auggpt_recovers_crash_between_append_and_sidecar(fake_env, monkeypatch):
    scripted_model(monkeypatch, per_call=2)
    original_write_jsonl = generate.write_jsonl
    appends = 0

    def crash_on_second_append(path, rows, append=False):
        nonlocal appends
        original_write_jsonl(path, rows, append=append)
        if append:
            appends += 1
            if appends == 2:
                raise KeyboardInterrupt

    monkeypatch.setattr(generate, "write_jsonl", crash_on_second_append)
    with pytest.raises(KeyboardInterrupt):
        generate.generate_auggpt("trec", fake=True, seed=13)

    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    journal = path.with_suffix(".generation-journal.json")
    assert journal.exists()
    stale = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert stale["total_count"] == 2 and len(read_jsonl(path)) == 4
    with pytest.raises(ValueError, match="unrecovered interrupted batch"):
        generate.verify_auggpt("trec", 13, mode="fake")

    monkeypatch.setattr(generate, "write_jsonl", original_write_jsonl)
    scripted_model(monkeypatch, prefix="recovered")
    rows, provenance = generate.generate_auggpt("trec", fake=True, seed=13)
    assert not journal.exists()
    assert per_source_counts(rows) == {source_id: K for source_id in sources(13)}
    assert [s["status"] for s in provenance["sessions"]] == ["interrupted", "complete"]
    generate.verify_auggpt("trec", 13, mode="fake")


def test_overlapping_sources_are_generated_separately_per_repetition(fake_env, monkeypatch):
    """A source shared by two repetitions gets its own model calls and rows in each."""
    shared = next(iter(sources(13).values()))
    seed42 = read_jsonl(real_path("trec", "lowres_seed42"))
    if shared["id"] not in {row["id"] for row in seed42}:
        replace = next(i for i, row in enumerate(seed42) if row["label"] == shared["label"])
        seed42[replace] = shared
    write_jsonl(real_path("trec", "lowres_seed42"), seed42)
    assert shared["id"] in sources(42) and len(sources(42)) == len(sources(13))

    scripted_model(monkeypatch, prefix="rep13")
    rows13, _ = generate.generate_auggpt("trec", fake=True, seed=13)
    path13 = synth_path("trec", "auggpt", seed=13, mode="fake")
    snapshot13 = path13.read_bytes()

    calls = scripted_model(monkeypatch, prefix="rep42")
    rows42, prov42 = generate.generate_auggpt("trec", fake=True, seed=42)

    assert calls["calls"] == len(sources(42))  # every source, including the shared one, is generated
    assert path13.read_bytes() == snapshot13
    assert all(row["repetition"] == 42 and ".seed42.aug" in row["id"] for row in rows42)
    shared13 = [r for r in rows13 if r["source_ids"] == [shared["id"]]]
    shared42 = [r for r in rows42 if r["source_ids"] == [shared["id"]]]
    assert len(shared13) == len(shared42) == K
    assert not {r["text"] for r in shared13} & {r["text"] for r in shared42}
    assert set(prov42["counts_by_source"]) == set(sources(42))

    # Copying another repetition's rows into this artifact is rejected rather than counted.
    path42 = synth_path("trec", "auggpt", seed=42, mode="fake")
    write_jsonl(path42, shared13 + rows42[K:])
    with pytest.raises(ValueError, match="repetition does not match"):
        generate.verify_auggpt("trec", 42, mode="fake")
    with pytest.raises(ValueError, match="repetition does not match"):
        generate.generate_auggpt("trec", fake=True, seed=42)


def test_unusable_duplicate_leaked_and_copied_candidates_are_rejected(fake_env, monkeypatch):
    own = sources(13)
    test_texts = {" ".join(r["text"].lower().split()) for r in read_jsonl(real_path("trec", "test"))}
    source_texts = {" ".join(r["text"].lower().split()) for r in own.values()}
    clean = sorted((r for r in own.values() if " ".join(r["text"].lower().split()) not in test_texts),
                   key=lambda r: r["id"])
    first, other = clean[0], next(r for r in clean[1:] if r["text"] != clean[0]["text"])
    first_id = first["id"]
    own = {first_id: first, other["id"]: other}
    test_text = next(t for t in sorted(test_texts) if t not in source_texts)
    replies = iter([
        ["", "   ", "one", first["text"], other["text"].upper(), test_text, "a fine paraphrase"],
        ["A  fine paraphrase", "another fine paraphrase"],
    ])

    def scripted(prompt, fake, max_words=None, retries=3, stats=None):
        stats["calls"] += 1
        try:
            raw = json.dumps({"examples": next(replies)})
        except StopIteration:
            raw = "sorry, I cannot help"
        out = generate.parse(raw, max_words=max_words)
        if not out:
            stats["parse_failures"] += 1
        stats["candidates_received"] += len(out)
        return out

    monkeypatch.setattr(generate, "ask", scripted)
    monkeypatch.setattr(generate, "load_auggpt_sources", lambda task, seed: dict(own))
    rows, provenance = generate.generate_auggpt("trec", fake=True, seed=13)

    accepted = [r["text"] for r in rows if r["source_ids"] == [first_id]]
    assert accepted == ["a fine paraphrase", "another fine paraphrase"]
    stats = provenance["cumulative_stats"]
    assert stats["rejected_source_copy"] == 2
    assert stats["rejected_test_leak"] == 1
    assert stats["rejected_duplicate"] == 1
    assert provenance["is_complete"] is False


def test_verify_rejects_tampering_and_inexact_source_samples(fake_env, monkeypatch):
    scripted_model(monkeypatch)
    rows, _ = generate.generate_auggpt("trec", fake=True, seed=13)
    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    generate.verify_auggpt("trec", 13, mode="fake")

    with pytest.raises(ValueError, match="artifact or provenance sidecar is missing"):
        generate.verify_auggpt("trec", 7, mode="fake")
    with pytest.raises(ValueError, match="sidecar is missing"):
        generate.verify_auggpt("trec", 13, mode="real")

    monkeypatch.setitem(CFG, "lowres_per_label", 10)
    with pytest.raises(ValueError, match="not exactly 10 per label"):
        generate.verify_auggpt("trec", 13, mode="fake")
    monkeypatch.setitem(CFG, "lowres_per_label", 2)

    write_jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match="row count does not match"):
        generate.verify_auggpt("trec", 13, mode="fake")

    write_jsonl(path, rows)
    prov_path = path.with_suffix(".provenance.json")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    provenance["is_complete"] = False
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="completeness does not match"):
        generate.verify_auggpt("trec", 13, mode="fake")


def test_verify_cli_exit_status(fake_env, monkeypatch):
    scripted_model(monkeypatch)
    generate.generate_auggpt("trec", fake=True, seed=13)
    args = ["--setting", "auggpt", "--task", "trec", "--dry-run", "--verify"]

    with pytest.raises(SystemExit) as passed:
        generate.main(args + ["--seed", "13"])
    assert passed.value.code == 0
    with pytest.raises(SystemExit) as failed:
        generate.main(args + ["--seed", "all"])
    assert failed.value.code == 1
    with pytest.raises(SystemExit):
        generate.main(["--setting", "zero_shot", "--seed", "13", "--verify"])
