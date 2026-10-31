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


APPROVED = {
    "provider": "ollama",
    "configured_tag": "qwen3:4b",
    "resolved_name": "qwen3:4b",
    "digest": generate.APPROVED_GENERATORS["qwen3:4b"],
}
UNAPPROVED = {
    "provider": "ollama",
    "configured_tag": "llama3.2:3b",
    "resolved_name": "llama3.2:3b",
    "digest": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
}


def generate_real(monkeypatch, identity, seed=13, approve_anything=False):
    """Run 'real' AugGPT with a stand-in model reporting the given Ollama identity."""
    monkeypatch.setattr(generate, "resolve_model_identity", lambda: dict(identity))
    scripted_model(monkeypatch, prefix=f"real {identity['configured_tag']}")
    if approve_anything:
        with monkeypatch.context() as patched:
            patched.setattr(generate, "require_approved_model", lambda *args: None)
            return generate.generate_auggpt("trec", fake=False, seed=seed)
    return generate.generate_auggpt("trec", fake=False, seed=seed)


def test_valid_real_artifact_from_approved_generator_passes(fake_env, monkeypatch):
    rows, _ = generate_real(monkeypatch, APPROVED)
    summary = generate.verify_auggpt("trec", 13, mode="real")
    assert summary["real_run_evidence"] is True
    assert summary["model_identity"] == APPROVED
    assert summary["total_count"] == len(rows) == K * len(sources(13))

    prefixed = dict(APPROVED, digest="sha256:" + APPROVED["digest"])
    generate.require_approved_model(prefixed, "real")  # Ollama digest prefixes are equivalent


def test_verify_rejects_self_consistent_unapproved_model(fake_env, monkeypatch):
    """A complete, internally consistent artifact from another model is not acceptable."""
    generate_real(monkeypatch, UNAPPROVED, approve_anything=True)
    path = synth_path("trec", "auggpt", seed=13, mode="real")
    before = path.read_bytes(), path.with_suffix(".provenance.json").read_bytes()

    with pytest.raises(ValueError, match="not the approved generator qwen3:4b"):
        generate.verify_auggpt("trec", 13, mode="real")
    with pytest.raises(SystemExit) as cli:
        generate.main(["--setting", "auggpt", "--task", "trec", "--seed", "13", "--verify"])
    assert cli.value.code == 1
    assert (path.read_bytes(), path.with_suffix(".provenance.json").read_bytes()) == before


@pytest.mark.parametrize("field, value", [
    ("digest", "0" * 64),
    ("configured_tag", "qwen3:8b"),
    ("resolved_name", "qwen3:latest"),
    ("provider", "fixture"),
])
def test_verify_rejects_tampered_approved_identity(fake_env, monkeypatch, field, value):
    generate_real(monkeypatch, dict(APPROVED, **{field: value}), approve_anything=True)
    with pytest.raises(ValueError, match="not the approved generator"):
        generate.verify_auggpt("trec", 13, mode="real")


def test_real_generation_refuses_unapproved_model_before_calls(fake_env, monkeypatch):
    monkeypatch.setattr(generate, "resolve_model_identity", lambda: dict(UNAPPROVED))
    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="not the approved generator"):
        generate.generate_auggpt("trec", fake=False, seed=13)
    assert calls["calls"] == 0
    assert not synth_path("trec", "auggpt", seed=13, mode="real").exists()


def test_fake_artifacts_need_the_fixture_identity_and_never_count_as_real(fake_env, monkeypatch):
    scripted_model(monkeypatch)
    generate.generate_auggpt("trec", fake=True, seed=13)
    assert generate.verify_auggpt("trec", 13, mode="fake")["real_run_evidence"] is False
    with pytest.raises(ValueError, match="sidecar is missing"):
        generate.verify_auggpt("trec", 13, mode="real")

    prov_path = synth_path("trec", "auggpt", seed=13, mode="fake").with_suffix(".provenance.json")
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    provenance["model_identity"] = APPROVED
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="non-fixture model identity"):
        generate.verify_auggpt("trec", 13, mode="fake")


def contaminate(path, rows, test_text):
    """Replace one paraphrase with held-out text, keeping row count and sidecar self-consistent."""
    altered = [dict(row) for row in rows]
    altered[3]["text"] = "  " + test_text.upper() + "  "
    write_jsonl(path, altered)
    return altered[3]["id"]


def unused_test_text(rows):
    taken = {" ".join(r["text"].lower().split()) for r in rows}
    taken |= {" ".join(r["text"].lower().split()) for r in sources(13).values()}
    return next(r for r in read_jsonl(real_path("trec", "test"))
                if " ".join(r["text"].lower().split()) not in taken)


@pytest.mark.parametrize("real", [False, True])
def test_verify_rejects_held_out_overlap_read_only(fake_env, monkeypatch, real):
    if real:
        rows, _ = generate_real(monkeypatch, APPROVED)
    else:
        scripted_model(monkeypatch)
        rows, _ = generate.generate_auggpt("trec", fake=True, seed=13)
    mode = "real" if real else "fake"
    path = synth_path("trec", "auggpt", seed=13, mode=mode)
    test_row = unused_test_text(rows)
    bad_id = contaminate(path, rows, test_row["text"])
    before = path.read_bytes(), path.with_suffix(".provenance.json").read_bytes()

    with pytest.raises(ValueError) as error:
        generate.verify_auggpt("trec", 13, mode=mode)
    message = str(error.value)
    assert "held-out test text" in message and bad_id in message and test_row["id"] in message
    assert "regenerate" in message
    assert (path.read_bytes(), path.with_suffix(".provenance.json").read_bytes()) == before


def test_resume_rejects_held_out_overlap_in_loaded_rows(fake_env, monkeypatch):
    scripted_model(monkeypatch, per_call=1)
    rows, _ = generate.generate_auggpt("trec", fake=True, seed=13)  # incomplete: resumable
    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    contaminate(path, rows, unused_test_text(rows)["text"])
    before = path.read_bytes()

    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="contaminated artifact"):
        generate.generate_auggpt("trec", fake=True, seed=13)
    assert calls["calls"] == 0
    assert path.read_bytes() == before


def test_verify_rejects_invalid_and_extra_sources(fake_env, monkeypatch):
    scripted_model(monkeypatch)
    rows, _ = generate.generate_auggpt("trec", fake=True, seed=13)
    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    prov_path = path.with_suffix(".provenance.json")
    original_rows, original_prov = path.read_bytes(), prov_path.read_bytes()

    outside = [dict(row) for row in rows]
    outside[0]["source_ids"] = ["trec.pool.99999"]
    outside[0]["id"] = "trec.pool.99999.seed13.aug1"
    write_jsonl(path, outside)
    with pytest.raises(ValueError, match="outside the configured low-resource sample"):
        generate.verify_auggpt("trec", 13, mode="fake")
    path.write_bytes(original_rows)

    provenance = json.loads(original_prov)
    provenance["counts_by_source"]["trec.pool.99999"] = 0
    prov_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="completeness does not match"):
        generate.verify_auggpt("trec", 13, mode="fake")
    prov_path.write_bytes(original_prov)

    lowres = read_jsonl(real_path("trec", "lowres_seed13"))
    extra = dict(lowres[0], id="trec.pool.99998", text="an extra source question ?")
    write_jsonl(real_path("trec", "lowres_seed13"), lowres + [extra])
    with pytest.raises(ValueError, match="not exactly 2 per label"):
        generate.verify_auggpt("trec", 13, mode="fake")


LONG_SOURCE = ("why do so many of the people who live in the small towns along the river "
               "still use the old wooden ferry to cross it every single morning ?")


def test_long_source_prompt_asks_for_paraphrases_within_the_word_limit():
    max_words = 20
    assert len(LONG_SOURCE.split()) > max_words
    short = "what is the capital of peru ?"

    assert generate.auggpt_prompt(short, 6, max_words) == generate.prompt_auggpt(short, 6)
    guarded = generate.auggpt_prompt(LONG_SOURCE, 6, max_words)
    assert guarded.startswith("Please rephrase the following sentence 6 different ways")
    assert json.dumps(LONG_SOURCE) in guarded
    assert f"at most {max_words} words" in guarded


def test_generation_sends_the_length_guard_only_for_long_sources(fake_env, monkeypatch):
    rows = sorted(sources(13).values(), key=lambda r: r["id"])
    short_row, long_row = rows[0], dict(rows[1], text=LONG_SOURCE)
    prompts = {}

    def recording(prompt, fake, max_words=None, retries=3, stats=None):
        stats["calls"] += 1
        source = short_row if json.dumps(short_row["text"]) in prompt else long_row
        prompts[source["id"]] = prompt
        return [f"{source['id']} paraphrase {i}" for i in range(K)]

    monkeypatch.setattr(generate, "ask", recording)
    monkeypatch.setattr(generate, "load_auggpt_sources",
                        lambda task, seed: {short_row["id"]: short_row, long_row["id"]: long_row})
    generate.generate_auggpt("trec", fake=True, seed=13)

    max_words = CFG["tasks"]["trec"]["max_words"]
    assert prompts[short_row["id"]] == generate.prompt_auggpt(short_row["text"], K)
    assert prompts[long_row["id"]] == generate.auggpt_prompt(LONG_SOURCE, K, max_words)
    assert prompts[long_row["id"]] != generate.prompt_auggpt(LONG_SOURCE, K)


def pre_length_guard_identity(task, seed, mode, model_identity, sources_by_id):
    """The AugGPT identity exactly as computed before the length guard existed."""
    test_rows = read_jsonl(real_path(task, "test"))
    identity, fingerprint = generate.build_generation_identity(
        task, "auggpt", seed, mode, model_identity,
        {"low_resource_sample": generate.rows_identity(list(sources_by_id.values())),
         "held_out_test": generate.rows_identity(test_rows)},
    )
    return identity, fingerprint, test_rows


def test_resume_carries_forward_rows_from_the_pre_length_guard_identity(fake_env, monkeypatch):
    """Rows saved before the length guard keep their recorded fingerprint and still count."""
    with monkeypatch.context() as old_code:
        old_code.setattr(generate, "auggpt_generation_identity", pre_length_guard_identity)
        scripted_model(old_code, per_call=1)
        legacy_rows, legacy = generate.generate_auggpt("trec", fake=True, seed=13)
    legacy_fp = legacy["generation_fingerprint"]
    assert legacy["is_complete"] is False

    scripted_model(monkeypatch, prefix="resumed")
    rows, provenance = generate.generate_auggpt("trec", fake=True, seed=13)

    assert rows[:len(legacy_rows)] == legacy_rows  # never rewritten
    new_rows = rows[len(legacy_rows):]
    assert new_rows and all(row["generation_fingerprint"] == provenance["generation_fingerprint"]
                            for row in new_rows)
    assert provenance["generation_fingerprint"] != legacy_fp
    assert "auggpt_length_guard" in provenance["generation_identity"]["prompt_extensions"]
    carried = provenance["carried_forward_identities"]
    assert [c["generation_fingerprint"] for c in carried] == [legacy_fp]
    assert carried[0]["generation_identity"] == legacy["generation_identity"]
    assert carried[0]["row_count"] == len(legacy_rows)
    assert provenance["is_complete"] is True
    summary = generate.verify_auggpt("trec", 13, mode="fake")
    assert summary["total_count"] == len(rows)


def test_verify_accepts_a_complete_artifact_from_the_pre_length_guard_identity(fake_env, monkeypatch):
    with monkeypatch.context() as old_code:
        old_code.setattr(generate, "auggpt_generation_identity", pre_length_guard_identity)
        scripted_model(old_code)
        rows, _ = generate.generate_auggpt("trec", fake=True, seed=13)
    assert generate.verify_auggpt("trec", 13, mode="fake")["total_count"] == len(rows)


def test_resume_still_rejects_rows_from_any_other_identity(fake_env, monkeypatch):
    with monkeypatch.context() as other:
        other.setattr(generate, "PROMPT_PROTOCOL_VERSION", "some-other-protocol")
        scripted_model(other, per_call=1)
        generate.generate_auggpt("trec", fake=True, seed=13)
    path = synth_path("trec", "auggpt", seed=13, mode="fake")
    before = path.read_bytes()

    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="generation identity does not match"):
        generate.generate_auggpt("trec", fake=True, seed=13)
    with pytest.raises(ValueError, match="generation identity does not match"):
        generate.verify_auggpt("trec", 13, mode="fake")
    assert calls["calls"] == 0 and path.read_bytes() == before


def test_verify_cli_exit_status(fake_env, monkeypatch):
    scripted_model(monkeypatch)
    generate.generate_auggpt("trec", fake=True, seed=13)
    args = ["--setting", "auggpt", "--task", "trec", "--dry-run", "--verify"]

    with pytest.raises(SystemExit) as passed:
        generate.main(args + ["--seed", "13"])
    assert passed.value.code == 0
    with pytest.raises(SystemExit) as failed:  # emotion seed 13 is missing
        generate.main(["--setting", "auggpt", "--task", "all", "--dry-run", "--verify", "--seed", "13"])
    assert failed.value.code == 1
    with pytest.raises(SystemExit):
        generate.main(["--setting", "zero_shot", "--seed", "13", "--verify"])
