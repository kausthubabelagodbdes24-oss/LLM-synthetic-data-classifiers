"""Fake-model checks for the ADR 0013 TREC AugGPT answer-type correction (issues #9 / #11).

Scripted stand-in models only. These establish routing, identity, isolation, resume and file
preservation; they cannot establish that real paraphrases keep their answer type.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from common import CFG, read_jsonl, real_path, synth_path, write_jsonl
from ws1_generation import generate, promote_auggpt

ROOT = Path(__file__).resolve().parents[1]
K = CFG["auggpt_k"]
VERSION = generate.TREC_ANSWER_TYPE_VERSION
APPROVED = {"provider": "ollama", "configured_tag": "qwen3:4b", "resolved_name": "qwen3:4b",
            "digest": generate.APPROVED_GENERATORS["qwen3:4b"]}


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data" / "sample" / "real", tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setitem(CFG, "lowres_per_label", 2)
    return tmp_path


def scripted_model(monkeypatch, prefix="para", per_call=None, interrupt_after=None):
    """Unique candidates; records every prompt. Optionally raises KeyboardInterrupt."""
    state = {"calls": 0, "prompts": []}

    def fake_ask(prompt, fake, max_words=None, retries=3, stats=None):
        if interrupt_after is not None and state["calls"] >= interrupt_after:
            raise KeyboardInterrupt
        state["calls"] += 1
        state["prompts"].append(prompt)
        if stats is not None:
            stats["calls"] += 1
        n = int(re.search(r"(\d+) different ways", prompt).group(1))
        n = min(n, per_call) if per_call else n
        out = [f"{prefix} question {state['calls']} form {i} ?" for i in range(n)]
        if stats is not None:
            stats["candidates_received"] += len(out)
        return out

    monkeypatch.setattr(generate, "ask", fake_ask)
    return state


def sources(task="trec", seed=13):
    return {row["id"]: row for row in read_jsonl(real_path(task, f"lowres_seed{seed}"))}


def make_original(monkeypatch, mode="fake"):
    """The pre-correction active artifact, produced exactly as before ADR 0013."""
    with monkeypatch.context() as old_code:
        old_code.setattr(generate, "AUGGPT_CORRECTIONS", {})
        scripted_model(old_code, prefix="original")
        if mode == "real":
            old_code.setattr(generate, "resolve_model_identity", lambda: dict(APPROVED))
        rows, prov = generate.generate_auggpt("trec", fake=mode == "fake", seed=13)
    path = synth_path("trec", "auggpt", seed=13, mode=mode)
    return path, rows, prov


def snapshot(path):
    return path.read_bytes(), path.with_suffix(".provenance.json").read_bytes()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------- prompt routing and label definitions ----------------

@pytest.mark.parametrize("label", list(CFG["tasks"]["trec"]["labels"]))
def test_trec_prompt_carries_the_canonical_label_and_its_definition(label):
    definition = CFG["tasks"]["trec"]["labels"][label]
    text = "what does nasa stand for ?"
    prompt = generate.auggpt_prompt_for("trec", 13, {"text": text, "label": label}, 6)

    assert prompt.startswith("Please rephrase the following question 6 different ways")
    assert json.dumps(text) in prompt
    assert f'label is "{label}", meaning: {definition}.' in prompt
    for kept in ("expected answer type", "facts and relationships", "entities and participants",
                 "negation", "intent", "Do not answer the question", "do not add facts"):
        assert kept in prompt
    assert f"at most {CFG['tasks']['trec']['max_words']} words" in prompt


def test_abbreviation_and_description_rules_are_label_specific():
    abbr = generate.auggpt_prompt_for("trec", 13, {"text": "what is nato ?", "label": "abbreviation"}, 6)
    desc = generate.auggpt_prompt_for("trec", 13, {"text": "why is the sky blue ?", "label": "description"}, 6)
    assert "still ask what it stands for (its expansion), not for a definition" in abbr
    assert "remain a question asking for an explanation" in desc
    assert "remain a question asking for an explanation" not in abbr
    assert "its expansion" not in desc


def test_routing_is_scoped_to_trec_seed13_and_leaves_emotion_byte_identical():
    row = {"text": "i feel oddly calm today", "label": "joy"}
    max_e = CFG["tasks"]["emotion"]["max_words"]
    assert generate.auggpt_prompt_for("emotion", 13, row, 6) == generate.auggpt_prompt(row["text"], 6, max_e)
    trec_row = {"text": "what is nato ?", "label": "abbreviation"}
    # Other TREC repetitions are historical and keep the uncorrected prompt.
    assert generate.auggpt_prompt_for("trec", 42, trec_row, 6) == generate.prompt_auggpt(trec_row["text"], 6)


def test_generation_sends_every_trec_source_its_own_label_definition(fake_env, monkeypatch):
    state = scripted_model(monkeypatch)
    generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)

    # Two fixture sources share a text (same label), so prompts are counted per (text, label).
    expected = {}
    for r in sources().values():
        key = generate.prompt_auggpt_trec_answer_type(
            r["text"], K, r["label"], CFG["tasks"]["trec"]["labels"][r["label"]],
            CFG["tasks"]["trec"]["max_words"])
        expected[key] = expected.get(key, 0) + 1
    sent = {}
    for prompt in state["prompts"]:
        sent[prompt] = sent.get(prompt, 0) + 1
    assert sent == expected


# ---------------- identity separation ----------------

def test_corrected_identity_is_distinct_and_emotion_identity_is_unchanged(fake_env):
    fixture = dict(generate.FIXTURE_MODEL_IDENTITY)
    guard, guard_fp, test_rows = generate.auggpt_generation_identity("trec", 13, "fake", fixture, sources())
    pre_guard = generate.auggpt_carried_forward_identities(
        "trec", 13, "fake", fixture, sources(), test_rows, guard_fp)
    corrected, fp, _, carried, correction = generate.auggpt_identity("trec", 13, "fake", fixture, sources())

    assert correction["version"] == VERSION and carried == {}
    assert fp not in {guard_fp, *pre_guard}
    ext = corrected["prompt_extensions"]["trec_answer_type"]
    assert ext["label_definitions"] == CFG["tasks"]["trec"]["labels"]
    assert ext["carry_forward"] == "none"
    assert "auggpt_length_guard" not in corrected["prompt_extensions"]

    hist = generate.auggpt_identity("trec", 13, "fake", fixture, sources(), historical=True)
    assert (hist[1], hist[3], hist[4]) == (guard_fp, pre_guard, None)

    emo = sources("emotion")
    e_identity, e_fp, _ = generate.auggpt_generation_identity("emotion", 13, "fake", fixture, emo)
    assert generate.auggpt_identity("emotion", 13, "fake", fixture, emo)[:2] == (e_identity, e_fp)


def test_changing_the_corrected_template_changes_the_fingerprint(fake_env, monkeypatch):
    fixture = dict(generate.FIXTURE_MODEL_IDENTITY)
    _, before, *_ = generate.auggpt_identity("trec", 13, "fake", fixture, sources())
    real_getsource = generate.inspect.getsource
    monkeypatch.setattr(generate.inspect, "getsource", lambda f: real_getsource(f) + (
        "#edit" if f is generate.prompt_auggpt_trec_answer_type else ""))
    _, after, *_ = generate.auggpt_identity("trec", 13, "fake", fixture, sources())
    assert before != after


# ---------------- old-row rejection ----------------

def test_old_active_artifact_is_not_accepted_as_corrected_output(fake_env, monkeypatch):
    path, rows, _ = make_original(monkeypatch)
    before = snapshot(path)

    with pytest.raises(ValueError, match=f"superseded by {VERSION}"):
        generate.verify_auggpt("trec", 13, mode="fake")
    hist = generate.verify_auggpt("trec", 13, mode="fake", historical=True)
    assert hist["corrected_output"] is False and hist["historical_identity"] is True
    assert hist["total_count"] == len(rows)
    # Fixture (fake) generation in place cannot reuse the old rows either.
    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="generation identity does not match"):
        generate.generate_auggpt("trec", fake=True, seed=13)
    assert calls["calls"] == 0 and snapshot(path) == before


def test_old_rows_copied_into_the_candidate_are_rejected(fake_env, monkeypatch):
    path, rows, _ = make_original(monkeypatch)
    cand = generate.auggpt_candidate_path("trec", 13, "fake")
    cand.parent.mkdir(parents=True)
    shutil.copy(path, cand)
    shutil.copy(path.with_suffix(".provenance.json"), cand.with_suffix(".provenance.json"))

    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="generation identity does not match"):
        generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    assert calls["calls"] == 0
    with pytest.raises(ValueError, match="superseded"):
        generate.verify_auggpt("trec", 13, mode="fake", path=cand)


def test_a_single_old_row_inside_a_corrected_candidate_is_rejected(fake_env, monkeypatch):
    _, old_rows, _ = make_original(monkeypatch)
    scripted_model(monkeypatch)
    rows, _ = generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    cand = generate.auggpt_candidate_path("trec", 13, "fake")
    generate.verify_auggpt("trec", 13, mode="fake", path=cand)

    swapped = [dict(r) for r in rows]
    victim = next(i for i, r in enumerate(swapped) if r["id"] == old_rows[0]["id"])
    swapped[victim] = old_rows[0]
    write_jsonl(cand, swapped)
    with pytest.raises(ValueError, match="generation identity does not match"):
        generate.verify_auggpt("trec", 13, mode="fake", path=cand)


# ---------------- candidate isolation and resumability ----------------

def test_candidate_resumes_to_six_per_source_without_touching_the_original(fake_env, monkeypatch):
    path, _, _ = make_original(monkeypatch)
    before = snapshot(path)

    scripted_model(monkeypatch, per_call=1)  # three attempts x one = three per source
    first, prov = generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    assert prov["is_complete"] is False and snapshot(path) == before
    assert prov["prompt_correction"]["version"] == VERSION
    assert prov["prompt_correction"]["generated_as"] == "candidate"

    with pytest.raises(KeyboardInterrupt):
        scripted_model(monkeypatch, prefix="cut", interrupt_after=2)
        generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    cand = generate.auggpt_candidate_path("trec", 13, "fake")
    interrupted = json.loads(cand.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert interrupted["sessions"][-1]["status"] == "interrupted"
    assert interrupted["total_count"] == len(read_jsonl(cand))

    scripted_model(monkeypatch, prefix="resumed")
    rows, final = generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    assert rows[:len(first)] == first
    assert final["is_complete"] is True and len(rows) == K * len(sources())
    assert {r["generation_fingerprint"] for r in rows} == {final["generation_fingerprint"]}
    assert final["carried_forward_identities"] == []
    summary = generate.verify_auggpt("trec", 13, mode="fake", path=cand)
    assert summary["corrected_output"] is True and summary["prompt_correction"] == VERSION
    assert snapshot(path) == before
    assert not list(path.parent.glob("*.generation-journal.json"))


def test_candidate_journal_crash_is_recovered_in_the_candidate_only(fake_env, monkeypatch):
    path, _, _ = make_original(monkeypatch)
    before = snapshot(path)
    real_write = generate.write_labelled_batch
    calls = {"n": 0}

    def crash_after_append(p, existing, new, prov, fp):
        calls["n"] += 1
        if calls["n"] == 2:
            generate.atomic_write_json(generate.generation_journal_path(p), {
                "schema_version": 2, "artifact_name": p.name, "generation_fingerprint": fp,
                "rows_before": len(existing),
                "prefix_byte_count": len(p.read_bytes()),
                "prefix_bytes_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "batch_rows": new, "batch_sha256": generate._identity_hash(new),
                "batch_jsonl": generate.serialize_jsonl_rows(new).decode("utf-8"),
                "batch_bytes_sha256": hashlib.sha256(generate.serialize_jsonl_rows(new)).hexdigest(),
                "provenance_after": prov,
            })
            write_jsonl(p, new, append=True)
            raise RuntimeError("simulated crash")
        return real_write(p, existing, new, prov, fp)

    monkeypatch.setattr(generate, "write_labelled_batch", crash_after_append)
    scripted_model(monkeypatch)
    with pytest.raises(RuntimeError):
        generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    cand = generate.auggpt_candidate_path("trec", 13, "fake")
    assert generate.generation_journal_path(cand).exists()
    with pytest.raises(ValueError, match="unrecovered"):
        generate.verify_auggpt("trec", 13, mode="fake", path=cand)

    monkeypatch.setattr(generate, "write_labelled_batch", real_write)
    scripted_model(monkeypatch, prefix="after")
    rows, prov = generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    assert prov["is_complete"] and len(rows) == K * len(sources())
    assert "interrupted" in [s["status"] for s in prov["sessions"]]
    assert snapshot(path) == before


def test_real_trec_seed13_refuses_in_place_generation_before_model_calls(fake_env, monkeypatch):
    path, _, _ = make_original(monkeypatch, mode="real")
    before = snapshot(path)
    monkeypatch.setattr(generate, "resolve_model_identity", lambda: pytest.fail("model resolved"))
    calls = scripted_model(monkeypatch)
    with pytest.raises(ValueError, match="--candidate"):
        generate.generate_auggpt("trec", fake=False, seed=13)
    assert calls["calls"] == 0 and snapshot(path) == before


def test_real_candidate_keeps_the_approved_model_and_settings(fake_env, monkeypatch):
    make_original(monkeypatch, mode="real")
    monkeypatch.setattr(generate, "resolve_model_identity", lambda: dict(APPROVED))
    scripted_model(monkeypatch)
    _, prov = generate.generate_auggpt("trec", fake=False, seed=13, candidate=True)
    assert prov["model_identity"] == APPROVED and prov["temperature"] == CFG["temperature"]
    assert prov["generation_identity"]["options"]["temperature"] == CFG["temperature"]
    assert prov["generation_identity"]["options"]["per_call"] == K
    cand = generate.auggpt_candidate_path("trec", 13, "real")
    assert generate.verify_auggpt("trec", 13, mode="real", path=cand)["real_run_evidence"] is True


# ---------------- promotion ----------------

def complete_candidate(monkeypatch):
    path, rows, _ = make_original(monkeypatch)
    scripted_model(monkeypatch, prefix="corrected")
    generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    return path, generate.auggpt_candidate_path("trec", 13, "fake")


def test_promotion_preserves_the_original_and_records_hashes(fake_env, monkeypatch):
    path, cand = complete_candidate(monkeypatch)
    original = snapshot(path)
    candidate = snapshot(cand)

    plan = promote_auggpt.plan("trec", 13, mode="fake")
    assert snapshot(path) == original  # plan is read-only
    assert plan["confirm_with"] == ["--candidate-sha256", sha(cand), "--original-sha256", sha(path)]
    assert plan["original"]["generation_fingerprint"] != plan["candidate"]["generation_fingerprint"]

    with pytest.raises(promote_auggpt.PromotionError, match="rerun --plan"):
        promote_auggpt.promote("trec", 13, "0" * 64, sha(path), mode="fake")
    assert snapshot(path) == original

    result = promote_auggpt.promote("trec", 13, sha(cand), sha(path), mode="fake")
    loc = promote_auggpt.locations("trec", 13, "fake")
    assert snapshot(path) == candidate and snapshot(cand) == candidate
    assert snapshot(loc["archive"]) == original
    record = json.loads(loc["record"].read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["original"]["artifact"]["sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert record["original"]["provenance"]["sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert record["promoted"]["artifact"]["sha256"] == hashlib.sha256(candidate[0]).hexdigest()
    assert result["preserved_original"]["artifact_sha256"] == hashlib.sha256(original[0]).hexdigest()

    # The active artifact now verifies as corrected output and resolves its promotion record.
    summary = generate.verify_auggpt("trec", 13, mode="fake")
    assert summary["corrected_output"] is True and summary["promotion_record"]["status"] == "complete"
    # Historical evidence stays readable, as superseded output.
    assert generate.verify_auggpt("trec", 13, mode="fake", path=loc["archive"],
                                  historical=True)["corrected_output"] is False
    # Rerunning is a no-op; a second plan refuses.
    assert promote_auggpt.promote("trec", 13, sha(cand), hashlib.sha256(original[0]).hexdigest(),
                                  mode="fake")["active"] == result["active"]
    with pytest.raises(promote_auggpt.PromotionError, match="promotion was started"):
        promote_auggpt.plan("trec", 13, mode="fake")


def test_interrupted_promotion_completes_on_rerun(fake_env, monkeypatch):
    path, cand = complete_candidate(monkeypatch)
    original_sha, candidate_sha = sha(path), sha(cand)
    real_replace = promote_auggpt._replace
    calls = {"n": 0}

    def crash_on_sidecar(target, raw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash")
        real_replace(target, raw)

    monkeypatch.setattr(promote_auggpt, "_replace", crash_on_sidecar)
    with pytest.raises(OSError):
        promote_auggpt.promote("trec", 13, candidate_sha, original_sha, mode="fake")
    assert sha(path) == candidate_sha  # JSONL replaced, sidecar still original
    with pytest.raises(promote_auggpt.PromotionError, match="in_progress"):
        promote_auggpt.verify_promotion("trec", 13, mode="fake")

    monkeypatch.setattr(promote_auggpt, "_replace", real_replace)
    promote_auggpt.promote("trec", 13, candidate_sha, original_sha, mode="fake")
    assert promote_auggpt.verify_promotion("trec", 13, mode="fake")["active"]["artifact_sha256"] == candidate_sha


def test_promotion_refuses_incomplete_candidates_and_tampered_archives(fake_env, monkeypatch):
    path, _, _ = make_original(monkeypatch)
    original = snapshot(path)
    scripted_model(monkeypatch, per_call=1)
    generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    cand = generate.auggpt_candidate_path("trec", 13, "fake")
    with pytest.raises(promote_auggpt.PromotionError, match="sources lack"):
        promote_auggpt.promote("trec", 13, sha(cand), sha(path), mode="fake")
    assert snapshot(path) == original
    assert not promote_auggpt.locations("trec", 13, "fake")["record"].exists()

    scripted_model(monkeypatch, prefix="done")
    generate.generate_auggpt("trec", fake=True, seed=13, candidate=True)
    promote_auggpt.promote("trec", 13, sha(cand), sha(path), mode="fake")
    archive = promote_auggpt.locations("trec", 13, "fake")["archive"]
    archive.write_bytes(archive.read_bytes() + b"\n")
    with pytest.raises(promote_auggpt.PromotionError, match="does not match the promotion record"):
        promote_auggpt.verify_promotion("trec", 13, mode="fake")


def test_cli_rejects_unscoped_candidate_use(fake_env):
    for argv in (["--setting", "auggpt", "--task", "emotion", "--seed", "13", "--candidate"],
                 ["--setting", "auggpt", "--task", "trec", "--seed", "all", "--candidate"],
                 ["--setting", "auggpt", "--task", "trec", "--seed", "13", "--historical"]):
        with pytest.raises(SystemExit) as exc:
            generate.main(argv)
        assert exc.value.code == 2
