"""Issue #10: scope30-v1 deterministic selection (ADR 0012). Fake model / temp data only."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from common import labels, read_jsonl, real_path, synth_path, write_jsonl
from ws1_generation import generate
from ws1_generation import select_scope30 as sel

ROOT = Path(__file__).resolve().parents[1]
APPROVED = {
    "provider": "ollama",
    "configured_tag": "qwen3:4b",
    "resolved_name": "qwen3:4b",
    "digest": generate.APPROVED_GENERATORS["qwen3:4b"],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data" / "sample" / "real", tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    return tmp_path


def fake_generate(task, setting, target):
    generate.main(["--setting", setting, "--task", task, "--seed", "13", "--target", str(target), "--dry-run"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv):
    with pytest.raises(SystemExit) as exc:
        sel.main(argv)
    return exc.value.code


def expected_ids(task, setting, rows):
    out = set()
    for lab in labels(task):
        group = sorted((r["id"] for r in rows if r["label"] == lab),
                       key=lambda rid: (hashlib.sha256(f"scope30-v1|13|{task}|{setting}|{rid}".encode()).hexdigest(), rid))
        out.update(group[:30])
    return out


@pytest.mark.parametrize("task", ["trec", "emotion"])
@pytest.mark.parametrize("setting", ["zero_shot", "few_shot"])
def test_excess_selects_exact_hash_rule_and_preserves_originals(env, task, setting):
    fake_generate(task, setting, 34)
    path = synth_path(task, setting, seed=13, mode="fake")
    before = (digest(path), digest(path.with_suffix(".provenance.json")))
    rows = read_jsonl(path)

    summary = sel.write_selection(task, setting, mode="fake")

    assert (digest(path), digest(path.with_suffix(".provenance.json"))) == before   # originals untouched
    manifest = json.loads(sel.manifest_path(task, setting, "fake").read_text(encoding="utf-8"))
    selected = read_jsonl(sel.selected_path(task, setting, "fake"))
    assert summary["selected_total"] == 180 and len(selected) == 180
    assert all(manifest["selected_counts_by_label"][lab] == 30 for lab in labels(task))
    assert {r["id"] for r in selected} == expected_ids(task, setting, rows) == set(manifest["selected_ids"])
    assert all(r in rows for r in selected)                                         # rows unchanged
    assert manifest["original_artifact"]["sha256"] == before[0]
    assert manifest["original_artifact"]["provenance_sha256"] == before[1]
    assert manifest["original_artifact"]["row_count"] == 34 * 6
    assert manifest["producer"]["session_targets_per_label"] == [34]
    assert manifest["producer"]["generation_fingerprint"] == rows[0]["generation_fingerprint"]
    assert manifest["real_run_evidence"] is False
    if setting == "few_shot":
        demos = {s for r in selected for s in r["source_ids"]}
        assert manifest["selected_distinct_demonstrations_count"] == len(demos) > 0
    assert sel.verify_selection(task, setting, mode="fake") == summary


def test_selection_is_reproducible_and_independent_of_file_order(env):
    fake_generate("trec", "zero_shot", 33)
    first = sel.write_selection("trec", "zero_shot", mode="fake")
    m_bytes = sel.manifest_path("trec", "zero_shot", "fake").read_bytes()
    assert sel.write_selection("trec", "zero_shot", mode="fake") == first          # idempotent rewrite
    assert sel.manifest_path("trec", "zero_shot", "fake").read_bytes() == m_bytes
    rows = read_jsonl(synth_path("trec", "zero_shot", seed=13, mode="fake"))
    ranked = sel.select_rows(list(reversed(rows)), "trec", "zero_shot")
    assert {r["id"] for g in ranked.values() for r in g} == set(json.loads(m_bytes)["selected_ids"])


def test_shortfall_blocks_selection_then_compatible_resume_completes(env):
    fake_generate("emotion", "few_shot", 10)                  # like the accepted pilot
    path = synth_path("emotion", "few_shot", seed=13, mode="fake")
    pilot_rows = read_jsonl(path)
    assert run(["--seed", "13", "--task", "emotion", "--setting", "few_shot", "--dry-run"]) == 1
    assert not sel.manifest_path("emotion", "few_shot", "fake").exists()
    st = sel.status("emotion", "few_shot", mode="fake")
    assert st["shortfall_to_30"] == {lab: 20 for lab in labels("emotion")}
    assert run(["--seed", "13", "--task", "emotion", "--setting", "few_shot", "--status", "--dry-run"]) == 1

    fake_generate("emotion", "few_shot", 30)                  # resume the shortfall only
    rows = read_jsonl(path)
    assert rows[:len(pilot_rows)] == pilot_rows and len(rows) == 180
    assert run(["--seed", "13", "--task", "emotion", "--setting", "few_shot", "--dry-run"]) == 0
    manifest = json.loads(sel.manifest_path("emotion", "few_shot", "fake").read_text(encoding="utf-8"))
    assert manifest["producer"]["session_targets_per_label"] == [10, 30]       # history kept
    assert run(["--seed", "13", "--task", "emotion", "--setting", "few_shot", "--verify", "--dry-run"]) == 0


def test_mixed_excess_and_exact_labels(env):
    """Mirrors trec zero-shot: one label far above 30 (historical 100 target), the rest resumed to 30."""
    fake_generate("trec", "zero_shot", 10)
    generate.main(["--setting", "zero_shot", "--task", "trec", "--seed", "13", "--target", "45", "--dry-run"])
    fake_generate("trec", "zero_shot", 30)                    # nothing new: all labels >= 30
    rows = read_jsonl(synth_path("trec", "zero_shot", seed=13, mode="fake"))
    sel.write_selection("trec", "zero_shot", mode="fake")
    manifest = json.loads(sel.manifest_path("trec", "zero_shot", "fake").read_text(encoding="utf-8"))
    assert manifest["original_artifact"]["excess_by_label"] == {lab: 15 for lab in labels("trec")}
    assert manifest["producer"]["session_targets_per_label"] == [10, 45, 30]
    assert manifest["producer"]["recorded_target_per_label"] == 30
    assert set(manifest["selected_ids"]) == expected_ids("trec", "zero_shot", rows)


def test_only_seed_13_is_accepted(env):
    with pytest.raises(SystemExit) as exc:
        sel.main(["--seed", "42", "--dry-run"])
    assert exc.value.code == 2


@pytest.mark.parametrize("tamper", ["selected_row", "selected_drop", "manifest_id", "manifest_hash"])
def test_verify_rejects_tampered_derived_files(env, tamper):
    fake_generate("trec", "few_shot", 31)
    sel.write_selection("trec", "few_shot", mode="fake")
    m_path, s_path = sel.manifest_path("trec", "few_shot", "fake"), sel.selected_path("trec", "few_shot", "fake")
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    selected = read_jsonl(s_path)
    if tamper == "selected_row":
        selected[0]["text"] = "what is a tampered question ?"
        write_jsonl(s_path, selected)
    elif tamper == "selected_drop":
        write_jsonl(s_path, selected[1:])
    elif tamper == "manifest_id":
        manifest["selected_ids"][0] = "trec.few_shot.seed13.abbreviation.0031"
        m_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        manifest["original_artifact"]["sha256"] = "0" * 64
        m_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(sel.SelectionError):
        sel.verify_selection("trec", "few_shot", mode="fake")


def test_changed_original_invalidates_selection_and_is_not_overwritten(env):
    fake_generate("emotion", "zero_shot", 30)
    sel.write_selection("emotion", "zero_shot", mode="fake")
    old_manifest = sel.manifest_path("emotion", "zero_shot", "fake").read_bytes()
    fake_generate("emotion", "zero_shot", 32)                 # original grows after selection
    with pytest.raises(sel.SelectionError, match="manifest does not match"):
        sel.verify_selection("emotion", "zero_shot", mode="fake")
    with pytest.raises(sel.SelectionError, match="already exists with different content"):
        sel.write_selection("emotion", "zero_shot", mode="fake")
    assert sel.manifest_path("emotion", "zero_shot", "fake").read_bytes() == old_manifest


def test_pending_journal_and_test_overlap_fail_closed(env):
    fake_generate("trec", "zero_shot", 30)
    path = synth_path("trec", "zero_shot", seed=13, mode="fake")
    journal = generate.generation_journal_path(path)
    journal.write_text("{}", encoding="utf-8")
    with pytest.raises(sel.SelectionError, match="unrecovered"):
        sel.status("trec", "zero_shot", mode="fake")
    journal.unlink()
    test_rows = read_jsonl(real_path("trec", "test"))
    leaked = read_jsonl(path)[5]["text"]
    write_jsonl(real_path("trec", "test"), test_rows + [{"id": "trec.test.99999", "label": "entity", "text": leaked}])
    with pytest.raises(ValueError):                           # test identity and/or overlap rejected
        sel.write_selection("trec", "zero_shot", mode="fake")
    assert not sel.manifest_path("trec", "zero_shot", "fake").exists()


def test_corrupted_original_rows_fail_closed(env):
    fake_generate("emotion", "few_shot", 30)
    path = synth_path("emotion", "few_shot", seed=13, mode="fake")
    rows = read_jsonl(path)
    rows[3]["text"] = rows[2]["text"]                         # normalized duplicate
    write_jsonl(path, rows)
    with pytest.raises(sel.SelectionError, match="duplicate"):
        sel.write_selection("emotion", "few_shot", mode="fake")


def test_real_mode_requires_the_approved_generator(env, monkeypatch):
    """A real-mode artifact (mocked Ollama) selects only under the pinned qwen3:4b digest."""
    monkeypatch.setattr(generate, "resolve_model_identity", lambda: dict(APPROVED))
    monkeypatch.setattr(generate, "ask_ollama", lambda prompt, timeout=120, request=None: generate.ask_fake(prompt))
    generate.main(["--setting", "zero_shot", "--task", "emotion", "--seed", "13", "--target", "30"])
    summary = sel.write_selection("emotion", "zero_shot", mode="real")
    assert summary["real_run_evidence"] is True and summary["selected_total"] == 180
    assert sel.verify_selection("emotion", "zero_shot", mode="real") == summary

    prov_path = synth_path("emotion", "zero_shot", seed=13, mode="real").with_suffix(".provenance.json")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    prov["model_identity"] = {**APPROVED, "digest": "f" * 64}
    prov_path.write_text(json.dumps(prov), encoding="utf-8")
    with pytest.raises(sel.SelectionError, match="approved generator"):
        sel.verify_selection("emotion", "zero_shot", mode="real")


def test_fake_artifacts_cannot_satisfy_real_selection(env):
    fake_generate("trec", "few_shot", 30)
    with pytest.raises(sel.SelectionError, match="missing"):
        sel.write_selection("trec", "few_shot", mode="real")
