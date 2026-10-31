"""Issue #12 freeze: deterministic manifest and fail-closed verification. Fake fixtures only."""

import json
import shutil
from pathlib import Path

import pytest

from common import CFG, read_jsonl, synth_path, write_jsonl
from ws1_generation import freeze_scope30 as fz
from ws1_generation import generate, promote_auggpt
from ws1_generation import select_scope30 as sel
from ws3_evaluation import quality_review as qr

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = {"kind": "model-assisted", "provider": "test", "model": "none", "exact_version": "n/a",
            "human_review": False}


def judge(sample, sha, wrong=()):
    items = []
    for key, batch in sample["batches"].items():
        setting = key.split("/")[1]
        for item in batch["items"]:
            j = {"batch": key, "row_id": item["row_id"],
                 "label_fit": "wrong" if item["row_id"] in wrong else "fits",
                 "naturalness": "natural", "style_fit": "matches", "template_repetition": "none",
                 "copied_demonstration": "no" if setting == "few_shot" else "not_applicable",
                 "source_meaning": "preserved" if setting == "auggpt" else "not_applicable",
                 "explanation": "fixture"}
            j["clear_error"], j["ambiguous"] = qr._derive(j, setting)
            items.append(j)
    return {"round": sample["round"], "sample_sha256": sha, "reviewer": dict(REVIEWER), "items": items}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Fake verified handoff: selected zero/few-shot, promoted TREC AugGPT, Emotion AugGPT, round-1 review."""
    base = tmp_path_factory.mktemp("freeze")
    shutil.copytree(ROOT / "data" / "sample" / "real", base / "data" / "real")
    mp = pytest.MonkeyPatch()
    mp.setenv("SYNTHTEXT_DATA", str(base / "data"))
    mp.setitem(CFG, "lowres_per_label", 2)
    for task in ("trec", "emotion"):
        for setting in ("zero_shot", "few_shot"):
            generate.main(["--setting", setting, "--task", task, "--seed", "13", "--target", "32", "--dry-run"])
            sel.write_selection(task, setting, mode="fake")
        if generate.auggpt_correction(task, 13) is None:
            generate.generate_auggpt(task, fake=True, seed=13)
        else:
            old = pytest.MonkeyPatch()
            old.setattr(generate, "AUGGPT_CORRECTIONS", {})
            generate.generate_auggpt(task, fake=True, seed=13)
            old.undo()
            generate.generate_auggpt(task, fake=True, seed=13, candidate=True)
            loc = promote_auggpt.locations(task, 13, "fake")
            promote_auggpt.promote(task, 13, promote_auggpt._sha(loc["candidate"]),
                                   promote_auggpt._sha(loc["active"]), mode="fake")
    evidence = base / "evidence"
    sample, sha = qr.write_sample(1, "fake", evidence)
    qr.judgments_path(1, evidence).write_text(json.dumps(judge(sample, sha)), encoding="utf-8")
    qr._write_new_or_same(qr.outcome_path(1, evidence), qr._canonical_bytes(qr.check_round(1, "fake", evidence)))
    mp.undo()
    return base


REV = "a" * 40


@pytest.fixture
def env(built, tmp_path, monkeypatch):
    shutil.copytree(built / "data", tmp_path / "data")
    shutil.copytree(built / "evidence", tmp_path / "evidence")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setitem(CFG, "lowres_per_label", 2)
    monkeypatch.setattr(fz, "revision_in_history", lambda rev: None)   # git ancestry tested separately
    return {"evidence": tmp_path / "evidence", "manifest": tmp_path / "freeze.json", "data": tmp_path / "data"}


def write(env):
    return fz.write_freeze("fake", REV, env["manifest"], env["evidence"])


def verify(env):
    return fz.verify_freeze("fake", env["manifest"], env["evidence"])


def test_freeze_binds_all_six_artifacts_deterministically_without_text(env):
    first = fz._canonical(fz.build_freeze("fake", REV, env["evidence"]))
    assert fz._canonical(fz.build_freeze("fake", REV, env["evidence"])) == first   # deterministic rerun
    manifest, digest, written = write(env)
    assert written and env["manifest"].read_bytes() == first
    assert write(env)[2] is False                                                    # identical rewrite is a no-op
    assert sorted(manifest["artifacts"]) == sorted(f"{t}/{s}" for t in ("trec", "emotion")
                                                   for s in ("zero_shot", "few_shot", "auggpt"))
    for key, art in manifest["artifacts"].items():
        task, setting = key.split("/")
        path = (synth_path(task, "auggpt", seed=13, mode="fake") if setting == "auggpt"
                else sel.selected_path(task, setting, "fake"))
        rows = read_jsonl(path)
        assert art["consumed"]["ids_in_file_order"] == [r["id"] for r in rows]
        assert art["consumed"]["sha256"] == fz._sha(path.read_bytes())
        assert art["generation_fingerprint"] and art["provenance"]["sha256"] and art["settings"]["prompt_sha256"]
        assert manifest["issue11_evidence"]["latest_round_by_batch"][key]["outcome"] == "PASS"
    trec = manifest["artifacts"]["trec/auggpt"]
    assert trec["corrected_output"] is True and set(trec["promotion"]) >= {
        "record", "preserved_original", "retained_candidate"}
    assert manifest["artifacts"]["emotion/auggpt"]["paraphrases_per_source"] == CFG["auggpt_k"]
    assert manifest["decision"]["option"] == "A" and manifest["repository_revision"] == REV
    assert any("1/10 clear errors and 3" in x for x in manifest["limitations"])
    assert "harmless" not in json.dumps(manifest["decision"])
    assert '"text"' not in env["manifest"].read_text(encoding="utf-8")
    assert verify(env)["freeze_sha256"] == digest


def _reorder(path):
    rows = read_jsonl(path)
    write_jsonl(path, rows[::-1])


def _compact_sidecar(path):
    side = Path(path).with_suffix(".provenance.json")
    side.write_text(json.dumps(json.loads(side.read_text(encoding="utf-8"))), encoding="utf-8")


def _edit_manifest(env, change):
    m = json.loads(env["manifest"].read_text(encoding="utf-8"))
    change(m)
    env["manifest"].write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


TAMPERS = {
    "reordered emotion AugGPT rows": lambda env: _reorder(synth_path("emotion", "auggpt", seed=13, mode="fake")),
    "reordered selected corpus": lambda env: _reorder(sel.selected_path("trec", "few_shot", "fake")),
    "missing selection manifest": lambda env: sel.manifest_path("emotion", "zero_shot", "fake").unlink(),
    "missing AugGPT artifact": lambda env: synth_path("emotion", "auggpt", seed=13, mode="fake").unlink(),
    "reformatted provenance sidecar": lambda env: _compact_sidecar(synth_path("emotion", "auggpt", seed=13, mode="fake")),
    "tampered promotion archive": lambda env: promote_auggpt.locations("trec", 13, "fake")["archive"].write_bytes(
        promote_auggpt.locations("trec", 13, "fake")["archive"].read_bytes() + b"\n"),
    "changed real reference": lambda env: (env["data"] / "real" / "trec" / "pool.jsonl").write_bytes(
        (env["data"] / "real" / "trec" / "pool.jsonl").read_bytes() + b"\n"),
    "tampered review judgments": lambda env: qr.judgments_path(1, env["evidence"]).write_text(
        qr.judgments_path(1, env["evidence"]).read_text(encoding="utf-8").replace("fixture", "edited")),
    "extra evidence file": lambda env: (env["evidence"] / "extra.json").write_text("{}", encoding="utf-8"),
    "removed evidence file": lambda env: qr.judgments_path(1, env["evidence"]).unlink(),
    "edited manifest hash": lambda env: _edit_manifest(env, lambda m: m["artifacts"]["trec/zero_shot"]["consumed"].update(
        sha256="0" * 64)),
    "edited manifest limitations": lambda env: _edit_manifest(env, lambda m: m["limitations"].pop()),
    "edited manifest decision": lambda env: _edit_manifest(env, lambda m: m["decision"].update(option="B")),
    "reordered manifest ids": lambda env: _edit_manifest(env, lambda m: m["artifacts"]["emotion/few_shot"]["consumed"][
        "ids_in_file_order"].reverse()),
    "manifest not JSON": lambda env: env["manifest"].write_text("{", encoding="utf-8"),
    "manifest missing": lambda env: env["manifest"].unlink(),
}


@pytest.mark.parametrize("tamper", list(TAMPERS))
def test_verify_fails_closed_on_tampering(env, tamper):
    write(env)
    verify(env)
    TAMPERS[tamper](env)
    with pytest.raises(ValueError):
        verify(env)


def test_rows_reordered_after_review_cannot_be_frozen(env):
    """The review evidence binds the reviewed file's hash; a later reorder no longer matches it."""
    _reorder(synth_path("emotion", "auggpt", seed=13, mode="fake"))
    with pytest.raises(fz.FreezeError, match="emotion/auggpt: latest review round 1 reviewed"):
        fz.build_freeze("fake", REV, env["evidence"])


def test_flagged_latest_round_blocks_the_freeze(env):
    sample = json.loads(qr.sample_path(1, env["evidence"]).read_text(encoding="utf-8"))
    sha = qr._evidence_sha256(qr.sample_path(1, env["evidence"]).read_bytes())
    wrong = sample["batches"]["trec/zero_shot"]["selected_ids_in_order"][:2]
    qr.judgments_path(1, env["evidence"]).write_text(json.dumps(judge(sample, sha, wrong)), encoding="utf-8")
    qr.outcome_path(1, env["evidence"]).write_bytes(qr._canonical_bytes(qr.check_round(1, "fake", env["evidence"])))
    with pytest.raises(fz.FreezeError, match="trec/zero_shot: latest review round 1 is FLAGGED"):
        fz.build_freeze("fake", REV, env["evidence"])


def test_unbound_outcome_and_missing_required_evidence_block_the_freeze(env):
    with pytest.raises(fz.FreezeError, match="issue #11 evidence missing"):
        fz.build_freeze("fake", REV, env["evidence"], required_evidence=fz.REQUIRED_REAL_EVIDENCE)
    out = json.loads(qr.outcome_path(1, env["evidence"]).read_text(encoding="utf-8"))
    out["judgments_sha256"] = "0" * 64
    qr.outcome_path(1, env["evidence"]).write_text(json.dumps(out), encoding="utf-8")
    with pytest.raises(fz.FreezeError, match="not bound to its sample/judgments"):
        fz.build_freeze("fake", REV, env["evidence"])


def test_corpus_text_can_never_enter_the_manifest(env, monkeypatch):
    leaked = read_jsonl(sel.selected_path("trec", "zero_shot", "fake"))[0]["text"]
    monkeypatch.setattr(fz, "LIMITATIONS", fz.LIMITATIONS + [leaked])
    with pytest.raises(fz.FreezeError, match="would contain corpus text"):
        fz.build_freeze("fake", REV, env["evidence"])


def test_an_existing_freeze_is_never_overwritten(env):
    write(env)
    _edit_manifest(env, lambda m: m.update(note="older freeze"))
    before = env["manifest"].read_bytes()
    with pytest.raises(fz.FreezeError, match="never overwritten"):
        write(env)
    assert env["manifest"].read_bytes() == before


def test_revision_must_be_in_history_and_well_formed(env, monkeypatch):
    with pytest.raises(fz.FreezeError, match="40-hex"):
        fz.build_freeze("fake", "abc", env["evidence"])
    monkeypatch.undo()
    with pytest.raises(fz.FreezeError, match="not in this checkout's history"):
        fz.revision_in_history("0" * 40)


def test_crlf_checkout_of_the_manifest_still_verifies(env):
    write(env)
    env["manifest"].write_bytes(env["manifest"].read_bytes().replace(b"\n", b"\r\n"))
    verify(env)


def test_cli_write_on_an_existing_freeze_only_reverifies(env, capsys):
    write(env)
    before = env["manifest"].read_bytes()
    args = ["--write", "--dry-run", "--manifest", str(env["manifest"]), "--evidence-dir", str(env["evidence"])]
    fz.main(args)
    assert "FREEZE ALREADY PRESENT" in capsys.readouterr().out and env["manifest"].read_bytes() == before
    _reorder(synth_path("emotion", "auggpt", seed=13, mode="fake"))
    with pytest.raises(SystemExit) as exc:
        fz.main(args)
    assert exc.value.code == 1 and env["manifest"].read_bytes() == before


def test_cli_verify_exit_codes(env, capsys):
    write(env)
    args = ["--verify", "--dry-run", "--manifest", str(env["manifest"]), "--evidence-dir", str(env["evidence"])]
    fz.main(args)
    assert "VERIFY PASS (FAKE fixture" in capsys.readouterr().out
    _reorder(sel.selected_path("emotion", "zero_shot", "fake"))
    with pytest.raises(SystemExit) as exc:
        fz.main(args)
    out = capsys.readouterr()
    assert exc.value.code == 1 and out.out.startswith("FAIL") and "Traceback" not in out.out + out.err
