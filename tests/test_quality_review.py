"""Issue #11: reproducible model-assisted review tooling. Fake fixtures only; no quality finding here."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from common import CFG, read_jsonl, real_path, synth_path
from ws1_generation import generate
from ws1_generation import select_scope30 as sel
from ws3_evaluation import quality_review as qr

ROOT = Path(__file__).resolve().parents[1]
REVIEWER = {"kind": "model-assisted", "provider": "test", "model": "none", "exact_version": "n/a",
            "human_review": False}


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    """Build fake verified inputs once: selected zero/few-shot (32/label) and complete AugGPT."""
    base = tmp_path_factory.mktemp("qr")
    shutil.copytree(ROOT / "data" / "sample" / "real", base / "data" / "real")
    mp = pytest.MonkeyPatch()
    mp.setenv("SYNTHTEXT_DATA", str(base / "data"))
    mp.setitem(CFG, "lowres_per_label", 2)  # committed fixtures hold two sources per label
    for task in ("trec", "emotion"):
        for setting in ("zero_shot", "few_shot"):
            generate.main(["--setting", setting, "--task", task, "--seed", "13", "--target", "32", "--dry-run"])
            sel.write_selection(task, setting, mode="fake")
        generate.generate_auggpt(task, fake=True, seed=13)
    mp.undo()
    return base


@pytest.fixture
def env(corpora, tmp_path, monkeypatch):
    shutil.copytree(corpora / "data", tmp_path / "data")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setitem(CFG, "lowres_per_label", 2)
    return tmp_path / "evidence"


def judge(sample, sample_sha, wrong=(), ambiguous=(), changed=(), copied=()):
    items = []
    for key, batch in sample["batches"].items():
        setting = key.split("/")[1]
        for item in batch["items"]:
            rid = item["row_id"]
            j = {"batch": key, "row_id": rid,
                 "label_fit": "wrong" if rid in wrong else ("ambiguous" if rid in ambiguous else "fits"),
                 "naturalness": "natural", "style_fit": "matches", "template_repetition": "none",
                 "copied_demonstration": ("near_copy" if rid in copied else "no") if setting == "few_shot"
                 else "not_applicable",
                 "source_meaning": ("changed" if rid in changed else "preserved") if setting == "auggpt"
                 else "not_applicable",
                 "explanation": "fixture judgment"}
            j["clear_error"], j["ambiguous"] = qr._derive(j, setting)
            items.append(j)
    return {"round": sample["round"], "sample_sha256": sample_sha, "reviewer": dict(REVIEWER), "items": items}


def write_judgments(evidence, round_no, payload):
    path = qr.judgments_path(round_no, evidence)
    path.write_text(json.dumps(payload), encoding="utf-8")


def ids(sample, key):
    return sample["batches"][key]["selected_ids_in_order"]


def test_sample_follows_hash_rule_uses_verified_inputs_and_holds_no_text(env):
    sample, sha = qr.write_sample(1, "fake", env)
    assert len(sample["batches"]) == 6 and all(len(b["items"]) == 10 for b in sample["batches"].values())
    for key, batch in sample["batches"].items():
        task, setting = key.split("/")
        population = [r["id"] for r in read_jsonl(qr.batch_input_path(task, setting, "fake"))]
        expected = sorted(population, key=lambda rid: (hashlib.sha256(
            f"issue11-quality-v1|round1|13|{task}|{setting}|{rid}".encode()).hexdigest(), rid))[:10]
        assert batch["selected_ids_in_order"] == expected
        assert batch["population_rows"] == (180 if setting != "auggpt" else 72)
    # zero/few-shot sampled from the 180-row selected file, not the 192-row original
    zs = sample["inputs"]["batches"]["trec/zero_shot"]
    assert zs["input"]["path"].endswith("zero_shot_seed13.scope30-v1.selected.jsonl") and zs["original_rows"] == 192
    assert sample["real_run_evidence"] is False
    raw = qr.sample_path(1, env).read_text(encoding="utf-8")
    for row in read_jsonl(qr.batch_input_path("trec", "zero_shot", "fake"))[:50]:
        assert row["text"] not in raw                                   # committed evidence carries no text
    packet = json.loads((qr.packet_dir() / "round1_packet.json").read_text(encoding="utf-8"))
    few = packet["batches"]["emotion/few_shot"][0]
    assert len(few["demonstrations"]) == 3 and all(d["text"] for d in few["demonstrations"])
    aug = packet["batches"]["trec/auggpt"][0]
    assert aug["source"]["text"] == generate.load_auggpt_sources("trec", 13)[aug["source"]["id"]]["text"]
    # deterministic: a second derivation is byte-identical, and the file is not rewritten
    again, sha2 = qr.write_sample(1, "fake", env)
    assert again == sample and sha2 == sha


def test_later_round_is_fresh_and_never_overwrites_earlier_rounds(env):
    with pytest.raises(qr.ReviewError, match="round 1 sample is missing"):
        qr.write_sample(2, "fake", env)
    first, _ = qr.write_sample(1, "fake", env)
    before = qr.sample_path(1, env).read_bytes()
    second, _ = qr.write_sample(2, "fake", env)
    for key in first["batches"]:
        assert not set(ids(first, key)) & set(ids(second, key))
        assert second["excluded_prior_round_ids"][key] == sorted(ids(first, key))
    assert qr.sample_path(1, env).read_bytes() == before
    qr.sample_path(1, env).write_text("{}", encoding="utf-8")
    with pytest.raises(qr.ReviewError, match="different content"):
        qr.write_sample(1, "fake", env)


@pytest.mark.parametrize("target", ["selected", "auggpt", "manifest"])
def test_changed_inputs_block_sampling_and_completion(env, target):
    sample, sha = qr.write_sample(1, "fake", env)
    write_judgments(env, 1, judge(sample, sha))
    path = {"selected": sel.selected_path("trec", "few_shot", "fake"),
            "auggpt": synth_path("emotion", "auggpt", seed=13, mode="fake"),
            "manifest": sel.manifest_path("emotion", "zero_shot", "fake")}[target]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"text": "', b'"text": "X', 1) if target != "manifest" else raw + b" ")
    # Producer verifiers catch the selected-file edit; recorded hashes also catch edits they tolerate
    # (a still-valid paraphrase text, a whitespace-only manifest change).
    with pytest.raises(qr.ReviewError, match="input verification failed|inputs changed since sampling"):
        qr.check_round(1, "fake", env)
    assert not qr.outcome_path(1, env).exists()
    if target == "selected":
        with pytest.raises(qr.ReviewError, match="input verification failed"):
            qr.write_sample(2, "fake", env)


def test_missing_input_blocks_review(env):
    sel.selected_path("trec", "zero_shot", "fake").unlink()
    with pytest.raises(qr.ReviewError, match="trec/zero_shot"):
        qr.write_sample(1, "fake", env)


def test_two_clear_errors_flag_a_batch_but_ambiguity_and_copies_do_not(env):
    sample, sha = qr.write_sample(1, "fake", env)
    tz, ef, ta = ids(sample, "trec/zero_shot"), ids(sample, "emotion/few_shot"), ids(sample, "trec/auggpt")
    payload = judge(sample, sha, wrong={tz[0], tz[5], ef[1]}, ambiguous=set(ef[2:6]), copied=set(ef[6:9]),
                    changed={ta[3], ta[4]})
    write_judgments(env, 1, payload)
    outcome = qr.check_round(1, "fake", env)
    b = outcome["batches"]
    assert b["trec/zero_shot"]["outcome"] == "FLAGGED" and b["trec/zero_shot"]["clear_error_ids"] == [tz[0], tz[5]]
    assert b["trec/auggpt"]["outcome"] == "FLAGGED"                     # changed meaning counts
    assert b["emotion/few_shot"]["outcome"] == "PASS"                    # one error, ambiguity/copies separate
    assert b["emotion/few_shot"]["ambiguous"] == 4 and len(b["emotion/few_shot"]["copied_demonstration_ids"]) == 3
    assert sorted(outcome["flagged_batches"]) == ["trec/auggpt", "trec/zero_shot"]
    assert outcome["reviewer"]["human_review"] is False


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["items"].pop(), "does not equal the sample"),
    (lambda p: p["items"].reverse(), "does not equal the sample"),
    (lambda p: p["items"][0].update(clear_error=True), "clear_error/ambiguous"),
    (lambda p: p["items"][0].update(copied_demonstration="no"), "few_shot only"),
    (lambda p: next(j for j in p["items"] if "auggpt" in j["batch"]).update(source_meaning="not_applicable",
                                                                            clear_error=False), "auggpt only"),
    (lambda p: p["items"][3].update(explanation=" "), "explanation is required"),
    (lambda p: p["items"][3].update(label_fit="maybe"), "label_fit"),
    (lambda p: p["reviewer"].update(human_review=True), "human_review false"),
    (lambda p: p.update(sample_sha256="0" * 64), "not bound"),
])
def test_incomplete_or_inconsistent_judgments_are_rejected(env, mutate, message):
    sample, sha = qr.write_sample(1, "fake", env)
    payload = judge(sample, sha)
    mutate(payload)
    write_judgments(env, 1, payload)
    with pytest.raises(qr.ReviewError, match=message):
        qr.check_round(1, "fake", env)


def test_cli_exit_codes_and_outcome_preservation(env, capsys):
    def run(*argv):
        with pytest.raises(SystemExit) as exc:
            qr.main([*argv, "--dry-run", "--evidence-dir", str(env)])
        return exc.value.code
    assert run("check") == 1                                            # no sample yet
    qr.main(["sample", "--dry-run", "--evidence-dir", str(env)])
    assert run("check") == 1                                            # no judgments: incomplete
    sample = json.loads(qr.sample_path(1, env).read_text(encoding="utf-8"))
    sha = hashlib.sha256(qr.sample_path(1, env).read_bytes()).hexdigest()
    write_judgments(env, 1, judge(sample, sha))
    assert run("check") == 0
    assert "FAKE fixture" in capsys.readouterr().out
    assert json.loads(qr.outcome_path(1, env).read_text())["flagged_batches"] == []
    write_judgments(env, 1, judge(sample, sha, wrong=set(ids(sample, "emotion/zero_shot")[:2])))
    assert run("check") == 1                                            # earlier outcome is never overwritten


def test_targeted_check_is_separate_from_the_random_sample(env):
    sample, sha = qr.write_targeted("fake", env, max_words=5)
    sources = generate.load_auggpt_sources("emotion", 13)
    long_ids = {sid for sid, s in sources.items() if len(s["text"].split()) > 5}
    assert long_ids and set(sample["long_sources"]) == long_ids
    assert sample["counts_toward_required_random_60"] is False
    assert len(sample["items"]) == 6 * len(long_ids)
    items = []
    for item in sample["items"]:
        j = {"batch": "emotion/auggpt", "row_id": item["row_id"], "label_fit": "fits", "naturalness": "natural",
             "style_fit": "matches", "template_repetition": "none", "copied_demonstration": "not_applicable",
             "source_meaning": "partially_changed", "explanation": "dropped a detail"}
        j["clear_error"], j["ambiguous"] = qr._derive(j, "auggpt")
        items.append(j)
    qr.targeted_paths(env)[1].write_text(json.dumps({"sample_sha256": sha, "reviewer": REVIEWER, "items": items}))
    result = qr.check_targeted("fake", env)
    assert result["reviewed"] == len(items) and result["clear_errors"] == 0 and result["ambiguous"] == len(items)
    assert not qr.sample_path(1, env).exists()                          # no effect on round files


def test_crlf_checkout_of_committed_evidence_still_binds(env):
    sample, sha = qr.write_sample(1, "fake", env)
    write_judgments(env, 1, judge(sample, sha))
    expected = qr.check_round(1, "fake", env)
    for path in (qr.sample_path(1, env), qr.judgments_path(1, env)):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))   # e.g. Windows core.autocrlf
    assert qr.check_round(1, "fake", env) == expected
    qr.write_sample(1, "fake", env)                                        # CRLF copy is not "different content"
