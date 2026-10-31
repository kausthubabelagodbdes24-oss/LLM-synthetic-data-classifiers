"""Issue #13: freeze-bound twelve-key training grid. Fake fixtures and the hashing embedder only.

The fixture builds a complete fake handoff at the exact scope30-v1 counts (40 real train / 15 pool / 10 lowres /
5 test rows per label, 30 selected zero/few-shot rows per label, six AugGPT paraphrases per source), reviews it
and freezes it with the real ``freeze_scope30`` tooling. Every score here is a fixture, never evidence.
"""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from common import CFG, labels, norm, read_jsonl, synth_path, write_jsonl
from ws1_generation import freeze_scope30 as fz
from ws1_generation import generate, promote_auggpt
from ws1_generation import select_scope30 as sel
from ws2_training import train as legacy
from ws2_training import train_scope30 as grid
from ws3_evaluation import quality_review as qr

REV = "a" * 40
REPO = {"revision": "b" * 40, "clean": True}


def _judge(sample, sha):
    items = []
    for key, batch in sample["batches"].items():
        setting = key.split("/")[1]
        for item in batch["items"]:
            j = {"batch": key, "row_id": item["row_id"], "label_fit": "fits", "naturalness": "natural",
                 "style_fit": "matches", "template_repetition": "none",
                 "copied_demonstration": "no" if setting == "few_shot" else "not_applicable",
                 "source_meaning": "preserved" if setting == "auggpt" else "not_applicable",
                 "explanation": "fixture"}
            j["clear_error"], j["ambiguous"] = qr._derive(j, setting)
            items.append(j)
    return {"round": sample["round"], "sample_sha256": sha, "items": items,
            "reviewer": {"kind": "model-assisted", "provider": "test", "model": "none", "exact_version": "n/a",
                         "human_review": False}}


def _write_real(data):
    for task in ("trec", "emotion"):
        train, pool, test, low = [], [], [], []
        n = 0
        for lab in labels(task):
            for i in range(55):
                n += 1
                row = {"id": f"{task}.train.{n:05d}", "text": f"{task} fixture {lab} real number {n} item {i}",
                       "label": lab}
                (pool if i < 15 else train).append(row)
                if i < 10:
                    low.append(row)
            for i in range(5):
                test.append({"id": f"{task}.test.{len(test):05d}", "text": f"{task} heldout {lab} case {i}",
                             "label": lab})
        for name, rows in (("train", train), ("pool", pool), ("test", test), ("lowres_seed13", low)):
            write_jsonl(data / "real" / task / f"{name}.jsonl", rows)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("grid")
    mp = pytest.MonkeyPatch()
    mp.setenv("SYNTHTEXT_DATA", str(base / "data"))
    mp.delenv("SYNTHTEXT_GENERATION_MODE", raising=False)
    _write_real(base / "data")
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
    qr.judgments_path(1, evidence).write_text(json.dumps(_judge(sample, sha)), encoding="utf-8")
    qr._write_new_or_same(qr.outcome_path(1, evidence), qr._canonical_bytes(qr.check_round(1, "fake", evidence)))
    fz.write_freeze("fake", REV, base / "freeze.json", evidence)
    mp.undo()
    return base


class Env:
    def __init__(self, root):
        self.root = root
        self.data = root / "data"
        self.evidence = root / "evidence"
        self.freeze = root / "freeze.json"
        self.results = root / "results" / "grid"
        self.log = []

    def kw(self):
        return {"mode": "fake", "manifest_path": self.freeze, "evidence_dir": self.evidence}

    def run(self, **kw):
        return grid.run_grid(self.results, log=self.log.append, **self.kw(), **kw)

    def validate(self, mode="fake"):
        return grid.validate_results(self.results, mode, self.freeze, self.evidence)

    def record(self, task, cond):
        return self.results / grid.RECORDS / grid.record_filename(task, cond)


@pytest.fixture
def env(built, tmp_path, monkeypatch):
    for name in ("data", "evidence"):
        shutil.copytree(built / name, tmp_path / name)
    shutil.copy(built / "freeze.json", tmp_path / "freeze.json")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("SYNTHTEXT_GENERATION_MODE", raising=False)
    monkeypatch.delenv("SYNTHTEXT_EMBEDDER", raising=False)
    monkeypatch.setattr(fz, "revision_in_history", lambda rev: None)       # git ancestry is tested in #12
    monkeypatch.setattr(grid, "repository_state", lambda mode: dict(REPO))
    return Env(tmp_path)


@pytest.fixture(scope="module")
def completed(built, tmp_path_factory):
    """One complete fake grid, copied by tests that need finished results."""
    root = tmp_path_factory.mktemp("done")
    for name in ("data", "evidence"):
        shutil.copytree(built / name, root / name)
    shutil.copy(built / "freeze.json", root / "freeze.json")
    mp = pytest.MonkeyPatch()
    mp.setenv("SYNTHTEXT_DATA", str(root / "data"))
    mp.setattr(fz, "revision_in_history", lambda rev: None)
    mp.setattr(grid, "repository_state", lambda mode: dict(REPO))
    Env(root).run()
    mp.undo()
    return root / "results" / "grid"


@pytest.fixture
def done(env, completed):
    shutil.copytree(completed, env.results)
    return env


def _no_fit(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a classifier was fitted")
    monkeypatch.setattr(grid, "fit_and_score", boom)


def _rehash(path, change):
    rec = json.loads(path.read_text(encoding="utf-8"))
    change(rec)
    rec.pop("record_sha256")
    rec["record_sha256"] = grid._sha(grid._canonical(rec))
    path.write_bytes(grid._canonical(rec))


# ---------------------------------------------------------------- inputs

def test_exact_counts_labels_and_composition(env):
    ctx = grid.prepare(**env.kw(), need_model=False)
    assert sorted(ctx["sets"]) == sorted((t, c) for t, c, _ in grid.all_keys()) and len(ctx["sets"]) == 12
    for (task, cond), (rows, _) in ctx["sets"].items():
        counts = Counter(r["label"] for r in rows)
        assert len(rows) == grid.EXPECTED_COUNTS[cond] == {"C5a": 60, "C5b": 420}.get(cond, 180)
        assert set(counts) == set(labels(task)) and set(counts.values()) == {grid.EXPECTED_PER_LABEL[cond]}
        parts = Counter(r["_part"] for r in rows)
        expected = {"C1": {"real_train": 180}, "C2": {"zero_shot": 180}, "C3": {"few_shot": 180},
                    "C4": {"real_train": 90, "few_shot": 90}, "C5a": {"lowres_source": 60},
                    "C5b": {"lowres_source": 60, "auggpt": 360}}[cond]
        assert dict(parts) == expected
        if cond == "C4":
            for part in ("real_train", "few_shot"):
                assert set(Counter(r["label"] for r in rows if r["_part"] == part).values()) == {15}
        inp = ctx["inputs"][task]
        assert not {r["id"] for r in rows} & inp["test_ids"]
        assert not {norm(r["text"]) for r in rows} & set(inp["test_texts"])


def test_c1_matches_the_baseline_sampler(env):
    from common import per_label_sample
    ctx = grid.prepare(**env.kw(), need_model=False)
    for task in ("trec", "emotion"):
        rows = ctx["sets"][(task, "C1")][0]
        want = per_label_sample(ctx["inputs"][task]["train"], 30, 13, required_labels=labels(task), exact=True)
        assert [r["id"] for r in rows] == [r["id"] for r in want]


def test_c4_selection_is_deterministic_and_within_the_c3_corpus(env):
    ctx = grid.prepare(**env.kw(), need_model=False)
    for task in ("trec", "emotion"):
        few = ctx["inputs"][task]["few_shot"]
        picked = grid.select_c4_synthetic(task, few)
        assert [r["id"] for r in grid.select_c4_synthetic(task, list(reversed(few)))] == [r["id"] for r in picked]
        assert {r["id"] for r in picked} <= {r["id"] for r in ctx["sets"][(task, "C3")][0]}
        for lab in labels(task):
            group = sorted((r for r in few if r["label"] == lab),
                           key=lambda r: (sel.selection_key(task, "few_shot", r["id"]), r["id"]))
            assert [r["id"] for r in picked if r["label"] == lab] == [r["id"] for r in group[:15]]
        c4_synth = [r["id"] for r in ctx["sets"][(task, "C4")][0] if r["_part"] == "few_shot"]
        assert c4_synth == [r["id"] for r in picked]


def test_c5b_is_exact_c5a_plus_six_distinct_paraphrases_each(env):
    ctx = grid.prepare(**env.kw(), need_model=False)
    for task in ("trec", "emotion"):
        c5a = ctx["sets"][(task, "C5a")][0]
        c5b = ctx["sets"][(task, "C5b")][0]
        assert [r["id"] for r in c5b[:60]] == [r["id"] for r in c5a]
        low = {r["id"]: r for r in read_jsonl(env.data / "real" / task / "lowres_seed13.jsonl")}
        assert [r["id"] for r in c5a] == list(low)
        groups = {}
        for p in c5b[60:]:
            groups.setdefault(p["source_ids"][0], []).append(p)
        assert set(groups) == set(low)
        for sid, group in groups.items():
            texts = {norm(p["text"]) for p in group}
            assert len(group) == 6 and len(texts) == 6 and norm(low[sid]["text"]) not in texts
            assert {p["label"] for p in group} == {low[sid]["label"]}


@pytest.mark.parametrize("damage", ["drop_one", "five_paraphrases", "copy_source", "foreign_source", "relabel"])
def test_incomplete_or_wrong_c5b_is_rejected(env, damage):
    ctx = grid.prepare(**env.kw(), need_model=False)
    inp = ctx["inputs"]["emotion"]
    rows = [dict(r) for r in ctx["sets"][("emotion", "C5b")][0]]
    para = rows[60:]
    if damage == "drop_one":
        rows.pop()
    elif damage == "five_paraphrases":
        para[0]["source_ids"] = [para[6]["source_ids"][0]]
    elif damage == "copy_source":
        para[0]["text"] = rows[[r["id"] for r in rows].index(para[0]["source_ids"][0])]["text"]
    elif damage == "foreign_source":
        para[0]["source_ids"] = ["emotion.train.99999"]
    else:
        para[0]["label"] = next(lab for lab in labels("emotion") if lab != para[0]["label"])
    with pytest.raises(grid.GridError):
        grid.validate_training_set("emotion", "C5b", rows, inp)


@pytest.mark.parametrize("cond,damage", [
    ("C1", "drop"), ("C1", "duplicate_id"), ("C1", "missing_label"), ("C1", "test_text"), ("C1", "test_id"),
    ("C1", "pool_row"), ("C2", "test_text"), ("C3", "reorder"), ("C4", "unselected_few_shot"), ("C5a", "extra"),
])
def test_incomplete_and_contaminated_sets_are_rejected(env, cond, damage):
    ctx = grid.prepare(**env.kw(), need_model=False)
    inp = ctx["inputs"]["trec"]
    rows = [dict(r) for r in ctx["sets"][("trec", cond)][0]]
    if damage == "drop":
        rows.pop()
    elif damage == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif damage == "missing_label":
        lab = rows[0]["label"]
        other = next(x for x in labels("trec") if x != lab)
        for r in rows:
            if r["label"] == lab:
                r["label"] = other
    elif damage == "test_text":
        rows[0]["text"] = inp["test"][0]["text"].upper()
    elif damage == "test_id":
        rows[0]["id"] = inp["test"][0]["id"]
    elif damage == "pool_row":
        rows[0] = dict(next(p for p in inp["pool"] if p["label"] == rows[0]["label"]), _part="real_train")
    elif damage == "reorder":
        rows.reverse()
    elif damage == "unselected_few_shot":
        i = next(i for i, r in enumerate(rows) if r["_part"] == "few_shot")
        chosen = {r["id"] for r in rows}
        rows[i] = dict(next(r for r in inp["few_shot"] if r["id"] not in chosen and r["label"] == rows[i]["label"]),
                       _part="few_shot")
    elif damage == "extra":
        rows.append(dict(inp["auggpt"][0], _part="auggpt"))
    with pytest.raises(grid.GridError):
        grid.validate_training_set("trec", cond, rows, inp)


def test_only_the_pinned_official_duplicate_is_disclosed_not_rejected(env, monkeypatch):
    ctx = grid.prepare(**env.kw(), need_model=False)
    inp = ctx["inputs"]["trec"]
    rows = [dict(r) for r in ctx["sets"][("trec", "C1")][0]]
    victim, test_row = rows[0], inp["test"][0]
    victim["text"] = test_row["text"]
    inp = dict(inp, test_texts={norm(r["text"]): r["id"] for r in inp["test"]})
    pin = {"task": "trec", "train_id": victim["id"], "test_id": test_row["id"],
           "normalized_text_sha256": grid._sha(norm(test_row["text"]).encode("utf-8")),
           "conditions": ["C1", "C4"], "decision": "fixture"}
    with pytest.raises(grid.GridError, match="test text overlap"):      # unpinned: fails closed
        grid.validate_training_set("trec", "C1", rows, inp)
    monkeypatch.setattr(grid, "PINNED_OFFICIAL_DUPLICATES", (pin,))
    grid.validate_training_set("trec", "C1", rows, inp)                 # the exact pinned pair is allowed
    assert grid.disclosed_overlap("trec", "C1", rows, inp)[0]["test_id"] == test_row["id"]
    for other in (dict(pin, normalized_text_sha256="0" * 64), dict(pin, test_id="trec.test.99999"),
                  dict(pin, conditions=["C4"]), dict(pin, task="emotion")):
        monkeypatch.setattr(grid, "PINNED_OFFICIAL_DUPLICATES", (other,))
        with pytest.raises(grid.GridError, match="test text overlap"):
            grid.validate_training_set("trec", "C1", rows, inp)
    monkeypatch.setattr(grid, "PINNED_OFFICIAL_DUPLICATES", (pin,))
    synthetic = [dict(r) for r in ctx["sets"][("trec", "C2")][0]]
    synthetic[0]["text"] = test_row["text"]
    with pytest.raises(grid.GridError, match="test text overlap"):      # never allowed in synthetic rows
        grid.validate_training_set("trec", "C2", synthetic, inp)


def test_real_pins_reference_the_known_trec_pair():
    (pin,) = grid.PINNED_OFFICIAL_DUPLICATES
    assert (pin["train_id"], pin["test_id"], pin["conditions"]) == ("trec.train.05262", "trec.test.00378", ["C1", "C4"])


# ---------------------------------------------------------------- freeze enforcement

def test_missing_freeze_is_rejected_before_anything(env, monkeypatch):
    _no_fit(monkeypatch)
    env.freeze.unlink()
    with pytest.raises(grid.GridError, match="missing"):
        env.run()
    assert not env.results.exists()


@pytest.mark.parametrize("tamper", ["manifest_edit", "selected_corpus", "auggpt_reordered", "evidence", "lowres"])
def test_tampered_freeze_or_inputs_are_rejected_before_fitting(env, monkeypatch, tamper):
    _no_fit(monkeypatch)
    if tamper == "manifest_edit":
        m = json.loads(env.freeze.read_text(encoding="utf-8"))
        m["artifacts"]["trec/few_shot"]["consumed"]["ids_in_file_order"].reverse()
        env.freeze.write_bytes(fz._canonical(m))
    elif tamper == "selected_corpus":
        p = sel.selected_path("trec", "few_shot", "fake")
        p.write_bytes(p.read_bytes().replace(b'"text": "', b'"text": "x', 1))
    elif tamper == "auggpt_reordered":
        p = synth_path("emotion", "auggpt", seed=13, mode="fake")
        write_jsonl(p, read_jsonl(p)[::-1])
    elif tamper == "evidence":
        (env.evidence / "round1_outcome.json").write_text("{}", encoding="utf-8")
    else:
        p = env.data / "real" / "trec" / "lowres_seed13.jsonl"
        write_jsonl(p, read_jsonl(p)[:-1])
    with pytest.raises(grid.GridError, match="freeze verification failed"):
        env.run()
    assert not env.results.exists()


def test_real_mode_accepts_only_the_accepted_freeze(env, monkeypatch):
    with pytest.raises(grid.GridError, match="accepted freeze"):
        grid.load_freeze("real", env.freeze, None)
    with pytest.raises(grid.GridError, match="accepted freeze"):
        grid.load_freeze("real", None, env.evidence)
    with pytest.raises(SystemExit):
        grid.main(["--freeze-manifest", str(env.freeze), "--check-inputs"])


def test_fake_freeze_is_never_accepted_as_real(env, monkeypatch):
    monkeypatch.setattr(grid, "ACCEPTED_FREEZE_PATH", env.freeze)
    with pytest.raises(grid.GridError, match="freeze verification failed"):
        grid.load_freeze("real", env.freeze, None)
    # Even a freeze that verified would need the accepted canonical hash.
    monkeypatch.setattr(fz, "verify_freeze", lambda *a, **k: {"freeze_sha256": fz._sha(fz._lf(env.freeze.read_bytes()))})
    with pytest.raises(grid.GridError, match="not the accepted"):
        grid.load_freeze("real", env.freeze, None)


def test_real_mode_refuses_fixture_settings(env, monkeypatch):
    monkeypatch.setenv("SYNTHTEXT_EMBEDDER", "hashing")
    with pytest.raises(grid.GridError, match="SYNTHTEXT_EMBEDDER"):
        grid.check_mode("real")
    monkeypatch.delenv("SYNTHTEXT_EMBEDDER")
    monkeypatch.setitem(CFG, "train_per_label", 100)
    with pytest.raises(grid.GridError, match="frozen scope30-v1 settings"):
        grid.check_mode("real")


def test_wrong_seed_is_rejected_everywhere(env, monkeypatch):
    _no_fit(monkeypatch)
    with pytest.raises(grid.GridError, match="seed 42"):
        env.run(seed=42)
    with pytest.raises(SystemExit):
        grid.main(["--seed", "42", "--check-inputs"])
    with pytest.raises(grid.GridError, match="only seed 13"):
        grid.parse_keys("trec/C1/42")
    assert not env.results.exists()


def test_legacy_or_unaccepted_synthetic_paths_are_rejected(env, monkeypatch):
    m = json.loads(env.freeze.read_text(encoding="utf-8"))
    freeze = {"manifest": m, "sha256": "x", "path": "freeze.json"}
    m["artifacts"]["trec/few_shot"]["consumed"]["path"] = m["artifacts"]["trec/few_shot"]["original_artifact"]["path"]
    with pytest.raises(grid.GridError, match="legacy/default synthetic paths are never consumed"):
        grid.load_task_inputs("trec", freeze, "fake")
    m["artifacts"]["trec/few_shot"]["consumed"]["path"] = "../outside.jsonl"
    with pytest.raises(grid.GridError):
        grid.load_task_inputs("trec", freeze, "fake")


def test_legacy_trainer_refuses_real_synthetic_conditions(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTHTEXT_GENERATION_MODE", raising=False)
    monkeypatch.delenv("SYNTHTEXT_EMBEDDER", raising=False)
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    for cond in ("C2", "C3", "C4", "C5b"):
        with pytest.raises(ValueError, match="train_scope30"):
            legacy.training_set("trec", cond, 13)
    results = tmp_path / "results"
    with pytest.raises(ValueError, match="train_scope30"):
        legacy.main(["--conditions", "C1,C3", "--results-dir", str(results)])
    assert not (results / "runs.csv").exists()


# ---------------------------------------------------------------- results

def test_fake_end_to_end_twelve_keys_with_full_provenance(env):
    out = env.run()
    assert out["complete"] and len(out["valid"]) == 12
    result = env.validate()
    assert len(result["records"]) == 12
    rows = list(csv.DictReader(open(env.results / grid.RUNS_CSV, encoding="utf-8")))
    assert [(r["task"], r["condition"], r["seed"]) for r in rows] == [(t, c, "13") for t, c, _ in grid.all_keys()]
    assert b"\r" not in (env.results / grid.RUNS_CSV).read_bytes()
    for key, rec in result["records"].items():
        task, cond, _ = key
        assert rec["real_run_evidence"] is False and rec["generation_mode"] == "fake"
        assert rec["classifier"]["embedding"]["kind"] == "hashing-fixture"
        assert rec["classifier"]["classifier"] == {"type": "sklearn.linear_model.LogisticRegression", "C": 1.0,
                                                   "max_iter": 2000, "random_state": 13}
        assert rec["freeze"]["sha256"] == fz._sha(fz._lf(env.freeze.read_bytes()))
        assert rec["training"]["n_train"] == grid.EXPECTED_COUNTS[cond]
        assert rec["training"]["ids_sha256"] == grid._ids_sha(rec["training"]["ids"])
        assert rec["test"]["sha256"] and rec["inputs"] and rec["repository"] == REPO
        assert rec["environment"]["python"] and 0 <= rec["metrics"]["macro_f1"] <= 1
        assert rec["runtime_seconds"] >= 0
    complete = json.loads((env.results / grid.COMPLETE).read_text(encoding="utf-8"))
    assert complete["keys"] == 12 and complete["real_run_evidence"] is False
    lines = []
    grid.report(result, log=lines.append)
    assert "FAKE FIXTURE RESULTS - NOT REAL EVIDENCE" in lines[0] and "(FAKE)" in lines[-1]


def test_fake_results_are_refused_as_real(done):
    ctx = grid.prepare(**done.kw())
    assert len(grid.validate_results(done.results, ctx=ctx)["records"]) == 12
    real_ctx = dict(ctx, mode="real")                  # as if a real run pointed at the fake directory
    with pytest.raises(grid.GridError, match="fake and real results are never mixed"):
        grid.validate_results(done.results, ctx=real_ctx)
    # Fake records copied into a real-marked directory are invalid, not reused.
    real_dir = done.root / "real-grid"
    grid.open_results_dir(real_dir, real_ctx, create=True)
    shutil.copytree(done.results / grid.RECORDS, real_dir / grid.RECORDS)
    with pytest.raises(grid.GridError, match="fake/real evidence flag"):
        grid.validate_results(real_dir, ctx=real_ctx)


def test_resume_reuses_only_exact_matches(done, monkeypatch):
    before = {p.name: p.read_bytes() for p in (done.results / grid.RECORDS).iterdir()}
    _no_fit(monkeypatch)
    out = done.run()
    assert out["complete"]
    assert {p.name: p.read_bytes() for p in (done.results / grid.RECORDS).iterdir()} == before
    assert sum("resumed" in line for line in done.log) == 12


def test_partial_record_is_resumed_only_after_explicit_replace(done):
    done.record("emotion", "C5b").write_bytes(done.record("emotion", "C5b").read_bytes()[:200])
    with pytest.raises(grid.GridError, match="emotion/C5b/13: invalid"):
        done.run()
    with pytest.raises(grid.GridError, match="emotion/C5b/13: invalid"):
        done.validate()
    done.run(replace=[("emotion", "C5b", 13)])
    assert len(done.validate()["records"]) == 12


@pytest.mark.parametrize("field", ["repository", "freeze", "training_ids", "test", "environment", "classifier"])
def test_stale_provenance_is_rejected(done, monkeypatch, field):
    _no_fit(monkeypatch)
    changes = {
        "repository": lambda r: r["repository"].update(revision="c" * 40),
        "freeze": lambda r: r["freeze"].update(sha256="0" * 64),
        "training_ids": lambda r: r["training"]["ids"].reverse(),
        "test": lambda r: r["test"].update(sha256="0" * 64),
        "environment": lambda r: r["environment"]["packages"].update({"scikit-learn": "0.0"}),
        "classifier": lambda r: r["classifier"]["classifier"].update(C=2.0),
    }
    _rehash(done.record("trec", "C3"), changes[field])
    with pytest.raises(grid.GridError, match="trec/C3/13: stale"):
        done.run()
    with pytest.raises(grid.GridError, match="trec/C3/13: stale"):
        done.validate()


def test_tampered_metrics_and_wrong_key_records_are_invalid(done):
    path = done.record("trec", "C2")
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["metrics"]["macro_f1"] = 0.99
    path.write_bytes(grid._canonical(rec))
    with pytest.raises(grid.GridError, match="record hash mismatch"):
        done.validate()
    _rehash(path, lambda r: r["metrics"].update(macro_f1=1.5))
    with pytest.raises(grid.GridError, match="not a metric"):
        done.validate()
    shutil.copy(done.record("trec", "C1"), path)              # duplicate key under another file name
    with pytest.raises(grid.GridError, match="does not match its file"):
        done.validate()


def test_targeted_replacement_changes_only_the_named_key(done):
    before = {p.name: p.read_bytes() for p in (done.results / grid.RECORDS).iterdir()}
    done.run(replace=grid.parse_keys("trec/C2"))
    after = {p.name: p.read_bytes() for p in (done.results / grid.RECORDS).iterdir()}
    changed = {n for n in before if before[n] != after[n]}
    assert changed == {"trec.C2.seed13.json"}
    superseded = list((done.results / grid.SUPERSEDED).glob("trec.C2.seed13.*.json"))
    assert len(superseded) == 1 and superseded[0].read_bytes() == before["trec.C2.seed13.json"]
    assert len(done.validate()["records"]) == 12


def test_requested_subset_leaves_unrequested_stale_rows_untouched(done):
    _rehash(done.record("emotion", "C4"), lambda r: r["repository"].update(revision="c" * 40))
    stale = done.record("emotion", "C4").read_bytes()
    done.run(keys=grid.parse_keys("trec/C1"), replace=grid.parse_keys("trec/C1"))
    assert done.record("emotion", "C4").read_bytes() == stale
    assert not (done.results / grid.COMPLETE).exists()
    with pytest.raises(grid.GridError, match="emotion/C4/13: stale"):
        done.validate()


def test_missing_duplicate_and_foreign_keys_fail_validation(done):
    done.record("emotion", "C1").unlink()
    with pytest.raises(grid.GridError, match="emotion/C1/13: missing"):
        done.validate()


def test_duplicate_runs_csv_rows_fail_validation(done):
    runs = done.results / grid.RUNS_CSV
    lines = runs.read_bytes().splitlines(keepends=True)
    runs.write_bytes(b"".join(lines + [lines[1]]))
    with pytest.raises(grid.GridError, match="duplicate keys"):
        done.validate()


def test_wrong_seed_record_file_is_rejected(done, monkeypatch):
    _no_fit(monkeypatch)
    shutil.copy(done.record("trec", "C1"), done.results / grid.RECORDS / "trec.C1.seed42.json")
    with pytest.raises(grid.GridError, match="unexpected"):
        done.run()
    with pytest.raises(grid.GridError, match="unexpected"):
        done.validate()


def test_incomplete_grid_is_never_reported_complete(env):
    out = env.run(keys=grid.parse_keys("trec/C1,emotion/C5b"))
    assert not out["complete"] and len(out["valid"]) == 2
    assert not (env.results / grid.COMPLETE).exists()
    with pytest.raises(grid.GridError, match="missing"):
        env.validate()


def test_historical_and_unrelated_result_dirs_are_refused(env, monkeypatch):
    _no_fit(monkeypatch)
    for protected in grid.PROTECTED_RESULTS:
        with pytest.raises(grid.GridError, match="historical"):
            grid.run_grid(protected, log=env.log.append, **env.kw())
    env.results.mkdir(parents=True)
    (env.results / "runs.csv").write_text("task,condition,seed,n_train,macro_f1,accuracy\n", encoding="utf-8")
    with pytest.raises(grid.GridError, match="not empty"):
        env.run()
    assert (env.results / "runs.csv").read_text(encoding="utf-8").count("\n") == 1


def test_parse_keys_rejects_unknown_and_duplicate_keys():
    assert len(grid.parse_keys(None)) == 12
    for bad in ("trec/C6", "sst/C1", "trec/C1,trec/C1", "", "trec"):
        with pytest.raises(grid.GridError):
            grid.parse_keys(bad)


MODEL_FILES = ("modules.json", "config.json", "tokenizer.json", "config_sentence_transformers.json",
               "model.safetensors")


def _fake_hub(root, revision, weights=b"approved weights", main=None):
    repo = root / "models--sentence-transformers--all-MiniLM-L6-v2"
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        (snap / name).write_bytes(weights if name == "model.safetensors" else name.encode())
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(main or revision, encoding="utf-8")
    return snap


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer; records what was loaded."""
    loaded = []
    on_load = None

    def __init__(self, path, device=None):
        type(self).loaded.append((Path(path), device))
        if type(self).on_load:
            type(self).on_load(Path(path))

    def encode(self, texts, **kw):
        import numpy as np
        return np.ones((len(texts), 4))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import sys
    import types
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeSentenceTransformer.loaded, _FakeSentenceTransformer.on_load = [], None
    return tmp_path / "hub"


def test_model_identity_pins_the_approved_snapshot(hub):
    with pytest.raises(grid.GridError, match="not in the local Hugging Face cache"):
        grid.model_identity("real")
    snap = _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    ident = grid.model_identity("real")
    assert ident["revision"] == grid.APPROVED_MINILM_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert ident["files_sha256"]["model.safetensors"] == grid._sha(b"approved weights")
    assert ident["fine_tuned"] is False and ident["normalize_embeddings"] is True
    with pytest.raises(grid.GridError, match="not the approved"):
        grid.model_identity("real", "f" * 40)
    (snap / "tokenizer.json").unlink()
    with pytest.raises(grid.GridError, match="incomplete"):
        grid.model_identity("real")


def test_loader_uses_the_recorded_snapshot_even_if_refs_main_moves(hub):
    approved = _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    ident = grid.model_identity("real")
    _fake_hub(hub, "e" * 40, weights=b"newer weights")              # refs/main now points elsewhere
    assert (hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "refs" / "main").read_text() == "e" * 40
    grid.Embedder("real", ident)(["a", "b"])
    assert _FakeSentenceTransformer.loaded == [(approved, "cpu")]


@pytest.mark.parametrize("change", ["weights", "added_file", "removed_file", "snapshot_gone"])
def test_loader_fails_closed_if_snapshot_changes_after_recording(hub, change):
    snap = _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    ident = grid.model_identity("real")                                 # recorded during preparation
    if change == "weights":
        (snap / "model.safetensors").write_bytes(b"swapped weights")
    elif change == "added_file":
        (snap / "pytorch_model.bin").write_bytes(b"extra")
    elif change == "removed_file":
        (snap / "config.json").unlink()
    else:
        shutil.rmtree(snap)
    with pytest.raises(grid.GridError):
        grid.Embedder("real", ident)(["a"])
    assert _FakeSentenceTransformer.loaded == []                         # never loaded


def test_loader_fails_closed_if_snapshot_changes_while_loading(hub):
    snap = _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    ident = grid.model_identity("real")
    _FakeSentenceTransformer.on_load = lambda path: (path / "model.safetensors").write_bytes(b"raced")
    with pytest.raises(grid.GridError, match="changed since it was recorded"):
        grid.Embedder("real", ident)(["a"])


def test_loader_refuses_missing_or_unapproved_identity(hub):
    _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    ident = grid.model_identity("real")
    with pytest.raises(grid.GridError, match="needs the recorded model identity"):
        grid.Embedder("real")
    for bad in (dict(ident, revision="e" * 40), dict(ident, name="other/model"), {"kind": "hashing-fixture"}):
        with pytest.raises(grid.GridError, match="not the approved"):
            grid.Embedder("real", bad)(["a"])
    forged = dict(ident, files_sha256=dict(ident["files_sha256"], **{"model.safetensors": "0" * 64}))
    with pytest.raises(grid.GridError, match="changed since it was recorded"):
        grid.Embedder("real", forged)(["a"])
    assert _FakeSentenceTransformer.loaded == []


def test_run_grid_passes_the_recorded_identity_to_the_loader(env, hub, monkeypatch):
    """Real-mode wiring: the embedder built by run_grid receives exactly ctx['model']."""
    _fake_hub(hub, grid.APPROVED_MINILM_REVISION)
    seen = []

    class Spy(grid.Embedder):
        def __init__(self, mode, identity=None):
            seen.append((mode, identity))
            super().__init__("fake")

    monkeypatch.setattr(grid, "Embedder", Spy)
    real_prepare = grid.prepare

    def prepare_as_real(*a, **k):
        ctx = real_prepare(*a, **k)
        ctx["mode"] = "real"
        ctx["model"] = grid.model_identity("real")
        return ctx

    monkeypatch.setattr(grid, "prepare", prepare_as_real)
    monkeypatch.setattr(grid, "fit_and_score", lambda ctx, task, cond, emb: (
        {"macro_f1": 0.5, "accuracy": 0.5}, {"n_iter": 1, "converged": True, "test_rows": 1}))
    grid.run_grid(env.results, keys=grid.parse_keys("trec/C1"), log=env.log.append, **env.kw())
    assert seen == [("real", grid.model_identity("real"))]


def test_cli_dry_run_end_to_end_and_validate(env, capsys):
    args = ["--dry-run", "--freeze-manifest", str(env.freeze), "--evidence-dir", str(env.evidence),
            "--results-dir", str(env.results)]
    grid.main(args)
    out = capsys.readouterr().out
    assert "FAKE FIXTURE RESULTS - NOT REAL EVIDENCE" in out and "COMPLETE: exactly 12" in out
    grid.main(args + ["--validate"])
    assert "COMPLETE: exactly 12 unique valid keys (FAKE)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        grid.main(["--dry-run", "--freeze-manifest", str(env.freeze), "--evidence-dir", str(env.evidence)])
