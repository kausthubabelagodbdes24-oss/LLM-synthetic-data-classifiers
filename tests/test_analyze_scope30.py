"""Issue #14: single-seed gap analysis of the frozen twelve-key grid. Fixture grids only; no score here is evidence.

Most tests build a grid with the producer's own helpers (``train_scope30`` record hashing, ``runs.csv``,
``grid_complete.json`` and ``grid.json``) at chosen metrics, so calculations can be checked against known values.
One test runs on the fake grid produced end to end by the real #13 pipeline fixture.
"""

import json
import math

import pytest
from matplotlib.container import ErrorbarContainer

from ws1_generation import freeze_scope30 as fz
from ws2_training import train_scope30 as ts
from ws3_evaluation import analyze_scope30 as an
# The #13 fixtures build a complete fake grid with the real pipeline (fake generator + hashing embedder).
from test_train_scope30 import REPO as GRID_REPO, built, completed  # noqa: F401

COMPOSITION = {"C1": {"real_train": 180}, "C2": {"zero_shot": 180}, "C3": {"few_shot": 180},
               "C4": {"few_shot": 90, "real_train": 90}, "C5a": {"lowres_source": 60},
               "C5b": {"auggpt": 360, "lowres_source": 60}}
DEMOS = {"trec": 59, "emotion": 54}
METRICS = {("trec", "C1"): (0.60, 0.65), ("trec", "C2"): (0.63, 0.66), ("trec", "C3"): (0.66, 0.70),
           ("trec", "C4"): (0.70, 0.72), ("trec", "C5a"): (0.55, 0.60), ("trec", "C5b"): (0.58, 0.61),
           ("emotion", "C1"): (0.40, 0.44), ("emotion", "C2"): (0.30, 0.43), ("emotion", "C3"): (0.36, 0.42),
           ("emotion", "C4"): (0.40, 0.45), ("emotion", "C5a"): (0.34, 0.37), ("emotion", "C5b"): (0.34, 0.36)}
FAKE_MODEL = {"kind": "hashing-fixture", "name": ts.HASHING_FIXTURE, "real_model": False}
REAL_MODEL = {"kind": "sentence-transformers", "name": ts.EMBEDDING_MODEL, "revision": ts.APPROVED_MINILM_REVISION,
              "files_sha256": {"model.safetensors": "0" * 64}, "identity_sha256": "1" * 64, "real_model": True,
              **ts.EMBEDDING_SETTINGS}


def write_freeze(path):
    manifest = {"freeze_id": fz.FREEZE_ID, "artifacts": {
        f"{t}/{s}": {"selected_distinct_demonstrations": DEMOS[t] if s == "few_shot" else 0}
        for t in an.TASKS for s in ("zero_shot", "few_shot")}}
    path.write_bytes(ts._canonical(manifest))
    return ts._sha(fz._lf(path.read_bytes()))


def write_record(path, rec):
    rec = {k: v for k, v in rec.items() if k != "record_sha256"}
    rec["record_sha256"] = ts._sha(ts._canonical(rec))
    path.write_bytes(ts._canonical(rec))
    return rec


def finish(grid_dir, records, ctx):
    """Derived files exactly as train_scope30.refresh_derived writes them."""
    (grid_dir / ts.RUNS_CSV).write_bytes(ts.runs_csv_bytes(records))
    (grid_dir / ts.COMPLETE).write_bytes(ts._canonical(ts.complete_summary(ctx, grid_dir, records)))


def make_grid(root, metrics=METRICS, mode="fake"):
    root.mkdir(parents=True, exist_ok=True)
    freeze_path = root / "freeze.json"
    freeze_sha = write_freeze(freeze_path)
    repo = {"revision": an.ACCEPTED_GRID_REVISION, "clean": True} if mode == "real" else {"revision": "c" * 40,
                                                                                          "clean": True}
    model = REAL_MODEL if mode == "real" else FAKE_MODEL
    ctx = {"mode": mode, "freeze": {"manifest": {"freeze_id": fz.FREEZE_ID}, "sha256": freeze_sha},
           "model": model, "repository": repo}
    grid_dir = root / "grid"
    (grid_dir / ts.RECORDS).mkdir(parents=True)
    (grid_dir / ts.MARKER).write_bytes(ts._canonical(ts._marker(ctx)))
    records = {}
    for key in ts.all_keys():
        task, cond, seed = key
        if (task, cond) not in metrics:
            continue
        f1, acc = metrics[(task, cond)]
        rec = {"schema_version": ts.SCHEMA_VERSION, "result_kind": ts.RESULT_KIND, "scope_id": ts.SCOPE_ID,
               "generation_mode": mode, "real_run_evidence": mode == "real",
               "key": {"task": task, "condition": cond, "seed": seed},
               "freeze": {"freeze_id": fz.FREEZE_ID, "sha256": freeze_sha, "path": "freeze.json",
                          "audit_revision": "d" * 40},
               "training": {"n_train": ts.EXPECTED_COUNTS[cond], "composition": COMPOSITION[cond],
                            "ids_sha256": ts._sha(f"{task}/{cond}".encode())},
               "classifier": {"embedding": model, "classifier": ts.CLASSIFIER}, "repository": repo,
               "metrics": {"macro_f1": f1, "accuracy": acc}, "fit": {"n_iter": 5, "converged": True, "test_rows": 30},
               "runtime_seconds": 1.5}
        records[key] = write_record(grid_dir / ts.RECORDS / ts.record_filename(*key), rec)
    if len(records) == len(ts.all_keys()):
        finish(grid_dir, records, ctx)
    return grid_dir, freeze_path, ctx


@pytest.fixture
def fake(tmp_path):
    return make_grid(tmp_path / "fixture")


@pytest.fixture
def real(tmp_path, monkeypatch):
    """A fixture grid that looks like the accepted real grid; the pins are pointed at it."""
    grid_dir, freeze_path, ctx = make_grid(tmp_path / "fixture", mode="real")
    monkeypatch.setattr(ts, "ACCEPTED_FREEZE_PATH", freeze_path)
    monkeypatch.setattr(ts, "ACCEPTED_FREEZE_SHA256", ctx["freeze"]["sha256"])
    monkeypatch.setattr(an, "ACCEPTED_GRID_COMPLETE_SHA256", ts._sha((grid_dir / ts.COMPLETE).read_bytes()))
    return grid_dir, freeze_path, ctx


def rec_path(grid_dir, task, cond):
    return grid_dir / ts.RECORDS / ts.record_filename(task, cond)


def rewrite(grid_dir, ctx, task, cond, change, refresh=True):
    """Change one record, re-seal its self-hash and (optionally) re-derive runs.csv/grid_complete.json."""
    p = rec_path(grid_dir, task, cond)
    rec = json.loads(p.read_text(encoding="utf-8"))
    change(rec)
    write_record(p, rec)
    if refresh:
        records = {k: json.loads(rec_path(grid_dir, k[0], k[1]).read_text(encoding="utf-8")) for k in ts.all_keys()}
        finish(grid_dir, records, ctx)


def analysed(grid_dir, freeze_path, mode="fake"):
    grid = an.load_grid(grid_dir, mode)
    return grid, an.analyse(grid, an.load_freeze_costs(grid, freeze_path if mode == "fake" else None))


# ---------------------------------------------------------------- calculations

def test_relative_gap_known_values_and_zero_denominator():
    assert an.relative_gap(0.63, 0.60) == pytest.approx(0.05)
    assert an.relative_gap(0.30, 0.40) == pytest.approx(-0.25)
    assert an.relative_gap(0.0, 0.40) == pytest.approx(-1.0)
    assert an.relative_gap(0.25, 0.0) is None              # undefined, never divided
    assert an.relative_gap(0.0, 0.0) is None


def test_gaps_pair_c2_c3_with_same_task_c1_at_full_precision(fake):
    _, result = analysed(fake[0], fake[1])
    by = {(g["task"], g["condition"]): g for g in result["gaps"]}
    assert set(by) == {(t, c) for t in an.TASKS for c in ("C2", "C3")}
    g = by[("emotion", "C2")]
    assert g["c1_macro_f1"] == 0.40 and g["macro_f1"] == 0.30
    assert g["absolute_gap"] == 0.30 - 0.40                  # exact float, no rounding before presentation
    assert g["relative_gap"] == (0.30 - 0.40) / 0.40
    assert by[("trec", "C3")]["relative_gap"] == pytest.approx(0.10)
    assert all(g["status"] == "defined" for g in result["gaps"])


def test_four_planned_comparisons_are_within_task_differences(fake):
    _, result = analysed(fake[0], fake[1])
    comps = {c["id"]: c for c in result["comparisons"]}
    assert set(comps) == {"cross_task_gap_C2", "cross_task_gap_C3", "few_vs_zero", "mix_vs_real",
                          "auggpt_vs_sources"}
    assert comps["few_vs_zero"]["per_task"]["trec"]["macro_f1_diff"] == pytest.approx(0.03)
    assert comps["mix_vs_real"]["per_task"]["emotion"]["macro_f1_diff"] == pytest.approx(0.0)
    assert comps["auggpt_vs_sources"]["per_task"]["trec"]["n_train"] == [420, 60]
    assert comps["auggpt_vs_sources"]["per_task"]["emotion"]["accuracy_diff"] == pytest.approx(-0.01)
    cross = comps["cross_task_gap_C2"]
    assert cross["emotion_minus_trec_relative"] == pytest.approx(-0.25 - 0.05)
    assert cross["emotion_minus_trec_absolute"] == pytest.approx(-0.10 - 0.03)


def test_zero_c1_keeps_the_task_with_undefined_relative_gap(tmp_path):
    metrics = dict(METRICS)
    metrics[("emotion", "C1")] = (0.0, 0.10)
    grid_dir, freeze_path, _ = make_grid(tmp_path / "zero", metrics)
    grid, result = analysed(grid_dir, freeze_path)
    emo = [g for g in result["gaps"] if g["task"] == "emotion"]
    assert len(emo) == 2                                     # retained, not dropped
    assert all(g["relative_gap"] is None and g["status"].startswith("undefined") for g in emo)
    assert [g["absolute_gap"] for g in emo] == [0.30, 0.36]  # absolute difference still reported
    cross = {c["id"]: c for c in result["comparisons"]}["cross_task_gap_C2"]
    assert cross["emotion_minus_trec_relative"] is None and cross["status"].startswith("relative difference undef")
    text = an.summary_markdown(result, an.provenance(grid))
    assert "undefined (C1 = 0)" in text and "relative difference undefined" in text
    _, ax = an.gap_chart(result, "fake")
    labels = [t.get_text() for t in ax.texts]
    assert sum("undefined" in s for s in labels) == 2
    heights = [p.get_height() for p in ax.patches if p.get_width() > 0]
    assert len(heights) == 2                                 # only TREC's two defined bars; no zero-height stand-ins


def test_missing_c1_blocks_analysis(tmp_path):
    metrics = {k: v for k, v in METRICS.items() if k != ("trec", "C1")}
    grid_dir, freeze_path, _ = make_grid(tmp_path / "noc1", metrics)
    with pytest.raises(an.AnalysisError, match="grid_complete.json"):
        an.load_grid(grid_dir, "fake")                       # no complete summary: incomplete grid
    table = {(t, c): {"macro_f1": f, "accuracy": a, "n_train": ts.EXPECTED_COUNTS[c], "composition": COMPOSITION[c]}
             for (t, c), (f, a) in metrics.items()}
    with pytest.raises(an.AnalysisError, match="C1 is missing"):
        an.gaps(table)


def test_missing_c1_record_in_a_complete_grid_is_named(fake):
    grid_dir = fake[0]
    rec_path(grid_dir, "trec", "C1").unlink()
    with pytest.raises(an.AnalysisError, match="trec/C1/13: record missing - a missing C1 blocks"):
        an.load_grid(grid_dir, "fake")


def test_costs_keep_demonstrations_separate_from_training_counts(fake):
    _, result = analysed(fake[0], fake[1])
    by = {(r["task"], r["condition"]): r for r in result["costs"]}
    assert by[("trec", "C3")] | {} == {**by[("trec", "C3")], "n_train": 180, "real_training_rows": 0,
                                       "synthetic_training_rows": 180, "real_generation_demonstrations": 59}
    assert by[("emotion", "C4")]["real_training_rows"] == 90 and by[("emotion", "C4")]["synthetic_training_rows"] == 90
    assert by[("emotion", "C4")]["real_generation_demonstrations"] == 54
    assert by[("trec", "C2")]["real_generation_demonstrations"] == 0
    assert by[("trec", "C5b")]["real_training_rows"] == 60 and by[("trec", "C5b")]["synthetic_training_rows"] == 360


# ---------------------------------------------------------------- grid validation

def test_fake_grid_built_by_the_real_pipeline_is_accepted(built, completed):
    grid = an.load_grid(completed, "fake")
    assert len(grid["records"]) == 12
    assert grid["complete"]["repository"] == GRID_REPO
    result = an.analyse(grid, an.load_freeze_costs(grid, built / "freeze.json"))
    assert len(result["gaps"]) == 4 and len(result["costs"]) == 12


def test_fake_grid_is_never_accepted_as_real(fake, monkeypatch):
    grid_dir = fake[0]
    monkeypatch.setattr(an, "ACCEPTED_GRID_COMPLETE_SHA256", ts._sha((grid_dir / ts.COMPLETE).read_bytes()))
    with pytest.raises(an.AnalysisError, match="not a complete real"):
        an.load_grid(grid_dir, "real")


def test_real_mode_accepts_only_the_pinned_grid(real, monkeypatch):
    grid_dir = real[0]
    grid = an.load_grid(grid_dir, "real")
    assert grid["complete_sha256"] == an.ACCEPTED_GRID_COMPLETE_SHA256
    monkeypatch.setattr(an, "ACCEPTED_GRID_COMPLETE_SHA256", "f" * 64)
    with pytest.raises(an.AnalysisError, match="is not the accepted"):
        an.load_grid(grid_dir, "real")


def test_resealed_metric_change_is_rejected_in_real_mode(real):
    grid_dir, _, ctx = real
    rewrite(grid_dir, ctx, "trec", "C2", lambda r: r["metrics"].update(macro_f1=0.99))   # fully re-derived chain
    with pytest.raises(an.AnalysisError, match="is not the accepted"):
        an.load_grid(grid_dir, "real")


@pytest.mark.parametrize("module, name, value, message", [
    (an, "ACCEPTED_GRID_REVISION", "e" * 40, "clean checkout"),
    (ts, "ACCEPTED_FREEZE_SHA256", "e" * 64, "accepted freeze"),
    (ts, "APPROVED_MINILM_REVISION", "e" * 40, "approved MiniLM"),
])
def test_real_mode_requires_the_accepted_revision_freeze_and_model(real, monkeypatch, module, name, value, message):
    monkeypatch.setattr(module, name, value)
    with pytest.raises(an.AnalysisError, match=message):
        an.load_grid(real[0], "real")


@pytest.mark.parametrize("damage, message", [
    (lambda r: r["metrics"].update(macro_f1=0.99), "record hash mismatch"),
    (lambda r: r.update(runtime_seconds=9.0), "record hash mismatch"),
])
def test_tampered_record_without_reseal_is_rejected(fake, damage, message):
    p = rec_path(fake[0], "emotion", "C3")
    rec = json.loads(p.read_text(encoding="utf-8"))
    damage(rec)
    p.write_bytes(ts._canonical(rec))
    with pytest.raises(an.AnalysisError, match=message):
        an.load_grid(fake[0], "fake")


def test_resealed_record_that_disagrees_with_grid_complete_is_rejected(fake):
    grid_dir, _, ctx = fake
    rewrite(grid_dir, ctx, "emotion", "C3", lambda r: r["metrics"].update(macro_f1=0.99), refresh=False)
    with pytest.raises(an.AnalysisError, match="differs from the hash recorded in grid_complete.json"):
        an.load_grid(grid_dir, "fake")


@pytest.mark.parametrize("change, message", [
    (lambda r: r["training"].update(n_train=179), "n_train 179 != 180"),
    (lambda r: r["metrics"].update(macro_f1=1.5), "metrics must be exactly"),
    (lambda r: r["metrics"].update(macro_f1=float("nan")), "metrics must be exactly"),
    (lambda r: r["metrics"].update(precision=0.5), "metrics must be exactly"),
    (lambda r: r.update(key={"task": "trec", "condition": "C2", "seed": 13}), "does not match its file"),
    (lambda r: r.update(generation_mode="real", real_run_evidence=True), "generation mode"),
    (lambda r: r["freeze"].update(sha256="0" * 64), "freeze binding"),
    (lambda r: r["repository"].update(revision="0" * 40), "repository binding"),
    (lambda r: r["classifier"]["classifier"].update(C=2.0), "classifier/embedder identity"),
    (lambda r: r["training"].update(composition={"real_train": 100}), "composition"),
])
def test_consistently_resealed_but_invalid_records_are_rejected(fake, change, message):
    grid_dir, _, ctx = fake
    p = rec_path(grid_dir, "trec", "C3")
    rec = json.loads(p.read_text(encoding="utf-8"))
    change(rec)
    sealed = write_record(p, rec)
    # Make grid_complete.json agree with the altered record so only the record's own content is at fault.
    complete = json.loads((grid_dir / ts.COMPLETE).read_text(encoding="utf-8"))
    for e in complete["results"]:
        if e["key"] == "trec/C3/13":
            e.update(record_sha256=sealed["record_sha256"], record_file_sha256=ts._sha(p.read_bytes()),
                     n_train=sealed["training"]["n_train"], macro_f1=sealed["metrics"].get("macro_f1"))
    (grid_dir / ts.COMPLETE).write_bytes(ts._canonical(complete))
    with pytest.raises(an.AnalysisError, match=message):
        an.load_grid(grid_dir, "fake")


def test_foreign_or_duplicate_record_files_are_rejected(fake):
    grid_dir = fake[0]
    src = rec_path(grid_dir, "trec", "C1")
    (grid_dir / ts.RECORDS / "trec.C1.seed42.json").write_bytes(src.read_bytes())
    with pytest.raises(an.AnalysisError, match="unexpected file records/trec.C1.seed42.json"):
        an.load_grid(grid_dir, "fake")


def test_runs_csv_must_match_records(fake):
    grid_dir = fake[0]
    runs = grid_dir / ts.RUNS_CSV
    runs.write_bytes(runs.read_bytes() + runs.read_bytes().splitlines(keepends=True)[1])   # duplicate a row
    with pytest.raises(an.AnalysisError, match="runs.csv does not match"):
        an.load_grid(grid_dir, "fake")
    runs.unlink()
    with pytest.raises(an.AnalysisError, match="runs.csv missing"):
        an.load_grid(grid_dir, "fake")


def test_grid_marker_must_identify_this_grid(fake):
    grid_dir = fake[0]
    marker = json.loads((grid_dir / ts.MARKER).read_text(encoding="utf-8"))
    marker["freeze_sha256"] = "0" * 64
    (grid_dir / ts.MARKER).write_bytes(ts._canonical(marker))
    with pytest.raises(an.AnalysisError, match="grid.json does not identify"):
        an.load_grid(grid_dir, "fake")


def test_freeze_for_costs_must_be_the_grids_freeze(fake, tmp_path):
    grid = an.load_grid(fake[0], "fake")
    other = tmp_path / "other.json"
    other.write_bytes(ts._canonical({"freeze_id": "x", "artifacts": {}}))
    with pytest.raises(an.AnalysisError, match="not the one the grid was trained on"):
        an.load_freeze_costs(grid, other)
    with pytest.raises(an.AnalysisError, match="needs the fixture's --freeze-manifest"):
        an.load_freeze_costs(grid, None)


# ---------------------------------------------------------------- charts and outputs

def test_single_seed_charts_have_no_error_bars(fake):
    _, result = analysed(fake[0], fake[1])
    for draw in (an.f1_chart, an.gap_chart):
        fig, ax = draw(result, "fake")
        assert not [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        assert not ax.collections and not ax.lines[1:]       # no error-bar caps/segments; only the gap zero line
        assert "FAKE FIXTURE" in ax.get_title(loc="left")
    _, ax = an.f1_chart(result, "fake")
    bars = [p for p in ax.patches if p.get_height() > 0]
    assert len(bars) == 12
    assert sorted(round(b.get_height(), 6) for b in bars) == sorted(round(v[0], 6) for v in METRICS.values())
    _, ax = an.gap_chart(result, "fake")
    assert sorted(round(p.get_height(), 6) for p in ax.patches if p.get_width() > 0) == \
        sorted(round(g["relative_gap"] * 100, 6) for g in result["gaps"])


def test_outputs_are_written_once_with_hashes_and_fake_tags(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(an, "analysis_repository", lambda mode: {"revision": "a" * 40, "clean": True})
    grid, result = analysed(fake[0], fake[1])
    out = an.write_outputs(grid, result, tmp_path / "out")
    names = sorted(p.name for p in out.iterdir())
    assert names == ["analysis.json", "f1_by_condition.png", "gap.png", "summary.md"]
    doc = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert doc["generation_mode"] == "fake" and doc["real_run_evidence"] is False and doc["single_seed"] is True
    for name, sha in doc["outputs_sha256"].items():
        assert ts._sha((out / name).read_bytes()) == sha
    assert doc["grid"]["grid_complete_sha256"] == grid["complete_sha256"]
    assert {r["condition"] for r in doc["results"]} == set(an.CONDITIONS) and len(doc["results"]) == 12
    assert all(math.isfinite(r["macro_f1"]) for r in doc["results"])
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "FAKE FIXTURE ANALYSIS - NOT REAL EVIDENCE" in text
    for banned in ("±", "std", "standard deviation of", "significant", "confidence interval of"):
        assert banned not in text.replace("No standard deviation, confidence interval, significance", "")
    with pytest.raises(an.AnalysisError, match="already has content"):
        an.write_outputs(grid, result, out)


def test_real_analysis_requires_a_clean_checkout(monkeypatch):
    calls = iter(["a" * 40, " M ws3_evaluation/analyze_scope30.py"])

    class Done:
        def __init__(self):
            self.stdout = next(calls)

    monkeypatch.setattr(an.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(an.AnalysisError, match="clean checkout"):
        an.analysis_repository("real")


def test_cli_guards_and_check_mode(fake, tmp_path, capsys, monkeypatch):
    grid_dir, freeze_path, _ = fake
    with pytest.raises(SystemExit):
        an.main(["--results-dir", str(grid_dir), "--freeze-manifest", str(freeze_path)])
    assert "real analysis reads only the accepted freeze" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        an.main(["--dry-run", "--results-dir", str(grid_dir), "--freeze-manifest", str(freeze_path)])
    assert "needs an explicit --out" in capsys.readouterr().out
    an.main(["--dry-run", "--check", "--results-dir", str(grid_dir), "--freeze-manifest", str(freeze_path)])
    out = capsys.readouterr().out
    assert "GRID VERIFIED (FAKE FIXTURE" in out and "nothing written" in out
    monkeypatch.setattr(an, "analysis_repository", lambda mode: {"revision": "a" * 40, "clean": True})
    an.main(["--dry-run", "--results-dir", str(grid_dir), "--freeze-manifest", str(freeze_path),
             "--out", str(tmp_path / "cli")])
    assert "ANALYSIS WRITTEN (FAKE FIXTURE" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        an.main(["--results-dir", str(grid_dir)])            # a fake grid in real mode
    assert "FAIL" in capsys.readouterr().out
