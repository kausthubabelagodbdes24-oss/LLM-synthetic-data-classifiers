from collections import Counter

from synthtext import config
from synthtext.io import read_csv, read_jsonl
from synthtext.paths import real_dir, results_root
from workstreams.ws1_generation import filter as filt
from workstreams.ws1_generation import generate
from workstreams.ws1_generation.llm_client import FakeClient
from workstreams.ws2_training import conditions, run_matrix
from workstreams.ws2_training.sampling import proportional_counts
from workstreams.ws3_evaluation import analyze, metrics


def _make_synthetic(task="emotion", per_label=12):
    cfg = config.generation()
    for setting in ("zero_shot", "few_shot"):
        generate.generate_labelled(task, setting, per_label, cfg, FakeClient())
        filt.main(["--task", task, "--setting", setting])
    generate.generate_auggpt(task, cfg, FakeClient(), pilot=False)
    filt.main(["--task", task, "--setting", "auggpt"])


def test_proportional_counts_sum_exactly():
    counts = proportional_counts({"a": 7, "b": 2, "c": 1}, 5)
    assert sum(counts.values()) == 5 and counts["a"] >= counts["b"] >= counts["c"]


def test_mix_conditions_keep_label_counts(sample_env):
    _make_synthetic()
    c1 = conditions.build("emotion", "C1", 13, n_train=24)
    for cond in ("C3", "C4a", "C4b", "C4c"):
        ts = conditions.build("emotion", cond, 13, n_train=24)
        assert ts.label_counts == c1.label_counts, cond
    c4b = conditions.build("emotion", "C4b", 13, n_train=24)
    assert sum(1 for r in c4b.records if "setting" in r) == 12  # half synthetic


def test_lowres_conditions(sample_env):
    _make_synthetic()
    c5a = conditions.build("emotion", "C5a", 42)
    c5b = conditions.build("emotion", "C5b", 42)
    c5c = conditions.build("emotion", "C5c", 42)
    base_ids = {r["id"] for r in c5a.records}
    added = [r for r in c5b.records if r["id"] not in base_ids]
    assert added and all(r["source_ids"][0] in base_ids for r in added)
    assert len(c5c.records) <= len(c5b.records)
    ceil = conditions.build("emotion", "CEIL", 13)
    assert len(ceil.records) == len(read_jsonl(real_dir("emotion") / "train.jsonl")) + len(
        read_jsonl(real_dir("emotion") / "pool.jsonl"))


def test_run_matrix_metrics_and_analysis_end_to_end(sample_env):
    _make_synthetic()
    run_matrix.main(["--task", "emotion", "--conditions", "C1,C2,C3,C4b,CEIL", "--n-train", "24"])
    runs = read_csv(results_root() / "runs.csv")
    assert {(r["condition"]) for r in runs} == {"C1", "C2", "C3", "C4b", "CEIL"}
    assert sum(r["condition"] == "C1" for r in runs) == len(config.experiments()["seeds"])
    assert sum(r["condition"] == "CEIL" for r in runs) == 1

    run_matrix.main(["--task", "emotion", "--conditions", "C1"])  # resumable: nothing new
    assert len(read_csv(results_root() / "runs.csv")) == len(runs)

    metrics.main(["--task", "emotion"])
    quality = read_csv(results_root() / "quality.csv")
    assert {q["setting"] for q in quality} == {"real", "zero_shot", "few_shot", "auggpt"}
    assert all(q["label_fidelity"] != "" for q in quality if q["setting"] != "real")

    analyze.main([])
    assert (results_root() / "summary.md").exists()
    assert Counter(p.suffix for p in (results_root() / "figures").iterdir())[".png"] >= 2
