from pathlib import Path

from synthtext import config
from synthtext.contracts import check_file, check_real, check_synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_configs_are_consistent():
    exp = config.experiments()
    assert set(exp["tasks"]) <= set(config.tasks())
    for t in exp["tasks"]:
        cfg = config.task(t)
        assert cfg["labels"], t
        assert "label_map" in cfg or "label_contains" in cfg, t
        mapped = set((cfg.get("label_map") or cfg.get("label_contains")).values())
        assert mapped == set(cfg["labels"]), f"{t}: label mapping must cover exactly the labels"


def test_committed_sample_files_follow_the_contracts():
    for t in config.task_ids("all"):
        for f in sorted((ROOT / "data" / "sample" / "real" / t).glob("*.jsonl")):
            assert check_file(f, "real", t) == [], f
        for f in sorted((ROOT / "data" / "sample" / "synthetic" / t).glob("*.jsonl")):
            assert check_file(f, "synthetic", t) == [], f


def test_checkers_catch_problems():
    assert check_real({"id": "x", "text": "hi there", "label": "not_a_label"}, "trec")
    assert check_real({"id": "x", "text": "", "label": "human"}, "trec")
    assert check_real({"id": "x", "text": "who is it ?", "label": "human"}, "trec") == []
    assert check_synthetic({"id": "x", "text": "who is it ?", "label": "human"}, "trec")  # missing fields
