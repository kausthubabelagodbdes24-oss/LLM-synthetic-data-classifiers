"""Runs the whole pipeline on fake sample data with the fake model and offline embedder."""

import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_whole_pipeline(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "data" / "sample" / "real", tmp_path / "data" / "real")
    monkeypatch.setenv("SYNTHTEXT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHTEXT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("SYNTHTEXT_EMBEDDER", "hashing")

    from common import CFG, labels, read_jsonl, synth_path
    from ws1_generation import generate
    from ws2_training import train
    from ws3_evaluation import analyze

    monkeypatch.setitem(CFG, "per_label", 12)
    for setting in ("zero_shot", "few_shot", "auggpt"):
        generate.main(["--setting", setting, "--dry-run"])
    generate.main(["--setting", "few_shot", "--dry-run"])          # resumable: adds nothing
    few = read_jsonl(synth_path("emotion", "few_shot"))
    assert len(few) == 12 * len(labels("emotion"))
    assert all(len(r["source_ids"]) == CFG["few_shot_k"] for r in few)

    train.main([])
    rows = list(csv.DictReader(open(tmp_path / "results" / "runs.csv")))
    assert len(rows) == 2 * len(CFG["conditions"]) * len(CFG["seeds"])
    analyze.main()
    assert (tmp_path / "results" / "gap.png").exists()


def test_parse_handles_messy_replies():
    from ws1_generation.generate import parse
    assert parse('{"examples": ["one two", "three four"]}') == ["one two", "three four"]
    assert parse("1. first example\n2. second example") == ["first example", "second example"]
    assert parse("sorry") == []
