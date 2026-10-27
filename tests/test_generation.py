from collections import Counter

from synthtext import config
from synthtext.contracts import check_file
from synthtext.io import read_jsonl
from synthtext.paths import synthetic_dir
from workstreams.ws1_generation import filter as filt
from workstreams.ws1_generation import generate, prompts
from workstreams.ws1_generation.llm_client import FakeClient


def test_parse_examples_handles_common_outputs():
    assert prompts.parse_examples('{"examples": ["one two", "three four"]}') == (["one two", "three four"], "json")
    got, how = prompts.parse_examples('<think>hmm</think>\nSure! {"examples": ["a b c"]} hope this helps')
    assert got == ["a b c"] and how == "json-embedded"
    got, how = prompts.parse_examples("1. first example here\n2. second example here")
    assert got == ["first example here", "second example here"] and how == "lines"
    assert prompts.parse_examples("Sorry, I can't help with that.") == ([], "failed")


def test_few_shot_prompt_shows_examples_and_forbids_copying():
    msgs = prompts.few_shot(config.task("emotion"), "joy", 10, ["i feel great", "i feel glad"], 0)
    user = msgs[-1]["content"]
    assert "i feel great" in user and "do not copy" in user and '"joy"' in user


def test_generate_is_resumable_and_filter_writes_valid_file(sample_env):
    gen_cfg = config.generation()
    client = FakeClient(fail_every=4)  # every 4th call fails -> exercises retries
    generate.generate_labelled("trec", "few_shot", 5, gen_cfg, client)
    raw = read_jsonl(generate.raw_path("trec", "few_shot"))
    assert Counter(r["label"] for r in raw) == {lab: 5 for lab in config.labels("trec")}
    assert all(len(r["source_ids"]) == gen_cfg["few_shot_k"] for r in raw)

    generate.generate_labelled("trec", "few_shot", 5, gen_cfg, FakeClient())  # already complete
    assert len(read_jsonl(generate.raw_path("trec", "few_shot"))) == len(raw)

    filt.main(["--task", "trec", "--setting", "few_shot"])
    out = synthetic_dir("trec") / "few_shot.jsonl"
    assert read_jsonl(out) and check_file(out, "synthetic", "trec") == []


def test_auggpt_paraphrases_each_lowres_seed(sample_env):
    generate.generate_auggpt("sst5", config.generation(), FakeClient(), pilot=False)
    rows = read_jsonl(generate.raw_path("sst5", "auggpt"))
    k = config.generation()["auggpt_paraphrases"]
    per_source = Counter(r["source_ids"][0] for r in rows)
    assert per_source and set(per_source.values()) == {k}
