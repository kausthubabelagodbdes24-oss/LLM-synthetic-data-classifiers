"""Generate synthetic training text with the open LLM.

    # Oct 29 pilot: 20 per label, check by eye before the full run
    python -m workstreams.ws1_generation.generate --task all --setting few_shot --pilot
    # full run (resumable: stop with Ctrl-C and run the same command again)
    python -m workstreams.ws1_generation.generate --task all --setting few_shot
    python -m workstreams.ws1_generation.generate --task all --setting zero_shot
    # AugGPT needs the low-resource seed files from WS2 (data/real/<task>/lowres_seed*.jsonl)
    python -m workstreams.ws1_generation.generate --task all --setting auggpt
    # no GPU? check the plumbing with the fake model:
    python -m workstreams.ws1_generation.generate --task trec --setting few_shot --pilot --dry-run

Raw output goes to data/synthetic/<task>/raw/<setting>.jsonl; run `filter` afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from synthtext import config
from synthtext.io import append_jsonl, read_jsonl
from synthtext.paths import real_dir, synthetic_dir
from workstreams.ws1_generation import prompts
from workstreams.ws1_generation.llm_client import LLMError, make_client

SETTINGS = ("zero_shot", "few_shot", "auggpt")


def raw_path(task_id: str, setting: str) -> Path:
    return synthetic_dir(task_id) / "raw" / f"{setting}.jsonl"


def stats_path(task_id: str, setting: str) -> Path:
    return synthetic_dir(task_id) / "raw" / f"{setting}.stats.json"


class Stats:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("calls", 0)
        self.data.setdefault("failed_calls", 0)
        self.data.setdefault("parse", {})
        self.data.setdefault("seconds", 0.0)

    def record(self, how: str, seconds: float, failed: bool) -> None:
        self.data["calls"] += 1
        self.data["failed_calls"] += int(failed)
        self.data["parse"][how] = self.data["parse"].get(how, 0) + 1
        self.data["seconds"] = round(self.data["seconds"] + seconds, 1)

    def save(self, **extra) -> None:
        self.data.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))


def call_with_retries(client, messages, meta, max_retries, stats) -> list[str]:
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            raw = client.complete(messages, schema=prompts.EXAMPLES_SCHEMA, meta=meta)
            examples, how = prompts.parse_examples(raw)
        except LLMError as exc:
            examples, how = [], "error"
            print(f"    ! {exc}", file=sys.stderr)
            if "Cannot reach" in str(exc):
                raise
        stats.record(how, time.time() - start, failed=not examples)
        if examples:
            return examples
        print(f"    retry {attempt}/{max_retries} ({how})", file=sys.stderr)
    return []


def generate_labelled(task_id: str, setting: str, target: int, gen_cfg: dict, client, labels_only=None) -> None:
    task_cfg = config.task(task_id)
    out = raw_path(task_id, setting)
    existing = read_jsonl(out)
    version = gen_cfg["prompt_version"]
    counts = Counter(r["label"] for r in existing if r.get("prompt_version") == version)
    stale = sum(1 for r in existing if r.get("prompt_version") != version)
    if stale:
        print(f"  note: {stale} rows from an older prompt_version are kept but not counted")

    pool_by_label: dict[str, list[dict]] = defaultdict(list)
    if setting == "few_shot":
        pool = read_jsonl(real_dir(task_id) / "pool.jsonl")
        if not pool:
            sys.exit(f"No example pool at {real_dir(task_id) / 'pool.jsonl'}. WS2 runs prepare_data first.")
        for r in pool:
            pool_by_label[r["label"]].append(r)

    stats = Stats(stats_path(task_id, setting))
    per_call = gen_cfg["examples_per_call"]
    k = gen_cfg["few_shot_k"]
    model_name = client.describe()

    for label in config.labels(task_id):
        if labels_only and label not in labels_only:
            continue
        call_index = 0
        while counts[label] < target:
            need = min(per_call, target - counts[label])
            meta = {"label": label, "n": per_call, "task": task_id}
            shown: list[dict] = []
            if setting == "zero_shot":
                messages = prompts.zero_shot(task_cfg, label, per_call, call_index + counts[label])
            else:
                rng = random.Random(f"{task_id}|{label}|{counts[label]}|{call_index}")
                candidates = pool_by_label[label]
                shown = rng.sample(candidates, min(k, len(candidates)))
                messages = prompts.few_shot(
                    task_cfg, label, per_call, [r["text"] for r in shown], call_index + counts[label]
                )
            examples = call_with_retries(client, messages, meta, gen_cfg["max_retries"], stats)
            call_index += 1
            if not examples:
                if call_index > 3 * (target // per_call + 1):
                    print(f"  ! giving up on {label}: too many failed calls", file=sys.stderr)
                    break
                continue
            rows = []
            for text in examples[:need]:
                counts[label] += 1
                rows.append({
                    "id": f"{task_id}.{setting}.{label}.{version}.{counts[label]:05d}",
                    "text": text,
                    "label": label,
                    "setting": setting,
                    "source_ids": [r["id"] for r in shown],
                    "model": model_name,
                    "prompt_version": version,
                })
            append_jsonl(out, rows)
            print(f"  {task_id}/{setting}/{label}: {counts[label]}/{target}", flush=True)
        stats.save(model=model_name, prompt_version=version)
    stats.save(model=model_name, prompt_version=version)


def generate_auggpt(task_id: str, gen_cfg: dict, client, pilot: bool) -> None:
    seed_files = sorted(real_dir(task_id).glob("lowres_seed*.jsonl"))
    if not seed_files:
        sys.exit(f"No lowres_seed*.jsonl in {real_dir(task_id)}. WS2 runs prepare_data first.")
    seeds: dict[str, dict] = {}
    for f in seed_files:
        for r in read_jsonl(f):
            seeds[r["id"]] = r
    out = raw_path(task_id, "auggpt")
    version = gen_cfg["prompt_version"]
    done = {r["source_ids"][0] for r in read_jsonl(out) if r.get("prompt_version") == version}
    todo = [s for sid, s in sorted(seeds.items()) if sid not in done]
    if pilot:
        todo = todo[: len(config.labels(task_id)) * 2]
    k = gen_cfg["auggpt_paraphrases"]
    stats = Stats(stats_path(task_id, "auggpt"))
    model_name = client.describe()
    print(f"  {task_id}/auggpt: {len(done)} seeds done, {len(todo)} to go")
    for i, seed in enumerate(todo, start=1):
        meta = {"label": seed["label"], "n": k, "task": task_id, "source_text": seed["text"]}
        examples = call_with_retries(client, prompts.auggpt(seed["text"], k), meta, gen_cfg["max_retries"], stats)
        rows = [{
            "id": f"{seed['id']}.aug.{version}.{j}",
            "text": text,
            "label": seed["label"],
            "setting": "auggpt",
            "source_ids": [seed["id"]],
            "model": model_name,
            "prompt_version": version,
        } for j, text in enumerate(examples[:k], start=1)]
        append_jsonl(out, rows)
        if i % 10 == 0 or i == len(todo):
            print(f"  {task_id}/auggpt: {i}/{len(todo)} seeds", flush=True)
            stats.save(model=model_name, prompt_version=version)
    stats.save(model=model_name, prompt_version=version)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", help="all, or comma-separated task ids")
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--pilot", action="store_true", help="only pilot_per_label per label")
    ap.add_argument("--per-label", type=int, default=None, help="override per_label from the config")
    ap.add_argument("--labels", default=None, help="comma-separated subset of labels")
    ap.add_argument("--model", default=None, help="override the model (e.g. the backup)")
    ap.add_argument("--dry-run", action="store_true", help="use the fake model (no GPU needed)")
    args = ap.parse_args(argv)

    gen_cfg = config.generation()
    client = make_client(gen_cfg, dry_run=args.dry_run, model=args.model)
    target = args.per_label or (gen_cfg["pilot_per_label"] if args.pilot else gen_cfg["per_label"])
    labels_only = set(args.labels.split(",")) if args.labels else None

    for task_id in config.task_ids(args.task):
        print(f"[{task_id}] {args.setting} with {client.describe()}")
        if args.setting == "auggpt":
            generate_auggpt(task_id, gen_cfg, client, pilot=args.pilot)
        else:
            generate_labelled(task_id, args.setting, target, gen_cfg, client, labels_only)
    print("done. Next: python -m workstreams.ws1_generation.filter --task", args.task, "--setting", args.setting)


if __name__ == "__main__":
    main()
