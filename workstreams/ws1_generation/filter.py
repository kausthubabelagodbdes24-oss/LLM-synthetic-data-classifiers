"""Clean a raw synthetic file into the file everyone else uses.

    python -m workstreams.ws1_generation.filter --task all --setting few_shot

Drops, in order: empty / too short / far too long texts; exact duplicates
(case and spacing ignored); exact copies of any real text; texts with cosine
>= test_overlap_cosine to a test item (leakage guard); few-shot texts with
cosine >= copy_cosine to an example they were shown (copies). AugGPT
paraphrases are meant to be close to their source, so only exact copies of the
source are dropped there; their closeness is reported, not filtered.

Writes data/synthetic/<task>/<setting>.jsonl and <setting>.filter.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from synthtext import config
from synthtext.embeddings import embed, resolve_model
from synthtext.io import normalise, read_jsonl, write_jsonl
from synthtext.paths import real_dir, synthetic_dir
from workstreams.ws1_generation.generate import SETTINGS, raw_path


def max_cosine(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """For each row of a, the highest cosine to any row of b (inputs L2-normalised)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a))
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = (a[i:i + chunk] @ b.T).max(axis=1)
    return out


def filter_records(task_id: str, setting: str, rows: list[dict], fcfg: dict) -> tuple[list[dict], dict]:
    task_cfg = config.task(task_id)
    real = {name: read_jsonl(real_dir(task_id) / f"{name}.jsonl") for name in ("train", "pool", "test")}
    real_norm = {normalise(r["text"]) for split in real.values() for r in split}
    source_text = {r["id"]: r["text"] for split in real.values() for r in split}
    max_words = int(task_cfg["max_words"] * 2)
    dropped: Counter = Counter()

    kept, seen = [], set()
    for r in rows:
        text = " ".join(r["text"].split()).strip().strip('"').strip()
        n_words = len(text.split())
        key = normalise(text)
        if n_words < fcfg["min_words"]:
            dropped["too_short"] += 1
        elif n_words > max_words:
            dropped["too_long"] += 1
        elif r["label"] not in task_cfg["labels"]:
            dropped["bad_label"] += 1
        elif key in seen:
            dropped["duplicate"] += 1
        elif key in real_norm:
            dropped["copy_of_real_text"] += 1
        else:
            seen.add(key)
            kept.append({**r, "text": text})

    model = fcfg["embedding_model"]
    report: dict = {}
    if kept and real["test"]:
        syn_vecs = embed([r["text"] for r in kept], model)
        test_vecs = embed([r["text"] for r in real["test"]], model)
        near_test = max_cosine(syn_vecs, test_vecs)

        copy_sim = np.zeros(len(kept))
        has_source = [i for i, r in enumerate(kept) if r["source_ids"]]
        if has_source:
            src_texts = sorted({source_text[s] for i in has_source for s in kept[i]["source_ids"] if s in source_text})
            src_index = {t: j for j, t in enumerate(src_texts)}
            src_vecs = embed(src_texts, model) if src_texts else np.zeros((0, syn_vecs.shape[1]))
            for i in has_source:
                idx = [src_index[source_text[s]] for s in kept[i]["source_ids"] if s in source_text]
                if idx:
                    copy_sim[i] = float((src_vecs[idx] @ syn_vecs[i]).max())

        final = []
        for i, r in enumerate(kept):
            if near_test[i] >= fcfg["test_overlap_cosine"]:
                dropped["near_test_item"] += 1
            elif setting == "few_shot" and copy_sim[i] >= fcfg["copy_cosine"]:
                dropped["copied_shown_example"] += 1
            else:
                final.append(r)
        if has_source:
            report["mean_cosine_to_source"] = round(float(copy_sim[has_source].mean()), 4)
            report["share_cosine_to_source_ge_copy"] = round(
                float((copy_sim[has_source] >= fcfg["copy_cosine"]).mean()), 4)
        kept = final
    elif not real["test"]:
        print(f"  warning: no test file for {task_id}; skipped the embedding checks")

    report.update({
        "task": task_id,
        "setting": setting,
        "raw": len(rows),
        "kept": len(kept),
        "dropped": dict(dropped),
        "kept_per_label": dict(Counter(r["label"] for r in kept)),
        "embedding_model": resolve_model(model),
    })
    return kept, report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all")
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    args = ap.parse_args(argv)
    fcfg = config.generation()["filter"]

    for task_id in config.task_ids(args.task):
        rows = read_jsonl(raw_path(task_id, args.setting))
        if not rows:
            print(f"[{task_id}] no raw {args.setting} file yet; skipping")
            continue
        kept, report = filter_records(task_id, args.setting, rows, fcfg)
        out_dir = synthetic_dir(task_id)
        write_jsonl(out_dir / f"{args.setting}.jsonl", kept)
        (out_dir / f"{args.setting}.filter.json").write_text(json.dumps(report, indent=2))
        print(f"[{task_id}] {args.setting}: kept {report['kept']}/{report['raw']} "
              f"dropped {report['dropped'] or 'none'}")


if __name__ == "__main__":
    main()
