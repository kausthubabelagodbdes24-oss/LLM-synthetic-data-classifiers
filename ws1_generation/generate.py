"""WS1 (Kausthuba): generate synthetic training text with a small local model.

Setup once:  install Ollama (https://ollama.com), then  `ollama pull llama3.2:3b`
After that everything runs offline.

    python -m ws1_generation.generate --setting few_shot --pilot     # 10 per label, check by eye
    python -m ws1_generation.generate --setting few_shot             # then zero_shot, then auggpt
    python -m ws1_generation.generate --setting zero_shot --dry-run  # fake model, no Ollama needed

Resumable: stop it any time and run the same command again.
Writes data/synthetic/<task>/<setting>.jsonl with rows {id, text, label, setting, source_ids}.
AugGPT needs WS2's data/real/<task>/lowres_seed*.jsonl first.
"""

import argparse
import json
import random
import re

import requests

from common import CFG, TASKS, labels, norm, read_jsonl, real_path, synth_path, write_jsonl

# ---------------- prompts (Li et al. 2023; Brown et al. 2020; Dai et al. 2023) ----------------

NUDGES = ["Pick specific, uncommon subjects.", "Mix short and longer examples.",
          "Use a different sentence structure each time.", "Avoid repeating the same opening words."]


def prompt_zero_shot(task, label, n, i):
    t = CFG["tasks"][task]
    return (f"You are building a dataset of {t['context']}.\n"
            f"Write {n} different examples of {t['style']}.\n"
            f'Every example must have the label "{label}", meaning: {t["labels"][label]}.\n'
            f"At most {t['max_words']} words each. {NUDGES[i % len(NUDGES)]}\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')


def prompt_few_shot(task, label, n, examples, i):
    shown = "\n".join(f"- {json.dumps(e)}" for e in examples)
    return (prompt_zero_shot(task, label, n, i).replace(
        "Reply only", f"Here are real examples with this label:\n{shown}\n"
        "Write NEW examples in the same style. Do not copy or paraphrase these.\nReply only"))


def prompt_auggpt(text, k):
    return (f"Please rephrase the following sentence {k} different ways, keeping its meaning:\n"
            f"{json.dumps(text)}\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')


# ---------------- model ----------------

def ask_ollama(prompt):
    r = requests.post(f"{CFG['ollama_url']}/api/chat", timeout=300, json={
        "model": CFG["model"], "stream": False, "format": "json",
        "options": {"temperature": CFG["temperature"]},
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask_fake(prompt):
    """Stand-in for tests: obviously fake text, never use for results."""
    n = int(re.search(r"(\d+) different", prompt).group(1))
    word = random.choice(["river", "violin", "planet", "garden", "museum", "station"])
    return json.dumps({"examples": [f"fake example {random.randint(0, 99999)} about a {word}" for _ in range(n)]})


def parse(reply):
    """List of strings from the model's reply; [] if unusable."""
    try:
        obj = json.loads(reply)
        items = obj.get("examples", next(iter(obj.values()), [])) if isinstance(obj, dict) else obj
        return [" ".join(s.split()).strip('"') for s in items if isinstance(s, str) and len(s.split()) >= 2]
    except (json.JSONDecodeError, StopIteration, AttributeError, TypeError):
        lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", ln).strip() for ln in reply.splitlines()]
        return [ln for ln in lines if len(ln.split()) >= 2]


def ask(prompt, fake, retries=3):
    for _ in range(retries):
        out = parse(ask_fake(prompt) if fake else ask_ollama(prompt))
        if out:
            return out
    return []


# ---------------- generation ----------------

def generate_labelled(task, setting, target, fake):
    path = synth_path(task, setting)
    rows = read_jsonl(path)
    seen = {norm(r["text"]) for r in rows} | {norm(r["text"]) for r in read_jsonl(real_path(task, "test"))}
    pool = read_jsonl(real_path(task, "pool"))
    if setting == "few_shot" and not pool:
        raise SystemExit(f"No example pool at {real_path(task, 'pool')}: run ws2_training.prepare_data first.")
    for label in labels(task):
        have = sum(r["label"] == label for r in rows)
        calls = 0
        while have < target and calls < 3 * target:
            calls += 1
            shown = []
            if setting == "few_shot":
                shown = random.Random(f"{label}{have}{calls}").sample(
                    [r for r in pool if r["label"] == label], CFG["few_shot_k"])
                prompt = prompt_few_shot(task, label, CFG["per_call"], [r["text"] for r in shown], calls)
            else:
                prompt = prompt_zero_shot(task, label, CFG["per_call"], calls)
            new = []
            for text in ask(prompt, fake):
                if have >= target or norm(text) in seen:
                    continue
                seen.add(norm(text))
                have += 1
                new.append({"id": f"{task}.{setting}.{label}.{have:04d}", "text": text, "label": label,
                            "setting": setting, "source_ids": [r["id"] for r in shown]})
            write_jsonl(path, new, append=True)
            rows += new
        print(f"  {task} {setting} {label}: {have}/{target}")


def generate_auggpt(task, fake):
    path = synth_path(task, "auggpt")
    done = {r["source_ids"][0] for r in read_jsonl(path)}
    seeds = {r["id"]: r for f in sorted(real_path(task, "x").parent.glob("lowres_seed*.jsonl")) for r in read_jsonl(f)}
    if not seeds:
        raise SystemExit(f"No lowres_seed*.jsonl for {task}: run ws2_training.prepare_data first.")
    for sid, seed in sorted(seeds.items()):
        if sid in done:
            continue
        paraphrases = ask(prompt_auggpt(seed["text"], CFG["auggpt_k"]), fake)[:CFG["auggpt_k"]]
        write_jsonl(path, [{"id": f"{sid}.aug{j}", "text": p, "label": seed["label"], "setting": "auggpt",
                            "source_ids": [sid]} for j, p in enumerate(paraphrases, 1)], append=True)
    print(f"  {task} auggpt: {len(seeds)} seed examples paraphrased")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=["zero_shot", "few_shot", "auggpt"])
    ap.add_argument("--task", default="all", choices=["all"] + TASKS)
    ap.add_argument("--pilot", action="store_true", help="only 10 per label")
    ap.add_argument("--dry-run", action="store_true", help="fake model, for testing")
    args = ap.parse_args(argv)
    for task in TASKS if args.task == "all" else [args.task]:
        if args.setting == "auggpt":
            generate_auggpt(task, args.dry_run)
        else:
            generate_labelled(task, args.setting, 10 if args.pilot else CFG["per_label"], args.dry_run)


if __name__ == "__main__":
    main()
