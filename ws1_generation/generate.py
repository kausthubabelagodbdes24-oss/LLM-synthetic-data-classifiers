"""WS1 (Harshita): generate synthetic training text with a small local model.

Setup once:  install Ollama (https://ollama.com), then `ollama pull llama3.2:3b`
After that everything runs offline.

    python -m ws1_generation.generate --setting few_shot --pilot --seed 13    # 10 per label pilot
    python -m ws1_generation.generate --setting few_shot --seed all          # full generation across repetitions
    python -m ws1_generation.generate --smoke-ollama                         # check Ollama status & model readiness
    python -m ws1_generation.generate --setting zero_shot --dry-run          # fake model, for testing

Resumable: stop it any time and run the same command again; resume is isolated to task x setting x repetition.
Writes data/synthetic/<task>/<setting>_seed<seed>.jsonl (or <setting>.jsonl in legacy mode)
with rows {id, text, label, setting, repetition, source_ids}.
Also writes sidecar <file>.provenance.json recording model, parameters, stats and demonstration IDs.
"""

import argparse
import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

from common import CFG, TASKS, labels, norm, read_jsonl, real_path, synth_path, write_jsonl

# ---------------- prompts (Li et al. 2023; Brown et al. 2020; Dai et al. 2023) ----------------

NUDGES = [
    "Pick specific, uncommon subjects.",
    "Mix short and longer examples.",
    "Use a different sentence structure each time.",
    "Avoid repeating the same opening words."
]

NON_EXAMPLE_PATTERNS = [
    re.compile(r"^(?:here (?:are|is)|sure,?|certainly,?|below (?:are|is)|following are|as requested)", re.I),
    re.compile(r"^(?:i hope|hope this helps|let me know|feel free|note:|please note)", re.I),
    re.compile(r"^(?:i\s+(?:cannot|can't|am unable|apologize|apologise)|sorry,?|apolog(?:y|ies|ize|ise)|as an ai)", re.I),
    re.compile(r"^(?:examples?:|questions?:|tweets?:|sentences?:)$", re.I),
]


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


# ---------------- model & reply parsing ----------------

def ask_ollama(prompt, timeout=120):
    url = f"{CFG['ollama_url']}/api/chat"
    r = requests.post(url, timeout=timeout, json={
        "model": CFG["model"],
        "stream": False,
        "format": "json",
        "options": {"temperature": CFG["temperature"]},
        "messages": [{"role": "user", "content": prompt}]
    })
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask_fake(prompt):
    """Stand-in for tests: obviously fake text, never use for results."""
    m = re.search(r"(\d+) different", prompt)
    n = int(m.group(1)) if m else 5
    words = ["river", "violin", "planet", "garden", "museum", "station"]
    # Check if TREC question style is requested
    is_trec = "question" in prompt.lower()
    examples = []
    for _ in range(n):
        w = random.choice(words)
        rand_num = random.randint(1000, 99999)
        if is_trec:
            examples.append(f"what is the fake fact {rand_num} about a {w} ?")
        else:
            examples.append(f"i feel fake emotion {rand_num} about a {w}")
    return json.dumps({"examples": examples})


def clean_markdown_fences(reply: str) -> str:
    """Extract inner content if wrapped in markdown code fences."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", reply)
    if m:
        return m.group(1).strip()
    return reply.strip()


def parse(reply, max_words=None):
    """List of cleaned strings from the model's reply; [] if unusable."""
    if not reply or not reply.strip():
        return []
    cleaned = clean_markdown_fences(reply)
    items = []

    # 1. Try structured JSON parsing
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            items = obj.get("examples", next(iter(obj.values()), []))
        elif isinstance(obj, list):
            items = obj
    except (json.JSONDecodeError, StopIteration, AttributeError, TypeError):
        # Try finding JSON substring
        sub_m = re.search(r"(\{[\s\S]*\"examples\"\s*:\s*\[[\s\S]*\][\s\S]*\}|\[[\s\S]*\])", cleaned)
        if sub_m:
            try:
                sub_obj = json.loads(sub_m.group(1))
                if isinstance(sub_obj, dict):
                    items = sub_obj.get("examples", [])
                elif isinstance(sub_obj, list):
                    items = sub_obj
            except Exception:
                items = []

    # 2. Fallback to line-by-line parsing if no JSON items extracted
    if not items or not isinstance(items, list):
        raw_lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", ln).strip() for ln in cleaned.splitlines()]
        items = [ln for ln in raw_lines if ln]

    # 3. Clean and filter candidates
    out = []
    for s in items:
        if not isinstance(s, str):
            continue
        text = " ".join(s.split()).strip('"\'')
        words = text.split()
        if len(words) < 2:
            continue
        if max_words is not None and len(words) > max_words:
            continue
        if any(pat.search(text) for pat in NON_EXAMPLE_PATTERNS):
            continue
        out.append(text)
    return out


def ask(prompt, fake, max_words=None, retries=3, stats=None):
    """Query model with retries; handles connection failures and malformed output."""
    for attempt in range(1, retries + 1):
        if stats is not None:
            stats["calls"] += 1
        try:
            raw = ask_fake(prompt) if fake else ask_ollama(prompt)
        except requests.RequestException as e:
            if stats is not None:
                stats["call_failures"] += 1
            print(f"    [Warning] Model call attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                return []
            time.sleep(1.0 * attempt)
            continue

        out = parse(raw, max_words=max_words)
        if out:
            if stats is not None:
                stats["candidates_received"] += len(out)
            return out
        else:
            if stats is not None:
                stats["parse_failures"] += 1
    return []


# ---------------- diagnostics & smoke check ----------------

def check_ollama_status():
    """Verify Ollama connectivity, model availability, and a smoke completion."""
    url = CFG.get("ollama_url", "http://localhost:11434")
    model = CFG.get("model", "llama3.2:3b")
    result = {
        "ollama_running": False,
        "model_available": False,
        "smoke_call_success": False,
        "message": ""
    }
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code != 200:
            result["message"] = f"Ollama returned HTTP {r.status_code} at {url}"
            return result
        result["ollama_running"] = True
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        model_found = any(m == model or m.startswith(f"{model}:") or m.startswith(model) for m in tags)
        result["model_available"] = model_found
        if not model_found:
            result["message"] = f"Model '{model}' not found in Ollama tags: {tags}. Run `ollama pull {model}`."
            return result
        test_reply = ask_ollama('Reply only with JSON: {"examples": ["what is the capital of france ?"]}', timeout=30)
        parsed = parse(test_reply)
        result["smoke_call_success"] = len(parsed) > 0
        if result["smoke_call_success"]:
            result["message"] = f"Ollama service and model '{model}' verified successfully."
        else:
            result["message"] = f"Smoke call succeeded but reply could not be parsed: {test_reply[:100]}"
        return result
    except requests.RequestException as e:
        result["message"] = f"Could not connect to Ollama at {url}: {e}. Ensure Ollama is running (`ollama serve`)."
        return result


# ---------------- generation ----------------

def generate_labelled(task, setting, target, fake, seed=None):
    """Generate zero-shot or few-shot examples for a task and repetition seed."""
    path = synth_path(task, setting, seed=seed)
    rows = read_jsonl(path)

    # Validate repetition isolation on existing rows
    if seed is not None:
        for r in rows:
            if r.get("repetition") is not None and r["repetition"] != seed:
                raise ValueError(
                    f"Repetition isolation mismatch in {path}: expected seed {seed}, found {r['repetition']}"
                )

    pool = read_jsonl(real_path(task, "pool"))
    if setting == "few_shot" and not pool:
        raise SystemExit(f"No example pool at {real_path(task, 'pool')}: run ws2_training.prepare_data first.")

    test_rows = read_jsonl(real_path(task, "test"))
    test_seen = {norm(r["text"]) for r in test_rows}
    seen = {norm(r["text"]) for r in rows}
    max_words = CFG["tasks"][task]["max_words"]

    stats = {
        "calls": 0, "call_failures": 0, "parse_failures": 0,
        "candidates_received": 0, "rejected_duplicate": 0,
        "rejected_test_leak": 0, "rejected_demo_leak": 0,
        "rejected_too_long": 0, "rejected_too_short": 0,
        "accepted_new": 0
    }

    demonstrations_used = set()
    for r in rows:
        demonstrations_used.update(r.get("source_ids", []))

    for label in labels(task):
        have = sum(r["label"] == label for r in rows)
        calls = 0
        max_calls = max(3 * target, 15)
        while have < target and calls < max_calls:
            calls += 1
            shown = []
            if setting == "few_shot":
                label_pool = [r for r in pool if r["label"] == label]
                if len(label_pool) < CFG["few_shot_k"]:
                    raise ValueError(
                        f"Insufficient demonstrations for {task} label {label}: "
                        f"have {len(label_pool)}, need {CFG['few_shot_k']}"
                    )
                # Seed-dependent deterministic demonstration selection per repetition
                rng_seed = f"{task}_{setting}_{seed}_{label}_{have}_{calls}"
                shown = random.Random(rng_seed).sample(label_pool, CFG["few_shot_k"])
                for s in shown:
                    demonstrations_used.add(s["id"])
                prompt = prompt_few_shot(task, label, CFG["per_call"], [r["text"] for r in shown], calls)
            else:
                prompt = prompt_zero_shot(task, label, CFG["per_call"], calls)

            candidates = ask(prompt, fake, max_words=max_words, stats=stats)
            shown_texts = {norm(r["text"]) for r in shown} if shown else set()
            new = []
            for text in candidates:
                if have >= target:
                    break
                n_text = norm(text)
                if n_text in seen:
                    stats["rejected_duplicate"] += 1
                    continue
                if n_text in test_seen:
                    stats["rejected_test_leak"] += 1
                    continue
                if n_text in shown_texts:
                    stats["rejected_demo_leak"] += 1
                    continue
                words = text.split()
                if len(words) > max_words:
                    stats["rejected_too_long"] += 1
                    continue
                if len(words) < 2:
                    stats["rejected_too_short"] += 1
                    continue

                seen.add(n_text)
                have += 1
                stats["accepted_new"] += 1

                row_id = (f"{task}.{setting}.seed{seed}.{label}.{have:04d}"
                          if seed is not None
                          else f"{task}.{setting}.{label}.{have:04d}")
                row = {
                    "id": row_id,
                    "text": text,
                    "label": label,
                    "setting": setting,
                    "repetition": seed,
                    "source_ids": [r["id"] for r in shown]
                }
                new.append(row)

            if new:
                write_jsonl(path, new, append=True)
                rows += new

        print(f"  {task} {setting} (seed={seed}) {label}: {have}/{target}")

    # Write provenance manifest sidecar
    prov_path = path.with_suffix(".provenance.json")
    provenance = {
        "task": task,
        "setting": setting,
        "repetition": seed,
        "model": CFG.get("model", "llama3.2:3b"),
        "temperature": CFG.get("temperature", 0.9),
        "target_per_label": target,
        "counts_by_label": {lab: sum(r["label"] == lab for r in rows) for lab in labels(task)},
        "total_count": len(rows),
        "is_complete": all(sum(r["label"] == lab for r in rows) >= target for lab in labels(task)),
        "distinct_demonstrations_count": len(demonstrations_used),
        "distinct_demonstration_ids": sorted(demonstrations_used),
        "stats": stats,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return rows, provenance


def generate_auggpt(task, fake, seed=None):
    """Generate AugGPT paraphrases for lowres sources."""
    path = synth_path(task, "auggpt", seed=seed)
    existing = read_jsonl(path)
    done = {r["source_ids"][0] for r in existing}

    # Find source files
    if seed is not None:
        source_files = [real_path(task, f"lowres_seed{seed}")]
    else:
        source_files = sorted(real_path(task, "x").parent.glob("lowres_seed*.jsonl"))

    seeds = {r["id"]: r for f in source_files for r in read_jsonl(f)}
    if not seeds:
        raise SystemExit(f"No lowres_seed*.jsonl for {task}: run ws2_training.prepare_data first.")

    test_rows = read_jsonl(real_path(task, "test"))
    test_seen = {norm(r["text"]) for r in test_rows}

    for sid, s_row in sorted(seeds.items()):
        if sid in done:
            continue
        paraphrases = ask(prompt_auggpt(s_row["text"], CFG["auggpt_k"]), fake)[:CFG["auggpt_k"]]
        valid = []
        for j, p in enumerate(paraphrases, 1):
            if norm(p) in test_seen or norm(p) == norm(s_row["text"]):
                continue
            aug_id = f"{sid}.aug{j}" if seed is None else f"{sid}.seed{seed}.aug{j}"
            valid.append({
                "id": aug_id,
                "text": p,
                "label": s_row["label"],
                "setting": "auggpt",
                "repetition": seed,
                "source_ids": [sid]
            })
        if valid:
            write_jsonl(path, valid, append=True)
    print(f"  {task} auggpt (seed={seed}): {len(seeds)} seed examples processed")


def inspect_pilot(task, setting, seed=None):
    """Print a concise inspection summary of pilot examples."""
    path = synth_path(task, setting, seed=seed)
    rows = read_jsonl(path)
    if not rows:
        print(f"No pilot rows found at {path}")
        return
    print(f"\n--- Pilot inspection: {task} {setting} (seed={seed}) [{len(rows)} rows] ---")
    for lab in labels(task):
        lab_rows = [r for r in rows if r["label"] == lab]
        print(f"\nLabel: {lab} ({len(lab_rows)} examples):")
        for r in lab_rows[:2]:
            demo_info = f" [demos: {len(r.get('source_ids', []))}]" if setting == "few_shot" else ""
            print(f"  - {r['text']}{demo_info}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["zero_shot", "few_shot", "auggpt", "all"], help="generation setting")
    ap.add_argument("--task", default="all", choices=["all"] + TASKS)
    ap.add_argument("--seed", default=None, help="repetition seed (e.g. 13, 42, 7, or 'all')")
    ap.add_argument("--pilot", action="store_true", help="only 10 per label (pilot mode)")
    ap.add_argument("--target", type=int, default=None, help="explicit target per label")
    ap.add_argument("--dry-run", action="store_true", help="fake model, for testing")
    ap.add_argument("--smoke-ollama", action="store_true", help="verify local Ollama service and model readiness")
    ap.add_argument("--inspect", action="store_true", help="inspect sample pilot rows across tasks/settings")
    args = ap.parse_args(argv)

    if args.smoke_ollama:
        print("Checking Ollama service and model status...")
        res = check_ollama_status()
        print(f"Status: {res['message']}")
        if not res["smoke_call_success"]:
            sys.exit(1)
        sys.exit(0)

    if args.inspect:
        for t in TASKS if args.task == "all" else [args.task]:
            for s in ["zero_shot", "few_shot"]:
                inspect_pilot(t, s, seed=int(args.seed) if args.seed and args.seed != "all" else None)
        return

    if not args.setting:
        ap.error("--setting is required unless --smoke-ollama or --inspect is given")

    # Determine seeds to run
    if args.seed == "all":
        seeds_to_run = [int(s) for s in CFG["seeds"]]
    elif args.seed is not None:
        seeds_to_run = [int(args.seed)]
    else:
        seeds_to_run = [None]  # Legacy unseeded mode

    # Determine target per label
    if args.target is not None:
        target = args.target
    elif args.pilot:
        target = 10
    else:
        target = CFG["per_label"]

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    settings_to_run = ["zero_shot", "few_shot", "auggpt"] if args.setting == "all" else [args.setting]

    for seed in seeds_to_run:
        for task in tasks_to_run:
            for setting in settings_to_run:
                if setting == "auggpt":
                    generate_auggpt(task, args.dry_run, seed=seed)
                else:
                    generate_labelled(task, setting, target, args.dry_run, seed=seed)


if __name__ == "__main__":
    main()
