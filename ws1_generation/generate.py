"""WS1 (Harshita): generate synthetic training text with a small local model.

Setup once:  install Ollama (https://ollama.com), then `ollama pull llama3.2:3b`
After that everything runs offline.

    python -m ws1_generation.generate --setting few_shot --pilot --seed 13    # 10 per label pilot
    python -m ws1_generation.generate --setting few_shot --seed all          # full generation across repetitions
    python -m ws1_generation.generate --smoke-ollama                         # check Ollama status & model readiness
    python -m ws1_generation.generate --setting zero_shot --dry-run          # fake model, for testing
    python -m ws1_generation.generate --setting auggpt --seed all            # six paraphrases per exact source
    python -m ws1_generation.generate --setting auggpt --seed all --verify   # fail-closed AugGPT handoff check

Resumable: stop it any time and run the same command again; resume is isolated to task x setting x repetition.
Writes real output under data/synthetic/ and --dry-run fixtures under data/synthetic-fixtures/.
Rows and sidecars identify their mode and immutable generation fingerprint. Sidecars preserve
model digest, prompt/options/source identity, run timing, accepted IDs and session history.
"""

import argparse
import datetime
import hashlib
import inspect
import json
import os
import random
import re
import sys
import time
import uuid
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

ROW_SCHEMA_VERSION = 2
PROMPT_PROTOCOL_VERSION = "issue-6-v2"


def prompt_zero_shot(task, label, n, i):
    t = CFG["tasks"][task]
    if task == "trec":
        boundaries = {
            "abbreviation": "Ask explicitly for an acronym's expansion or a shortened form. "
                            "Do not ask for a general definition or explanation.",
            "entity": "The answer must name a non-human thing, object, product, animal or substance. "
                      "Do not ask for a definition, acronym expansion or chemical symbol.",
            "description": "Ask for an explanation, definition, reason or how something works. "
                           "Do not ask what an acronym stands for.",
            "human": "The answer must name a person, team, group or organisation. "
                     "Mentioning a person does not suffice when the requested answer is their work or creation.",
            "location": "The answer must be a geographic place such as a city, country, mountain or river. "
                        "Do not ask for a person, definition, count, age or acronym expansion.",
            "numeric": "The answer must be a number, count, date, age, duration or measurement. "
                       "Do not ask for a product name or a place name.",
        }[label] + " Ask open questions, not yes/no questions."
    else:
        boundaries = {
            "sadness": "Clearly express feeling sad, lonely, hurt or low in a concrete situation.",
            "joy": "Clearly express happiness, delight, contentment or excitement. "
                   "The dominant feeling should be happiness rather than affection toward someone.",
            "love": "Clearly express affection, tenderness or care toward someone or something. "
                    "A good event or feeling happy alone is not sufficient.",
            "anger": "Clearly express anger, irritation, frustration or annoyance about a situation. "
                     "Do not rely on a situation alone without expressing the writer's reaction.",
            "fear": "Clearly express feeling afraid, anxious, nervous or worried about a threat or uncertainty.",
            "surprise": "Clearly express being amazed, shocked or caught off guard by an unexpected event. "
                        "Mentioning a surprise party alone does not establish the writer's reaction.",
        }[label] + " Make this the clear dominant emotion; use varied situations and natural wording."
    return (f"You are building a dataset of {t['context']}.\n"
            f"Write {n} different examples of {t['style']}.\n"
            f'Every example must have the label "{label}", meaning: {t["labels"][label]}.\n'
            f"Requirements for this label: {boundaries}\n"
            f"At most {t['max_words']} words each. {NUDGES[i % len(NUDGES)]}\n"
            "Check each candidate against the requested label before returning it. "
            "Return only examples that clearly fit; do not include answers or explanations.\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')


def prompt_few_shot(task, label, n, examples, i):
    shown = "\n".join(f"- {json.dumps(e)}" for e in examples)
    return (prompt_zero_shot(task, label, n, i).replace(
        "Reply only", f"Here are real examples with this label:\n{shown}\n"
        "Use these for style, while following the requested label's requirements above.\n"
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
    structured = False

    # 1. Try structured JSON parsing
    try:
        obj = json.loads(cleaned)
        structured = True
    except json.JSONDecodeError:
        # Try finding JSON substring
        sub_m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if sub_m:
            try:
                obj = json.loads(sub_m.group(1))
                structured = True
            except json.JSONDecodeError:
                return []
        elif cleaned.startswith(("{", "[")):
            return []

    # A structured empty/error reply is not a line of training text.
    if structured:
        if isinstance(obj, dict):
            items = obj.get("examples", [])
        elif isinstance(obj, list):
            items = obj
        else:
            return []
        if not isinstance(items, list):
            return []
    else:
        # Preserve the legacy plain/numbered-line reply format.
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
    started_clock = time.perf_counter()
    url = CFG.get("ollama_url", "http://localhost:11434")
    model = CFG.get("model", "llama3.2:3b")
    result = {
        "ollama_running": False,
        "model_available": False,
        "smoke_call_success": False,
        "configured_model": model,
        "model_identity": None,
        "elapsed_seconds": None,
        "message": ""
    }
    def finish():
        result["elapsed_seconds"] = round(time.perf_counter() - started_clock, 6)
        return result

    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code != 200:
            result["message"] = f"Ollama returned HTTP {r.status_code} at {url}"
            return finish()
        result["ollama_running"] = True
        available_models = r.json().get("models", [])
        matched = next((entry for entry in available_models
                        if entry.get("name") == model or entry.get("model") == model), None)
        result["model_available"] = matched is not None
        if not matched:
            tags = [entry.get("name", "") for entry in available_models]
            result["message"] = f"Model '{model}' not found in Ollama tags: {tags}. Run `ollama pull {model}`."
            return finish()
        result["model_identity"] = {
            "provider": "ollama",
            "configured_tag": model,
            "resolved_name": matched.get("model") or matched.get("name"),
            "digest": matched.get("digest"),
        }
        if not result["model_identity"]["digest"]:
            result["message"] = f"Ollama did not report a digest for configured model '{model}'."
            return finish()
        test_reply = ask_ollama('Reply only with JSON: {"examples": ["what is the capital of france ?"]}', timeout=30)
        parsed = parse(test_reply)
        result["smoke_call_success"] = len(parsed) > 0
        if result["smoke_call_success"]:
            result["message"] = f"Ollama service and model '{model}' verified successfully."
        else:
            result["message"] = f"Smoke call succeeded but reply could not be parsed: {test_reply[:100]}"
        return finish()
    except requests.RequestException as e:
        result["message"] = f"Could not connect to Ollama at {url}: {e}. Ensure Ollama is running (`ollama serve`)."
        return finish()


def resolve_model_identity():
    """Resolve the configured Ollama tag to the immutable digest used for generation."""
    model = CFG.get("model", "llama3.2:3b")
    response = requests.get(f"{CFG['ollama_url']}/api/tags", timeout=5)
    response.raise_for_status()
    models = response.json().get("models", [])
    for available in models:
        if available.get("name") == model or available.get("model") == model:
            digest = available.get("digest")
            if not digest:
                raise RuntimeError(f"Ollama did not report a digest for configured model {model!r}")
            return {
                "provider": "ollama",
                "configured_tag": model,
                "resolved_name": available.get("model") or available.get("name"),
                "digest": digest,
            }
    raise RuntimeError(f"Configured model {model!r} is not available in Ollama")


# Generators approved for full generation, by configured tag. config.yaml names the tag; the
# immutable digest is recorded only in ADR 0011 (31 Oct 2026), so it is pinned here. Adding or
# changing an entry requires a recorded lead decision.
APPROVED_GENERATORS = {
    "qwen3:4b": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
}

# The only identity fake (--dry-run) artifacts may carry; it can never satisfy a real target.
FIXTURE_MODEL_IDENTITY = {
    "provider": "fixture",
    "configured_tag": None,
    "resolved_name": "synthetic-fake-model",
    "digest": "fixture-v1",
}


def require_approved_model(model_identity, mode):
    """Raise ValueError unless the identity is the approved generator (real) or the fixture (fake)."""
    if mode == "fake":
        if model_identity != FIXTURE_MODEL_IDENTITY:
            raise ValueError(f"fake artifact carries a non-fixture model identity {model_identity!r}")
        return
    tag = CFG["model"]
    approved_digest = APPROVED_GENERATORS.get(tag)
    if approved_digest is None:
        raise ValueError(f"configured model {tag!r} has no approved digest; record a lead decision first")
    identity = model_identity if isinstance(model_identity, dict) else {}
    digest = str(identity.get("digest") or "").removeprefix("sha256:")
    if (identity.get("provider") != "ollama" or identity.get("configured_tag") != tag
            or identity.get("resolved_name") != tag or digest != approved_digest):
        raise ValueError(
            f"model identity {model_identity!r} is not the approved generator "
            f"{tag} @ {approved_digest} (ADR 0011)"
        )


def _identity_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_generation_identity(task, setting, seed, mode, model_identity, source_identity):
    """Build the stable identity which must match across resumptions."""
    task_cfg = CFG["tasks"][task]
    prompt_functions = {
        "zero_shot": [prompt_zero_shot],
        "few_shot": [prompt_zero_shot, prompt_few_shot],
        "auggpt": [prompt_auggpt],
    }[setting]
    template_components = [
        {
            "name": function.__name__,
            "sha256": hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest(),
        }
        for function in prompt_functions
    ]
    prompt_spec = {
        "protocol": PROMPT_PROTOCOL_VERSION,
        "task": task,
        "setting": setting,
        "context": task_cfg["context"],
        "style": task_cfg["style"],
        "max_words": task_cfg["max_words"],
        "labels": task_cfg["labels"],
        "few_shot_k": CFG["few_shot_k"] if setting == "few_shot" else 0,
        "auggpt_k": CFG["auggpt_k"] if setting == "auggpt" else 0,
        "nudges": NUDGES,
        "template_components": template_components,
        "template_sha256": _identity_hash(template_components),
    }
    options_spec = {
        "response_format": "json",
        "stream": False,
        "temperature": CFG["temperature"] if mode == "real" else None,
        "per_call": CFG["auggpt_k"] if setting == "auggpt" else CFG["per_call"],
    }
    identity = {
        "schema_version": 1,
        "mode": mode,
        "task": task,
        "setting": setting,
        "repetition": seed,
        "model": model_identity,
        "sources": source_identity,
        "prompt": {"protocol": PROMPT_PROTOCOL_VERSION, "sha256": _identity_hash(prompt_spec)},
        "options": {**options_spec, "sha256": _identity_hash(options_spec)},
    }
    return identity, _identity_hash(identity)


def validate_existing_rows(rows, path, task, setting, seed, mode, fingerprint, pool):
    """Fail closed before any existing row can count toward a resume target."""
    expected_labels = set(labels(task))
    pool_by_id = {row["id"]: row for row in pool}
    seen_ids = set()
    seen_texts = set()
    required = {
        "id", "task", "text", "label", "setting", "repetition", "source_ids",
        "schema_version", "generation_mode", "generation_fingerprint",
    }
    for index, row in enumerate(rows, 1):
        reason = None
        missing = sorted(required - set(row))
        if missing:
            reason = f"missing fields {missing}"
        elif row["schema_version"] != ROW_SCHEMA_VERSION:
            reason = f"schema_version is {row['schema_version']!r}"
        elif row["task"] != task or row["setting"] != setting or row["repetition"] != seed:
            reason = "task, setting, or repetition does not match the artifact path"
        elif row["generation_mode"] != mode or row["generation_fingerprint"] != fingerprint:
            reason = "generation identity does not match this invocation"
        elif row["label"] not in expected_labels:
            reason = f"unknown label {row['label']!r}"
        elif not isinstance(row["text"], str) or not row["text"].strip():
            reason = "text is missing or empty"
        else:
            seed_part = f".seed{seed}" if seed is not None else ""
            expected_prefix = f"{task}.{setting}{seed_part}.{row['label']}."
            if (not isinstance(row["id"], str)
                    or re.fullmatch(re.escape(expected_prefix) + r"\d{4}", row["id"]) is None):
                reason = "row id does not match task, setting, repetition, and label"
            elif row["id"] in seen_ids:
                reason = f"duplicate row id {row['id']!r}"
            elif norm(row["text"]) in seen_texts:
                reason = "duplicate normalized text"
            elif not isinstance(row["source_ids"], list):
                reason = "source_ids is not a list"
            elif setting == "zero_shot" and row["source_ids"]:
                reason = "zero-shot row has source identifiers"
            elif setting == "few_shot":
                source_ids = row["source_ids"]
                if len(source_ids) != CFG["few_shot_k"] or len(set(source_ids)) != len(source_ids):
                    reason = "few-shot row does not have the required distinct demonstrations"
                elif any(source_id not in pool_by_id for source_id in source_ids):
                    reason = "few-shot row references a source outside the real pool"
                elif any(pool_by_id[source_id].get("label") != row["label"] for source_id in source_ids):
                    reason = "few-shot row references a demonstration from another label"
        if reason:
            raise ValueError(f"incompatible existing row {index} in {path}: {reason}")
        seen_ids.add(row["id"])
        seen_texts.add(norm(row["text"]))


def read_existing_provenance(path, rows, generation_identity, fingerprint):
    prov_path = path.with_suffix(".provenance.json")
    if not rows and not prov_path.exists():
        return None
    if not prov_path.exists():
        raise ValueError(f"incompatible existing artifact {path}: provenance sidecar is missing")
    try:
        provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"incompatible existing artifact {path}: invalid provenance sidecar") from exc
    if provenance.get("generation_fingerprint") != fingerprint:
        raise ValueError(f"incompatible existing artifact {path}: generation identity does not match")
    identity_fields_match = (
        provenance.get("task") == generation_identity["task"]
        and provenance.get("setting") == generation_identity["setting"]
        and provenance.get("repetition") == generation_identity["repetition"]
        and provenance.get("generation_mode") == generation_identity["mode"]
        and provenance.get("generation_identity") == generation_identity
    )
    if not identity_fields_match:
        raise ValueError(f"incompatible existing artifact {path}: provenance identity does not match rows")
    if provenance.get("total_count") != len(rows):
        raise ValueError(f"incompatible existing artifact {path}: provenance row count does not match JSONL")
    if not isinstance(provenance.get("sessions"), list) or not provenance["sessions"]:
        raise ValueError(f"incompatible existing artifact {path}: provenance session history is missing")
    if not isinstance(provenance.get("cumulative_stats"), dict) or not provenance.get("created_at"):
        raise ValueError(f"incompatible existing artifact {path}: provenance history is incomplete")
    return provenance


def read_generation_rows(path):
    try:
        return read_jsonl(path)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"incompatible existing artifact {path}: invalid JSONL") from exc


def rows_identity(rows):
    selected = [
        {"id": row.get("id"), "label": row.get("label"), "text": row.get("text")}
        for row in rows
    ]
    return {"row_count": len(selected), "sha256": _identity_hash(selected)}


def atomic_write_json(path, value):
    """Replace a JSON file atomically after writing its complete temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp_path.replace(path)


def generation_journal_path(path):
    return path.with_suffix(".generation-journal.json")


def serialize_jsonl_rows(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def write_labelled_batch(path, existing_rows, new_rows, provenance_after, fingerprint):
    """Write a recoverable batch across the JSONL/provenance file boundary."""
    journal_path = generation_journal_path(path)
    prefix_bytes = path.read_bytes() if path.exists() else b""
    batch_bytes = serialize_jsonl_rows(new_rows)
    journal = {
        "schema_version": 2,
        "artifact_name": path.name,
        "generation_fingerprint": fingerprint,
        "rows_before": len(existing_rows),
        "prefix_byte_count": len(prefix_bytes),
        "prefix_bytes_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
        "batch_rows": new_rows,
        "batch_sha256": _identity_hash(new_rows),
        "batch_jsonl": batch_bytes.decode("utf-8"),
        "batch_bytes_sha256": hashlib.sha256(batch_bytes).hexdigest(),
        "provenance_after": provenance_after,
    }
    atomic_write_json(journal_path, journal)
    write_jsonl(path, new_rows, append=True)
    atomic_write_json(path.with_suffix(".provenance.json"), provenance_after)
    journal_path.unlink()


def recover_labelled_journal(path, generation_identity, fingerprint):
    """Reconcile a durable batch journal without accepting unrelated artifact rows."""
    journal_path = generation_journal_path(path)
    journal_temp = journal_path.with_suffix(journal_path.suffix + ".tmp")
    provenance_path = path.with_suffix(".provenance.json")
    provenance_temp = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    if not journal_path.exists():
        # A journal temp without its atomic replacement cannot precede a data append.
        journal_temp.unlink(missing_ok=True)
        if provenance_path.exists():
            provenance_temp.unlink(missing_ok=True)
        return read_generation_rows(path)

    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"incompatible recovery journal for {path}: invalid JSON") from exc

    required = {
        "schema_version", "artifact_name", "generation_fingerprint", "rows_before",
        "prefix_byte_count", "prefix_bytes_sha256", "batch_rows", "batch_sha256",
        "batch_jsonl", "batch_bytes_sha256", "provenance_after",
    }
    if required - set(journal):
        raise ValueError(f"incompatible recovery journal for {path}: required identity is missing")
    batch_rows = journal["batch_rows"]
    provenance_after = journal["provenance_after"]
    rows_before = journal["rows_before"]
    prefix_byte_count = journal["prefix_byte_count"]
    expected_batch_bytes = serialize_jsonl_rows(batch_rows) if isinstance(batch_rows, list) else b""
    if (
        journal["schema_version"] != 2
        or journal["artifact_name"] != path.name
        or journal["generation_fingerprint"] != fingerprint
        or not isinstance(rows_before, int)
        or rows_before < 0
        or not isinstance(prefix_byte_count, int)
        or prefix_byte_count < 0
        or not isinstance(batch_rows, list)
        or not batch_rows
        or journal["batch_sha256"] != _identity_hash(batch_rows)
        or not isinstance(journal["batch_jsonl"], str)
        or journal["batch_jsonl"].encode("utf-8") != expected_batch_bytes
        or journal["batch_bytes_sha256"] != hashlib.sha256(expected_batch_bytes).hexdigest()
        or provenance_after.get("generation_identity") != generation_identity
        or provenance_after.get("generation_fingerprint") != fingerprint
        or provenance_after.get("total_count") != rows_before + len(batch_rows)
    ):
        raise ValueError(f"incompatible recovery journal for {path}: identity or counts do not match")

    persisted_bytes = path.read_bytes() if path.exists() else b""
    if len(persisted_bytes) < prefix_byte_count:
        raise ValueError(f"incompatible recovery journal for {path}: persisted prefix does not match")
    persisted_prefix = persisted_bytes[:prefix_byte_count]
    if hashlib.sha256(persisted_prefix).hexdigest() != journal["prefix_bytes_sha256"]:
        raise ValueError(f"incompatible recovery journal for {path}: persisted prefix does not match")
    persisted_tail = persisted_bytes[prefix_byte_count:]
    if (len(persisted_tail) > len(expected_batch_bytes)
            or persisted_tail != expected_batch_bytes[:len(persisted_tail)]):
        raise ValueError(f"incompatible recovery journal for {path}: persisted batch does not match")
    missing_bytes = expected_batch_bytes[len(persisted_tail):]
    if missing_bytes:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(missing_bytes)

    rows = read_generation_rows(path)
    if len(rows) != rows_before + len(batch_rows) or rows[rows_before:] != batch_rows:
        raise ValueError(f"incompatible recovery journal for {path}: recovered rows do not match")

    recovered_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sessions = provenance_after.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError(f"incompatible recovery journal for {path}: session evidence is missing")
    sessions[-1]["status"] = "interrupted"
    sessions[-1]["finished_at"] = recovered_at
    sessions[-1]["recovered_at"] = recovered_at
    provenance_after["updated_at"] = recovered_at
    atomic_write_json(provenance_path, provenance_after)
    provenance_temp.unlink(missing_ok=True)
    journal_temp.unlink(missing_ok=True)
    journal_path.unlink()
    return rows


# ---------------- generation ----------------

def generate_labelled(task, setting, target, fake, seed=None):
    """Generate zero-shot or few-shot examples for a task and repetition seed."""
    session_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    session_started_clock = time.perf_counter()
    run_id = str(uuid.uuid4())
    mode = "fake" if fake else "real"
    model_identity = ({
        "provider": "fixture",
        "configured_tag": None,
        "resolved_name": "synthetic-fake-model",
        "digest": "fixture-v1",
    } if fake else resolve_model_identity())
    path = synth_path(task, setting, seed=seed, mode=mode)

    pool = read_jsonl(real_path(task, "pool"))
    if setting == "few_shot" and not pool:
        raise SystemExit(f"No example pool at {real_path(task, 'pool')}: run ws2_training.prepare_data first.")
    test_rows = read_jsonl(real_path(task, "test"))
    source_identity = {
        "demonstration_pool": rows_identity(pool) if setting == "few_shot" else None,
        "held_out_test": rows_identity(test_rows),
    }
    generation_identity, generation_fingerprint = build_generation_identity(
        task, setting, seed, mode, model_identity, source_identity
    )
    rows = recover_labelled_journal(path, generation_identity, generation_fingerprint)
    rows_before = len(rows)
    validate_existing_rows(
        rows, path, task, setting, seed, mode, generation_fingerprint, pool
    )
    previous_provenance = read_existing_provenance(
        path, rows, generation_identity, generation_fingerprint
    )

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

    prior_sessions = list(previous_provenance.get("sessions", [])) if previous_provenance else []
    prior_cumulative = dict(previous_provenance.get("cumulative_stats", {})) if previous_provenance else {}
    prov_path = path.with_suffix(".provenance.json")

    def build_provenance(status, rows_snapshot):
        checkpointed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        complete = all(sum(r["label"] == lab for r in rows_snapshot) >= target for lab in labels(task))
        cumulative_stats = {
            key: prior_cumulative.get(key, 0) + value
            for key, value in stats.items()
        }
        session = {
            "run_id": run_id,
            "generation_mode": mode,
            "started_at": session_started_at,
            "finished_at": checkpointed_at if status != "in_progress" else None,
            "elapsed_seconds": round(time.perf_counter() - session_started_clock, 6),
            "target_per_label": target,
            "rows_before": rows_before,
            "rows_after": len(rows_snapshot),
            "accepted_row_ids": [row["id"] for row in rows_snapshot[rows_before:]],
            "stats": dict(stats),
            "status": status,
        }
        provenance = {
            "task": task,
            "setting": setting,
            "repetition": seed,
            "model": CFG.get("model", "llama3.2:3b") if not fake else "synthetic-fake-model",
            "model_identity": model_identity,
            "generation_mode": mode,
            "generation_identity": generation_identity,
            "generation_fingerprint": generation_fingerprint,
            "temperature": CFG.get("temperature", 0.9) if not fake else None,
            "target_per_label": target,
            "counts_by_label": {lab: sum(r["label"] == lab for r in rows_snapshot) for lab in labels(task)},
            "total_count": len(rows_snapshot),
            "is_complete": complete,
            "distinct_demonstrations_count": len(demonstrations_used),
            "distinct_demonstration_ids": sorted(demonstrations_used),
            "stats": cumulative_stats,
            "cumulative_stats": cumulative_stats,
            "sessions": prior_sessions + [session],
            "created_at": previous_provenance.get("created_at", session_started_at) if previous_provenance else session_started_at,
            "updated_at": checkpointed_at,
        }
        return provenance

    def persist_provenance(status):
        """Atomically checkpoint the current invocation's evidence."""
        provenance = build_provenance(status, rows)
        atomic_write_json(prov_path, provenance)
        return provenance

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

            try:
                candidates = ask(prompt, fake, max_words=max_words, stats=stats)
            except KeyboardInterrupt:
                persist_provenance("interrupted")
                raise
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
                    "task": task,
                    "text": text,
                    "label": label,
                    "setting": setting,
                    "repetition": seed,
                    "source_ids": [r["id"] for r in shown],
                    "generation_mode": mode,
                    "schema_version": ROW_SCHEMA_VERSION,
                    "generation_fingerprint": generation_fingerprint,
                }
                new.append(row)

            if new:
                checkpoint = build_provenance("in_progress", rows + new)
                write_labelled_batch(
                    path, rows, new, checkpoint, generation_fingerprint
                )
                rows += new

        print(f"  {task} {setting} (seed={seed}) {label}: {have}/{target}")

    complete = all(sum(r["label"] == lab for r in rows) >= target for lab in labels(task))
    final_status = "no_op" if stats["accepted_new"] == 0 and complete else ("complete" if complete else "incomplete")
    provenance = persist_provenance(final_status)
    return rows, provenance


def load_auggpt_sources(task, seed):
    """Return this repetition's exact low-resource sources keyed by ID."""
    source_files = ([real_path(task, f"lowres_seed{seed}")] if seed is not None else
                    sorted(real_path(task, "x").parent.glob("lowres_seed*.jsonl")))
    sources = {}
    for source_file in source_files:
        for row in read_jsonl(source_file):
            prior = sources.get(row["id"])
            if prior and (prior.get("text"), prior.get("label")) != (row.get("text"), row.get("label")):
                raise ValueError(f"Conflicting low-resource source ID {row['id']!r} for {task} seed {seed}")
            sources.setdefault(row["id"], row)
    return sources


def validate_existing_auggpt_rows(rows, path, task, seed, mode, fingerprint, sources, test_rows):
    required = {
        "id", "task", "text", "label", "setting", "repetition", "source_ids",
        "schema_version", "generation_mode", "generation_fingerprint",
    }
    test_ids = {}
    for row in test_rows:
        test_ids.setdefault(norm(row["text"]), row.get("id"))
    leaks = [
        f"row {index} ({row.get('id')}) = held-out {test_ids[norm(row['text'])]}"
        for index, row in enumerate(rows, 1)
        if isinstance(row.get("text"), str) and norm(row["text"]) in test_ids
    ]
    if leaks:
        raise ValueError(
            f"contaminated artifact {path}: {len(leaks)} row(s) match held-out test text after "
            f"normalization: {'; '.join(leaks[:10])}. Do not edit rows in place: move this artifact and "
            "its sidecar out of the data root and regenerate this task x repetition before the freeze."
        )
    source_texts = {norm(row["text"]) for row in sources.values()}
    seen_ids = set()
    seen_texts = set()
    per_source = {}
    for index, row in enumerate(rows, 1):
        reason = None
        missing = sorted(required - set(row))
        source_ids = row.get("source_ids")
        source_id = source_ids[0] if isinstance(source_ids, list) and len(source_ids) == 1 else None
        if missing:
            reason = f"missing fields {missing}"
        elif row["schema_version"] != ROW_SCHEMA_VERSION:
            reason = f"schema_version is {row['schema_version']!r}"
        elif row["task"] != task or row["setting"] != "auggpt" or row["repetition"] != seed:
            reason = "task, setting, or repetition does not match the artifact path"
        elif row["generation_mode"] != mode or row["generation_fingerprint"] != fingerprint:
            reason = "generation identity does not match this invocation"
        elif source_id not in sources:
            reason = "source identity is missing or outside the configured low-resource sample"
        elif row["label"] != sources[source_id].get("label"):
            reason = "label does not match the source row"
        elif not isinstance(row["text"], str) or not row["text"].strip():
            reason = "text is missing or empty"
        else:
            id_prefix = f"{source_id}.seed{seed}.aug" if seed is not None else f"{source_id}.aug"
            if (not isinstance(row["id"], str)
                    or re.fullmatch(re.escape(id_prefix) + r"\d+", row["id"]) is None):
                reason = "row id does not match the source and repetition"
            elif row["id"] in seen_ids:
                reason = f"duplicate row id {row['id']!r}"
            elif norm(row["text"]) in seen_texts:
                reason = "duplicate normalized text"
            elif norm(row["text"]) == norm(sources[source_id]["text"]):
                reason = "paraphrase copies its source"
            elif norm(row["text"]) in source_texts:
                reason = "paraphrase copies another low-resource source"
        if reason:
            raise ValueError(f"incompatible existing row {index} in {path}: {reason}")
        seen_ids.add(row["id"])
        seen_texts.add(norm(row["text"]))
        per_source[source_id] = per_source.get(source_id, 0) + 1
        if per_source[source_id] > CFG["auggpt_k"]:
            raise ValueError(f"incompatible existing row {index} in {path}: too many paraphrases for {source_id}")


def auggpt_generation_identity(task, seed, mode, model_identity, sources):
    test_rows = read_jsonl(real_path(task, "test"))
    source_identity = {
        "low_resource_sample": rows_identity(list(sources.values())),
        "held_out_test": rows_identity(test_rows),
    }
    identity, fingerprint = build_generation_identity(
        task, "auggpt", seed, mode, model_identity, source_identity
    )
    return identity, fingerprint, test_rows


def generate_auggpt(task, fake, seed=None):
    """Generate provenance-checked AugGPT paraphrases for this repetition's exact sources.

    Each source resumes until it has auggpt_k valid paraphrases; a partially populated source
    is never treated as complete. Batches use the same write-ahead journal and atomic sidecar
    checkpoint as zero/few-shot generation, so interruption cannot inflate counts.
    """
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started_clock = time.perf_counter()
    run_id = str(uuid.uuid4())
    mode = "fake" if fake else "real"
    model_identity = dict(FIXTURE_MODEL_IDENTITY) if fake else resolve_model_identity()
    # Refuse an unapproved generator before any model call or write.
    require_approved_model(model_identity, mode)
    path = synth_path(task, "auggpt", seed=seed, mode=mode)

    sources = load_auggpt_sources(task, seed)
    if not sources:
        raise SystemExit(f"No lowres_seed*.jsonl for {task}: run ws2_training.prepare_data first.")
    generation_identity, fingerprint, test_rows = auggpt_generation_identity(
        task, seed, mode, model_identity, sources
    )
    existing = recover_labelled_journal(path, generation_identity, fingerprint)
    rows_before = len(existing)
    validate_existing_auggpt_rows(existing, path, task, seed, mode, fingerprint, sources, test_rows)
    previous_provenance = read_existing_provenance(
        path, existing, generation_identity, fingerprint
    )

    test_seen = {norm(row["text"]) for row in test_rows}
    source_texts = {norm(row["text"]) for row in sources.values()}
    seen = {norm(row["text"]) for row in existing}
    stats = {
        "calls": 0, "call_failures": 0, "parse_failures": 0,
        "candidates_received": 0, "rejected_duplicate": 0,
        "rejected_test_leak": 0, "rejected_source_copy": 0,
        "accepted_new": 0,
    }
    prior_sessions = list(previous_provenance.get("sessions", [])) if previous_provenance else []
    prior_cumulative = dict(previous_provenance.get("cumulative_stats", {})) if previous_provenance else {}
    prov_path = path.with_suffix(".provenance.json")

    def build_provenance(status, rows_snapshot):
        checkpointed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        counts_by_source = {
            source_id: sum(row["source_ids"] == [source_id] for row in rows_snapshot)
            for source_id in sorted(sources)
        }
        complete = all(count == CFG["auggpt_k"] for count in counts_by_source.values())
        cumulative_stats = {key: prior_cumulative.get(key, 0) + value for key, value in stats.items()}
        session = {
            "run_id": run_id,
            "generation_mode": mode,
            "started_at": started_at,
            "finished_at": checkpointed_at if status != "in_progress" else None,
            "elapsed_seconds": round(time.perf_counter() - started_clock, 6),
            "target_per_source": CFG["auggpt_k"],
            "rows_before": rows_before,
            "rows_after": len(rows_snapshot),
            "accepted_row_ids": [row["id"] for row in rows_snapshot[rows_before:]],
            "stats": dict(stats),
            "status": status,
        }
        return {
            "task": task,
            "setting": "auggpt",
            "repetition": seed,
            "model": CFG.get("model") if not fake else "synthetic-fake-model",
            "model_identity": model_identity,
            "generation_mode": mode,
            "generation_identity": generation_identity,
            "generation_fingerprint": fingerprint,
            "temperature": CFG.get("temperature") if not fake else None,
            "target_per_source": CFG["auggpt_k"],
            "source_count": len(sources),
            "counts_by_source": counts_by_source,
            "incomplete_source_ids": [sid for sid, count in counts_by_source.items() if count != CFG["auggpt_k"]],
            "total_count": len(rows_snapshot),
            "is_complete": complete,
            "cumulative_stats": cumulative_stats,
            "stats": cumulative_stats,
            "sessions": prior_sessions + [session],
            "created_at": previous_provenance.get("created_at", started_at) if previous_provenance else started_at,
            "updated_at": checkpointed_at,
        }

    for source_id, source_row in sorted(sources.items()):
        have = sum(row["source_ids"] == [source_id] for row in existing)
        attempts = 0
        while have < CFG["auggpt_k"] and attempts < 3:
            attempts += 1
            try:
                candidates = ask(
                    prompt_auggpt(source_row["text"], CFG["auggpt_k"] - have),
                    fake,
                    max_words=CFG["tasks"][task]["max_words"],
                    stats=stats,
                )
            except KeyboardInterrupt:
                atomic_write_json(prov_path, build_provenance("interrupted", existing))
                raise
            new = []
            for candidate in candidates:
                if have >= CFG["auggpt_k"]:
                    break
                normalized = norm(candidate)
                if normalized in seen:
                    stats["rejected_duplicate"] += 1
                    continue
                if normalized in test_seen:
                    stats["rejected_test_leak"] += 1
                    continue
                if normalized in source_texts:
                    stats["rejected_source_copy"] += 1
                    continue
                have += 1
                stats["accepted_new"] += 1
                seen.add(normalized)
                aug_id = (f"{source_id}.seed{seed}.aug{have}" if seed is not None
                          else f"{source_id}.aug{have}")
                new.append({
                    "id": aug_id,
                    "task": task,
                    "text": candidate,
                    "label": source_row["label"],
                    "setting": "auggpt",
                    "repetition": seed,
                    "source_ids": [source_id],
                    "generation_mode": mode,
                    "schema_version": ROW_SCHEMA_VERSION,
                    "generation_fingerprint": fingerprint,
                })
            if new:
                checkpoint = build_provenance("in_progress", existing + new)
                write_labelled_batch(path, existing, new, checkpoint, fingerprint)
                existing += new

    provenance = build_provenance("finished", existing)
    complete = provenance["is_complete"]
    status = "no_op" if stats["accepted_new"] == 0 and complete else ("complete" if complete else "incomplete")
    provenance["sessions"][-1]["status"] = status
    atomic_write_json(prov_path, provenance)
    print(f"  {task} auggpt (seed={seed}): {len(sources)} source examples, {len(existing)} paraphrases; "
          f"complete={complete}")
    if not complete:
        print(f"    {len(provenance['incomplete_source_ids'])} incomplete sources: rerun the same command to resume")
    return existing, provenance


def verify_auggpt(task, seed, mode="real"):
    """Fail-closed, read-only handoff check for one task x repetition AugGPT artifact.

    Requires the recorded model to be the approved generator (APPROVED_GENERATORS / ADR 0011) for
    real artifacts, or the fixture identity for fake ones, then recomputes the fingerprint from that
    identity and the current exact sources/prompts. Revalidates every row (including held-out test
    overlap) and requires auggpt_k paraphrases for every source of an exact lowres_seed<seed> sample.
    Raises ValueError on any problem; never writes or repairs data. Fake results are never
    real-run evidence.
    """
    if seed is None:
        raise ValueError("AugGPT verification requires a repetition seed")
    path = synth_path(task, "auggpt", seed=seed, mode=mode)
    prov_path = path.with_suffix(".provenance.json")
    if generation_journal_path(path).exists():
        raise ValueError(f"{path}: unrecovered interrupted batch; rerun generation to recover it")
    if not path.exists() or not prov_path.exists():
        raise ValueError(f"{path}: artifact or provenance sidecar is missing")
    try:
        recorded = json.loads(prov_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid provenance sidecar") from exc
    model_identity = recorded.get("model_identity")
    try:
        require_approved_model(model_identity, mode)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    sources = load_auggpt_sources(task, seed)
    expected = CFG["lowres_per_label"]
    label_counts = {lab: sum(row["label"] == lab for row in sources.values()) for lab in labels(task)}
    if any(count != expected for count in label_counts.values()) or len(sources) != expected * len(labels(task)):
        raise ValueError(f"{task} seed {seed}: low-resource sample is not exactly {expected} per label: {label_counts}")

    identity, fingerprint, test_rows = auggpt_generation_identity(task, seed, mode, model_identity, sources)
    rows = read_generation_rows(path)
    validate_existing_auggpt_rows(rows, path, task, seed, mode, fingerprint, sources, test_rows)
    provenance = read_existing_provenance(path, rows, identity, fingerprint)
    counts = {sid: sum(row["source_ids"] == [sid] for row in rows) for sid in sorted(sources)}
    short = {sid: count for sid, count in counts.items() if count != CFG["auggpt_k"]}
    if short:
        raise ValueError(f"{path}: {len(short)} of {len(sources)} sources lack {CFG['auggpt_k']} paraphrases: "
                         f"{dict(list(short.items())[:5])}")
    if provenance.get("counts_by_source") != counts or provenance.get("is_complete") is not True:
        raise ValueError(f"{path}: provenance completeness does not match the rows")
    return {
        "task": task,
        "repetition": seed,
        "generation_mode": mode,
        "real_run_evidence": mode == "real",
        "path": str(path),
        "source_count": len(sources),
        "total_count": len(rows),
        "model_identity": model_identity,
        "generation_fingerprint": fingerprint,
        "sessions": len(provenance["sessions"]),
        "elapsed_seconds": round(sum(s.get("elapsed_seconds") or 0 for s in provenance["sessions"]), 6),
        "cumulative_stats": provenance["cumulative_stats"],
    }


def inspect_pilot(task, setting, seed=None, mode="real"):
    """Print a concise inspection summary of pilot examples."""
    path = synth_path(task, setting, seed=seed, mode=mode)
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
            print(f"  - [{r['id']}] {r['text']}{demo_info}")


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
    ap.add_argument("--verify", action="store_true",
                    help="fail-closed AugGPT completeness/provenance check (with --setting auggpt --seed)")
    args = ap.parse_args(argv)

    if args.smoke_ollama:
        print("Checking Ollama service and model status...")
        res = check_ollama_status()
        print(f"Status: {res['message']}")
        print(json.dumps(res, indent=2))
        if not res["smoke_call_success"]:
            sys.exit(1)
        sys.exit(0)

    if args.inspect:
        for t in TASKS if args.task == "all" else [args.task]:
            for s in ["zero_shot", "few_shot"]:
                inspect_pilot(
                    t, s,
                    seed=int(args.seed) if args.seed and args.seed != "all" else None,
                    mode="fake" if args.dry_run else "real",
                )
        return

    if not args.setting:
        ap.error("--setting is required unless --smoke-ollama or --inspect is given")

    if args.verify:
        if args.setting != "auggpt" or args.seed is None:
            ap.error("--verify supports --setting auggpt with --seed 13|42|7|all")
        verify_seeds = [int(s) for s in CFG["seeds"]] if args.seed == "all" else [int(args.seed)]
        failures = 0
        for seed in verify_seeds:
            for task in TASKS if args.task == "all" else [args.task]:
                try:
                    summary = verify_auggpt(task, seed, mode="fake" if args.dry_run else "real")
                    note = "" if summary["real_run_evidence"] else " (FAKE fixture; not real-run evidence)"
                    print(f"PASS{note} {task} auggpt seed {seed}: {json.dumps(summary)}")
                except ValueError as exc:
                    failures += 1
                    print(f"FAIL {task} auggpt seed {seed}: {exc}")
        sys.exit(1 if failures else 0)

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
