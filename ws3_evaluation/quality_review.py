"""Issue #11: disclosed model-assisted quality review of seed-13 generated text (ADR 0012).

Six batches (trec/emotion x zero_shot/few_shot/auggpt), ten rows each, 60 reviews per round.

Inputs are the verified scope30-v1 *selected* zero/few-shot corpora and the complete AugGPT
artifacts. Before sampling, and again before outcomes are accepted, every input is re-verified
read-only with ``select_scope30.verify_selection`` / ``generate.verify_auggpt`` and re-hashed.
Missing, changed or invalid inputs fail closed.

Sampling (``issue11-quality-v1``): for each batch, rank the row IDs of the input file by ascending
SHA-256 hex of UTF-8 ``issue11-quality-v1|round<R>|13|<task>|<setting>|<row-id>`` (row ID as
tie-breaker) and take the first ten. Rows sampled in earlier rounds of the same batch are excluded
from a later round, so a fresh audit never re-uses already-reviewed rows. The rule depends only on
row IDs, never on text, labels or apparent quality; nothing is filtered or replaced.

    python -m ws3_evaluation.quality_review sample --round 1     # verify inputs, write sample + local packet
    python -m ws3_evaluation.quality_review check  --round 1     # re-verify inputs, validate judgments, outcomes
    python -m ws3_evaluation.quality_review targeted             # supplementary long-source AugGPT packet
    python -m ws3_evaluation.quality_review check-targeted       # validate the supplementary judgments

The committed evidence (``docs/reviews/issue11/``) holds IDs, hashes, the rubric and judgments, no
corpus text. The text-bearing packet (texts, few-shot demonstrations, exact AugGPT sources) is
written under the data root (``<SYNTHTEXT_DATA>/reviews/issue11``), which is git-ignored for the
accepted real root. ``--dry-run`` uses fake fixtures and is never real-run evidence.

Batch outcome: at least two *clear* errors (wrong intended label, or for AugGPT a changed source
meaning/label) among a batch's ten -> FLAGGED (investigation, producer correction and a fresh
round required). Ambiguities, copied demonstrations, unnatural wording and repeated templates are
recorded separately and do not count toward the trigger. Model judgments are not independent-human
annotation, human agreement, Krippendorff alpha or a population accuracy estimate.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from common import CFG, ROOT, TASKS, data_dir, labels, norm, read_jsonl, real_path, synth_path
from ws1_generation import generate as gen
from ws1_generation import select_scope30 as sel

REVIEW_ID = "issue11-quality-v1"
SEED = 13
PER_BATCH = 10
SETTINGS = ("zero_shot", "few_shot", "auggpt")
CLEAR_ERROR_TRIGGER = 2
SCHEMA_VERSION = 1
EVIDENCE_DIR = ROOT / "docs" / "reviews" / "issue11"
SAMPLING_RULE = (
    "per task x setting, ascending SHA-256 hex of UTF-8 "
    "'issue11-quality-v1|round<R>|13|<task>|<setting>|<row-id>', row id as tie-breaker; first 10, "
    "excluding row ids sampled in earlier rounds of the same batch"
)

# Judgment vocabulary (rubric). Labels are judged against config.yaml's existing definitions.
FIELDS = {
    "label_fit": ("fits", "ambiguous", "wrong"),
    "naturalness": ("natural", "minor_issues", "unnatural"),
    "style_fit": ("matches", "partial", "mismatch"),
    "template_repetition": ("none", "repeated_frame"),
    "copied_demonstration": ("not_applicable", "no", "near_copy", "exact_copy"),
    "source_meaning": ("not_applicable", "preserved", "partially_changed", "changed"),
}
RUBRIC = {
    "label_definitions": "config.yaml tasks.<task>.labels (unchanged; annoyance under anger, worry under fear)",
    "label_fit": "fits: clearly the intended label; ambiguous: plausible under the intended label but another "
                 "label is comparably plausible or the label is under-determined; wrong: clearly a different label",
    "naturalness": "fluency and plausibility as a real question/tweet",
    "style_fit": "matches the task/dataset style in config.yaml (TREC short search question; Emotion lower-case "
                 "first-person tweet)",
    "template_repetition": "repeated_frame when the text reuses an opening/frame that recurs across the artifact "
                           "with little lexical variation",
    "copied_demonstration": "few-shot only: compared with the row's own three source demonstrations",
    "source_meaning": "AugGPT only: compared with the exact low-resource source text and label; changed means the "
                      "meaning or the expressed label changed; partially_changed means details dropped/added but the "
                      "label and core meaning survive",
    "clear_error": "derived: label_fit == wrong or source_meaning == changed; counts toward the two-error trigger",
    "ambiguous": "derived: not a clear error, and label_fit == ambiguous or source_meaning == partially_changed",
    "trigger": f">= {CLEAR_ERROR_TRIGGER} clear errors among a batch's {PER_BATCH} -> FLAGGED",
}


class ReviewError(ValueError):
    """Inputs or judgments cannot support a valid review."""


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(Path(path).read_bytes())


def _canonical_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sample_key(round_no, task, setting, row_id, seed=SEED):
    return _sha256_bytes(f"{REVIEW_ID}|round{round_no}|{seed}|{task}|{setting}|{row_id}".encode("utf-8"))


def batch_input_path(task, setting, mode="real"):
    if setting == "auggpt":
        return synth_path(task, "auggpt", seed=SEED, mode=mode)
    return sel.selected_path(task, setting, mode)


def _rel(path):
    try:
        return Path(path).resolve().relative_to(data_dir().resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _file_record(path):
    path = Path(path)
    if not path.exists():
        raise ReviewError(f"{path}: required review input is missing")
    raw = path.read_bytes()
    return {"path": _rel(path), "sha256": _sha256_bytes(raw), "bytes": len(raw),
            "rows": sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())}


def verify_inputs(mode="real"):
    """Run the read-only producer verifiers and record exact hashes/fingerprints for every batch."""
    batches, failures = {}, []
    for task in TASKS:
        for setting in SETTINGS:
            key = f"{task}/{setting}"
            try:
                if setting == "auggpt":
                    summary = gen.verify_auggpt(task, SEED, mode=mode)
                    path = batch_input_path(task, setting, mode)
                    record = {
                        "input": _file_record(path),
                        "provenance_sidecar": _file_record(path.with_suffix(".provenance.json")),
                        "generation_fingerprint": summary["generation_fingerprint"],
                        "row_fingerprints": summary["row_fingerprints"],
                        "model_identity": summary["model_identity"],
                        "source_count": summary["source_count"],
                        "verifier": "ws1_generation.generate.verify_auggpt",
                    }
                else:
                    summary = sel.verify_selection(task, setting, mode=mode)
                    path = batch_input_path(task, setting, mode)
                    record = {
                        "input": _file_record(path),
                        "selection_manifest": _file_record(sel.manifest_path(task, setting, mode)),
                        "original_artifact_sha256": summary["original_sha256"],
                        "original_provenance_sha256": summary["provenance_sha256"],
                        "original_rows": summary["original_rows"],
                        "generation_fingerprint": summary["generation_fingerprint"],
                        "selected_distinct_demonstrations": summary["selected_distinct_demonstrations"],
                        "verifier": "ws1_generation.select_scope30.verify_selection",
                    }
                    if record["input"]["sha256"] != summary["selected_jsonl_sha256"]:
                        raise ReviewError(f"{path}: hash differs from its verified selection manifest")
                if summary.get("real_run_evidence") is not (mode == "real"):
                    raise ReviewError(f"{key}: real_run_evidence does not match mode {mode!r}")
                batches[key] = record
            except (ValueError, OSError) as exc:
                failures.append(f"{key}: {exc}")
    if failures:
        raise ReviewError("input verification failed; review cannot proceed:\n  " + "\n  ".join(failures))
    references = {}
    for task in TASKS:
        for name in ("pool", f"lowres_seed{SEED}"):
            references[f"{task}/{name}"] = _file_record(real_path(task, name))
    overlap = {task: sel.cross_setting_overlap(task, mode) for task in TASKS}
    return {"batches": batches, "reference_inputs": references,
            "selected_zero_few_shared_normalized_texts": overlap}


def draw_sample(row_ids, round_no, task, setting, exclude=(), n=PER_BATCH):
    excluded = set(exclude)
    eligible = sorted(set(row_ids) - excluded)
    if len(eligible) < n:
        raise ReviewError(f"{task}/{setting}: only {len(eligible)} eligible rows for a {n}-row sample")
    return sorted(eligible, key=lambda rid: (sample_key(round_no, task, setting, rid), rid))[:n]


def sample_path(round_no, evidence_dir=None):
    return Path(evidence_dir or EVIDENCE_DIR) / f"round{round_no}_sample.json"


def judgments_path(round_no, evidence_dir=None):
    return Path(evidence_dir or EVIDENCE_DIR) / f"round{round_no}_judgments.json"


def outcome_path(round_no, evidence_dir=None):
    return Path(evidence_dir or EVIDENCE_DIR) / f"round{round_no}_outcome.json"


def packet_dir():
    return data_dir() / "reviews" / "issue11"


def _write_new_or_same(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ReviewError(f"{path}: exists with different content; earlier rounds are never overwritten")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def _prior_sampled(round_no, evidence_dir):
    prior = {}
    for r in range(1, round_no):
        path = sample_path(r, evidence_dir)
        if not path.exists():
            raise ReviewError(f"{path}: round {r} sample is missing; rounds must be preserved in order")
        recorded = json.loads(path.read_text(encoding="utf-8"))
        for key, batch in recorded["batches"].items():
            prior.setdefault(key, set()).update(item["row_id"] for item in batch["items"])
    return prior


def _tokens(text):
    return set(norm(text).replace("?", " ").replace(",", " ").replace(".", " ").split())


def _jaccard(a, b):
    ta, tb = _tokens(a), _tokens(b)
    return round(len(ta & tb) / len(ta | tb), 3) if ta | tb else 0.0


def _opening(text, n=3):
    return " ".join(norm(text).split()[:n])


def build_sample(round_no, mode="real", evidence_dir=None):
    """Verify inputs and derive the round's sample (IDs only) and its text-bearing packet."""
    inputs = verify_inputs(mode)
    prior = _prior_sampled(round_no, evidence_dir)
    pools = {task: {r["id"]: r for r in read_jsonl(real_path(task, "pool"))} for task in TASKS}
    sources = {task: gen.load_auggpt_sources(task, SEED) for task in TASKS}
    zero_few_texts = {
        task: {s: {norm(r["text"]) for r in read_jsonl(batch_input_path(task, s, mode))}
               for s in ("zero_shot", "few_shot")}
        for task in TASKS
    }
    sample = {
        "schema_version": SCHEMA_VERSION, "review_id": REVIEW_ID, "round": round_no, "seed": SEED,
        "generation_mode": mode, "real_run_evidence": mode == "real",
        "sampling_rule": SAMPLING_RULE, "per_batch": PER_BATCH,
        "excluded_prior_round_ids": {k: sorted(v) for k, v in sorted(prior.items())},
        "inputs": inputs, "batches": {},
    }
    packet = {"review_id": REVIEW_ID, "round": round_no, "generation_mode": mode,
              "label_definitions": {t: CFG["tasks"][t]["labels"] for t in TASKS},
              "style": {t: CFG["tasks"][t]["style"] for t in TASKS}, "batches": {}}
    for task in TASKS:
        max_words = CFG["tasks"][task]["max_words"]
        for setting in SETTINGS:
            key = f"{task}/{setting}"
            rows = read_jsonl(batch_input_path(task, setting, mode))
            by_id = {r["id"]: r for r in rows}
            if len(by_id) != len(rows):
                raise ReviewError(f"{key}: duplicate row ids")
            chosen = draw_sample(by_id, round_no, task, setting, prior.get(key, ()))
            openings = Counter(_opening(r["text"]) for r in rows)
            items, packet_items = [], []
            for order, rid in enumerate(chosen, 1):
                row = by_id[rid]
                item = {"order": order, "row_id": rid, "label": row["label"],
                        "sample_key": sample_key(round_no, task, setting, rid),
                        "text_sha256": _sha256_bytes(row["text"].encode("utf-8")),
                        "source_ids": list(row.get("source_ids", []))}
                aid = {"opening_trigram": _opening(row["text"]),
                       "opening_trigram_count_in_input": openings[_opening(row["text"])],
                       "nearest_other_row_jaccard": max(
                           (_jaccard(row["text"], o["text"]) for o in rows if o["id"] != rid), default=0.0),
                       "word_count": len(row["text"].split())}
                entry = {"order": order, "row_id": rid, "label": row["label"], "text": row["text"]}
                if setting == "few_shot":
                    demos = []
                    for sid in row["source_ids"]:
                        if sid not in pools[task]:
                            raise ReviewError(f"{key}: {rid} demonstration {sid} is not in the real pool")
                        demo = pools[task][sid]
                        demos.append({"id": sid, "label": demo["label"], "text": demo["text"],
                                      "jaccard": _jaccard(row["text"], demo["text"]),
                                      "normalized_exact": norm(row["text"]) == norm(demo["text"])})
                    entry["demonstrations"] = demos
                if setting in ("zero_shot", "few_shot"):
                    other = "few_shot" if setting == "zero_shot" else "zero_shot"
                    item["shared_with_other_selected_setting"] = norm(row["text"]) in zero_few_texts[task][other]
                    aid["shared_with_other_selected_setting"] = item["shared_with_other_selected_setting"]
                if setting == "auggpt":
                    sid = row["source_ids"][0]
                    if sid not in sources[task]:
                        raise ReviewError(f"{key}: {rid} source {sid} is not in lowres_seed{SEED}")
                    src = sources[task][sid]
                    item["source_word_count"] = len(src["text"].split())
                    item["long_source"] = item["source_word_count"] > max_words
                    item["generation_fingerprint"] = row["generation_fingerprint"]
                    entry["source"] = {"id": sid, "label": src["label"], "text": src["text"],
                                       "word_count": item["source_word_count"], "long_source": item["long_source"]}
                    aid["jaccard_to_source"] = _jaccard(row["text"], src["text"])
                entry["aids"] = aid
                items.append(item)
                packet_items.append(entry)
            sample["batches"][key] = {
                "population_rows": len(rows), "eligible_rows": len(rows) - len(prior.get(key, ())),
                "selected_ids_in_order": chosen,
                "sampled_label_counts": dict(sorted(Counter(by_id[r]["label"] for r in chosen).items())),
                "items": items,
            }
            packet["batches"][key] = packet_items
    return sample, packet


def _packet_markdown(packet):
    lines = [f"# Issue #11 review packet — {packet['review_id']} round {packet['round']} "
             f"({packet['generation_mode']}; LOCAL ONLY, contains corpus text)", ""]
    for task, defs in packet["label_definitions"].items():
        lines.append(f"## {task} labels — style: {packet['style'][task]}")
        lines += [f"- **{lab}**: {d}" for lab, d in defs.items()] + [""]
    for key, entries in packet["batches"].items():
        lines += [f"## Batch {key}", ""]
        for e in entries:
            lines.append(f"### {key} #{e['order']:02d} `{e['row_id']}` — intended **{e['label']}**")
            lines.append(f"> {e['text']}")
            for d in e.get("demonstrations", []):
                lines.append(f"- demo `{d['id']}` [{d['label']}] (jaccard {d['jaccard']}): {d['text']}")
            if "source" in e:
                s = e["source"]
                lines.append(f"- SOURCE `{s['id']}` [{s['label']}] ({s['word_count']} words"
                             f"{', LONG' if s['long_source'] else ''}): {s['text']}")
            lines.append(f"- aids: `{json.dumps(e['aids'], ensure_ascii=False)}`")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_sample(round_no, mode="real", evidence_dir=None):
    sample, packet = build_sample(round_no, mode, evidence_dir)
    data = _canonical_bytes(sample)
    _write_new_or_same(sample_path(round_no, evidence_dir), data)
    out = packet_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"round{round_no}_packet.json").write_bytes(_canonical_bytes(packet))
    (out / f"round{round_no}_packet.md").write_text(_packet_markdown(packet), encoding="utf-8")
    return json.loads(data), _sha256_bytes(data)


def _derive(item, setting):
    clear = item["label_fit"] == "wrong" or item["source_meaning"] == "changed"
    ambiguous = not clear and (item["label_fit"] == "ambiguous" or item["source_meaning"] == "partially_changed")
    return clear, ambiguous


def validate_items(items, expected, setting_of, where):
    """Validate a list of judgments against expected (key, row_id) pairs, in order."""
    problems = []
    if [(j.get("batch"), j.get("row_id")) for j in items] != expected:
        problems.append(f"{where}: judged (batch,row_id) list does not equal the sample in order")
    for j in items:
        tag = f"{where} {j.get('batch')} {j.get('row_id')}"
        setting = setting_of(j.get("batch"))
        for field, allowed in FIELDS.items():
            if j.get(field) not in allowed:
                problems.append(f"{tag}: {field}={j.get(field)!r} not in {allowed}")
        if problems and problems[-1].startswith(tag):
            continue
        if (setting == "few_shot") != (j["copied_demonstration"] != "not_applicable"):
            problems.append(f"{tag}: copied_demonstration must be judged for few_shot only")
        if (setting == "auggpt") != (j["source_meaning"] != "not_applicable"):
            problems.append(f"{tag}: source_meaning must be judged for auggpt only")
        clear, ambiguous = _derive(j, setting)
        if j.get("clear_error") is not clear or j.get("ambiguous") is not ambiguous:
            problems.append(f"{tag}: clear_error/ambiguous must be {clear}/{ambiguous} under the rubric")
        if not str(j.get("explanation", "")).strip():
            problems.append(f"{tag}: explanation is required")
    return problems


def _check_reviewer(reviewer, problems):
    required = ("kind", "provider", "model", "exact_version", "human_review")
    if not isinstance(reviewer, dict) or any(k not in reviewer for k in required):
        problems.append(f"reviewer must record {required}")
    elif reviewer["kind"] != "model-assisted" or reviewer["human_review"] is not False:
        problems.append("reviewer.kind must be 'model-assisted' and human_review false")


def check_round(round_no, mode="real", evidence_dir=None):
    """Re-verify inputs against the recorded sample, validate judgments, compute batch outcomes."""
    s_path, j_path = sample_path(round_no, evidence_dir), judgments_path(round_no, evidence_dir)
    if not s_path.exists():
        raise ReviewError(f"{s_path}: sample is missing")
    s_raw = s_path.read_bytes()
    sample = json.loads(s_raw)
    rederived, _ = build_sample(round_no, mode, evidence_dir)   # re-runs the read-only verifiers
    rederived = json.loads(_canonical_bytes(rederived))
    current = rederived["inputs"]
    if current != sample.get("inputs"):
        recorded = sample.get("inputs", {}).get("batches", {})
        changed = sorted(k for k in set(current["batches"]) | set(recorded)
                         if current["batches"].get(k) != recorded.get(k))
        raise ReviewError(f"inputs changed since sampling (batches {changed or 'reference inputs'}); "
                          "the round is blocked")
    if rederived != sample:
        raise ReviewError(f"{s_path}: recorded sample does not match re-derivation")
    if not j_path.exists():
        raise ReviewError(f"{j_path}: judgments are missing; review incomplete")
    judged = json.loads(j_path.read_text(encoding="utf-8"))
    problems = []
    if judged.get("sample_sha256") != _sha256_bytes(s_raw) or judged.get("round") != round_no:
        problems.append("judgments are not bound to this round's sample (sample_sha256/round)")
    _check_reviewer(judged.get("reviewer"), problems)
    expected = [(k, item["row_id"]) for k, b in sample["batches"].items() for item in b["items"]]
    items = judged.get("items", [])
    problems += validate_items(items, expected, lambda k: (k or "/").split("/")[1], f"round {round_no}")
    if problems:
        raise ReviewError("invalid judgments:\n  " + "\n  ".join(problems))
    batches = {}
    for key in sample["batches"]:
        mine = [j for j in items if j["batch"] == key]
        clear = [j["row_id"] for j in mine if j["clear_error"]]
        batches[key] = {
            "reviewed": len(mine),
            "clear_errors": len(clear), "clear_error_ids": clear,
            "ambiguous": sum(j["ambiguous"] for j in mine),
            "ambiguous_ids": [j["row_id"] for j in mine if j["ambiguous"]],
            "copied_demonstration_ids": [j["row_id"] for j in mine
                                         if j["copied_demonstration"] in ("near_copy", "exact_copy")],
            "unnatural_ids": [j["row_id"] for j in mine if j["naturalness"] == "unnatural"],
            "style_mismatch_ids": [j["row_id"] for j in mine if j["style_fit"] == "mismatch"],
            "repeated_frame_ids": [j["row_id"] for j in mine if j["template_repetition"] == "repeated_frame"],
            "partially_changed_meaning_ids": [j["row_id"] for j in mine
                                              if j["source_meaning"] == "partially_changed"],
            "outcome": "FLAGGED" if len(clear) >= CLEAR_ERROR_TRIGGER else "PASS",
        }
    return {
        "schema_version": SCHEMA_VERSION, "review_id": REVIEW_ID, "round": round_no,
        "generation_mode": mode, "real_run_evidence": mode == "real",
        "sample_sha256": _sha256_bytes(s_raw),
        "judgments_sha256": _sha256_file(j_path),
        "reviewer": judged["reviewer"], "trigger": RUBRIC["trigger"], "batches": batches,
        "flagged_batches": [k for k, b in batches.items() if b["outcome"] == "FLAGGED"],
        "note": "Model-assisted screen; not independent-human annotation, agreement or population accuracy.",
    }


# --- supplementary targeted check (kept separate from the random 60) -----------------------------

TARGETED_ID = "issue11-targeted-long-sources-v1"


def targeted_paths(evidence_dir=None):
    base = Path(evidence_dir or EVIDENCE_DIR)
    return base / "targeted_long_sources_sample.json", base / "targeted_long_sources_judgments.json"


def build_targeted(mode="real", task="emotion", max_words=None):
    """All AugGPT paraphrases of sources longer than max_words (default: the task's guard limit)."""
    inputs = verify_inputs(mode)
    max_words = CFG["tasks"][task]["max_words"] if max_words is None else max_words
    sources = gen.load_auggpt_sources(task, SEED)
    long_ids = sorted(sid for sid, s in sources.items() if len(s["text"].split()) > max_words)
    rows = [r for r in read_jsonl(batch_input_path(task, "auggpt", mode)) if r["source_ids"][0] in long_ids]
    rows.sort(key=lambda r: r["id"])
    sample = {
        "schema_version": SCHEMA_VERSION, "review_id": TARGETED_ID, "seed": SEED, "task": task,
        "generation_mode": mode, "real_run_evidence": mode == "real",
        "counts_toward_required_random_60": False,
        "selection": f"every {task}/auggpt row whose exact source exceeds max_words={max_words}",
        "max_words": max_words,
        "input": inputs["batches"][f"{task}/auggpt"],
        "long_sources": {sid: {"label": sources[sid]["label"], "word_count": len(sources[sid]["text"].split())}
                         for sid in long_ids},
        "items": [{"order": i, "row_id": r["id"], "source_id": r["source_ids"][0], "label": r["label"],
                   "word_count": len(r["text"].split()), "generation_fingerprint": r["generation_fingerprint"],
                   "text_sha256": _sha256_bytes(r["text"].encode("utf-8"))}
                  for i, r in enumerate(rows, 1)],
    }
    packet = {"review_id": TARGETED_ID, "sources": {sid: sources[sid] for sid in long_ids},
              "items": [{"row_id": r["id"], "source_id": r["source_ids"][0], "label": r["label"],
                         "text": r["text"], "fingerprint": r["generation_fingerprint"][:8]} for r in rows]}
    return sample, packet


def write_targeted(mode="real", evidence_dir=None, max_words=None):
    sample, packet = build_targeted(mode, max_words=max_words)
    data = _canonical_bytes(sample)
    _write_new_or_same(targeted_paths(evidence_dir)[0], data)
    out = packet_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "targeted_long_sources_packet.json").write_bytes(_canonical_bytes(packet))
    return json.loads(data), _sha256_bytes(data)


def check_targeted(mode="real", evidence_dir=None):
    s_path, j_path = targeted_paths(evidence_dir)
    if not s_path.exists() or not j_path.exists():
        raise ReviewError("targeted sample or judgments are missing")
    s_raw = s_path.read_bytes()
    sample = json.loads(s_raw)
    rederived, _ = build_targeted(mode, sample["task"], sample["max_words"])
    rederived = json.loads(_canonical_bytes(rederived))
    if rederived != sample:
        raise ReviewError(f"{s_path}: targeted sample/inputs changed since it was recorded")
    judged = json.loads(j_path.read_text(encoding="utf-8"))
    problems = []
    if judged.get("sample_sha256") != _sha256_bytes(s_raw):
        problems.append("targeted judgments are not bound to the targeted sample")
    _check_reviewer(judged.get("reviewer"), problems)
    key = f"{sample['task']}/auggpt"
    expected = [(key, item["row_id"]) for item in sample["items"]]
    items = judged.get("items", [])
    problems += validate_items(items, expected, lambda k: "auggpt", "targeted")
    if problems:
        raise ReviewError("invalid targeted judgments:\n  " + "\n  ".join(problems))
    by_source = {}
    for j, item in zip(items, sample["items"]):
        b = by_source.setdefault(item["source_id"], {"clear_errors": [], "partially_changed": [], "ambiguous": []})
        if j["clear_error"]:
            b["clear_errors"].append(j["row_id"])
        if j["source_meaning"] == "partially_changed":
            b["partially_changed"].append(j["row_id"])
        if j["ambiguous"]:
            b["ambiguous"].append(j["row_id"])
    return {"review_id": TARGETED_ID, "counts_toward_required_random_60": False,
            "sample_sha256": _sha256_bytes(s_raw), "judgments_sha256": _sha256_file(j_path),
            "reviewed": len(items), "clear_errors": sum(j["clear_error"] for j in items),
            "ambiguous": sum(j["ambiguous"] for j in items), "by_source": by_source}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "check", "targeted", "check-targeted"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--evidence-dir", default=None, help="default docs/reviews/issue11")
    ap.add_argument("--dry-run", action="store_true", help="fake fixtures (never real-run evidence)")
    args = ap.parse_args(argv)
    if args.round < 1:
        ap.error("--round must be >= 1")
    mode = "fake" if args.dry_run else "real"
    note = "" if mode == "real" else " (FAKE fixture; not real-run evidence)"
    try:
        if args.command == "sample":
            sample, digest = write_sample(args.round, mode, args.evidence_dir)
            for key, batch in sample["batches"].items():
                print(f"SAMPLED{note} {key} round {args.round}: {batch['selected_ids_in_order']}")
            print(f"sample sha256 {digest}; local packet in {packet_dir()}")
        elif args.command == "check":
            outcome = check_round(args.round, mode, args.evidence_dir)
            data = _canonical_bytes(outcome)
            _write_new_or_same(outcome_path(args.round, args.evidence_dir), data)
            for key, batch in outcome["batches"].items():
                print(f"{batch['outcome']}{note} {key} round {args.round}: clear_errors={batch['clear_errors']} "
                      f"ambiguous={batch['ambiguous']}")
            sys.exit(2 if outcome["flagged_batches"] else 0)
        elif args.command == "targeted":
            sample, digest = write_targeted(mode, args.evidence_dir)
            print(f"TARGETED{note} {len(sample['items'])} rows from {len(sample['long_sources'])} long sources; "
                  f"sha256 {digest}")
        else:
            result = check_targeted(mode, args.evidence_dir)
            out = targeted_paths(args.evidence_dir)[0].with_name("targeted_long_sources_outcome.json")
            _write_new_or_same(out, _canonical_bytes(result))
            print(f"TARGETED CHECK{note}: {json.dumps({k: result[k] for k in ('reviewed', 'clear_errors', 'ambiguous')})}")
    except ReviewError as exc:
        print(f"FAIL{note}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
