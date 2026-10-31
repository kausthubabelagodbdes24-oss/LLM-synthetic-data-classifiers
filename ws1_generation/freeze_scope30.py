"""Issue #12: audit and freeze the selected seed-13 synthetic handoff (ADR 0012, lead Option A).

Six accepted artifacts: trec/emotion x zero_shot/few_shot (the scope30-v1 *selected* corpora) and
trec/emotion AugGPT (the complete artifacts; TREC is the promoted ADR 0013 v2 replacement).

    python -m ws1_generation.freeze_scope30 --audit     # read-only: full audit, prints a summary
    python -m ws1_generation.freeze_scope30 --write     # explicit: write the freeze manifest (never overwrites)
    python -m ws1_generation.freeze_scope30 --verify    # read-only, fail-closed check of a written freeze

The audit re-runs the producers' read-only verifiers (``select_scope30.verify_selection``,
``generate.verify_auggpt``, ``promote_auggpt.verify_promotion``), then independently re-reads every
consumed file: exact counts, label coverage, unique IDs and texts, ID order, source identity, held-out
test exclusion, six distinct paraphrases per exact AugGPT source, model/prompt/settings identity,
fingerprints and provenance. It binds the complete issue #11 evidence by hash and requires, for every
batch, that its latest review round PASSed against the exact artifact being frozen.

The manifest (``docs/freeze/scope30-v1-seed13.freeze.json``) is deterministic (no timestamps): the same
files and repository revision produce the same bytes. It holds IDs, hashes and counts, never corpus
text. ``--verify`` rebuilds it from the current files and requires byte equality, so a missing,
replaced, reordered, tampered or incompatible artifact or evidence file fails. ``--dry-run`` uses
fake fixtures and is never real-run evidence.

Option A (issue #12): known review/scan findings are retained and disclosed, not removed or corrected.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from common import CFG, ROOT, TASKS, data_dir, labels, norm, read_jsonl, real_path, synth_path
from ws1_generation import generate as gen
from ws1_generation import promote_auggpt
from ws1_generation import select_scope30 as sel

FREEZE_ID = "scope30-v1-seed13-freeze-v1"
SCHEMA_VERSION = 1
SEED = 13
SETTINGS = ("zero_shot", "few_shot", "auggpt")
FREEZE_PATH = ROOT / "docs" / "freeze" / "scope30-v1-seed13.freeze.json"
EVIDENCE_DIR = ROOT / "docs" / "reviews" / "issue11"
# Evidence that must exist for a real freeze (fake fixtures may bind a smaller set).
REQUIRED_REAL_EVIDENCE = (
    "README.md", "round1_sample.json", "round1_judgments.json", "round1_outcome.json",
    "round2_sample.json", "round2_judgments.json", "round2_outcome.json",
    "round2_recheck_sample.json", "round2_recheck_judgments.json", "round2_recheck_outcome.json",
    "targeted_long_sources_sample.json", "targeted_long_sources_judgments.json",
    "targeted_long_sources_outcome.json", "targeted_long_sources_adjudication_2026-11-01.json",
    "targeted_long_sources_adjudication_2026-11-01_outcome.json", "scope_scan_2026-11-01_ids.json",
)

DECISION = {
    "option": "A",
    "summary": "Accept and disclose: the current verified selected seed-13 corpora are accepted for the "
               "exploratory scope30-v1 study without regeneration, relabelling or row deletion; known "
               "problems are retained and disclosed.",
    "decided_by": "Kausthuba (@kausthubabelagodbdes24-oss), sole operator/project lead",
    "record": "https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/12#issuecomment-6547818106",
    "recorded_at": "2026-10-31T19:02:43Z",
    "rejected_alternatives": ["v3 regeneration of TREC AugGPT", "row deletion"],
    "valid_only_for": "scope30-v1, seed 13, exploratory single-seed comparison (ADR 0012)",
    "post_freeze_changes": "require explicit lead approval and affected retraining/reanalysis",
}
LIMITATIONS = [
    "Corrected TREC AugGPT (ADR 0013 v2) passed the formal round-2 screen with 1/10 clear errors and 3 "
    "ambiguous (two-clear-errors trigger not reached); the clear error trec.train.04163.seed13.aug4 changes "
    "the named entity 'Hot Wheels' to 'Hot and Wheels'.",
    "The targeted re-check of the 10 round-1 IDs found trec.train.00845.seed13.aug3 still changes the "
    "expected answer type (description to a named event); this re-check is outside the random denominator.",
    "A read-only heuristic scope scan found approximately 13 likely problematic TREC AugGPT rows plus 6 "
    "borderline rows of 360: answer-type changes, corrupted entity or text names (appearing only in the v2 "
    "output), a reversed relation and abbreviation questions drifting toward definition requests. Around 3-4 "
    "of these rows may change the intended label or answer type.",
    "All 12 Emotion AugGPT paraphrases of emotion.train.07790 and emotion.train.15223 change the source "
    "meaning (agent reversal; third-party strength made self-attributed) while retaining their labels.",
    "These rows are retained, not relabelled, deleted, corrected or hidden. They are accepted only for the "
    "exploratory scope30-v1, seed-13 study and are not described as harmless or scientifically negligible.",
    "All findings are model-assisted (Claude Opus 5.5; one judgment per item) or heuristic. They are not "
    "independent-human judgments, human agreement, or population error estimates, and ten-row batches cannot "
    "estimate corpus error rates.",
    "The freeze does not establish corpus correctness, semantic quality, robustness, statistical stability or "
    "generalization. Other disclosed #11 flags (ambiguous rows, Emotion style mismatch and inherited weak "
    "source labels, 18 TREC texts shared by selected zero- and few-shot corpora, few-shot labelling cost) "
    "remain as recorded in docs/reviews/issue11/README.md.",
]


class FreezeError(ValueError):
    """The handoff cannot be frozen or does not match its freeze."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _canonical(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _rel(path):
    try:
        return Path(path).resolve().relative_to(data_dir().resolve()).as_posix()
    except ValueError as exc:
        raise FreezeError(f"{path} is outside the data root") from exc


def _file(path, rows=True):
    path = Path(path)
    if not path.exists():
        raise FreezeError(f"{path}: required file is missing")
    raw = path.read_bytes()
    record = {"path": _rel(path), "sha256": _sha(raw), "bytes": len(raw)}
    if rows:
        record["rows"] = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
    return record


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().decode("utf-8").splitlines() if line.strip()]


def _provenance(path):
    sidecar = Path(path).with_suffix(".provenance.json")
    rec = _file(sidecar, rows=False)
    prov = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = prov.get("generation_identity") or {}
    rec.update({
        "updated_at": prov.get("updated_at"),
        "session_count": len(prov.get("sessions") or []),
        "last_session_status": (prov.get("sessions") or [{}])[-1].get("status"),
        "generation_fingerprint": prov.get("generation_fingerprint"),
    })
    settings = {
        "prompt_protocol": (identity.get("prompt") or {}).get("protocol"),
        "prompt_sha256": (identity.get("prompt") or {}).get("sha256"),
        "options_sha256": (identity.get("options") or {}).get("sha256"),
        "options": {k: v for k, v in (identity.get("options") or {}).items() if k != "sha256"},
        "prompt_extensions_sha256": gen._identity_hash(identity.get("prompt_extensions")),
        "sources_sha256": gen._identity_hash(identity.get("sources")),
        "temperature": prov.get("temperature"),
    }
    return rec, prov, settings


def _audit_rows(task, setting, rows, test_texts, problems, mode):
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    if len({norm(r["text"]) for r in rows}) != len(rows):
        problems.append("duplicate normalized texts")
    leaks = [r["id"] for r in rows if norm(r["text"]) in test_texts]
    if leaks:
        problems.append(f"held-out test overlap {leaks[:5]}")
    if any(r.get("task") != task or r.get("setting") != setting or r.get("repetition") != SEED for r in rows):
        problems.append("task/setting/repetition mismatch")
    if any(r.get("generation_mode") != mode for r in rows):
        problems.append("generation_mode mismatch")



def audit_selected(task, setting, mode):
    try:
        summary = sel.verify_selection(task, setting, mode)
    except ValueError as exc:
        raise FreezeError(f"{task}/{setting}: {exc}") from exc
    s_path, m_path = sel.selected_path(task, setting, mode), sel.manifest_path(task, setting, mode)
    orig = synth_path(task, setting, seed=SEED, mode=mode)
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    rows = _jsonl(s_path)
    test_texts = {norm(r["text"]) for r in read_jsonl(real_path(task, "test"))}
    pool = {r["id"]: r for r in read_jsonl(real_path(task, "pool"))}
    problems = []
    _audit_rows(task, setting, rows, test_texts, problems, mode)
    counts = Counter(r["label"] for r in rows)
    if {lab: counts.get(lab, 0) for lab in labels(task)} != {lab: sel.SELECTED_PER_LABEL for lab in labels(task)}:
        problems.append(f"label counts {dict(counts)}")
    if sorted(r["id"] for r in rows) != manifest["selected_ids"]:
        problems.append("selected ids differ from the selection manifest")
    if setting == "zero_shot" and any(r["source_ids"] for r in rows):
        problems.append("zero-shot rows with source ids")
    if setting == "few_shot":
        for r in rows:
            if len(r["source_ids"]) != CFG["few_shot_k"] or any(
                    s not in pool or pool[s]["label"] != r["label"] for s in r["source_ids"]):
                problems.append(f"{r['id']}: demonstrations are not {CFG['few_shot_k']} same-label pool rows")
                break
    orig_rec, prov, settings = _provenance(orig)
    if summary["selected_jsonl_sha256"] != _sha(s_path.read_bytes()):
        problems.append("selected file hash differs from its verified manifest")
    if {r["generation_fingerprint"] for r in rows} - {prov["generation_fingerprint"]}:
        problems.append("selected rows carry another fingerprint than the producer sidecar")
    if problems:
        raise FreezeError(f"{task}/{setting}: audit failed: {problems}")
    return {
        "kind": "scope30-v1 selected corpus",
        "consumed": {**_file(s_path), "ids_in_file_order": [r["id"] for r in rows],
                     "counts_by_label": {lab: counts[lab] for lab in labels(task)}},
        "selection_manifest": _file(m_path, rows=False),
        "original_artifact": _file(orig),
        "provenance": orig_rec,
        "generation_fingerprint": prov["generation_fingerprint"],
        "row_fingerprints": dict(sorted(Counter(r["generation_fingerprint"] for r in rows).items())),
        "model_identity": prov["model_identity"],
        "settings": settings,
        "selected_distinct_demonstrations": summary["selected_distinct_demonstrations"],
        "checks": {"held_out_test_overlap": 0, "duplicate_ids": 0, "duplicate_normalized_texts": 0,
                   "labels_covered": len(labels(task)), "per_label": sel.SELECTED_PER_LABEL},
    }


def audit_auggpt(task, mode):
    try:
        summary = gen.verify_auggpt(task, SEED, mode=mode)
        promotion = (promote_auggpt.verify_promotion(task, SEED, mode)
                     if gen.auggpt_correction(task, SEED) else None)
    except ValueError as exc:
        raise FreezeError(f"{task}/auggpt: {exc}") from exc
    path = synth_path(task, "auggpt", seed=SEED, mode=mode)
    rows = _jsonl(path)
    sources = gen.load_auggpt_sources(task, SEED)
    lowres = read_jsonl(real_path(task, f"lowres_seed{SEED}"))
    test_texts = {norm(r["text"]) for r in read_jsonl(real_path(task, "test"))}
    problems = []
    _audit_rows(task, "auggpt", rows, test_texts, problems, mode)
    per_source = Counter(r["source_ids"][0] for r in rows)
    distinct = {sid: len({norm(r["text"]) for r in rows if r["source_ids"] == [sid]}) for sid in sources}
    k = CFG["auggpt_k"]
    if set(sources) != {r["id"] for r in lowres} or len(sources) != CFG["lowres_per_label"] * len(labels(task)):
        problems.append("AugGPT sources are not the exact low-resource sample")
    if dict(per_source) != {sid: k for sid in sources} or any(n != k for n in distinct.values()):
        problems.append(f"not exactly {k} distinct paraphrases per source")
    if any(r["label"] != sources[r["source_ids"][0]]["label"] for r in rows):
        problems.append("paraphrase label differs from its source")
    if len(rows) != k * len(sources):
        problems.append(f"total {len(rows)}")
    src_counts = Counter(s["label"] for s in sources.values())
    if {lab: src_counts.get(lab, 0) for lab in labels(task)} != {lab: CFG["lowres_per_label"] for lab in labels(task)}:
        problems.append(f"source label coverage {dict(src_counts)}")
    prov_rec, prov, settings = _provenance(path)
    if summary["artifact_sha256"] != _sha(path.read_bytes()):
        problems.append("artifact hash differs from verify_auggpt")
    if problems:
        raise FreezeError(f"{task}/auggpt: audit failed: {problems}")
    record = {
        "kind": "complete AugGPT artifact",
        "consumed": {**_file(path), "ids_in_file_order": [r["id"] for r in rows],
                     "counts_by_label": dict(sorted(Counter(r["label"] for r in rows).items()))},
        "provenance": prov_rec,
        "generation_fingerprint": prov["generation_fingerprint"],
        "row_fingerprints": dict(sorted(Counter(r["generation_fingerprint"] for r in rows).items())),
        "model_identity": prov["model_identity"],
        "settings": settings,
        "prompt_correction": summary["prompt_correction"],
        "corrected_output": summary["corrected_output"],
        "source_count": len(sources),
        "paraphrases_per_source": k,
        "checks": {"held_out_test_overlap": 0, "duplicate_ids": 0, "duplicate_normalized_texts": 0,
                   "sources_exact_lowres_seed13": True, "distinct_paraphrases_per_source": k},
    }
    if promotion is not None:
        loc = promote_auggpt.locations(task, SEED, mode)
        record["promotion"] = {
            "record": _file(loc["record"], rows=False),
            "preserved_original": _file(loc["archive"]),
            "preserved_original_provenance": _file(loc["archive"].with_suffix(".provenance.json"), rows=False),
            "preserved_original_fingerprint": promotion["preserved_original"]["generation_fingerprint"],
            "retained_candidate": _file(loc["candidate"]),
            "retained_candidate_provenance": _file(loc["candidate"].with_suffix(".provenance.json"), rows=False),
        }
    return record


def bind_evidence(evidence_dir, artifacts, required=()):
    """Hash every #11 evidence file and require each batch's latest round to PASS on the frozen file."""
    evidence_dir = Path(evidence_dir)
    missing = [name for name in required if not (evidence_dir / name).exists()]
    if missing:
        raise FreezeError(f"issue #11 evidence missing: {missing}")
    files = {p.name: _sha(_lf(p.read_bytes())) for p in sorted(evidence_dir.iterdir()) if p.is_file()}
    load = lambda name: json.loads((evidence_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    rounds = sorted(int(n[5:-12]) for n in files if n.startswith("round") and n.endswith("_sample.json")
                    and "_recheck_" not in n)
    batches = {}
    for r in rounds:
        sample, outcome_name = load(f"round{r}_sample.json"), f"round{r}_outcome.json"
        if outcome_name not in files:
            raise FreezeError(f"round {r} has a sample but no outcome")
        outcome = load(outcome_name)
        if (outcome.get("sample_sha256") != files[f"round{r}_sample.json"]
                or outcome.get("judgments_sha256") != files.get(f"round{r}_judgments.json")):
            raise FreezeError(f"round {r} outcome is not bound to its sample/judgments")
        for key in sample["batches"]:
            batches[key] = (r, sample, outcome)       # later rounds replace earlier ones
    summary = {}
    for key, consumed in artifacts.items():
        if key not in batches:
            raise FreezeError(f"{key}: no review round covers this batch")
        r, sample, outcome = batches[key]
        result = outcome["batches"][key]
        if result["outcome"] != "PASS":
            raise FreezeError(f"{key}: latest review round {r} is {result['outcome']}; flagged batches block the freeze")
        reviewed_sha = sample["inputs"]["batches"][key]["input"]["sha256"]
        if reviewed_sha != consumed["consumed"]["sha256"]:
            raise FreezeError(f"{key}: latest review round {r} reviewed {reviewed_sha}, not the frozen file")
        summary[key] = {"latest_round": r, "outcome": result["outcome"], "reviewed": result["reviewed"],
                        "clear_errors": result["clear_errors"], "clear_error_ids": result["clear_error_ids"],
                        "ambiguous": result["ambiguous"], "reviewed_input_sha256": reviewed_sha}
    extra = {}
    if "round2_recheck_outcome.json" in files:
        out = load("round2_recheck_outcome.json")
        if out.get("sample_sha256") != files.get("round2_recheck_sample.json"):
            raise FreezeError("re-check outcome is not bound to its sample")
        extra["trec_auggpt_recheck"] = {k: v["clear_error_ids"] for k, v in out["batches"].items()}
    if "targeted_long_sources_adjudication_2026-11-01_outcome.json" in files:
        out = load("targeted_long_sources_adjudication_2026-11-01_outcome.json")
        if out.get("adjudication_sha256") != files.get("targeted_long_sources_adjudication_2026-11-01.json"):
            raise FreezeError("adjudication outcome is not bound to its adjudication file")
        extra["emotion_adjudication_clear_errors"] = sorted(
            rid for b in out["by_source"].values() for rid in b["clear_errors"])
    if "scope_scan_2026-11-01_ids.json" in files:
        scan = load("scope_scan_2026-11-01_ids.json")
        for task in ("trec", "emotion"):
            frozen = artifacts.get(f"{task}/auggpt", {}).get("consumed", {}).get("sha256")
            if scan.get(f"{task}_artifact_sha256") != frozen:
                raise FreezeError(f"scope scan was run on another {task}/auggpt artifact")
        extra["scope_scan_curated"] = scan.get("curated_followup_2026-11-01")
    return {"directory": "docs/reviews/issue11", "files_sha256_lf": files, "latest_round_by_batch": summary,
            "supplementary": extra,
            "note": "model-assisted (Claude Opus 5.5) and heuristic findings; not independent-human judgments"}


def build_freeze(mode="real", revision=None, evidence_dir=None, required_evidence=None):
    """Read-only full audit; returns the deterministic freeze manifest."""
    if not isinstance(revision, str) or len(revision) != 40:
        raise FreezeError("a 40-hex repository revision is required")
    artifacts = {}
    for task in TASKS:
        for setting in SETTINGS:
            artifacts[f"{task}/{setting}"] = (audit_auggpt(task, mode) if setting == "auggpt"
                                              else audit_selected(task, setting, mode))
    identities = {json.dumps(a["model_identity"], sort_keys=True) for a in artifacts.values()}
    if len(identities) != 1:
        raise FreezeError(f"artifacts carry different model identities: {identities}")
    references = {f"{t}/{n}": _file(real_path(t, n)) for t in TASKS for n in ("pool", f"lowres_seed{SEED}", "test")}
    if required_evidence is None:
        required_evidence = REQUIRED_REAL_EVIDENCE if mode == "real" else ()
    evidence = bind_evidence(evidence_dir or EVIDENCE_DIR, artifacts, required_evidence)
    manifest = {
        "schema_version": SCHEMA_VERSION, "freeze_id": FREEZE_ID, "scope_id": sel.SCOPE_ID, "seed": SEED,
        "generation_mode": mode, "real_run_evidence": mode == "real",
        "authority": ["docs/adr/0012-eod-exploratory-30-seed13.md", "docs/adr/0013-trec-auggpt-answer-type.md"],
        "repository_revision": revision,
        "accepted_model_identity": json.loads(identities.pop()),
        "approved_generator_pin": gen.APPROVED_GENERATORS.get(CFG["model"]) if mode == "real" else "fixture",
        "real_references": references,
        "artifacts": artifacts,
        "totals": {"selected_zero_few_shot_texts": sum(len(a["consumed"]["ids_in_file_order"])
                                                       for k, a in artifacts.items() if not k.endswith("auggpt")),
                   "auggpt_paraphrases": sum(len(a["consumed"]["ids_in_file_order"])
                                             for k, a in artifacts.items() if k.endswith("auggpt"))},
        "issue11_evidence": evidence,
        "decision": DECISION,
        "limitations": LIMITATIONS,
        "contains_corpus_text": False,
        "post_freeze_rule": DECISION["post_freeze_changes"],
    }
    _assert_no_corpus_text(manifest, mode)
    return manifest


def _assert_no_corpus_text(manifest, mode):
    texts = set()
    for task in TASKS:
        for name in ("pool", f"lowres_seed{SEED}", "test", "train"):
            texts |= {norm(r["text"]) for r in read_jsonl(real_path(task, name))}
        for setting in SETTINGS:
            path = (synth_path(task, "auggpt", seed=SEED, mode=mode) if setting == "auggpt"
                    else sel.selected_path(task, setting, mode))
            texts |= {norm(r["text"]) for r in _jsonl(path)}
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "text":
                    found.append("key 'text'")
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and norm(x) in texts:
            found.append(x[:40])
    walk(manifest)
    if found:
        raise FreezeError(f"freeze manifest would contain corpus text: {found[:3]}")


def repo_revision(require_clean=True):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeError("cannot read the repository revision with git") from exc
    if dirty and require_clean:
        raise FreezeError(f"tracked files are modified; commit or discard them before freezing:\n{dirty}")
    return rev


def revision_in_history(revision):
    """The recorded revision must exist and be an ancestor of (or equal to) HEAD."""
    try:
        ok = subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT,
                            capture_output=True).returncode == 0
    except OSError as exc:
        raise FreezeError("cannot check the recorded revision with git") from exc
    if not ok:
        raise FreezeError(f"recorded repository revision {revision} is not in this checkout's history")


def write_freeze(mode="real", revision=None, path=None, evidence_dir=None, required_evidence=None):
    path = Path(path or FREEZE_PATH)
    data = _canonical(build_freeze(mode, revision, evidence_dir, required_evidence))
    if path.exists():
        if _lf(path.read_bytes()) != data:
            raise FreezeError(f"{path} exists with different content; a freeze is never overwritten")
        return json.loads(data), _sha(data), False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return json.loads(data), _sha(data), True


def _differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _differences(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if a == b else [prefix.rstrip(".")]


def verify_freeze(mode="real", path=None, evidence_dir=None, required_evidence=None, check_revision=True):
    """Read-only: rebuild from the current files and require byte equality with the recorded freeze."""
    path = Path(path or FREEZE_PATH)
    if not path.exists():
        raise FreezeError(f"{path}: freeze manifest is missing")
    raw = _lf(path.read_bytes())
    try:
        recorded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FreezeError(f"{path}: freeze manifest is not valid JSON") from exc
    if not isinstance(recorded, dict) or recorded.get("schema_version") != SCHEMA_VERSION \
            or recorded.get("freeze_id") != FREEZE_ID or recorded.get("generation_mode") != mode:
        raise FreezeError(f"{path}: not a {FREEZE_ID} manifest for mode {mode!r}")
    if check_revision:
        revision_in_history(recorded.get("repository_revision"))
    rebuilt = build_freeze(mode, recorded.get("repository_revision"), evidence_dir, required_evidence)
    if _canonical(rebuilt) != raw:
        diffs = _differences(recorded, rebuilt)
        raise FreezeError(f"{path}: frozen handoff does not match the current files; differing: "
                          f"{diffs[:12] or ['serialization']}")
    return {"freeze_sha256": _sha(raw), "artifacts": {k: a["consumed"]["sha256"] for k, a in rebuilt["artifacts"].items()},
            "repository_revision": recorded["repository_revision"], "totals": rebuilt["totals"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true", help="read-only full audit; writes nothing")
    action.add_argument("--write", action="store_true", help="write the freeze manifest (refuses to overwrite)")
    action.add_argument("--verify", action="store_true", help="read-only fail-closed check of the written freeze")
    ap.add_argument("--manifest", default=None, help=f"default {FREEZE_PATH.relative_to(ROOT).as_posix()}")
    ap.add_argument("--evidence-dir", default=None, help="default docs/reviews/issue11")
    ap.add_argument("--dry-run", action="store_true", help="fake fixtures (never real-run evidence)")
    args = ap.parse_args(argv)
    mode = "fake" if args.dry_run else "real"
    note = "" if mode == "real" else " (FAKE fixture; not real-run evidence)"
    try:
        if args.audit:
            manifest = build_freeze(mode, repo_revision(require_clean=False), args.evidence_dir)
            for key, a in manifest["artifacts"].items():
                ev = manifest["issue11_evidence"]["latest_round_by_batch"][key]
                print(f"AUDIT PASS{note} {key}: rows={a['consumed']['rows']} sha256={a['consumed']['sha256']} "
                      f"fingerprint={a['generation_fingerprint'][:12]} review=round{ev['latest_round']} "
                      f"{ev['outcome']} clear_errors={ev['clear_errors']}")
            print(f"AUDIT PASS{note}: totals {manifest['totals']}; nothing written")
        elif args.write:
            if Path(args.manifest or FREEZE_PATH).exists():
                # Never rewritten: an existing freeze is only re-verified under its recorded revision.
                result = verify_freeze(mode, args.manifest, args.evidence_dir)
                print(f"FREEZE ALREADY PRESENT{note} (not rewritten): sha256 {result['freeze_sha256']} "
                      f"revision {result['repository_revision']}")
            else:
                manifest, digest, _ = write_freeze(mode, repo_revision(), args.manifest, args.evidence_dir)
                print(f"FREEZE WRITTEN{note}: sha256 {digest} revision {manifest['repository_revision']}")
                result = verify_freeze(mode, args.manifest, args.evidence_dir)
            print(f"VERIFY PASS{note}: {json.dumps(result)}")
        else:
            result = verify_freeze(mode, args.manifest, args.evidence_dir)
            print(f"VERIFY PASS{note}: {json.dumps(result)}")
    except (FreezeError, ValueError, OSError) as exc:
        print(f"FAIL{note}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
