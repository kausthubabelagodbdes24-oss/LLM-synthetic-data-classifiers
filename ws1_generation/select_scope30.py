"""Issue #10: deterministic scope30-v1 selection of seed-13 zero-shot / few-shot corpora (ADR 0012).

Original producer artifacts (``<setting>_seed13.jsonl`` + ``.provenance.json``) are read-only here.
For each task x setting this tool validates the whole original artifact, then selects exactly 30
rows per label by ascending SHA-256 of the UTF-8 string
``scope30-v1|13|<task>|<setting>|<row-id>`` (row ID as tie-breaker), and writes two derived files
next to the original:

    <setting>_seed13.scope30-v1.selection.json   manifest: selected IDs, original hashes/fingerprints
    <setting>_seed13.scope30-v1.selected.jsonl   the selected rows, byte-identical to their originals

    python -m ws1_generation.select_scope30 --seed 13 --status    # read-only per-label counts/shortfall
    python -m ws1_generation.select_scope30 --seed 13             # validate + write derived selection
    python -m ws1_generation.select_scope30 --seed 13 --verify    # read-only, independent re-derivation

Selection never looks at test performance or text content; it depends only on row IDs. A label
with fewer than 30 valid rows is a shortfall: resume compatible generation with
``python -m ws1_generation.generate --setting <setting> --task <task> --seed 13 --target 30``.
The producer's recorded targets (pilot 10, historical 100) stay in its sidecar sessions; the
selected 30 is a separate, independently validated count. ``--dry-run`` uses fake fixtures,
which are never real-run evidence. Nothing here freezes data or claims quality acceptance.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter

from common import CFG, TASKS, labels, norm, read_jsonl, real_path, synth_path
from ws1_generation import generate as gen

SCOPE_ID = "scope30-v1"
SCOPE_SEED = 13
SELECTED_PER_LABEL = 30
SETTINGS = ("zero_shot", "few_shot")
MANIFEST_SCHEMA_VERSION = 1
SELECTION_RULE = (
    "per label, ascending SHA-256 hex of UTF-8 'scope30-v1|13|<task>|<setting>|<row-id>', "
    "row id as tie-breaker; first 30 (ADR 0012)"
)


class SelectionError(ValueError):
    """The artifact cannot (yet) yield a valid scope30-v1 selection."""


def selection_key(task, setting, row_id, seed=SCOPE_SEED):
    return hashlib.sha256(f"{SCOPE_ID}|{seed}|{task}|{setting}|{row_id}".encode("utf-8")).hexdigest()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def manifest_path(task, setting, mode="real"):
    path = synth_path(task, setting, seed=SCOPE_SEED, mode=mode)
    return path.with_name(f"{setting}_seed{SCOPE_SEED}.{SCOPE_ID}.selection.json")


def selected_path(task, setting, mode="real"):
    path = synth_path(task, setting, seed=SCOPE_SEED, mode=mode)
    return path.with_name(f"{setting}_seed{SCOPE_SEED}.{SCOPE_ID}.selected.jsonl")


def _check_setting(setting):
    if setting not in SETTINGS:
        raise SelectionError(f"scope30-v1 selection applies only to {SETTINGS}, not {setting!r}")


def load_validated_artifact(task, setting, mode="real"):
    """Read-only, fail-closed validation of a complete original artifact.

    Requires the approved generator (or the fixture identity for fake artifacts), recomputes the
    generation fingerprint from the current prompts/options/sources, revalidates every row with the
    producer's resume rules (schema, IDs, labels, normalized duplicates, demonstrations), checks the
    sidecar counts, and rejects held-out test overlap and copied few-shot demonstrations.
    """
    _check_setting(setting)
    path = synth_path(task, setting, seed=SCOPE_SEED, mode=mode)
    prov_path = path.with_suffix(".provenance.json")
    if gen.generation_journal_path(path).exists():
        raise SelectionError(f"{path}: unrecovered interrupted batch; rerun the same generation command first")
    if not path.exists() or not prov_path.exists():
        raise SelectionError(f"{path}: artifact or provenance sidecar is missing")
    raw = path.read_bytes()
    prov_raw = prov_path.read_bytes()
    try:
        recorded = json.loads(prov_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{path}: invalid provenance sidecar") from exc
    model_identity = recorded.get("model_identity")
    try:
        gen.require_approved_model(model_identity, mode)
    except ValueError as exc:
        raise SelectionError(f"{path}: {exc}") from exc

    pool = read_jsonl(real_path(task, "pool"))
    test_rows = read_jsonl(real_path(task, "test"))
    if not test_rows:
        raise SelectionError(f"{task}: held-out test split is missing; cannot check test-text exclusion")
    if setting == "few_shot" and not pool:
        raise SelectionError(f"{task}: demonstration pool is missing")
    source_identity = {
        "demonstration_pool": gen.rows_identity(pool) if setting == "few_shot" else None,
        "held_out_test": gen.rows_identity(test_rows),
    }
    identity, fingerprint = gen.build_generation_identity(
        task, setting, SCOPE_SEED, mode, model_identity, source_identity
    )
    rows = gen.read_generation_rows(path)
    try:
        gen.validate_existing_rows(rows, path, task, setting, SCOPE_SEED, mode, fingerprint, pool)
        provenance = gen.read_existing_provenance(path, rows, identity, fingerprint)
    except ValueError as exc:
        raise SelectionError(str(exc)) from exc
    if provenance is None:
        raise SelectionError(f"{path}: artifact is empty")
    counts = {lab: sum(row["label"] == lab for row in rows) for lab in labels(task)}
    if provenance.get("counts_by_label") != counts:
        raise SelectionError(f"{path}: provenance counts_by_label does not match the rows")

    test_by_text = {}
    for row in test_rows:
        test_by_text.setdefault(norm(row["text"]), row["id"])
    leaks = [(row["id"], test_by_text[norm(row["text"])]) for row in rows if norm(row["text"]) in test_by_text]
    if leaks:
        raise SelectionError(
            f"{path}: {len(leaks)} row(s) match held-out test text (first 10: {leaks[:10]}); "
            "move the artifact and sidecar aside and regenerate before the freeze"
        )
    if setting == "few_shot":
        pool_text = {row["id"]: norm(row["text"]) for row in pool}
        copies = [row["id"] for row in rows if norm(row["text"]) in {pool_text[s] for s in row["source_ids"]}]
        if copies:
            raise SelectionError(f"{path}: rows copy their own demonstrations: {copies[:10]}")

    return {
        "task": task,
        "setting": setting,
        "mode": mode,
        "path": path,
        "raw": raw,
        "prov_raw": prov_raw,
        "rows": rows,
        "provenance": provenance,
        "fingerprint": fingerprint,
        "counts": counts,
        "test_texts": set(test_by_text),
    }


def shortfall(counts, per_label=SELECTED_PER_LABEL):
    return {lab: per_label - n for lab, n in counts.items() if n < per_label}


def select_rows(rows, task, setting, per_label=SELECTED_PER_LABEL):
    """Return {label: [rows in rank order]} using only the scope30-v1 ID hash."""
    ranked = {}
    for lab in labels(task):
        group = [row for row in rows if row["label"] == lab]
        if len(group) < per_label:
            raise SelectionError(
                f"{task} {setting} seed {SCOPE_SEED} label {lab!r}: {len(group)} valid rows, need {per_label}; "
                "resume compatible generation for the shortfall"
            )
        group.sort(key=lambda row: (selection_key(task, setting, row["id"]), row["id"]))
        ranked[lab] = group[:per_label]
    return ranked


def build_selection(artifact):
    """Deterministically derive (manifest, selected_jsonl_bytes) from a validated artifact."""
    task, setting, rows = artifact["task"], artifact["setting"], artifact["rows"]
    provenance = artifact["provenance"]
    ranked = select_rows(rows, task, setting)
    selected_ids = {row["id"] for group in ranked.values() for row in group}
    selected = [row for row in rows if row["id"] in selected_ids]   # original file order
    selected_bytes = gen.serialize_jsonl_rows(selected)

    # Independent count/uniqueness checks on the selected view itself.
    sel_counts = Counter(row["label"] for row in selected)
    if (set(sel_counts) != set(labels(task))
            or any(sel_counts[lab] != SELECTED_PER_LABEL for lab in labels(task))
            or len(selected) != SELECTED_PER_LABEL * len(labels(task))
            or len({row["id"] for row in selected}) != len(selected)
            or len({norm(row["text"]) for row in selected}) != len(selected)):
        raise SelectionError(f"{task} {setting}: selected view failed exact-count/uniqueness checks")
    original_lines = set(artifact["raw"].decode("utf-8").splitlines())
    if any(line not in original_lines for line in selected_bytes.decode("utf-8").splitlines()):
        raise SelectionError(f"{task} {setting}: selected rows are not byte-identical to the original")

    sessions = provenance["sessions"]
    selected_demos = sorted({sid for row in selected for sid in row["source_ids"]})
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "scope_id": SCOPE_ID,
        "authority": "docs/adr/0012-eod-exploratory-30-seed13.md",
        "selection_rule": SELECTION_RULE,
        "task": task,
        "setting": setting,
        "repetition": SCOPE_SEED,
        "generation_mode": artifact["mode"],
        "real_run_evidence": artifact["mode"] == "real",
        "selected_per_label": SELECTED_PER_LABEL,
        "selected_total": len(selected),
        "selected_counts_by_label": {lab: sel_counts[lab] for lab in labels(task)},
        "selected_ids_by_label": {lab: [row["id"] for row in ranked[lab]] for lab in labels(task)},
        "selected_ids": sorted(selected_ids),
        "selected_rows_identity": gen.rows_identity(selected),
        "selected_jsonl": {
            "name": selected_path(task, setting, artifact["mode"]).name,
            "sha256": _sha256(selected_bytes),
            "byte_count": len(selected_bytes),
        },
        "selected_distinct_demonstrations_count": len(selected_demos),
        "selected_distinct_demonstration_ids": selected_demos,
        "original_artifact": {
            "name": artifact["path"].name,
            "sha256": _sha256(artifact["raw"]),
            "byte_count": len(artifact["raw"]),
            "row_count": len(rows),
            "counts_by_label": artifact["counts"],
            "excess_by_label": {lab: n - SELECTED_PER_LABEL for lab, n in artifact["counts"].items()},
            "provenance_name": artifact["path"].with_suffix(".provenance.json").name,
            "provenance_sha256": _sha256(artifact["prov_raw"]),
        },
        "producer": {
            "generation_fingerprint": provenance["generation_fingerprint"],
            "row_fingerprints": dict(Counter(row["generation_fingerprint"] for row in rows)),
            "model_identity": provenance["model_identity"],
            "temperature": provenance.get("temperature"),
            "recorded_target_per_label": provenance.get("target_per_label"),
            "recorded_is_complete": provenance.get("is_complete"),
            "session_targets_per_label": [s.get("target_per_label") for s in sessions],
            "session_statuses": [s.get("status") for s in sessions],
            "session_count": len(sessions),
            "elapsed_seconds_total": round(sum(s.get("elapsed_seconds") or 0 for s in sessions), 6),
            "cumulative_stats": provenance.get("cumulative_stats"),
            "distinct_demonstrations_count": provenance.get("distinct_demonstrations_count"),
        },
        "checks": {
            "approved_model_identity": True,
            "fingerprint_recomputed_from_current_prompts_and_sources": True,
            "all_rows_valid_schema_ids_labels": True,
            "no_normalized_duplicates_in_original": True,
            "no_held_out_test_text_overlap": True,
            "no_copied_demonstrations": True,
            "exact_selected_counts": True,
        },
        "not_claimed": [
            "semantic/quality acceptance (issue #11/#12)",
            "frozen status (issue #12)",
        ],
    }
    return manifest, selected_bytes


def _manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_selection(task, setting, mode="real"):
    """Validate, select and write the derived files. Never touches the original artifact."""
    artifact = load_validated_artifact(task, setting, mode)
    manifest, selected_bytes = build_selection(artifact)
    m_path, s_path = manifest_path(task, setting, mode), selected_path(task, setting, mode)
    m_bytes = _manifest_bytes(manifest)
    for existing, new in ((m_path, m_bytes), (s_path, selected_bytes)):
        if existing.exists() and existing.read_bytes() != new:
            raise SelectionError(
                f"{existing} already exists with different content (the original changed since it was "
                "written). Move the old derived files aside to preserve them, then rerun."
            )
    _atomic_write_bytes(s_path, selected_bytes)
    _atomic_write_bytes(m_path, m_bytes)
    return summarize(manifest)


def verify_selection(task, setting, mode="real"):
    """Read-only: re-derive the selection from the originals and require exact equality."""
    artifact = load_validated_artifact(task, setting, mode)
    expected_manifest, expected_selected = build_selection(artifact)
    m_path, s_path = manifest_path(task, setting, mode), selected_path(task, setting, mode)
    if not m_path.exists() or not s_path.exists():
        raise SelectionError(f"{m_path.name} / {s_path.name}: derived selection files are missing")
    try:
        recorded = json.loads(m_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{m_path}: invalid manifest") from exc
    if recorded != expected_manifest:
        differing = sorted(k for k in set(recorded) | set(expected_manifest)
                           if recorded.get(k) != expected_manifest.get(k))
        raise SelectionError(f"{m_path}: manifest does not match re-derivation; differing keys {differing}")
    selected_raw = s_path.read_bytes()
    if selected_raw != expected_selected:
        raise SelectionError(f"{s_path}: selected JSONL does not match re-derivation")
    # Independent checks on the file as consumers will read it.
    rows = [json.loads(line) for line in selected_raw.decode("utf-8").splitlines() if line.strip()]
    counts = Counter(row["label"] for row in rows)
    problems = []
    if {lab: counts.get(lab, 0) for lab in labels(task)} != {lab: SELECTED_PER_LABEL for lab in labels(task)}:
        problems.append(f"counts {dict(counts)}")
    if len(rows) != SELECTED_PER_LABEL * len(labels(task)):
        problems.append(f"total {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        problems.append("duplicate ids")
    if len({norm(row["text"]) for row in rows}) != len(rows):
        problems.append("duplicate normalized text")
    if any(norm(row["text"]) in artifact["test_texts"] for row in rows):
        problems.append("held-out test overlap")
    if sorted(row["id"] for row in rows) != recorded["selected_ids"]:
        problems.append("ids differ from manifest")
    if problems:
        raise SelectionError(f"{s_path}: {problems}")
    return summarize(recorded)


def status(task, setting, mode="real"):
    """Read-only: per-label counts and shortfall against the selected target of 30."""
    artifact = load_validated_artifact(task, setting, mode)
    return {
        "task": task, "setting": setting, "repetition": SCOPE_SEED, "generation_mode": mode,
        "total": len(artifact["rows"]), "counts_by_label": artifact["counts"],
        "shortfall_to_30": shortfall(artifact["counts"]),
        "recorded_target_per_label": artifact["provenance"].get("target_per_label"),
        "generation_fingerprint": artifact["fingerprint"],
    }


def summarize(manifest):
    return {
        "task": manifest["task"],
        "setting": manifest["setting"],
        "repetition": manifest["repetition"],
        "real_run_evidence": manifest["real_run_evidence"],
        "selected_total": manifest["selected_total"],
        "selected_counts_by_label": manifest["selected_counts_by_label"],
        "original_rows": manifest["original_artifact"]["row_count"],
        "original_sha256": manifest["original_artifact"]["sha256"],
        "provenance_sha256": manifest["original_artifact"]["provenance_sha256"],
        "generation_fingerprint": manifest["producer"]["generation_fingerprint"],
        "selected_jsonl_sha256": manifest["selected_jsonl"]["sha256"],
        "selected_distinct_demonstrations": manifest["selected_distinct_demonstrations_count"],
    }


def cross_setting_overlap(task, mode="real"):
    """Informational: normalized texts shared by the two selected corpora of one task."""
    texts = {}
    for setting in SETTINGS:
        path = selected_path(task, setting, mode)
        if not path.exists():
            return None
        texts[setting] = {norm(row["text"]) for row in read_jsonl(path)}
    return len(texts["zero_shot"] & texts["few_shot"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", required=True, help="must be 13 (scope30-v1 is seed 13 only)")
    ap.add_argument("--task", default="all", choices=["all"] + TASKS)
    ap.add_argument("--setting", default="all", choices=["all", *SETTINGS])
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="read-only counts and shortfall to 30/label")
    group.add_argument("--verify", action="store_true", help="read-only re-derivation of written selections")
    ap.add_argument("--dry-run", action="store_true", help="use fake fixtures (never real-run evidence)")
    args = ap.parse_args(argv)
    if str(args.seed) != str(SCOPE_SEED):
        ap.error(f"scope30-v1 selection is defined for seed {SCOPE_SEED} only")
    if CFG.get("per_label") != SELECTED_PER_LABEL or CFG.get("train_per_label") != SELECTED_PER_LABEL:
        ap.error("config.yaml per_label/train_per_label are not 30; scope30-v1 does not apply")
    mode = "fake" if args.dry_run else "real"
    action = "STATUS" if args.status else ("VERIFY" if args.verify else "SELECT")
    func = status if args.status else (verify_selection if args.verify else write_selection)
    failures = 0
    tasks = TASKS if args.task == "all" else [args.task]
    for task in tasks:
        for setting in SETTINGS if args.setting == "all" else [args.setting]:
            try:
                summary = func(task, setting, mode)
                note = "" if mode == "real" else " (FAKE fixture; not real-run evidence)"
                ok = not (args.status and summary["shortfall_to_30"])
                failures += 0 if ok else 1
                word = "PASS" if ok else "SHORT"
                print(f"{action} {word}{note} {task} {setting} seed {SCOPE_SEED}: {json.dumps(summary)}")
            except (SelectionError, ValueError) as exc:
                failures += 1
                print(f"{action} FAIL {task} {setting} seed {SCOPE_SEED}: {exc}")
    if args.verify and not failures:
        for task in tasks:
            overlap = cross_setting_overlap(task, mode)
            if overlap is not None:
                print(f"INFO {task}: {overlap} normalized text(s) shared by selected zero_shot and few_shot")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
