"""Reviewed, explicit promotion of a corrected AugGPT candidate (ADR 0013, issue #9 / #11).

The corrected artifact is generated into an isolated candidate location
(``generate --setting auggpt --task trec --seed 13 --candidate``) while the active artifact stays
untouched. This module replaces the active artifact only on explicit request, after the operator has
reviewed a read-only plan and passes back the exact hashes it printed:

    python -m ws1_generation.promote_auggpt --task trec --seed 13 --plan
    python -m ws1_generation.promote_auggpt --task trec --seed 13 --promote \
        --candidate-sha256 <sha> --original-sha256 <sha>
    python -m ws1_generation.promote_auggpt --task trec --seed 13 --verify-promotion

Promotion copies the original JSONL and sidecar byte-for-byte to
``<task>/superseded/<correction>/`` (never moved or rewritten), checks the copies, writes
``auggpt_seed<seed>.promotion.json`` with every hash, then atomically replaces the active JSONL and
sidecar with the candidate's exact bytes. The candidate stays in place. An interrupted promotion is
completed by rerunning the same command. No model is called and no row is edited. Passing these checks
is structural evidence only: semantic acceptance needs the fresh #11 review round.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

from common import data_dir, norm, read_jsonl, synth_path
from ws1_generation import generate as gen

RECORD_SCHEMA_VERSION = 1


class PromotionError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path):
    try:
        return Path(path).resolve().relative_to(data_dir().resolve()).as_posix()
    except ValueError:
        return str(path)


def _sidecar(path):
    return Path(path).with_suffix(".provenance.json")


def _file(path):
    raw = Path(path).read_bytes()
    return {"path": _rel(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def locations(task, seed, mode="real"):
    correction = gen.auggpt_correction(task, seed)
    if correction is None:
        raise PromotionError(f"no approved AugGPT correction for {task} seed {seed}")
    active = synth_path(task, "auggpt", seed=seed, mode=mode)
    return {
        "correction": correction,
        "active": active,
        "candidate": gen.auggpt_candidate_path(task, seed, mode),
        "archive": active.parent / "superseded" / correction["version"] / active.name,
        "record": gen.promotion_record_path(task, seed, mode),
    }


def _no_pending_journals(*paths):
    for path in paths:
        if gen.generation_journal_path(path).exists():
            raise PromotionError(f"{path}: unrecovered generation journal; resolve it before promotion")


def _verify(task, seed, mode, path, historical):
    try:
        return gen.verify_auggpt(task, seed, mode=mode, path=path, historical=historical)
    except ValueError as exc:
        raise PromotionError(str(exc)) from exc


def plan(task, seed, mode="real"):
    """Read-only: verify the candidate and the original, and report what promotion would do."""
    loc = locations(task, seed, mode)
    active, candidate = loc["active"], loc["candidate"]
    _no_pending_journals(active, candidate)
    if loc["record"].exists():
        raise PromotionError(f"{loc['record']} exists: a promotion was started; use --verify-promotion "
                             "or rerun --promote with the recorded hashes")
    cand = _verify(task, seed, mode, candidate, historical=False)
    if not active.exists() or not _sidecar(active).exists():
        raise PromotionError(f"{active}: original active artifact or sidecar is missing")
    orig = _verify(task, seed, mode, active, historical=True)
    if orig["corrected_output"] is not False:
        raise PromotionError(f"{active}: the active artifact is not a pre-correction original")
    sources_c = {r["source_ids"][0] for r in read_jsonl(candidate)}
    sources_o = {r["source_ids"][0] for r in read_jsonl(active)}
    if sources_c != sources_o:
        raise PromotionError("candidate and original cover different sources")
    shared = {norm(r["text"]) for r in read_jsonl(candidate)} & {norm(r["text"]) for r in read_jsonl(active)}
    return {
        "task": task, "repetition": seed, "generation_mode": mode,
        "real_run_evidence": mode == "real",
        "correction": loc["correction"]["version"], "decision": loc["correction"]["decision"],
        "original": {"artifact": _file(active), "provenance": _file(_sidecar(active)),
                     "generation_fingerprint": orig["generation_fingerprint"],
                     "row_fingerprints": orig["row_fingerprints"], "total_count": orig["total_count"]},
        "candidate": {"artifact": _file(candidate), "provenance": _file(_sidecar(candidate)),
                      "generation_fingerprint": cand["generation_fingerprint"],
                      "total_count": cand["total_count"], "source_count": cand["source_count"]},
        "archive_destination": _rel(loc["archive"]),
        "record_destination": _rel(loc["record"]),
        "candidate_texts_also_in_original": len(shared),
        "confirm_with": ["--candidate-sha256", cand["artifact_sha256"],
                         "--original-sha256", orig["artifact_sha256"]],
    }


def _write_new_or_same(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise PromotionError(f"{path} exists with different content; refusing to overwrite it")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _replace(path, raw):
    tmp = path.with_name(path.name + ".promote.tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _write_record(path, record):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def promote(task, seed, candidate_sha256, original_sha256, mode="real"):
    """Replace the active artifact with the verified candidate, preserving the original."""
    loc = locations(task, seed, mode)
    active, candidate, archive, record_path = loc["active"], loc["candidate"], loc["archive"], loc["record"]
    _no_pending_journals(active, candidate)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if (record["candidate"]["artifact"]["sha256"] != candidate_sha256
                or record["original"]["artifact"]["sha256"] != original_sha256):
            raise PromotionError(f"{record_path} records different hashes than the ones confirmed")
        if record["status"] == "complete":
            return verify_promotion(task, seed, mode)
    else:
        if not candidate.exists() or _sha(candidate) != candidate_sha256:
            raise PromotionError("candidate hash differs from --candidate-sha256; rerun --plan and review")
        if not active.exists() or _sha(active) != original_sha256:
            raise PromotionError("active artifact hash differs from --original-sha256; rerun --plan and review")
        reviewed = plan(task, seed, mode)
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "status": "in_progress",
            "task": task, "repetition": seed, "generation_mode": mode,
            "correction": reviewed["correction"], "decision": reviewed["decision"],
            "original": {**reviewed["original"], "archived_artifact": _rel(archive),
                         "archived_provenance": _rel(_sidecar(archive))},
            "candidate": reviewed["candidate"],
            "candidate_texts_also_in_original": reviewed["candidate_texts_also_in_original"],
            "started_at": now, "completed_at": None,
            "note": "Original preserved byte-for-byte; candidate bytes copied unchanged. Structural "
                    "evidence only; semantic acceptance requires a fresh issue #11 review round.",
        }
        # 1. Preserve the original before anything changes.
        _write_new_or_same(archive, active.read_bytes())
        _write_new_or_same(_sidecar(archive), _sidecar(active).read_bytes())
        _write_record(record_path, record)

    original, cand = record["original"], record["candidate"]
    if (_sha(archive) != original["artifact"]["sha256"]
            or _sha(_sidecar(archive)) != original["provenance"]["sha256"]):
        raise PromotionError(f"{archive}: preserved original does not match the recorded hashes")
    if (_sha(candidate) != cand["artifact"]["sha256"]
            or _sha(_sidecar(candidate)) != cand["provenance"]["sha256"]):
        raise PromotionError(f"{candidate}: candidate changed since promotion started")
    # 2. Each active file must be either the recorded original or the candidate (resumable).
    for target, src, orig_sha in ((active, candidate, original["artifact"]["sha256"]),
                                  (_sidecar(active), _sidecar(candidate), original["provenance"]["sha256"])):
        current = _sha(target) if target.exists() else None
        if current not in (orig_sha, _sha(src)):
            raise PromotionError(f"{target}: unexpected content; refusing to replace it")
        if current != _sha(src):
            _replace(target, src.read_bytes())

    summary = _verify(task, seed, mode, active, historical=False)
    record.update(status="complete", completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  promoted={"artifact": _file(active), "provenance": _file(_sidecar(active)),
                            "generation_fingerprint": summary["generation_fingerprint"]})
    _write_record(record_path, record)
    return verify_promotion(task, seed, mode)


def verify_promotion(task, seed, mode="real"):
    """Read-only: the active artifact is the recorded candidate and the original is preserved."""
    loc = locations(task, seed, mode)
    active, candidate, archive, record_path = loc["active"], loc["candidate"], loc["archive"], loc["record"]
    if not record_path.exists():
        raise PromotionError(f"{record_path}: no promotion record")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "complete":
        raise PromotionError(f"{record_path}: promotion is {record.get('status')!r}; rerun --promote")
    checks = (
        (active, record["candidate"]["artifact"]["sha256"]),
        (_sidecar(active), record["candidate"]["provenance"]["sha256"]),
        (candidate, record["candidate"]["artifact"]["sha256"]),
        (_sidecar(candidate), record["candidate"]["provenance"]["sha256"]),
        (archive, record["original"]["artifact"]["sha256"]),
        (_sidecar(archive), record["original"]["provenance"]["sha256"]),
    )
    for path, expected in checks:
        if not path.exists() or _sha(path) != expected:
            raise PromotionError(f"{path}: missing or does not match the promotion record")
    promoted = _verify(task, seed, mode, active, historical=False)
    preserved = _verify(task, seed, mode, archive, historical=True)
    if promoted["corrected_output"] is not True or preserved["corrected_output"] is not False:
        raise PromotionError("promoted/preserved identities are not corrected/superseded as required")
    return {
        "task": task, "repetition": seed, "generation_mode": mode,
        "real_run_evidence": mode == "real", "correction": record["correction"],
        "promotion_record": {"path": _rel(record_path), "sha256": _sha(record_path)},
        "active": {"artifact_sha256": promoted["artifact_sha256"],
                   "provenance_sha256": promoted["provenance_sha256"],
                   "generation_fingerprint": promoted["generation_fingerprint"],
                   "total_count": promoted["total_count"]},
        "preserved_original": {"path": _rel(archive), "artifact_sha256": preserved["artifact_sha256"],
                               "provenance_sha256": preserved["provenance_sha256"],
                               "generation_fingerprint": preserved["generation_fingerprint"],
                               "total_count": preserved["total_count"]},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", required=True, type=int)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true", help="read-only review of what promotion would do")
    action.add_argument("--promote", action="store_true", help="replace the active artifact (needs both hashes)")
    action.add_argument("--verify-promotion", action="store_true", help="read-only check of a completed promotion")
    ap.add_argument("--candidate-sha256")
    ap.add_argument("--original-sha256")
    ap.add_argument("--dry-run", action="store_true", help="fixture storage (synthetic-fixtures/)")
    args = ap.parse_args(argv)
    mode = "fake" if args.dry_run else "real"
    if args.promote and not (args.candidate_sha256 and args.original_sha256):
        ap.error("--promote needs --candidate-sha256 and --original-sha256 from --plan")
    try:
        if args.plan:
            result = plan(args.task, args.seed, mode)
        elif args.promote:
            result = promote(args.task, args.seed, args.candidate_sha256, args.original_sha256, mode)
        else:
            result = verify_promotion(args.task, args.seed, mode)
    except (PromotionError, OSError) as exc:
        print(f"FAIL {args.task} auggpt seed {args.seed}: {exc}")
        sys.exit(1)
    note = "" if mode == "real" else " (FAKE fixture; not real-run evidence)"
    print(f"PASS{note} {args.task} auggpt seed {args.seed}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
