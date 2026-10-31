"""Issue #13: train the twelve-key scope30-v1 seed-13 grid on the accepted freeze only (ADR 0012).

    python -m ws2_training.train_scope30 --check-inputs                 # read-only: freeze + all 12 training sets
    python -m ws2_training.train_scope30 --results-dir DIR              # train missing keys, resume exact matches
    python -m ws2_training.train_scope30 --results-dir DIR --replace trec/C2   # explicit targeted rerun
    python -m ws2_training.train_scope30 --results-dir DIR --validate   # read-only: exactly 12 valid keys
    ... --dry-run                                                        # FAKE fixtures; never real evidence

Fail-closed order: ``freeze_scope30.verify_freeze`` must pass (and, for real runs, the manifest must be the
accepted ``docs/freeze/scope30-v1-seed13.freeze.json`` with canonical SHA-256 ``ACCEPTED_FREEZE_SHA256``)
before any training set is built, any classifier is fitted or any result is written. Synthetic inputs are read
only from the paths the freeze names and must match its hashes and ID order; default/legacy synthetic paths
are never consulted. Real train/test files must match the canonical #5 hashes.

Grid: tasks trec/emotion x conditions C1 C2 C3 C4 C5a C5b x seed 13 = exactly twelve keys.
    C1  180 = 30/label real, seed-13 ``per_label_sample`` of the unchanged training split (as #7)
    C2  180 = the frozen selected zero-shot corpus
    C3  180 = the frozen selected few-shot corpus
    C4  180 = 15/label real (seed-13 ``per_label_sample``) + 15/label of the C3 corpus (``C4_SYNTHETIC_RULE``)
    C5a  60 = the exact frozen ``lowres_seed13`` sources, 10/label
    C5b 420 = those 60 sources + their 360 frozen AugGPT paraphrases, six distinct valid ones per source

Classifier (frozen): sentence-transformers/all-MiniLM-L6-v2 embeddings (L2-normalised, not fine-tuned) and
LogisticRegression(C=1.0, max_iter=2000, random_state=13).

Results (fresh directory, never a historical one): ``grid.json`` (directory identity), one provenance record
per key in ``records/``, derived ``runs.csv`` (six columns, for #14) and ``grid_complete.json`` once exactly
twelve valid keys exist. A key is resumed only if its record's complete provenance binding equals the binding
recomputed now; a stale or invalid record stops the run unless the key is named in ``--replace``, in which case
the old record is moved to ``superseded/`` (never deleted). Unrequested keys are never touched.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from common import CFG, ROOT, TASKS, data_dir, labels, norm, per_label_sample, real_path, synth_path
from ws1_generation import freeze_scope30 as fz
from ws1_generation import select_scope30 as sel
from ws2_training import prepare_data

SCHEMA_VERSION = 1
RESULT_KIND = "scope30-v1-seed13-frozen-grid"
SCOPE_ID = "scope30-v1"
SEED = 13
TASK_ORDER = ("trec", "emotion")
CONDITIONS = ("C1", "C2", "C3", "C4", "C5a", "C5b")
PER_LABEL = 30
C4_REAL_PER_LABEL = 15
C4_SYNTH_PER_LABEL = 15
LOWRES_PER_LABEL = 10
AUGGPT_K = 6
N_LABELS = 6
EXPECTED_COUNTS = {"C1": 180, "C2": 180, "C3": 180, "C4": 180, "C5a": 60, "C5b": 420}
EXPECTED_PER_LABEL = {"C1": 30, "C2": 30, "C3": 30, "C4": 30, "C5a": 10, "C5b": 70}

ACCEPTED_FREEZE_PATH = fz.FREEZE_PATH
ACCEPTED_FREEZE_SHA256 = "dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1"
ACCEPTED_FREEZE_MERGE = "ef53a60e195b29c5562f16315785936eb8f09ddf"   # PR #32 merge accepting the freeze

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SETTINGS = {"normalize_embeddings": True, "batch_size": 64, "device": "cpu", "fine_tuned": False}
# Lead approval, PR #33 review 5724732904 (1 November 2026): this exact cached snapshot, pinned and re-verified.
APPROVED_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CLASSIFIER = {"type": "sklearn.linear_model.LogisticRegression", "C": 1.0, "max_iter": 2000, "random_state": SEED}
SHUFFLE = "random.Random(13).shuffle of the construction order before fitting (as #7)"
HASHING_FIXTURE = "HashingVectorizer(n_features=1024, ngram_range=(1, 2), alternate_sign=False)"

C4_SYNTHETIC_RULE = (
    "per label, the first 15 rows of the frozen selected few-shot (C3) corpus in ascending SHA-256 hex of UTF-8 "
    "'scope30-v1|13|<task>|few_shot|<row-id>', row id as tie-breaker: the first half of the same ADR 0012 rank "
    "order that selected the 30; no test data or performance is used")
CONSTRUCTION = {
    "C1": "per_label_sample(real train, 30, seed 13, exact) - identical to the #7 baseline sampler",
    "C2": "frozen selected zero-shot corpus, file order",
    "C3": "frozen selected few-shot corpus, file order",
    "C4": "per_label_sample(real train, 15, seed 13, exact) followed by " + C4_SYNTHETIC_RULE,
    "C5a": "frozen lowres_seed13 sources, file order",
    "C5b": "C5a rows followed by the frozen AugGPT artifact in file order",
}

# Lead decision (issue #13, 1 November 2026, "Pin + disclose"): the canonical, unchanged seed-13 TREC sample
# contains one official-dataset train/test duplicate. Only this exact pair, matched by both IDs and the
# normalized-text hash, may appear in TREC C1/C4 real rows; it is recorded in every affected result.
# Every other train/test overlap (any ID, any synthetic/C5 text, any other real text) fails closed.
PINNED_OFFICIAL_DUPLICATES = (
    {"task": "trec", "train_id": "trec.train.05262", "test_id": "trec.test.00378",
     "normalized_text_sha256": "ba6af17c366034f8dca2abff920ba01cc773fc77eda6188a1127b8190b324d2e",
     "conditions": ["C1", "C4"],
     "decision": "lead decision on issue #13 (1 November 2026): pin and disclose; canonical sampling unchanged"},
)

MARKER = "grid.json"
RECORDS = "records"
SUPERSEDED = "superseded"
RUNS_CSV = "runs.csv"
COMPLETE = "grid_complete.json"
RUN_COLUMNS = ["task", "condition", "seed", "n_train", "macro_f1", "accuracy"]
DEFAULT_RESULTS = ROOT / "results" / "scope30-v1-seed13-freeze-v1"
PROTECTED_RESULTS = (ROOT / "results", ROOT / "results" / "scope30-seed13", ROOT / "results" / "archive")


class GridError(ValueError):
    """Inputs, freeze or results are not acceptable for the frozen twelve-key grid."""


# ---------------------------------------------------------------- small helpers

def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _ids_sha(ids):
    return _sha("\n".join(ids).encode("utf-8"))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def key_name(task, cond, seed=SEED):
    return f"{task}/{cond}/{seed}"


def all_keys():
    return [(task, cond, SEED) for task in TASK_ORDER for cond in CONDITIONS]


def record_filename(task, cond, seed=SEED):
    return f"{task}.{cond}.seed{seed}.json"


def parse_keys(text):
    """'trec/C2,emotion/C4' (optionally with '/13') -> canonical ordered keys; anything else fails."""
    if text is None:
        return all_keys()
    out = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        bits = part.split("/")
        if len(bits) == 3:
            if bits[2] != str(SEED):
                raise GridError(f"{part}: only seed {SEED} is permitted")
            bits = bits[:2]
        if len(bits) != 2 or bits[0] not in TASK_ORDER or bits[1] not in CONDITIONS:
            raise GridError(f"{part!r} is not one of the twelve approved task/condition keys")
        out.append((bits[0], bits[1], SEED))
    if not out:
        raise GridError("no keys given")
    if len(set(out)) != len(out):
        raise GridError(f"duplicate keys requested: {text}")
    return [k for k in all_keys() if k in out]


# ---------------------------------------------------------------- mode, freeze, identity

def check_mode(mode, seed=SEED):
    if seed != SEED:
        raise GridError(f"seed {seed} is not permitted; scope30-v1 is seed {SEED} only")
    if mode not in ("real", "fake"):
        raise GridError(f"unknown mode {mode!r}")
    if tuple(TASKS) != TASK_ORDER or tuple(CFG["conditions"]) != CONDITIONS:
        raise GridError(f"config tasks/conditions {TASKS}/{CFG['conditions']} differ from the approved grid")
    for task in TASK_ORDER:
        if len(labels(task)) != N_LABELS:
            raise GridError(f"{task}: expected {N_LABELS} labels")
    if mode == "real":
        cfg = {k: CFG.get(k) for k in ("train_per_label", "lowres_per_label", "auggpt_k", "seeds", "embedding_model")}
        want = {"train_per_label": PER_LABEL, "lowres_per_label": LOWRES_PER_LABEL, "auggpt_k": AUGGPT_K,
                "seeds": [SEED], "embedding_model": EMBEDDING_MODEL}
        if cfg != want:
            raise GridError(f"config {cfg} differs from the frozen scope30-v1 settings {want}")
        if os.environ.get("SYNTHTEXT_EMBEDDER") or os.environ.get("SYNTHTEXT_GENERATION_MODE", "real") not in ("", "real"):
            raise GridError("real runs refuse SYNTHTEXT_EMBEDDER / fake SYNTHTEXT_GENERATION_MODE settings")
        if "sample" in data_dir().resolve().as_posix().lower().split("/"):
            raise GridError(f"real runs refuse the sample data root {data_dir()}")


def load_freeze(mode, manifest_path=None, evidence_dir=None):
    """Run the fail-closed freeze verification and return the verified manifest."""
    path = Path(manifest_path or ACCEPTED_FREEZE_PATH)
    if mode == "real":
        if path.resolve() != ACCEPTED_FREEZE_PATH.resolve() or evidence_dir is not None:
            raise GridError(f"real training consumes only the accepted freeze {ACCEPTED_FREEZE_PATH} "
                            "with its committed issue #11 evidence")
    if not path.exists():
        raise GridError(f"{path}: freeze manifest is missing")
    try:
        verified = fz.verify_freeze(mode, path, evidence_dir)
    except (fz.FreezeError, ValueError, OSError) as exc:
        raise GridError(f"freeze verification failed: {exc}") from exc
    raw = fz._lf(path.read_bytes())
    digest = _sha(raw)
    if digest != verified["freeze_sha256"]:
        raise GridError("freeze manifest changed during verification")
    if mode == "real" and digest != ACCEPTED_FREEZE_SHA256:
        raise GridError(f"freeze sha256 {digest} is not the accepted {ACCEPTED_FREEZE_SHA256}")
    manifest = json.loads(raw.decode("utf-8"))
    expected_artifacts = {f"{t}/{s}" for t in TASK_ORDER for s in ("zero_shot", "few_shot", "auggpt")}
    problems = []
    if manifest.get("seed") != SEED or manifest.get("scope_id") != SCOPE_ID or manifest.get("freeze_id") != fz.FREEZE_ID:
        problems.append("scope/seed/freeze id")
    if manifest.get("generation_mode") != mode or manifest.get("real_run_evidence") is not (mode == "real"):
        problems.append("generation mode / real-run flag")
    if set(manifest.get("artifacts", {})) != expected_artifacts:
        problems.append("artifact set")
    if problems:
        raise GridError(f"{path}: verified freeze is not the scope30-v1 seed-13 {mode} handoff: {problems}")
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:                       # fake fixture outside the repository: bound by hash, not location
        rel = f"<external>/{path.name}"
    return {"path": rel, "sha256": digest, "manifest": manifest, "verified": verified}


def repository_state(mode):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GridError("cannot read the repository revision with git") from exc
    if mode == "real" and dirty:
        raise GridError(f"real runs need a clean checkout; tracked files are modified:\n{dirty}")
    return {"revision": rev, "clean": not dirty}


def environment():
    packages = {}
    for name in ("scikit-learn", "numpy", "scipy", "sentence-transformers", "torch", "transformers",
                 "tokenizers", "huggingface-hub", "safetensors", "pyyaml"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(), "packages": packages}


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    return home / "hub"


def snapshot_dir(revision):
    """The cache snapshot of one exact revision; never resolved through the mutable refs/main pointer."""
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise GridError(f"invalid {EMBEDDING_MODEL} revision {revision!r}")
    return hub_cache() / ("models--" + EMBEDDING_MODEL.replace("/", "--")) / "snapshots" / revision


def snapshot_files(revision):
    """{relative path: sha256} of every file in the exact snapshot, or GridError if missing/incomplete."""
    snapshot = snapshot_dir(revision)
    if not snapshot.is_dir():
        raise GridError(f"{EMBEDDING_MODEL} revision {revision} is not in the local Hugging Face cache "
                        f"({snapshot}); download that exact revision before training")
    files = {p.relative_to(snapshot).as_posix(): _sha(p.read_bytes())
             for p in sorted(snapshot.rglob("*")) if p.is_file()}
    required = {"modules.json", "config.json", "tokenizer.json", "config_sentence_transformers.json"}
    if required - set(files) or not ({"model.safetensors", "pytorch_model.bin"} & set(files)):
        raise GridError(f"{snapshot}: incomplete {EMBEDDING_MODEL} snapshot")
    return files


def model_identity(mode, revision=APPROVED_MINILM_REVISION):
    """Identity of the exact embedder used. Real: the approved pinned MiniLM snapshot and every file hash."""
    if mode == "fake":
        return {"kind": "hashing-fixture", "name": HASHING_FIXTURE, "real_model": False}
    if revision != APPROVED_MINILM_REVISION:
        raise GridError(f"{EMBEDDING_MODEL} revision {revision} is not the approved {APPROVED_MINILM_REVISION}")
    files = snapshot_files(revision)
    return {"kind": "sentence-transformers", "name": EMBEDDING_MODEL, "revision": revision,
            "files_sha256": files, "identity_sha256": _sha(_canonical(files)), "real_model": True,
            **EMBEDDING_SETTINGS}


def verify_model_identity(identity):
    """Fail closed unless the cached snapshot still has exactly the recorded files and hashes."""
    if not isinstance(identity, dict) or identity.get("kind") != "sentence-transformers" \
            or identity.get("name") != EMBEDDING_MODEL or identity.get("revision") != APPROVED_MINILM_REVISION:
        raise GridError(f"recorded embedder identity is not the approved {EMBEDDING_MODEL} @ {APPROVED_MINILM_REVISION}")
    current = snapshot_files(identity["revision"])
    if current != identity.get("files_sha256") or _sha(_canonical(current)) != identity.get("identity_sha256"):
        changed = sorted(k for k in set(current) | set(identity.get("files_sha256") or {})
                         if current.get(k) != (identity.get("files_sha256") or {}).get(k))
        raise GridError(f"{EMBEDDING_MODEL} snapshot {identity['revision']} changed since it was recorded "
                        f"(files {changed}); refusing to train with an unrecorded model")


def require_real_dependencies():
    missing = [m for m in ("sentence_transformers", "torch", "sklearn", "numpy") if _missing(m)]
    if missing:
        raise GridError(f"real runs need the classifier dependencies; missing: {missing} "
                        "(install requirements.txt in the training environment)")


def _missing(module):
    import importlib.util
    return importlib.util.find_spec(module) is None


class Embedder:
    """Real mode loads only the snapshot named by the recorded identity, re-verified just before and after loading."""

    def __init__(self, mode, identity=None):
        if mode == "real" and identity is None:
            raise GridError("the real embedder needs the recorded model identity")
        self.mode = mode
        self.identity = identity
        self._model = None

    def _load(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        verify_model_identity(self.identity)                    # immediately before use
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(snapshot_dir(self.identity["revision"])), device=EMBEDDING_SETTINGS["device"])
        verify_model_identity(self.identity)                    # the files read while loading were the recorded ones
        return model

    def __call__(self, texts):
        import numpy as np
        if self.mode == "fake":
            from sklearn.feature_extraction.text import HashingVectorizer
            vec = HashingVectorizer(n_features=1024, ngram_range=(1, 2), alternate_sign=False)
            return vec.transform(texts).toarray()
        if self._model is None:
            self._model = self._load()
        return np.asarray(self._model.encode(list(texts), batch_size=EMBEDDING_SETTINGS["batch_size"],
                                             normalize_embeddings=EMBEDDING_SETTINGS["normalize_embeddings"],
                                             show_progress_bar=False))


# ---------------------------------------------------------------- frozen inputs

def _data_rel(path):
    try:
        return Path(path).resolve().relative_to(data_dir().resolve()).as_posix()
    except ValueError as exc:
        raise GridError(f"{path} is outside the data root {data_dir()}") from exc


def _read_bound(rel, expected_sha, role):
    """Read a file named by the freeze; its bytes must hash to the frozen value."""
    if not isinstance(rel, str) or rel.startswith(("/", "\\")) or ".." in Path(rel).parts or ":" in rel:
        raise GridError(f"{role}: unsafe frozen path {rel!r}")
    path = data_dir() / rel
    if not path.exists():
        raise GridError(f"{role}: frozen input {rel} is missing")
    raw = path.read_bytes()
    digest = _sha(raw)
    if expected_sha is not None and digest != expected_sha:
        raise GridError(f"{role}: {rel} sha256 {digest} differs from the bound {expected_sha}")
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridError(f"{role}: {rel} is not valid JSONL") from exc
    return rows, {"role": role, "path": rel, "sha256": digest, "rows": len(rows)}


def expected_consumed_path(task, setting, mode):
    path = (synth_path(task, "auggpt", seed=SEED, mode=mode) if setting == "auggpt"
            else sel.selected_path(task, setting, mode))
    return _data_rel(path)


def _pinned_sha(task, name, mode):
    if mode != "real":
        return None
    return prepare_data.CHECKSUMS[f"{task}/{name}.jsonl"]["sha256"]


def _check_rows(rows, task, role, setting=None, mode=None):
    problems = []
    valid = set(labels(task))
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"]:
            problems.append("row without an id")
            break
        if not isinstance(r.get("text"), str) or not r["text"].strip():
            problems.append(f"{r['id']}: empty text")
        if r.get("label") not in valid:
            problems.append(f"{r['id']}: invalid label {r.get('label')!r}")
        if setting is not None and (r.get("task") != task or r.get("setting") != setting
                                    or r.get("repetition") != SEED or r.get("generation_mode") != mode):
            problems.append(f"{r['id']}: task/setting/repetition/mode is not {task}/{setting}/{SEED}/{mode}")
    ids = [r.get("id") for r in rows if isinstance(r, dict)]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    if problems:
        raise GridError(f"{task} {role}: {problems[:5]}")


def load_task_inputs(task, freeze, mode):
    """Every input of one task, each read from its frozen/canonical location and hash-checked."""
    m = freeze["manifest"]
    refs, arts = m["real_references"], m["artifacts"]
    out = {"task": task, "records": {}}
    for name in ("test", f"lowres_seed{SEED}", "pool"):
        ref = refs[f"{task}/{name}"]
        expected = ref["path"]
        if expected != _data_rel(real_path(task, name)):
            raise GridError(f"{task}/{name}: frozen path {expected} is not the canonical real split")
        pinned = _pinned_sha(task, name, mode)
        if pinned is not None and pinned != ref["sha256"]:
            raise GridError(f"{task}/{name}: freeze binds {ref['sha256']}, canonical #5 hash is {pinned}")
        rows, rec = _read_bound(expected, ref["sha256"], f"{task}/{name}")
        if rec["rows"] != ref["rows"]:
            raise GridError(f"{task}/{name}: {rec['rows']} rows, freeze records {ref['rows']}")
        _check_rows(rows, task, name)
        out[name] = rows
        out["records"][name] = rec
    train_rel = _data_rel(real_path(task, "train"))
    out["train"], out["records"]["train"] = _read_bound(train_rel, _pinned_sha(task, "train", mode), f"{task}/train")
    _check_rows(out["train"], task, "train")
    for setting in ("zero_shot", "few_shot", "auggpt"):
        art = arts[f"{task}/{setting}"]
        consumed = art["consumed"]
        want = expected_consumed_path(task, setting, mode)
        if consumed["path"] != want:
            raise GridError(f"{task}/{setting}: freeze names {consumed['path']}, not the accepted {want}; "
                            "legacy/default synthetic paths are never consumed")
        rows, rec = _read_bound(consumed["path"], consumed["sha256"], f"{task}/{setting}")
        if [r.get("id") for r in rows] != consumed["ids_in_file_order"]:
            raise GridError(f"{task}/{setting}: row IDs or their order differ from the freeze")
        _check_rows(rows, task, setting, setting=setting, mode=mode)
        rec["generation_fingerprint"] = art["generation_fingerprint"]
        out[setting] = rows
        out["records"][setting] = rec
        if setting != "auggpt":
            out["records"][f"{setting}_selection"] = _bind_plain(art["selection_manifest"],
                                                                 f"{task}/{setting}_selection")
    labs = set(labels(task))
    if {r["label"] for r in out["test"]} != labs:
        raise GridError(f"{task}: held-out test set does not cover all six labels")
    out["test_ids"] = {r["id"] for r in out["test"]}
    out["test_texts"] = {norm(r["text"]): r["id"] for r in out["test"]}
    out["pool_ids"] = {r["id"] for r in out["pool"]}
    return out


def _bind_plain(ref, role):
    rel = ref.get("path")
    if not isinstance(rel, str) or rel.startswith(("/", "\\")) or ".." in Path(rel).parts or ":" in rel:
        raise GridError(f"{role}: unsafe frozen path {rel!r}")
    path = data_dir() / rel
    if not path.exists():
        raise GridError(f"{role}: {ref['path']} is missing")
    digest = _sha(path.read_bytes())
    if digest != ref["sha256"]:
        raise GridError(f"{role}: {ref['path']} sha256 {digest} differs from the freeze")
    return {"role": role, "path": ref["path"], "sha256": digest}


# ---------------------------------------------------------------- training sets

def select_c4_synthetic(task, few_rows):
    out = []
    for lab in sorted(labels(task)):                       # same label order as per_label_sample
        group = [r for r in few_rows if r["label"] == lab]
        if len(group) != PER_LABEL:
            raise GridError(f"{task} C4: selected few-shot corpus has {len(group)} {lab!r} rows, need {PER_LABEL}")
        group.sort(key=lambda r: (sel.selection_key(task, "few_shot", r["id"]), r["id"]))
        out += group[:C4_SYNTH_PER_LABEL]
    return out


def _real_sample(task, inp, n, cond):
    try:
        return per_label_sample(inp["train"], n, SEED, required_labels=labels(task), exact=True)
    except ValueError as exc:
        raise GridError(f"{task} {cond}: {exc}") from exc


def build_training_set(task, cond, inp):
    """Return (rows in construction order, [(part, n)], consumed input roles)."""
    if cond == "C1":
        rows = [dict(r, _part="real_train") for r in _real_sample(task, inp, PER_LABEL, cond)]
        roles = ["train"]
    elif cond == "C2":
        rows, roles = [dict(r, _part="zero_shot") for r in inp["zero_shot"]], ["zero_shot", "zero_shot_selection"]
    elif cond == "C3":
        rows, roles = [dict(r, _part="few_shot") for r in inp["few_shot"]], ["few_shot", "few_shot_selection"]
    elif cond == "C4":
        rows = ([dict(r, _part="real_train") for r in _real_sample(task, inp, C4_REAL_PER_LABEL, cond)]
                + [dict(r, _part="few_shot") for r in select_c4_synthetic(task, inp["few_shot"])])
        roles = ["train", "few_shot", "few_shot_selection"]
    elif cond == "C5a":
        rows, roles = [dict(r, _part="lowres_source") for r in inp[f"lowres_seed{SEED}"]], [f"lowres_seed{SEED}"]
    elif cond == "C5b":
        rows = ([dict(r, _part="lowres_source") for r in inp[f"lowres_seed{SEED}"]]
                + [dict(r, _part="auggpt") for r in inp["auggpt"]])
        roles = [f"lowres_seed{SEED}", "auggpt"]
    else:
        raise GridError(f"unknown condition {cond!r}")
    validate_training_set(task, cond, rows, inp)
    return rows, roles


def validate_training_set(task, cond, rows, inp):
    """Exact counts, all labels, composition, source identity and train/test separation, or GridError."""
    labs = sorted(labels(task))
    problems = []
    if len(rows) != EXPECTED_COUNTS[cond]:
        problems.append(f"{len(rows)} rows, need exactly {EXPECTED_COUNTS[cond]}")
    counts = Counter(r.get("label") for r in rows)
    if {lab: counts.get(lab, 0) for lab in labs} != {lab: EXPECTED_PER_LABEL[cond] for lab in labs} \
            or set(counts) != set(labs):
        problems.append(f"label counts {dict(sorted(counts.items()))}, need {EXPECTED_PER_LABEL[cond]} for each of {labs}")
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("duplicate ids")
    texts = [norm(r.get("text") or "") for r in rows]
    if len(set(texts)) != len(texts) or not all(texts):
        problems.append("duplicate or empty normalized texts")
    if set(ids) & inp["test_ids"]:
        problems.append(f"test ids in training: {sorted(set(ids) & inp['test_ids'])[:5]}")
    leaks = []
    for r in rows:
        hit = inp["test_texts"].get(norm(r.get("text") or ""))
        if hit is not None and not _pinned_duplicate(task, cond, r, hit):
            leaks.append(f"{r.get('id')}={hit}")
    if leaks:
        problems.append(f"held-out test text overlap: {leaks[:5]}")
    parts = Counter(r.get("_part") for r in rows)
    real = [r for r in rows if r.get("_part") == "real_train"]
    if real and {r["id"] for r in real} & inp["pool_ids"]:
        problems.append("real training rows taken from the demonstration pool")
    few_ids = [r["id"] for r in inp["few_shot"]]
    if cond in ("C1",) and parts != Counter({"real_train": 180}):
        problems.append(f"composition {dict(parts)}")
    if cond == "C2" and ids != [r["id"] for r in inp["zero_shot"]]:
        problems.append("not the frozen selected zero-shot corpus")
    if cond == "C3" and ids != few_ids:
        problems.append("not the frozen selected few-shot corpus")
    if cond == "C4":
        synth = [r for r in rows if r.get("_part") == "few_shot"]
        if Counter(r["label"] for r in real) != Counter({lab: C4_REAL_PER_LABEL for lab in labs}) or \
                Counter(r["label"] for r in synth) != Counter({lab: C4_SYNTH_PER_LABEL for lab in labs}) or \
                len(real) + len(synth) != len(rows):
            problems.append("C4 is not 15 real + 15 selected few-shot rows per label")
        if not {r["id"] for r in synth} <= set(few_ids):
            problems.append("C4 synthetic rows are not drawn from the C3 corpus")
        if [r["id"] for r in synth] != [r["id"] for r in select_c4_synthetic(task, inp["few_shot"])]:
            problems.append("C4 synthetic rows do not follow the approved deterministic rule")
    low = inp[f"lowres_seed{SEED}"]
    if cond in ("C5a", "C5b"):
        head = rows[:len(low)]
        if [r["id"] for r in head] != [r["id"] for r in low] or any(r.get("_part") != "lowres_source" for r in head):
            problems.append("C5 sources are not the exact frozen lowres_seed13 rows")
        if not {r["id"] for r in low} <= inp["pool_ids"]:
            problems.append("lowres sources are not pool rows")
        if Counter(r["label"] for r in low) != Counter({lab: LOWRES_PER_LABEL for lab in labs}):
            problems.append("lowres sources are not 10 per label")
    if cond == "C5a" and len(rows) != len(low):
        problems.append("C5a contains rows beyond its sources")
    if cond == "C5b":
        problems += _check_paraphrases(rows[len(low):], low)
    if problems:
        raise GridError(f"{task} {cond} seed {SEED}: {problems[:6]}")


def _check_paraphrases(paraphrases, sources):
    problems = []
    by_id = {s["id"]: s for s in sources}
    groups = {}
    for p in paraphrases:
        sids = p.get("source_ids")
        if p.get("_part") != "auggpt" or not isinstance(sids, list) or len(sids) != 1 or sids[0] not in by_id:
            problems.append(f"{p.get('id')}: not a paraphrase of exactly one C5a source")
            continue
        groups.setdefault(sids[0], []).append(p)
    for sid, src in by_id.items():
        group = groups.get(sid, [])
        texts = {norm(p["text"]) for p in group}
        if len(group) != AUGGPT_K or len(texts) != AUGGPT_K:
            problems.append(f"{sid}: {len(group)} paraphrases / {len(texts)} distinct, need {AUGGPT_K}")
        if norm(src["text"]) in texts:
            problems.append(f"{sid}: a paraphrase repeats its source")
        if any(p["label"] != src["label"] for p in group):
            problems.append(f"{sid}: paraphrase label differs from its source")
    if len(paraphrases) != AUGGPT_K * len(sources):
        problems.append(f"{len(paraphrases)} paraphrases, need {AUGGPT_K * len(sources)}")
    return problems


def _pinned_duplicate(task, cond, row, test_id):
    digest = _sha(norm(row["text"]).encode("utf-8"))
    return any(p["task"] == task and cond in p["conditions"] and row.get("_part") == "real_train"
               and p["train_id"] == row.get("id") and p["test_id"] == test_id
               and p["normalized_text_sha256"] == digest for p in PINNED_OFFICIAL_DUPLICATES)


def disclosed_overlap(task, cond, rows, inp):
    out = []
    for r in rows:
        hit = inp["test_texts"].get(norm(r["text"]))
        if hit is not None:
            pin = next(p for p in PINNED_OFFICIAL_DUPLICATES if p["train_id"] == r["id"])
            out.append({"train_id": r["id"], "test_id": hit, "normalized_text_sha256": pin["normalized_text_sha256"],
                        "decision": pin["decision"]})
    return out


# ---------------------------------------------------------------- context and bindings

def prepare(mode="real", manifest_path=None, evidence_dir=None, seed=SEED, need_model=True):
    """Freeze verification first, then every input and all twelve training sets. Writes nothing."""
    check_mode(mode, seed)
    freeze = load_freeze(mode, manifest_path, evidence_dir)
    inputs = {task: load_task_inputs(task, freeze, mode) for task in TASK_ORDER}
    sets = {}
    for task, cond, _ in all_keys():
        sets[(task, cond)] = build_training_set(task, cond, inputs[task])
    ctx = {"mode": mode, "freeze": freeze, "inputs": inputs, "sets": sets}
    if need_model:
        if mode == "real":
            require_real_dependencies()
        ctx["repository"] = repository_state(mode)
        ctx["environment"] = environment()
        ctx["model"] = model_identity(mode)
    return ctx


def fit_order(rows):
    order = list(rows)
    random.Random(SEED).shuffle(order)
    return order


def expected_binding(ctx, task, cond):
    rows, roles = ctx["sets"][(task, cond)]
    inp = ctx["inputs"][task]
    ids = [r["id"] for r in rows]
    m = ctx["freeze"]["manifest"]
    test_rec = dict(inp["records"]["test"])
    test_rec.update({"ids_sha256": _ids_sha([r["id"] for r in inp["test"]]),
                     "counts_by_label": dict(sorted(Counter(r["label"] for r in inp["test"]).items()))})
    return {
        "schema_version": SCHEMA_VERSION, "result_kind": RESULT_KIND, "scope_id": SCOPE_ID,
        "generation_mode": ctx["mode"], "real_run_evidence": ctx["mode"] == "real",
        "key": {"task": task, "condition": cond, "seed": SEED},
        "freeze": {"freeze_id": m["freeze_id"], "sha256": ctx["freeze"]["sha256"], "path": ctx["freeze"]["path"],
                   "audit_revision": m["repository_revision"]},
        "training": {
            "n_train": len(rows), "counts_by_label": dict(sorted(Counter(r["label"] for r in rows).items())),
            "composition": dict(sorted(Counter(r["_part"] for r in rows).items())),
            "construction": CONSTRUCTION[cond], "ids": ids, "ids_sha256": _ids_sha(ids),
            "fit_order_ids_sha256": _ids_sha([r["id"] for r in fit_order(rows)]), "shuffle": SHUFFLE,
            "disclosed_test_text_overlap": disclosed_overlap(task, cond, rows, inp),
        },
        "inputs": [inp["records"][role] for role in roles],
        "test": test_rec,
        "classifier": {"embedding": ctx["model"], "classifier": CLASSIFIER},
        "repository": ctx["repository"],
        "environment": ctx["environment"],
    }


def fit_and_score(ctx, task, cond, embedder):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    rows, _ = ctx["sets"][(task, cond)]
    test = ctx["inputs"][task]["test"]
    order = fit_order(rows)
    clf = LogisticRegression(C=CLASSIFIER["C"], max_iter=CLASSIFIER["max_iter"], random_state=CLASSIFIER["random_state"])
    clf.fit(embedder([r["text"] for r in order]), [r["label"] for r in order])
    pred = clf.predict(embedder([r["text"] for r in test]))
    gold = [r["label"] for r in test]
    macro_f1 = float(f1_score(gold, pred, labels=labels(task), average="macro", zero_division=0))
    acc = float(accuracy_score(gold, pred))
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (macro_f1, acc)):
        raise GridError(f"{task} {cond}: metric out of range macro_f1={macro_f1} accuracy={acc}")
    n_iter = int(max(clf.n_iter_))
    return {"macro_f1": macro_f1, "accuracy": acc}, {"n_iter": n_iter, "converged": n_iter < CLASSIFIER["max_iter"],
                                                      "test_rows": len(test)}


def make_record(binding, metrics, fit, started, finished, seconds):
    record = dict(binding)
    record.update({"metrics": metrics, "fit": fit, "started_at": started, "finished_at": finished,
                   "runtime_seconds": round(seconds, 3)})
    record["record_sha256"] = _sha(_canonical(record))
    return record


# ---------------------------------------------------------------- results directory

def _marker(ctx):
    return {"result_kind": RESULT_KIND, "scope_id": SCOPE_ID, "seed": SEED, "generation_mode": ctx["mode"],
            "real_run_evidence": ctx["mode"] == "real", "freeze_id": ctx["freeze"]["manifest"]["freeze_id"],
            "freeze_sha256": ctx["freeze"]["sha256"], "keys": [key_name(*k) for k in all_keys()]}


def open_results_dir(path, ctx, create):
    path = Path(path)
    resolved = path.resolve()
    if any(resolved == p.resolve() for p in PROTECTED_RESULTS):
        raise GridError(f"{path} holds historical or baseline results; use a fresh scope-specific directory")
    marker = path / MARKER
    if marker.exists():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GridError(f"{marker}: unreadable directory identity") from exc
        if recorded != _marker(ctx):
            diffs = sorted(k for k in set(recorded) | set(_marker(ctx)) if recorded.get(k) != _marker(ctx).get(k))
            raise GridError(f"{path} belongs to another grid/freeze/mode (differing: {diffs}); "
                            "fake and real results are never mixed")
        return path
    if path.exists() and any(path.iterdir()):
        raise GridError(f"{path} is not empty and is not a {RESULT_KIND} directory; refusing to write into "
                        "historical or unrelated results")
    if not create:
        raise GridError(f"{path}: no {RESULT_KIND} results directory")
    _atomic_write(marker, _canonical(_marker(ctx)))
    return path


def scan_records(path):
    """{key: (filename, parsed record or None, error)}; unexpected files are errors."""
    rec_dir = Path(path) / RECORDS
    names = {record_filename(*k): k for k in all_keys()}
    found, unexpected = {}, []
    if rec_dir.exists():
        for p in sorted(rec_dir.iterdir()):
            if p.name not in names:
                unexpected.append(p.name)
                continue
            try:
                found[names[p.name]] = (p, json.loads(p.read_text(encoding="utf-8")), None)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                found[names[p.name]] = (p, None, f"unreadable or partial record ({exc.__class__.__name__})")
    return found, unexpected


def check_record(record, key, binding):
    """'valid', or ('invalid'|'stale', reason)."""
    task, cond, seed = key
    if not isinstance(record, dict):
        return "invalid", "not a JSON object"
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    if record.get("record_sha256") != _sha(_canonical(body)):
        return "invalid", "record hash mismatch (tampered or partial)"
    if record.get("key") != {"task": task, "condition": cond, "seed": seed}:
        return "invalid", f"record key {record.get('key')} does not match its file (duplicate/misplaced key)"
    metrics = record.get("metrics") or {}
    for name in ("macro_f1", "accuracy"):
        v = metrics.get(name)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0.0 <= v <= 1.0:
            return "invalid", f"{name}={v!r} is not a metric in [0, 1]"
    if (record.get("training") or {}).get("n_train") != EXPECTED_COUNTS[cond]:
        return "invalid", f"n_train {(record.get('training') or {}).get('n_train')} != {EXPECTED_COUNTS[cond]}"
    if record.get("real_run_evidence") is not binding["real_run_evidence"]:
        return "invalid", "fake/real evidence flag does not match this run"
    diffs = [k for k in sorted(set(binding) | set(body)) if k not in ("metrics", "fit", "started_at", "finished_at",
                                                                      "runtime_seconds") and body.get(k) != binding.get(k)]
    if diffs:
        return "stale", f"provenance differs from the current binding in {diffs}"
    return "valid"


def runs_csv_bytes(records):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=RUN_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for key in all_keys():
        if key in records:
            r = records[key]
            writer.writerow({"task": key[0], "condition": key[1], "seed": key[2], "n_train": r["training"]["n_train"],
                             "macro_f1": repr(r["metrics"]["macro_f1"]), "accuracy": repr(r["metrics"]["accuracy"])})
    return buf.getvalue().encode("utf-8")


def complete_summary(ctx, path, records):
    rec_dir = Path(path) / RECORDS
    return {
        "result_kind": RESULT_KIND, "scope_id": SCOPE_ID, "seed": SEED, "generation_mode": ctx["mode"],
        "real_run_evidence": ctx["mode"] == "real", "complete": True, "keys": len(records),
        "freeze": {"freeze_id": ctx["freeze"]["manifest"]["freeze_id"], "sha256": ctx["freeze"]["sha256"]},
        "classifier": {"embedding": ctx["model"], "classifier": CLASSIFIER},
        "repository": ctx["repository"],
        "runs_csv_sha256": _sha(runs_csv_bytes(records)),
        "results": [{"key": key_name(*k), "n_train": records[k]["training"]["n_train"],
                     "macro_f1": records[k]["metrics"]["macro_f1"], "accuracy": records[k]["metrics"]["accuracy"],
                     "training_ids_sha256": records[k]["training"]["ids_sha256"],
                     "record_sha256": records[k]["record_sha256"],
                     "record_file_sha256": _sha((rec_dir / record_filename(*k)).read_bytes()),
                     "runtime_seconds": records[k]["runtime_seconds"]} for k in all_keys()],
        "total_runtime_seconds": round(sum(records[k]["runtime_seconds"] for k in all_keys()), 3),
    }


def refresh_derived(ctx, path):
    """Rewrite runs.csv from the valid records; write/remove grid_complete.json."""
    found, _ = scan_records(path)
    valid = {k: rec for k, (_, rec, err) in found.items()
             if err is None and check_record(rec, k, expected_binding(ctx, k[0], k[1])) == "valid"}
    _atomic_write(Path(path) / RUNS_CSV, runs_csv_bytes(valid))
    complete = Path(path) / COMPLETE
    if len(valid) == len(all_keys()):
        _atomic_write(complete, _canonical(complete_summary(ctx, path, valid)))
    elif complete.exists():
        (Path(path) / SUPERSEDED).mkdir(parents=True, exist_ok=True)
        os.replace(complete, Path(path) / SUPERSEDED / f"{COMPLETE}.{_stamp()}")
    return valid


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


# ---------------------------------------------------------------- commands

def run_grid(results_dir, keys=None, replace=(), mode="real", manifest_path=None, evidence_dir=None, seed=SEED,
             embedder=None, log=print):
    keys = list(keys or all_keys())
    replace = list(replace or ())
    if not set(replace) <= set(keys):
        raise GridError("--replace keys must be among the requested keys")
    ctx = prepare(mode, manifest_path, evidence_dir, seed)          # freeze first; writes nothing
    path = open_results_dir(results_dir, ctx, create=True)
    found, unexpected = scan_records(path)
    if unexpected:
        raise GridError(f"{path / RECORDS}: unexpected files {unexpected} (wrong seed/key or foreign results)")
    plan, blocked = [], []
    for key in keys:
        binding = expected_binding(ctx, key[0], key[1])
        if key not in found:
            plan.append((key, "missing"))
            continue
        _, rec, err = found[key]
        status = ("invalid", err) if err else check_record(rec, key, binding)
        if key in replace:
            plan.append((key, "replace"))
        elif status == "valid":
            log(f"  {key_name(*key)}: complete provenance matches; resumed (not retrained)")
        else:
            blocked.append(f"{key_name(*key)}: {status[0]}: {status[1]}")
    if blocked:
        raise GridError("existing results are not reusable; nothing was trained. Name them explicitly in "
                        "--replace to retrain (old records are preserved in superseded/):\n  " + "\n  ".join(blocked))
    for key in [k for k in all_keys() if k not in keys and k in found]:
        log(f"  {key_name(*key)}: not requested; left untouched")
    # The loader receives the identity recorded in every binding; it never re-resolves the cache itself.
    embedder = embedder or Embedder(ctx["mode"], ctx["model"] if ctx["mode"] == "real" else None)
    note = "" if mode == "real" else " [FAKE fixture - not real evidence]"
    for key, why in plan:
        task, cond, _ = key
        binding = expected_binding(ctx, task, cond)
        started, t0 = _now(), time.perf_counter()
        metrics, fit = fit_and_score(ctx, task, cond, embedder)
        record = make_record(binding, metrics, fit, started, _now(), time.perf_counter() - t0)
        target = path / RECORDS / record_filename(*key)
        if target.exists():
            (path / SUPERSEDED).mkdir(parents=True, exist_ok=True)
            os.replace(target, path / SUPERSEDED / f"{target.stem}.{_stamp()}.json")
        _atomic_write(target, _canonical(record))
        refresh_derived(ctx, path)
        log(f"  {key_name(*key)}{note}: {why} -> macro-F1 {metrics['macro_f1']:.4f} accuracy {metrics['accuracy']:.4f} "
            f"n_train {binding['training']['n_train']} ({record['runtime_seconds']:.1f}s)")
    valid = refresh_derived(ctx, path)
    return {"path": path, "valid": sorted(valid), "complete": len(valid) == len(all_keys()), "ctx": ctx}


def validate_results(results_dir, mode="real", manifest_path=None, evidence_dir=None, seed=SEED, ctx=None):
    """Read-only: exactly twelve unique valid keys whose provenance matches the current binding."""
    ctx = ctx or prepare(mode, manifest_path, evidence_dir, seed)
    path = open_results_dir(results_dir, ctx, create=False)
    found, unexpected = scan_records(path)
    problems = [f"unexpected file {n}" for n in unexpected]
    records = {}
    for key in all_keys():
        if key not in found:
            problems.append(f"{key_name(*key)}: missing")
            continue
        _, rec, err = found[key]
        status = ("invalid", err) if err else check_record(rec, key, expected_binding(ctx, key[0], key[1]))
        if status != "valid":
            problems.append(f"{key_name(*key)}: {status[0]}: {status[1]}")
        else:
            records[key] = rec
    runs = Path(path) / RUNS_CSV
    if not runs.exists():
        problems.append("runs.csv missing")
    else:
        rows = list(csv.DictReader(io.StringIO(runs.read_bytes().decode("utf-8"))))
        keys = [(r.get("task"), r.get("condition"), r.get("seed")) for r in rows]
        if len(set(keys)) != len(keys):
            problems.append("runs.csv has duplicate keys")
        if sorted(keys) != sorted((t, c, str(s)) for t, c, s in all_keys()):
            problems.append(f"runs.csv has {len(rows)} rows / keys differ from the twelve approved keys")
        elif not problems and runs.read_bytes() != runs_csv_bytes(records):
            problems.append("runs.csv does not match the records")
    if not problems:
        complete = Path(path) / COMPLETE
        if not complete.exists() or complete.read_bytes() != _canonical(complete_summary(ctx, path, records)):
            problems.append("grid_complete.json missing or does not match the records")
    if problems:
        raise GridError(f"{path}: results are not a complete valid twelve-key grid:\n  " + "\n  ".join(problems))
    return {"path": path, "records": records, "ctx": ctx}


def report(result, log=print):
    ctx, path, records = result["ctx"], Path(result["path"]), result["records"]
    fake = ctx["mode"] != "real"
    tag = "FAKE FIXTURE RESULTS - NOT REAL EVIDENCE" if fake else "REAL RESULTS (seed 13, single run)"
    model = ctx["model"]
    log(f"==== {tag} ====")
    log(f"freeze      {ctx['freeze']['manifest']['freeze_id']} sha256 {ctx['freeze']['sha256']}")
    log(f"classifier  {model['name']} revision {model.get('revision', '-')} identity {model.get('identity_sha256', '-')}")
    log(f"            LogisticRegression(C={CLASSIFIER['C']}, max_iter={CLASSIFIER['max_iter']}, "
        f"random_state={CLASSIFIER['random_state']}); embeddings frozen, normalized")
    log(f"repository  {ctx['repository']['revision']} clean={ctx['repository']['clean']}")
    log(f"{'key':<17}{'n_train':>8}{'macro-F1':>10}{'accuracy':>10}{'seconds':>9}  record sha256")
    for key in all_keys():
        r = records[key]
        log(f"{key_name(*key):<17}{r['training']['n_train']:>8}{r['metrics']['macro_f1']:>10.4f}"
            f"{r['metrics']['accuracy']:>10.4f}{r['runtime_seconds']:>9.1f}  {r['record_sha256']}")
        for d in r["training"]["disclosed_test_text_overlap"]:
            log(f"{'':<17}disclosed pinned official duplicate {d['train_id']} = {d['test_id']}")
    log(f"total runtime {sum(records[k]['runtime_seconds'] for k in all_keys()):.1f}s over {len(records)} keys")
    for name in (RUNS_CSV, COMPLETE, MARKER):
        log(f"{name:<19}{(path / name).resolve()}  sha256 {_sha((path / name).read_bytes())}")
    log(f"records            {(path / RECORDS).resolve()}")
    log(f"COMPLETE: exactly {len(records)} unique valid keys" + (" (FAKE)" if fake else ""))


def check_inputs(mode="real", manifest_path=None, evidence_dir=None, seed=SEED, log=print):
    ctx = prepare(mode, manifest_path, evidence_dir, seed, need_model=False)
    note = "" if mode == "real" else " [FAKE fixture]"
    log(f"FREEZE VERIFIED{note}: {ctx['freeze']['path']} sha256 {ctx['freeze']['sha256']}")
    for (task, cond), (rows, roles) in ctx["sets"].items():
        parts = dict(sorted(Counter(r["_part"] for r in rows).items()))
        per_label = sorted(set(Counter(r["label"] for r in rows).values()))
        ids_sha = _ids_sha([r["id"] for r in rows])
        overlap = disclosed_overlap(task, cond, rows, ctx["inputs"][task])
        log(f"INPUTS PASS{note} {key_name(task, cond)}: n={len(rows)} per_label={per_label} parts={parts} "
            f"ids_sha256={ids_sha[:16]} inputs={[ctx['inputs'][task]['records'][r]['sha256'][:12] for r in roles]}"
            + (f" disclosed_overlap={[d['train_id'] for d in overlap]}" if overlap else ""))
    log(f"INPUTS PASS{note}: twelve training sets built and validated; nothing fitted or written")
    return ctx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--check-inputs", action="store_true", help="read-only: verify freeze and build all 12 sets")
    action.add_argument("--validate", action="store_true", help="read-only: require exactly 12 valid keys and report")
    ap.add_argument("--results-dir", default=None, help=f"fresh scope-specific directory (default {DEFAULT_RESULTS})")
    ap.add_argument("--keys", default=None, help="subset of task/condition keys, e.g. trec/C1,emotion/C5b (default all 12)")
    ap.add_argument("--replace", default=None, help="keys to retrain even if a record exists (old record preserved)")
    ap.add_argument("--seed", type=int, default=SEED, choices=[SEED], help="seed 13 only")
    ap.add_argument("--freeze-manifest", default=None, help="fake runs only; real runs use the accepted freeze")
    ap.add_argument("--evidence-dir", default=None, help="fake runs only")
    ap.add_argument("--dry-run", action="store_true", help="FAKE fixtures + hashing embedder; never real evidence")
    args = ap.parse_args(argv)
    mode = "fake" if args.dry_run else "real"
    try:
        if mode == "real" and (args.freeze_manifest or args.evidence_dir):
            raise GridError("real runs use only the accepted freeze and committed evidence")
        if mode == "fake" and not args.results_dir and not args.check_inputs:
            raise GridError("--dry-run needs an explicit --results-dir")
        results = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS
        if args.check_inputs:
            check_inputs(mode, args.freeze_manifest, args.evidence_dir, args.seed)
        elif args.validate:
            report(validate_results(results, mode, args.freeze_manifest, args.evidence_dir, args.seed))
        else:
            keys = parse_keys(args.keys)
            replace = parse_keys(args.replace) if args.replace else []
            out = run_grid(results, keys, replace, mode, args.freeze_manifest, args.evidence_dir, args.seed)
            if out["complete"]:
                report(validate_results(results, mode, args.freeze_manifest, args.evidence_dir, args.seed,
                                        ctx=out["ctx"]))
            else:
                missing = [key_name(*k) for k in all_keys() if k not in out["valid"]]
                print(f"INCOMPLETE: {len(out['valid'])}/12 valid keys; missing {missing}. Not reported as complete.")
                sys.exit(2)
    except GridError as exc:
        print(f"FAIL{' (FAKE fixture)' if mode == 'fake' else ''}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
