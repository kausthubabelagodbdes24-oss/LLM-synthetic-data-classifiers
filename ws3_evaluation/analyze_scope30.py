"""Issue #14: single-seed gap analysis of the accepted scope30-v1 seed-13 twelve-key grid.

    python -m ws3_evaluation.analyze_scope30 --check --results-dir <grid>     read-only validation; writes nothing
    python -m ws3_evaluation.analyze_scope30 --results-dir <grid>             tables, two charts, summary, JSON
    python -m ws3_evaluation.analyze_scope30 --dry-run --results-dir <fake grid> --freeze-manifest <fake freeze>
                                             --out <dir>                      FAKE fixtures; never real evidence

The grid is the frozen record set written by ``ws2_training.train_scope30`` at the PR #33 merge and accepted by the
lead on issue #13. Its records bind that commit and environment, so ``train_scope30 --validate`` (which compares
records with the *current* checkout) is run from the pinned run checkout, not from here. This module validates
the frozen records on their own: each record's self-hash, the ``grid_complete.json`` chain (record file hashes,
``runs.csv``), the exact twelve keys/counts/provenance, and - in real mode - the accepted ``grid_complete.json``
SHA-256, which transitively pins every record and ``runs.csv``.

Single run, one seed: values and paired gaps only. No standard deviation, confidence interval, significance,
stability or human-agreement figure is computed or drawn. A missing C1 blocks the analysis; a C1 macro-F1 of 0
makes that task's relative gap undefined (labelled, with the absolute difference reported) - never divided,
substituted or dropped.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from common import ROOT
from ws1_generation import freeze_scope30 as fz
from ws2_training import train_scope30 as ts

ANALYSIS_KIND = "scope30-v1-seed13-single-seed-gap-analysis"
SCHEMA_VERSION = 1
ACCEPTED_GRID_REVISION = "1354118b4b05bec31973c268057d8a713907f3a6"          # PR #33 merge that produced the grid
ACCEPTED_GRID_COMPLETE_SHA256 = "3b12bbb878d8191e387e3abc0f77fc02d9f081b98f0216f9b64dbb144bf40032"
ACCEPTED_GRID_DECISION = ("lead review on issue #13 (1 November 2026): real run accepted; evidence "
                          "https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/issues/13"
                          "#issuecomment-6553500190")
DEFAULT_OUT = ROOT / "docs" / "analysis" / "scope30-v1-seed13"

TASKS = ts.TASK_ORDER                                   # ("trec", "emotion")
CONDITIONS = ts.CONDITIONS                              # C1 C2 C3 C4 C5a C5b
SYNTHETIC_ONLY = ("C2", "C3")
NAMES = {"C1": "Real", "C2": "Zero-shot", "C3": "Few-shot", "C4": "Real + few-shot",
         "C5a": "Real sources", "C5b": "Sources + AugGPT"}
TASK_NAMES = {"trec": "TREC", "emotion": "Emotion"}
REAL_PARTS = ("real_train", "lowres_source")
SYNTHETIC_PARTS = ("zero_shot", "few_shot", "auggpt")
# Exact training composition per condition (ADR 0012), from the producer's own size constants. A record must match
# its condition's structure exactly - key set, values and JSON integer type - or the cost disclosure would be wrong.
_ALL, _C4_REAL = ts.PER_LABEL * ts.N_LABELS, ts.C4_REAL_PER_LABEL * ts.N_LABELS
_SOURCES = ts.LOWRES_PER_LABEL * ts.N_LABELS
EXPECTED_COMPOSITION = {
    "C1": {"real_train": _ALL},
    "C2": {"zero_shot": _ALL},
    "C3": {"few_shot": _ALL},
    "C4": {"few_shot": _ALL - _C4_REAL, "real_train": _C4_REAL},
    "C5a": {"lowres_source": _SOURCES},
    "C5b": {"auggpt": _SOURCES * ts.AUGGPT_K, "lowres_source": _SOURCES},
}
COMPLETE_KEYS = {"result_kind", "scope_id", "seed", "generation_mode", "real_run_evidence", "complete", "keys",
                 "freeze", "classifier", "repository", "runs_csv_sha256", "results", "total_runtime_seconds"}
ENTRY_KEYS = {"key", "n_train", "macro_f1", "accuracy", "training_ids_sha256", "record_sha256",
              "record_file_sha256", "runtime_seconds"}

# The planned comparisons (spec "Evaluation and reporting"; issue #14). Each is a difference within a task.
COMPARISONS = (
    ("few_vs_zero", "Few-shot versus zero-shot synthetic data", "C3", "C2"),
    ("mix_vs_real", "Real + few-shot mix versus real-only", "C4", "C1"),
    ("auggpt_vs_sources", "AugGPT-augmented versus real sources only", "C5b", "C5a"),
)

LIMITATIONS = (
    "One seed (13) and one run per key: no standard deviation, confidence interval, significance test or "
    "stability claim is possible, and none is made. Differences of a few hundredths of macro-F1 can come from "
    "sampling alone.",
    "Small data: 30 training examples per label (180 per task) in C1-C4, 10 real sources per label in C5a.",
    "C5a (60 rows) and C5b (420 rows) intentionally differ in training size; C5b - C5a mixes the effect of "
    "AugGPT content with the effect of more rows.",
    "Task confounding: TREC and Emotion differ in more than subjectivity (label semantics, text length and style, "
    "test-set size 500 vs 2,000 and label balance). A cross-task difference in gaps is not attributable to "
    "subjectivity alone.",
    "Labelling costs differ: C3/C4 synthetic rows were generated from real few-shot demonstrations and C5b from "
    "real sources; these real examples are disclosed separately from classifier training counts.",
    "Synthetic-data quality was screened by model-assisted and heuristic review (issue #11), not by independent "
    "human annotators; known flagged AugGPT rows are retained as disclosed in the freeze manifest.",
    "TREC C1 and C4 contain one disclosed official train/test duplicate (trec.train.05262 = trec.test.00378), "
    "kept by lead decision on issue #13.",
)


class AnalysisError(ValueError):
    """The grid, freeze or output location is not acceptable for the single-seed analysis."""


# ---------------------------------------------------------------- validation of the frozen grid

def _read_json(path, what):
    try:
        raw = Path(path).read_bytes()
        return raw, json.loads(raw.decode("utf-8"))
    except FileNotFoundError:
        raise AnalysisError(f"{path}: {what} is missing") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{path}: unreadable {what}: {exc}") from exc


def _metric_ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0


def _check_record(key, raw, rec, entry, complete, mode):
    """Problems with one frozen record, judged against itself and its grid_complete.json entry."""
    task, cond, seed = key
    name = ts.key_name(*key)
    if not isinstance(rec, dict):
        return [f"{name}: record is not a JSON object"]
    problems = []
    if raw != ts._canonical(rec):
        problems.append(f"{name}: record file is not in canonical form")
    body = {k: v for k, v in rec.items() if k != "record_sha256"}
    if rec.get("record_sha256") != ts._sha(ts._canonical(body)):
        problems.append(f"{name}: record hash mismatch (tampered or partial)")
    if ts._sha(raw) != entry.get("record_file_sha256") or rec.get("record_sha256") != entry.get("record_sha256"):
        problems.append(f"{name}: record differs from the hash recorded in grid_complete.json")
    if rec.get("key") != {"task": task, "condition": cond, "seed": seed}:
        problems.append(f"{name}: record key {rec.get('key')} does not match its file")
    if (rec.get("result_kind"), rec.get("scope_id")) != (ts.RESULT_KIND, ts.SCOPE_ID):
        problems.append(f"{name}: not a {ts.RESULT_KIND} {ts.SCOPE_ID} record")
    if rec.get("generation_mode") != mode or rec.get("real_run_evidence") is not (mode == "real"):
        problems.append(f"{name}: generation mode / real-run flag is not {mode}")
    metrics = rec.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"macro_f1", "accuracy"} \
            or not all(_metric_ok(metrics[m]) for m in metrics):
        problems.append(f"{name}: metrics must be exactly macro_f1 and accuracy in [0, 1], not {metrics!r}")
        metrics = {}
    training = rec.get("training") if isinstance(rec.get("training"), dict) else {}
    n_train = training.get("n_train")
    if n_train != ts.EXPECTED_COUNTS[cond]:
        problems.append(f"{name}: n_train {n_train} != {ts.EXPECTED_COUNTS[cond]}")
    composition = training.get("composition")
    if not ts._same(composition, EXPECTED_COMPOSITION[cond]):        # structural: no arithmetic on untrusted values
        problems.append(f"{name}: training composition {composition!r} is not the exact {cond} structure "
                        f"{EXPECTED_COMPOSITION[cond]}")
    if metrics and not all(ts._same(metrics[m], entry.get(m)) for m in ("macro_f1", "accuracy")):
        problems.append(f"{name}: metrics differ from grid_complete.json")
    if n_train != entry.get("n_train") or training.get("ids_sha256") != entry.get("training_ids_sha256") \
            or not ts._same(rec.get("runtime_seconds"), entry.get("runtime_seconds")):
        problems.append(f"{name}: counts / training ids / runtime differ from grid_complete.json")
    freeze = rec.get("freeze") if isinstance(rec.get("freeze"), dict) else {}
    if freeze.get("sha256") != complete["freeze"].get("sha256") \
            or freeze.get("freeze_id") != complete["freeze"].get("freeze_id"):
        problems.append(f"{name}: freeze binding differs from grid_complete.json")
    if not ts._same(rec.get("classifier"), complete["classifier"]):
        problems.append(f"{name}: classifier/embedder identity differs from grid_complete.json")
    if not ts._same(rec.get("repository"), complete["repository"]):
        problems.append(f"{name}: repository binding differs from grid_complete.json")
    return problems


def load_grid(results_dir, mode="real"):
    """Validate the frozen twelve-key grid read-only and return its records and identities. Writes nothing."""
    if mode not in ("real", "fake"):
        raise AnalysisError(f"unknown mode {mode!r}")
    path = Path(results_dir)
    complete_raw, complete = _read_json(path / ts.COMPLETE, "grid_complete.json (an incomplete grid blocks analysis)")
    complete_sha = ts._sha(complete_raw)
    if mode == "real" and complete_sha != ACCEPTED_GRID_COMPLETE_SHA256:
        raise AnalysisError(f"{path}: grid_complete.json sha256 {complete_sha} is not the accepted "
                            f"{ACCEPTED_GRID_COMPLETE_SHA256} ({ACCEPTED_GRID_DECISION})")
    if not isinstance(complete, dict) or set(complete) != COMPLETE_KEYS or complete_raw != ts._canonical(complete):
        raise AnalysisError(f"{path}: grid_complete.json is not a canonical {ts.RESULT_KIND} summary")
    expected_keys = [ts.key_name(*k) for k in ts.all_keys()]
    head = {k: complete[k] for k in ("result_kind", "scope_id", "seed", "generation_mode", "real_run_evidence",
                                     "complete", "keys")}
    want = {"result_kind": ts.RESULT_KIND, "scope_id": ts.SCOPE_ID, "seed": ts.SEED, "generation_mode": mode,
            "real_run_evidence": mode == "real", "complete": True, "keys": len(expected_keys)}
    if not ts._same(head, want):
        raise AnalysisError(f"{path}: grid_complete.json {head} is not a complete {mode} scope30-v1 seed-13 grid")
    if not isinstance(complete["freeze"], dict) or not isinstance(complete["repository"], dict) \
            or not isinstance(complete["classifier"], dict):
        raise AnalysisError(f"{path}: grid_complete.json freeze/repository/classifier sections are malformed")
    if mode == "real":
        if complete["freeze"].get("sha256") != ts.ACCEPTED_FREEZE_SHA256:
            raise AnalysisError(f"{path}: grid was not trained on the accepted freeze {ts.ACCEPTED_FREEZE_SHA256}")
        if complete["repository"] != {"revision": ACCEPTED_GRID_REVISION, "clean": True}:
            raise AnalysisError(f"{path}: grid was not produced by a clean checkout of {ACCEPTED_GRID_REVISION}")
        emb = complete["classifier"].get("embedding") or {}
        if emb.get("revision") != ts.APPROVED_MINILM_REVISION or emb.get("real_model") is not True:
            raise AnalysisError(f"{path}: grid embedder is not the approved MiniLM {ts.APPROVED_MINILM_REVISION}")

    marker_raw, marker = _read_json(path / ts.MARKER, "grid.json directory identity")
    want_marker = {"result_kind": ts.RESULT_KIND, "scope_id": ts.SCOPE_ID, "seed": ts.SEED, "generation_mode": mode,
                   "real_run_evidence": mode == "real", "freeze_id": complete["freeze"].get("freeze_id"),
                   "freeze_sha256": complete["freeze"].get("sha256"), "keys": expected_keys}
    if not ts._same(marker, want_marker) or marker_raw != ts._canonical(marker):
        raise AnalysisError(f"{path}: grid.json does not identify this {mode} grid/freeze")

    entries = complete["results"]
    if not isinstance(entries, list) or [e.get("key") if isinstance(e, dict) else None for e in entries] \
            != expected_keys or any(set(e) != ENTRY_KEYS for e in entries):
        raise AnalysisError(f"{path}: grid_complete.json results are not exactly the twelve approved keys in order")

    rec_dir = path / ts.RECORDS
    names = {ts.record_filename(*k): k for k in ts.all_keys()}
    present = sorted(p.name for p in rec_dir.iterdir()) if rec_dir.is_dir() else []
    problems = [f"unexpected file records/{n} (wrong seed/key, duplicate or foreign result)"
                for n in present if n not in names]
    records = {}
    for entry, key in zip(entries, ts.all_keys()):
        f = rec_dir / ts.record_filename(*key)
        if not f.is_file():
            note = " - a missing C1 blocks the analysis" if key[1] == "C1" else ""
            problems.append(f"{ts.key_name(*key)}: record missing{note}")
            continue
        raw = f.read_bytes()
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            problems.append(f"{ts.key_name(*key)}: record is not valid JSON")
            continue
        found = _check_record(key, raw, rec, entry, complete, mode)
        problems += found
        if not found:
            records[key] = rec
    runs = path / ts.RUNS_CSV
    if not runs.is_file():
        problems.append("runs.csv missing")
    elif not problems:
        runs_raw = runs.read_bytes()
        if runs_raw != ts.runs_csv_bytes(records) or ts._sha(runs_raw) != complete["runs_csv_sha256"]:
            problems.append("runs.csv does not match the records and grid_complete.json")
    if problems:
        raise AnalysisError(f"{path}: not the complete valid twelve-key grid; analysis blocked:\n  "
                            + "\n  ".join(problems))
    return {"path": path, "mode": mode, "records": records, "complete": complete,
            "complete_sha256": complete_sha, "runs_csv_sha256": complete["runs_csv_sha256"]}


def load_freeze_costs(grid, manifest_path=None):
    """Distinct real demonstrations behind each selected zero/few-shot corpus, from the grid's own freeze."""
    mode = grid["mode"]
    path = Path(manifest_path) if manifest_path else ts.ACCEPTED_FREEZE_PATH
    if mode == "real" and (manifest_path is not None and path.resolve() != ts.ACCEPTED_FREEZE_PATH.resolve()):
        raise AnalysisError("real analysis reads only the accepted freeze manifest")
    if mode == "fake" and manifest_path is None:
        raise AnalysisError("--dry-run needs the fixture's --freeze-manifest")
    try:
        raw = fz._lf(path.read_bytes())
    except OSError as exc:
        raise AnalysisError(f"{path}: freeze manifest unreadable: {exc}") from exc
    if ts._sha(raw) != grid["complete"]["freeze"]["sha256"]:
        raise AnalysisError(f"{path}: freeze manifest is not the one the grid was trained on")
    artifacts = json.loads(raw.decode("utf-8")).get("artifacts", {})
    demos = {}
    for task in TASKS:
        for setting in ("zero_shot", "few_shot"):
            n = (artifacts.get(f"{task}/{setting}") or {}).get("selected_distinct_demonstrations")
            if not isinstance(n, int) or isinstance(n, bool) or n < 0 or (setting == "zero_shot" and n != 0):
                raise AnalysisError(f"{path}: {task}/{setting} has no valid selected_distinct_demonstrations")
            demos[(task, setting)] = n
    return demos


# ---------------------------------------------------------------- calculations (pure)

def metrics_table(records):
    """{(task, cond): {"macro_f1", "accuracy", "n_train", "composition"}} at full precision."""
    return {(t, c): {"macro_f1": r["metrics"]["macro_f1"], "accuracy": r["metrics"]["accuracy"],
                     "n_train": r["training"]["n_train"], "composition": dict(r["training"]["composition"])}
            for (t, c, _), r in records.items()}


def relative_gap(synthetic, real):
    """(synthetic - real) / real, or None when the real-only denominator is 0 (undefined, never divided)."""
    return None if real == 0 else (synthetic - real) / real


def _need(table, task, cond):
    if (task, cond) not in table:
        if cond == "C1":
            raise AnalysisError(f"{task}: C1 is missing; a missing real-only baseline blocks the analysis")
        raise AnalysisError(f"{task}: {cond} is missing; all twelve keys are required")
    return table[(task, cond)]


def gaps(table):
    """C2/C3 versus same-task seed-13 C1: absolute and relative macro-F1 gap, with undefined cases kept."""
    out = []
    for task in TASKS:
        c1 = _need(table, task, "C1")["macro_f1"]
        for cond in SYNTHETIC_ONLY:
            f1 = _need(table, task, cond)["macro_f1"]
            rel = relative_gap(f1, c1)
            out.append({"task": task, "condition": cond, "c1_macro_f1": c1, "macro_f1": f1,
                        "absolute_gap": f1 - c1, "relative_gap": rel,
                        "status": "defined" if rel is not None else "undefined: C1 macro-F1 is 0"})
    return out


def _direction(diff):
    return "higher" if diff > 0 else "lower" if diff < 0 else "equal"


def comparisons(table, gap_rows):
    """The four planned comparisons as within-task differences; plain direction words, no inference."""
    rows = []
    by = {(g["task"], g["condition"]): g for g in gap_rows}
    for cond in SYNTHETIC_ONLY:
        t, e = by[("trec", cond)], by[("emotion", cond)]
        defined = t["relative_gap"] is not None and e["relative_gap"] is not None
        rows.append({"id": f"cross_task_gap_{cond}",
                     "title": f"{NAMES[cond]} ({cond}) gap versus real-only, Emotion compared with TREC",
                     "a": cond, "b": "C1",
                     "per_task": {task: {"absolute": by[(task, cond)]["absolute_gap"],
                                         "relative": by[(task, cond)]["relative_gap"]} for task in TASKS},
                     "emotion_minus_trec_relative": (e["relative_gap"] - t["relative_gap"]) if defined else None,
                     "emotion_minus_trec_absolute": e["absolute_gap"] - t["absolute_gap"],
                     "status": "defined" if defined else "relative difference undefined: a C1 macro-F1 is 0"})
    for cid, title, a, b in COMPARISONS:
        per_task = {}
        for task in TASKS:
            x, y = _need(table, task, a), _need(table, task, b)
            per_task[task] = {"macro_f1_diff": x["macro_f1"] - y["macro_f1"],
                              "accuracy_diff": x["accuracy"] - y["accuracy"],
                              "n_train": [x["n_train"], y["n_train"]]}
        rows.append({"id": cid, "title": title, "a": a, "b": b, "per_task": per_task})
    return rows


def costs(table, demos):
    """Real labelled examples consumed per condition, kept separate from classifier training counts."""
    out = []
    for task in TASKS:
        for cond in CONDITIONS:
            row = _need(table, task, cond)
            comp = row["composition"]
            real_rows = sum(comp.get(p, 0) for p in REAL_PARTS)
            synthetic_rows = sum(comp.get(p, 0) for p in SYNTHETIC_PARTS)
            if comp.get("few_shot"):
                demo_n, demo_note = demos[(task, "few_shot")], (
                    "distinct real demonstrations behind the selected few-shot corpus" if cond == "C3" else
                    "at most this many: the 90 few-shot rows are a subset of the C3 corpus")
            elif comp.get("zero_shot"):
                demo_n, demo_note = demos[(task, "zero_shot")], "zero-shot prompts use no real examples"
            elif comp.get("auggpt"):
                demo_n, demo_note = 0, "paraphrases of the same real sources counted in real training rows"
            else:
                demo_n, demo_note = 0, "real data only"
            out.append({"task": task, "condition": cond, "n_train": row["n_train"],
                        "real_training_rows": real_rows, "synthetic_training_rows": synthetic_rows,
                        "real_generation_demonstrations": demo_n, "note": demo_note})
    return out


def analyse(grid, demos):
    table = metrics_table(grid["records"])
    gap_rows = gaps(table)
    return {"table": table, "gaps": gap_rows, "comparisons": comparisons(table, gap_rows),
            "costs": costs(table, demos)}


# ---------------------------------------------------------------- presentation

def _f(v, digits=4):
    return f"{v:.{digits}f}"


def _signed(v, digits=4):
    return f"{v:+.{digits}f}"


def _pct(v):
    return "undefined (C1 = 0)" if v is None else f"{v * 100:+.2f}%"


def _tag(mode):
    return "REAL RESULTS (seed 13, single run)" if mode == "real" else "FAKE FIXTURE ANALYSIS - NOT REAL EVIDENCE"


def summary_markdown(result, provenance):
    mode = provenance["generation_mode"]
    table, lines = result["table"], []
    add = lines.append
    add(f"# Single-seed gap analysis: scope30-v1, seed 13 - {_tag(mode)}")
    add("")
    add("Generated by `python -m ws3_evaluation.analyze_scope30`. Values are single runs at full precision in "
        "`analysis.json`; shown here to four decimals. No standard deviation, confidence interval, significance "
        "or stability figure exists for one seed, and none is shown.")
    add("")
    add("## Inputs")
    add("")
    g = provenance["grid"]
    add(f"- Grid: `{g['directory']}`, produced at `{g['revision']}`, `grid_complete.json` sha256 "
        f"`{g['grid_complete_sha256']}`, `runs.csv` sha256 `{g['runs_csv_sha256']}`.")
    add(f"- Freeze: `{g['freeze']['freeze_id']}` sha256 `{g['freeze']['sha256']}`.")
    emb = g["classifier"]["embedding"]
    add(f"- Classifier: frozen `{emb.get('name')}`" + (f" revision `{emb['revision']}`" if emb.get("revision") else "")
        + f" + LogisticRegression {json.dumps(g['classifier']['classifier'], sort_keys=True)}.")
    add(f"- Acceptance: {g['decision']}.")
    add("")
    add("## Macro-F1 and accuracy on the official test sets")
    add("")
    add("| Task | Metric | " + " | ".join(f"{c} {NAMES[c]}" for c in CONDITIONS) + " |")
    add("| --- | --- |" + " ---: |" * len(CONDITIONS))
    for task in TASKS:
        add(f"| {TASK_NAMES[task]} | n_train | " + " | ".join(str(table[(task, c)]["n_train"]) for c in CONDITIONS)
            + " |")
        for m, label in (("macro_f1", "macro-F1"), ("accuracy", "accuracy")):
            add(f"| {TASK_NAMES[task]} | {label} | " + " | ".join(_f(table[(task, c)][m]) for c in CONDITIONS) + " |")
    add("")
    add("## Synthetic-only versus real-only (C1) gaps")
    add("")
    add("Relative gap = (synthetic macro-F1 - same-task seed-13 C1 macro-F1) / C1 macro-F1. When C1 is 0 the "
        "relative gap is undefined and only the absolute difference is reported.")
    add("")
    add("| Task | Condition | C1 macro-F1 | macro-F1 | Absolute gap | Relative gap |")
    add("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in result["gaps"]:
        add(f"| {TASK_NAMES[r['task']]} | {r['condition']} {NAMES[r['condition']]} | {_f(r['c1_macro_f1'])} | "
            f"{_f(r['macro_f1'])} | {_signed(r['absolute_gap'])} | {_pct(r['relative_gap'])} |")
    add("")
    add("## The four planned comparisons (within-task differences, single run)")
    add("")
    for i, c in enumerate(result["comparisons"], 1):
        if c["id"].startswith("cross_task_gap"):
            if i == 1:
                add("**1. Synthetic-versus-real gap across tasks** (planned expectation: a larger negative gap on "
                    "the more subjective Emotion task).")
                add("")
            pt = c["per_task"]
            rel = c["emotion_minus_trec_relative"]
            add(f"- {c['title']}: TREC {_pct(pt['trec']['relative'])} ({_signed(pt['trec']['absolute'])}), "
                f"Emotion {_pct(pt['emotion']['relative'])} ({_signed(pt['emotion']['absolute'])}). Emotion minus "
                "TREC: " + (f"{rel * 100:+.2f} percentage points of relative gap" if rel is not None
                            else "relative difference undefined") + f", {_signed(c['emotion_minus_trec_absolute'])} "
                "absolute macro-F1.")
            continue
        num = i - 1
        add("")
        add(f"**{num}. {c['title']}** ({c['a']} - {c['b']}).")
        add("")
        for task in TASKS:
            d = c["per_task"][task]
            n_a, n_b = d["n_train"]
            sizes = f" Training sizes differ: {n_a} vs {n_b} rows." if n_a != n_b else ""
            add(f"- {TASK_NAMES[task]}: {c['a']} macro-F1 is {_direction(d['macro_f1_diff'])} than {c['b']} by "
                f"{_f(abs(d['macro_f1_diff']))} ({_signed(d['macro_f1_diff'])}); accuracy "
                f"{_signed(d['accuracy_diff'])}.{sizes}")
    add("")
    add("## Real labelled examples used (separate from classifier training counts)")
    add("")
    add("| Task | Condition | n_train | Real training rows | Synthetic training rows | "
        "Real demonstrations for generation | Note |")
    add("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in result["costs"]:
        add(f"| {TASK_NAMES[r['task']]} | {r['condition']} | {r['n_train']} | {r['real_training_rows']} | "
            f"{r['synthetic_training_rows']} | {r['real_generation_demonstrations']} | {r['note']} |")
    add("")
    add("## Limitations")
    add("")
    for item in LIMITATIONS:
        add(f"- {item}")
    add("")
    add("## Charts")
    add("")
    add("- `f1_by_condition.png`: six-condition macro-F1 by task (single run; no error bars).")
    add("- `gap.png`: C2/C3 relative gap versus same-task C1 (undefined cases labelled, not drawn as zero).")
    add("")
    return "\n".join(lines)


INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
COLOURS = {"trec": "#2a78d6", "emotion": "#eb6834"}     # validated categorical slots 1-2 (CVD-safe pair)


def _style(plt):
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "legend.frameon": False})


def f1_chart(result, mode):
    """Grouped bars: macro-F1 per condition and task. Single run: bars only, never error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    table = result["table"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.38
    for i, task in enumerate(TASKS):
        xs = [j + (i - 0.5) * width for j in range(len(CONDITIONS))]
        vals = [table[(task, c)]["macro_f1"] for c in CONDITIONS]
        bars = ax.bar(xs, vals, width - 0.03, color=COLOURS[task], label=TASK_NAMES[task],
                      edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    ax.set_xticks(range(len(CONDITIONS)),
                  [f"{c}\n{NAMES[c]}\nn={table[(TASKS[0], c)]['n_train']}" for c in CONDITIONS])
    ax.set_ylabel("Macro-F1 on the official test set")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper right")
    title = "Macro-F1 by training condition"
    ax.set_title(title if mode == "real" else f"{title} [FAKE FIXTURE]", loc="left", fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Seed 13, one run per condition: no error bars or uncertainty estimate. "
             "C5a/C5b training sizes differ by design.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, ax


def gap_chart(result, mode):
    """Relative C2/C3 gap versus same-task C1; an undefined gap is a labelled empty slot, never a zero bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    by = {(g["task"], g["condition"]): g for g in result["gaps"]}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    width = 0.38
    defined = [g["relative_gap"] * 100 for g in result["gaps"] if g["relative_gap"] is not None]
    span = max([abs(v) for v in defined] + [5.0])
    for i, task in enumerate(TASKS):
        for j, cond in enumerate(SYNTHETIC_ONLY):
            g = by[(task, cond)]
            x = j + (i - 0.5) * width
            if g["relative_gap"] is None:
                ax.text(x, 0, f"undefined\n(C1 = 0)\nΔF1 {g['absolute_gap']:+.3f}", ha="center", va="center",
                        fontsize=7.5, color=INK2, bbox={"boxstyle": "round", "fc": SURFACE, "ec": COLOURS[task]})
                continue
            v = g["relative_gap"] * 100
            ax.bar(x, v, width - 0.03, color=COLOURS[task], edgecolor=SURFACE, linewidth=1,
                   label=TASK_NAMES[task] if j == 0 or not any(
                       by[(task, c)]["relative_gap"] is not None for c in SYNTHETIC_ONLY[:j]) else None)
            ax.text(x, v + (0.04 * span if v >= 0 else -0.04 * span), f"{v:+.1f}%\n(ΔF1 {g['absolute_gap']:+.3f})",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    for task in TASKS:                       # legend entries even when a task's bars are all undefined
        if not any(by[(task, c)]["relative_gap"] is not None for c in SYNTHETIC_ONLY):
            ax.bar([0], [0], width=0, color=COLOURS[task], label=f"{TASK_NAMES[task]} (undefined)")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(SYNTHETIC_ONLY)), [f"{c} {NAMES[c]}" for c in SYNTHETIC_ONLY])
    ax.set_xlim(-0.6, len(SYNTHETIC_ONLY) - 0.4)
    ax.set_ylim(-span * 1.45, span * 1.45)
    ax.set_ylabel("Relative macro-F1 gap vs real-only C1 (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper left")
    title = "Synthetic-only versus real-only data"
    ax.set_title(title if mode == "real" else f"{title} [FAKE FIXTURE]", loc="left", fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "(synthetic − C1) / C1 within task, seed 13, single run; no uncertainty shown.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, ax


# ---------------------------------------------------------------- outputs and provenance

def analysis_repository(mode):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AnalysisError("cannot read the analysis checkout revision with git") from exc
    if mode == "real" and dirty:
        raise AnalysisError(f"real analysis needs a clean checkout; tracked files are modified:\n{dirty}")
    return {"revision": rev, "clean": not dirty}


def provenance(grid):
    c = grid["complete"]
    return {"generation_mode": grid["mode"], "real_run_evidence": grid["mode"] == "real",
            "grid": {"directory": Path(grid["path"]).name, "revision": c["repository"]["revision"],
                     "grid_complete_sha256": grid["complete_sha256"], "runs_csv_sha256": grid["runs_csv_sha256"],
                     "freeze": c["freeze"], "classifier": c["classifier"],
                     "decision": ACCEPTED_GRID_DECISION if grid["mode"] == "real" else "fake fixture (no decision)",
                     "records": {e["key"]: {"record_sha256": e["record_sha256"],
                                            "record_file_sha256": e["record_file_sha256"]} for e in c["results"]}}}


def _fresh_out(out):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise AnalysisError(f"{out} already has content; analysis outputs are written once to a fresh directory")
    return out


def write_outputs(grid, result, out):
    """Write summary.md, f1_by_condition.png, gap.png, then analysis.json (with their hashes) to a fresh dir."""
    out = _fresh_out(out)
    mode = grid["mode"]
    repo = analysis_repository(mode)
    prov = provenance(grid)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_bytes(summary_markdown(result, prov).encode("utf-8"))
    import matplotlib.pyplot as plt
    for name, draw in (("f1_by_condition.png", f1_chart), ("gap.png", gap_chart)):
        fig, _ = draw(result, mode)
        fig.savefig(out / name, dpi=200, metadata={"Software": None})
        plt.close(fig)
    doc = {"analysis_kind": ANALYSIS_KIND, "schema_version": SCHEMA_VERSION, **prov,
           "analysis_code": repo, "single_seed": True, "uncertainty": "none computed (one seed, one run per key)",
           "results": [{"task": t, "condition": c, **{k: v for k, v in result["table"][(t, c)].items()}}
                       for t in TASKS for c in CONDITIONS],
           "gaps": result["gaps"], "comparisons": result["comparisons"], "costs": result["costs"],
           "limitations": list(LIMITATIONS),
           "outputs_sha256": {n: ts._sha((out / n).read_bytes())
                              for n in ("summary.md", "f1_by_condition.png", "gap.png")}}
    (out / "analysis.json").write_bytes(ts._canonical(doc))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="the frozen twelve-key grid directory")
    ap.add_argument("--out", default=None, help=f"fresh output directory (default {DEFAULT_OUT})")
    ap.add_argument("--check", action="store_true", help="read-only: validate the grid and freeze; write nothing")
    ap.add_argument("--dry-run", action="store_true", help="FAKE fixture grid; never real evidence")
    ap.add_argument("--freeze-manifest", default=None, help="fake runs only")
    args = ap.parse_args(argv)
    mode = "fake" if args.dry_run else "real"
    try:
        if mode == "real" and args.freeze_manifest:
            raise AnalysisError("real analysis reads only the accepted freeze manifest")
        if mode == "fake" and not args.out and not args.check:
            raise AnalysisError("--dry-run needs an explicit --out")
        grid = load_grid(args.results_dir, mode)
        demos = load_freeze_costs(grid, args.freeze_manifest)
        result = analyse(grid, demos)
        tag = _tag(mode)
        print(f"GRID VERIFIED ({tag}): {len(grid['records'])} keys, grid_complete.json sha256 "
              f"{grid['complete_sha256']}, produced at {grid['complete']['repository']['revision']}")
        if args.check:
            print("CHECK ONLY: nothing written.")
            return
        out = write_outputs(grid, result, Path(args.out) if args.out else DEFAULT_OUT)
        print((out / "summary.md").read_text(encoding="utf-8"))
        for name in ("summary.md", "f1_by_condition.png", "gap.png", "analysis.json"):
            print(f"{name:<20}{(out / name).resolve()}  sha256 {ts._sha((out / name).read_bytes())}")
        print(f"ANALYSIS WRITTEN ({tag}). Inspect both charts before reporting.")
    except AnalysisError as exc:
        print(f"FAIL{' (FAKE fixture)' if mode == 'fake' else ''}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
