"""Measure how subjective each task is, two ways (Li et al. 2023, sections 3 and 6).

1. Task level — pairwise survey ("which task is more objective?"):
    python -m workstreams.ws3_evaluation.subjectivity form
        prints the 6 questions to paste into a Google Form
    python -m workstreams.ws3_evaluation.subjectivity survey --csv reports/annotation/survey.csv
        reads the Form's CSV export and ranks the tasks by wins

2. Instance level — the three of us label the same 40 test items per task:
    python -m workstreams.ws3_evaluation.subjectivity make-sheet
        writes reports/annotation/annotation_sheet.csv (upload to Google Sheets;
        each person fills their own ann_<name> column; gold labels are in a separate key file)
    python -m workstreams.ws3_evaluation.subjectivity score --sheet reports/annotation/annotation_filled.csv
        per-item agreement + per-task Krippendorff's alpha

Both update results/subjectivity.csv; per-item agreement goes to results/item_agreement.csv.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np

from synthtext import config
from synthtext.io import read_csv, read_jsonl, write_csv
from synthtext.paths import annotation_dir, real_dir, results_root
from workstreams.ws2_training.sampling import stratified_sample

SUBJ_COLUMNS = [
    "task", "n_items", "n_annotators", "mean_agreement", "alpha_nominal", "alpha_ordinal",
    "subjectivity", "survey_respondents", "survey_wins_objective", "survey_rank_objective",
]
ORDINAL_TASKS = {"sst5"}


def _subj_path() -> Path:
    return results_root() / "subjectivity.csv"


def _update_subjectivity(updates: dict[str, dict]) -> None:
    rows = {r["task"]: r for r in read_csv(_subj_path())}
    for task_id, values in updates.items():
        rows.setdefault(task_id, {"task": task_id}).update(values)
    order = config.task_ids("all")
    write_csv(_subj_path(), [rows[t] for t in order if t in rows], SUBJ_COLUMNS)


# ---------- task level: pairwise survey ----------

def cmd_form(_args) -> None:
    tasks = config.task_ids("all")
    print("Google Form: one multiple-choice question per pair. Paste the description of each task at the top.\n")
    for t in tasks:
        cfg = config.task(t)
        print(f"- {cfg['title']}: {cfg['task_context'].strip()}; labels: {', '.join(cfg['labels'])}")
    print()
    for i, (a, b) in enumerate(itertools.combinations(tasks, 2), start=1):
        print(f"Q{i}. Which task is MORE OBJECTIVE (people would agree on the right label)?")
        print(f"     ( ) {config.task(a)['title']}")
        print(f"     ( ) {config.task(b)['title']}")


def _match_task(answer: str) -> str | None:
    answer = answer.strip().lower()
    if not answer:
        return None
    for t in config.task_ids("all"):
        if answer == t or answer == config.task(t)["title"].lower():
            return t
    for t in config.task_ids("all"):
        if config.task(t)["title"].lower() in answer or t in answer:
            return t
    return None


def survey_wins(rows: list[dict[str, str]]) -> tuple[Counter, int]:
    wins: Counter = Counter({t: 0 for t in config.task_ids("all")})
    respondents = 0
    for row in rows:
        answered = False
        for col, value in row.items():
            if col.lower().startswith("timestamp") or value is None:
                continue
            t = _match_task(value)
            if t:
                wins[t] += 1
                answered = True
        respondents += int(answered)
    return wins, respondents


def cmd_survey(args) -> None:
    rows = read_csv(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")
    wins, n = survey_wins(rows)
    ranked = sorted(wins, key=lambda t: (-wins[t], t))
    updates = {t: {"survey_respondents": n, "survey_wins_objective": wins[t],
                   "survey_rank_objective": ranked.index(t) + 1} for t in wins}
    _update_subjectivity(updates)
    print(f"{n} respondents. Most objective first: " + ", ".join(f"{t} ({wins[t]})" for t in ranked))


# ---------- instance level: annotation ----------

def cmd_make_sheet(args) -> None:
    exp = config.experiments()["annotation"]
    names = [n.strip() for n in args.annotators.split(",")]
    sheet, key = [], []
    for t in config.task_ids("all"):
        test = read_jsonl(real_dir(t) / "test.jsonl")
        if not test:
            print(f"  no test file for {t}; run prepare_data first")
            continue
        for r in stratified_sample(test, exp["items_per_task"], exp["sample_seed"]):
            sheet.append({"item_id": r["id"], "task": t, "text": r["text"],
                          "choose_one_of": " | ".join(config.labels(t)),
                          **{f"ann_{n}": "" for n in names}})
            key.append({"item_id": r["id"], "task": t, "gold": r["label"]})
    cols = ["item_id", "task", "text", "choose_one_of"] + [f"ann_{n}" for n in names]
    write_csv(annotation_dir() / "annotation_sheet.csv", sheet, cols)
    write_csv(annotation_dir() / "annotation_key.csv", key, ["item_id", "task", "gold"])
    print(f"wrote {len(sheet)} items to {annotation_dir() / 'annotation_sheet.csv'} "
          "(do not open annotation_key.csv until everyone has finished)")


def _norm_label(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", " ").split())


def item_agreement(labels: list[str]) -> tuple[str, float]:
    counts = Counter(labels)
    top, n = counts.most_common(1)[0]
    return top, n / len(labels)


def task_alpha(matrix: list[list[str | None]], label_order: list[str], level: str) -> float | None:
    import krippendorff

    index = {l: i for i, l in enumerate(label_order)}
    data = np.array([[index[v] if v in index else np.nan for v in row] for row in matrix], dtype=float)
    if data.shape[0] < 2 or np.isnan(data).all():
        return None
    try:
        return round(float(krippendorff.alpha(reliability_data=data, level_of_measurement=level)), 4)
    except Exception:
        return None


def cmd_score(args) -> None:
    rows = read_csv(args.sheet)
    if not rows:
        raise SystemExit(f"No rows in {args.sheet}")
    ann_cols = [c for c in rows[0] if c.startswith("ann_")]
    gold = {r["item_id"]: r["gold"] for r in read_csv(annotation_dir() / "annotation_key.csv")}
    items_out, updates = [], {}
    for t in config.task_ids("all"):
        valid = set(config.labels(t))
        task_rows = [r for r in rows if r["task"] == t]
        if not task_rows:
            continue
        matrix, agreements = [[] for _ in ann_cols], []
        for r in task_rows:
            given = []
            for j, c in enumerate(ann_cols):
                v = _norm_label(r.get(c) or "")
                v = v if v in valid else None
                matrix[j].append(v)
                if v:
                    given.append(v)
            if len(given) >= 2:
                majority, agree = item_agreement(given)
                agreements.append(agree)
                items_out.append({"item_id": r["item_id"], "task": t, "n_annotators": len(given),
                                  "majority": majority, "agreement": round(agree, 4),
                                  "gold": gold.get(r["item_id"], "")})
        alpha_nom = task_alpha(matrix, config.labels(t), "nominal")
        alpha_ord = task_alpha(matrix, config.labels(t), "ordinal") if t in ORDINAL_TASKS else None
        updates[t] = {
            "n_items": len(agreements), "n_annotators": len(ann_cols),
            "mean_agreement": round(float(np.mean(agreements)), 4) if agreements else "",
            "alpha_nominal": "" if alpha_nom is None else alpha_nom,
            "alpha_ordinal": "" if alpha_ord is None else alpha_ord,
            "subjectivity": "" if alpha_nom is None else round(1 - alpha_nom, 4),
        }
        print(f"[{t}] items {len(agreements)}, mean agreement {updates[t]['mean_agreement']}, "
              f"alpha {updates[t]['alpha_nominal']}")
    write_csv(results_root() / "item_agreement.csv", items_out,
              ["item_id", "task", "n_annotators", "majority", "agreement", "gold"])
    _update_subjectivity(updates)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("form").set_defaults(fn=cmd_form)
    p = sub.add_parser("survey"); p.add_argument("--csv", required=True); p.set_defaults(fn=cmd_survey)
    p = sub.add_parser("make-sheet")
    p.add_argument("--annotators", default="kausthuba,member_b,member_c")
    p.set_defaults(fn=cmd_make_sheet)
    p = sub.add_parser("score"); p.add_argument("--sheet", required=True); p.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
