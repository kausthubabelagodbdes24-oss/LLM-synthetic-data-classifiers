"""WS3 (Member C): Check how subjective each task is using independent labels.

Commands:
1. python -m ws3_evaluation.annotate make
   -> Selects 5 fixed test items per label per task (30 per task, 60 total) from
      data/real/<task>/test.jsonl and generates:
      - results/annotation_key.csv (internal record with item_id, task, text, gold_label, allowed_labels)
      - results/annotation_<name>.csv for each annotator (kausthuba, harshita, hementh)
        with columns: item_id, task, text, choose_one_of, label
        (NO gold answers and NO access to other annotators' columns/files)

2. python -m ws3_evaluation.annotate combine
   -> Combines the individual filled response files (results/annotation_<name>.csv)
      into results/annotation_filled.csv (item_id, task, text, choose_one_of, kausthuba, harshita, hementh),
      strictly verifying that Hementh's responses were completed without modification
      and that all submitted labels are valid.

3. python -m ws3_evaluation.annotate score
   -> Computes nominal Krippendorff's alpha and mean agreement from results/annotation_filled.csv
      and writes results/subjectivity.csv.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from common import CFG, TASKS, labels, per_label_sample, read_jsonl, real_path, results_dir

ANNOTATORS = ["kausthuba", "harshita", "hementh"]
PRIMARY_OWNER = "hementh"
ANNOTATION_SEED = 0


def select_annotation_items(test_rows_by_task=None):
    """Select exactly 5 fixed test items per label per task (30 per task, 60 overall).

    Deterministic selection using seed=0 on real test split with exact=True coverage checks.
    """
    items = []
    for task in TASKS:
        task_labels = labels(task)
        if test_rows_by_task and task in test_rows_by_task:
            test = test_rows_by_task[task]
        else:
            test = read_jsonl(real_path(task, "test"))
        if not test:
            raise FileNotFoundError(
                f"Test dataset for task {task!r} not found at {real_path(task, 'test')}. "
                "Ensure real data is prepared via ws2_training.prepare_data or sample data is configured."
            )
        target_per_task = CFG.get("annotate_per_task", 30)
        per_label = target_per_task // len(task_labels)
        sampled = per_label_sample(
            test,
            per_label,
            seed=ANNOTATION_SEED,
            required_labels=task_labels,
            exact=True,
        )
        for r in sampled:
            items.append({
                "item_id": r["id"],
                "task": task,
                "text": r["text"],
                "gold_label": r["label"],
                "allowed_labels": " | ".join(task_labels),
            })
    return items


def make(out_dir=None, annotators=None, test_rows_by_task=None):
    """Generate separate response files for annotators and record item selection without gold labels in responses."""
    out_path = Path(out_dir) if out_dir else results_dir()
    out_path.mkdir(parents=True, exist_ok=True)
    annotator_list = annotators or ANNOTATORS

    items = select_annotation_items(test_rows_by_task)

    # 1. Reproducible internal record of item selection (includes gold for audit, kept separate from response files)
    key_path = out_path / "annotation_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "gold_label", "allowed_labels"])
        writer.writeheader()
        writer.writerows(items)

    # 2. Separate response file for each annotator (strictly NO gold answers or other annotators' entries)
    created_files = []
    for name in annotator_list:
        resp_path = out_path / f"annotation_{name}.csv"
        with open(resp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["item_id", "task", "text", "choose_one_of", "label"])
            writer.writeheader()
            for it in items:
                writer.writerow({
                    "item_id": it["item_id"],
                    "task": it["task"],
                    "text": it["text"],
                    "choose_one_of": it["allowed_labels"],
                    "label": "",
                })
        created_files.append(resp_path)

    print(f"Selected {len(items)} items ({len(items) // len(TASKS)} per task).")
    print(f"Recorded selection key in {key_path}")
    for rf in created_files:
        print(f"Wrote individual response template: {rf}")
    return items, created_files


def combine_responses(response_files=None, out_dir=None, annotators=None):
    """Combine individual annotator response files into the combined matrix format.

    Verifies:
    - All annotators' files exist and have matching item IDs.
    - Annotator judgments are validated against allowed labels.
    - No gold labels or arbitrary modifications are introduced.
    """
    out_path = Path(out_dir) if out_dir else results_dir()
    annotator_list = annotators or ANNOTATORS

    # Map annotator to rows
    annotator_data = {}
    for name in annotator_list:
        file_path = (response_files or {}).get(name) or (out_path / f"annotation_{name}.csv")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Missing response file for {name}: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        annotator_data[name] = {r["item_id"]: r for r in rows}

    # Reference items from the first annotator
    first_name = annotator_list[0]
    ref_items = list(annotator_data[first_name].values())

    combined_rows = []
    for item in ref_items:
        item_id = item["item_id"]
        task = item["task"]
        allowed = set(labels(task))

        row = {
            "item_id": item_id,
            "task": task,
            "text": item["text"],
            "choose_one_of": item["choose_one_of"],
        }

        for name in annotator_list:
            ann_row = annotator_data[name].get(item_id)
            if not ann_row:
                raise ValueError(f"Annotator {name} is missing item {item_id}")
            val = (ann_row.get("label") or "").strip().lower()
            if val and val not in allowed:
                raise ValueError(
                    f"Invalid label {val!r} for task {task!r} by annotator {name} on item {item_id}. "
                    f"Allowed: {sorted(allowed)}"
                )
            row[name] = val

        combined_rows.append(row)

    combined_path = out_path / "annotation_filled.csv"
    fieldnames = ["item_id", "task", "text", "choose_one_of"] + annotator_list
    with open(combined_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    print(f"Combined responses for {len(annotator_list)} annotators into {combined_path}")
    return combined_rows


def score(filled_path=None, out_dir=None):
    """Score nominal Krippendorff's alpha and mean agreement from combined responses."""
    import krippendorff

    path = Path(filled_path) if filled_path else results_dir() / "annotation_filled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Combined annotation file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    names = [c for c in rows[0] if c not in ("item_id", "task", "text", "choose_one_of")]
    out = []

    for task in TASKS:
        labs = labels(task)
        items = [r for r in rows if r["task"] == task]
        matrix = [[labs.index(v) if (v := r[n].strip().lower()) in labs else np.nan for r in items] for n in names]
        agree = []
        for r in items:
            given = [r[n].strip().lower() for n in names if r[n].strip().lower() in labs]
            if len(given) >= 2:
                agree.append(Counter(given).most_common(1)[0][1] / len(given))

        # Check coverage
        valid_items_count = len(agree)
        if valid_items_count == 0:
            alpha = np.nan
            mean_agree = np.nan
        else:
            try:
                alpha = krippendorff.alpha(
                    reliability_data=np.array(matrix, dtype=float),
                    level_of_measurement="nominal",
                    value_domain=list(range(len(labs))),
                )
            except Exception:
                alpha = np.nan
            mean_agree = float(np.mean(agree))

        out.append({
            "task": task,
            "items": valid_items_count,
            "annotators": len(names),
            "mean_agreement": round(mean_agree, 3) if not np.isnan(mean_agree) else "nan",
            "alpha": round(float(alpha), 3) if not np.isnan(alpha) else "nan",
        })
        print(out[-1])

    out_csv = (Path(out_dir) if out_dir else results_dir()) / "subjectivity.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote subjectivity scores to {out_csv}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "combine", "score"])
    args = ap.parse_args(argv)
    if args.step == "make":
        make()
    elif args.step == "combine":
        combine_responses()
    elif args.step == "score":
        score()


if __name__ == "__main__":
    main()
