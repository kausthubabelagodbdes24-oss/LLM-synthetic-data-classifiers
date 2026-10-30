"""WS3 (Member C): Check how subjective each task is using independent labels.

Commands:
1. python -m ws3_evaluation.annotate make [--force]
   -> Selects 5 fixed test items per label per task (30 per task, 60 total) from
      data/real/<task>/test.jsonl and generates:
      - results/annotation_key.csv  [PRIVATE — contains gold labels; DO NOT share with annotators]
      - results/annotation_<name>.csv for each annotator (kausthuba, harshita, hementh)
        with columns: item_id, task, text, choose_one_of, label
        (NO gold answers and NO access to other annotators' columns/files)

   File distribution guide:
      SHARE with each annotator individually:
        annotation_kausthuba.csv  -> send only to Kausthuba
        annotation_harshita.csv   -> send only to Harshita
        annotation_hementh.csv    -> send only to Hementh
      DO NOT share with anyone:
        annotation_key.csv        -> internal audit record; contains gold_label column

   The --force flag is required to regenerate files that already exist.
   Without --force the command will abort rather than overwrite completed judgments.

2. python -m ws3_evaluation.annotate combine
   -> Combines the individual filled response files (results/annotation_<name>.csv)
      into results/annotation_filled.csv (item_id, task, text, choose_one_of, kausthuba, harshita, hementh),
      validating each submission against the canonical annotation_key.csv.

   NOTE — Hementh completing his judgments before this step is run is a HUMAN PROCESS
   REQUIREMENT (per ADR-0005 independent-annotation rule).  This code does not and cannot
   verify that order; the team lead must confirm it separately before running combine.

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

# Canonical columns for each file type
_KEY_FIELDS = ["item_id", "task", "text", "gold_label", "allowed_labels"]
_RESPONSE_FIELDS = ["item_id", "task", "text", "choose_one_of", "label"]


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


def make(out_dir=None, annotators=None, test_rows_by_task=None, force=False):
    """Generate separate response files for annotators and record item selection without gold labels in responses.

    Overwrite protection:
        If any output file already exists and ``force`` is False (the default),
        this function raises FileExistsError listing the conflicting paths rather
        than silently overwriting completed judgments.  Pass force=True (or
        --force on the CLI) only after confirming no judgments will be lost.

    File privacy:
        annotation_key.csv — contains gold_label column.
            DO NOT share with annotators.  Internal audit record only.
        annotation_<name>.csv — contains no gold labels and no other annotator
            columns.  Safe to share ONLY with the named annotator.
    """
    out_path = Path(out_dir) if out_dir else results_dir()
    out_path.mkdir(parents=True, exist_ok=True)
    annotator_list = annotators or ANNOTATORS

    # --- Overwrite protection ---
    # Check before doing any work so we never leave a partially-regenerated set.
    candidate_paths = [out_path / "annotation_key.csv"]
    for name in annotator_list:
        candidate_paths.append(out_path / f"annotation_{name}.csv")
    existing = [str(p) for p in candidate_paths if p.exists()]
    if existing and not force:
        existing_list = "\n  ".join(existing)
        raise FileExistsError(
            "The following annotation files already exist and may contain completed judgments:\n"
            f"  {existing_list}\n"
            "Pass --force (CLI) or force=True (API) to regenerate them, "
            "but only after confirming that no judgments will be lost."
        )

    items = select_annotation_items(test_rows_by_task)

    # 1. Reproducible internal record of item selection (includes gold for audit, kept separate from response files)
    # annotation_key.csv — PRIVATE. Contains gold_label. Do NOT distribute to annotators.
    key_path = out_path / "annotation_key.csv"
    with open(key_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_KEY_FIELDS)
        writer.writeheader()
        writer.writerows(items)

    # 2. Separate response file for each annotator (strictly NO gold answers or other annotators' entries)
    created_files = []
    for name in annotator_list:
        resp_path = out_path / f"annotation_{name}.csv"
        with open(resp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_RESPONSE_FIELDS)
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
    print(f"Recorded selection key in {key_path}  [PRIVATE — do NOT share with annotators]")
    for rf in created_files:
        annotator_name = rf.stem.removeprefix("annotation_")
        print(f"Wrote response template for {annotator_name}: {rf}  [share only with {annotator_name}]")
    return items, created_files


def combine_responses(response_files=None, out_dir=None, annotators=None, key_path=None):
    """Combine individual annotator response files into the combined matrix format.

    Validates each submission against the canonical annotation_key.csv so that
    no rows are silently discarded and no phantom rows are accepted.

    Checks performed per annotator file:
      - Exact header match: must be exactly ['item_id', 'task', 'text', 'choose_one_of', 'label']
      - No duplicate item_id rows within a single submission
      - No missing item IDs (items in canonical set absent from submission)
      - No extra item IDs (items in submission absent from canonical set)
      - task, text, and choose_one_of metadata match the canonical key exactly
      - Labels validated against the exact allowed values (strip() only — no case normalisation).
        Blank labels are accepted (treated as incomplete, NOT silently converted to any value).
        Invalid non-blank labels raise ValueError with an actionable message.

    NOTE — The order in which annotators submit (e.g. Hementh completing first) is a
    HUMAN PROCESS REQUIREMENT per ADR-0005.  This code does not enforce or verify that
    ordering.  The team lead must confirm it before running combine.
    """
    out_path = Path(out_dir) if out_dir else results_dir()
    annotator_list = annotators or ANNOTATORS

    # Load canonical item set from annotation_key.csv
    resolved_key_path = Path(key_path) if key_path else (out_path / "annotation_key.csv")
    if not resolved_key_path.exists():
        raise FileNotFoundError(
            f"Canonical annotation key not found: {resolved_key_path}. "
            "Run 'python -m ws3_evaluation.annotate make' first."
        )
    with open(resolved_key_path, "r", encoding="utf-8") as fh:
        key_rows = list(csv.DictReader(fh))

    # Build canonical lookup: item_id -> key row (task, text, allowed_labels)
    canonical = {r["item_id"]: r for r in key_rows}
    canonical_ids = set(canonical)

    # --- Load and validate each annotator's submission ---
    annotator_data = {}  # name -> {item_id: row}
    for name in annotator_list:
        file_path = (response_files or {}).get(name) or (out_path / f"annotation_{name}.csv")
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Missing response file for annotator '{name}': {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            actual_fields = list(reader.fieldnames or [])
            rows = list(reader)

        # 1. Header check
        if actual_fields != list(_RESPONSE_FIELDS):
            raise ValueError(
                f"Wrong headers in '{file_path}' for annotator '{name}'. "
                f"Expected: {list(_RESPONSE_FIELDS)}. Got: {actual_fields}."
            )

        # 2. Duplicate item_id check
        seen_ids = []
        for r in rows:
            seen_ids.append(r["item_id"])
        if len(seen_ids) != len(set(seen_ids)):
            dup_counts = Counter(seen_ids)
            dups = [iid for iid, cnt in dup_counts.items() if cnt > 1]
            raise ValueError(
                f"Annotator '{name}' submission has duplicate item_id(s): {sorted(dups)}. "
                "Each item must appear exactly once."
            )

        submitted_ids = set(seen_ids)

        # 3. Missing item IDs
        missing_ids = canonical_ids - submitted_ids
        if missing_ids:
            raise ValueError(
                f"Annotator '{name}' submission is missing {len(missing_ids)} item(s) "
                f"from the canonical set: {sorted(missing_ids)}"
            )

        # 4. Extra item IDs
        extra_ids = submitted_ids - canonical_ids
        if extra_ids:
            raise ValueError(
                f"Annotator '{name}' submission contains {len(extra_ids)} item(s) "
                f"not in the canonical set: {sorted(extra_ids)}"
            )

        # 5. Metadata match and label validation per row
        id_to_row = {r["item_id"]: r for r in rows}
        for item_id, sub_row in id_to_row.items():
            canon_row = canonical[item_id]

            # task metadata
            if sub_row["task"] != canon_row["task"]:
                raise ValueError(
                    f"Annotator '{name}', item '{item_id}': 'task' mismatch. "
                    f"Expected {canon_row['task']!r}, got {sub_row['task']!r}."
                )
            # text metadata
            if sub_row["text"] != canon_row["text"]:
                raise ValueError(
                    f"Annotator '{name}', item '{item_id}': 'text' mismatch. "
                    f"Expected {canon_row['text']!r}, got {sub_row['text']!r}."
                )
            # choose_one_of matches allowed_labels in key
            if sub_row["choose_one_of"] != canon_row["allowed_labels"]:
                raise ValueError(
                    f"Annotator '{name}', item '{item_id}': 'choose_one_of' mismatch. "
                    f"Expected {canon_row['allowed_labels']!r}, got {sub_row['choose_one_of']!r}."
                )

            # Label validation — strip only, no case normalisation
            raw_label = sub_row.get("label", "")
            val = raw_label.strip()
            if val == "":
                # Blank label: treat as incomplete. Do NOT raise — caller/score handles np.nan.
                pass
            else:
                task = sub_row["task"]
                allowed = set(labels(task))
                if val not in allowed:
                    raise ValueError(
                        f"Annotator '{name}' submitted label {val!r} for item '{item_id}' "
                        f"(task {task!r}). Valid labels are: {sorted(allowed)}. "
                        "Labels must match exactly (case-sensitive) with no extra whitespace."
                    )

        annotator_data[name] = id_to_row

    # --- Build combined matrix (canonical order) ---
    combined_rows = []
    for item_id, canon_row in canonical.items():
        row = {
            "item_id": item_id,
            "task": canon_row["task"],
            "text": canon_row["text"],
            "choose_one_of": canon_row["allowed_labels"],
        }
        for name in annotator_list:
            # Preserve exact stripped value (empty string for blank)
            raw = annotator_data[name][item_id].get("label", "")
            row[name] = raw.strip()
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
        lab_set = set(labs)
        items = [r for r in rows if r["task"] == task]

        # strip() only — no lowercasing; blank stays blank (np.nan in matrix)
        def to_index(val, labs=labs, lab_set=lab_set):
            v = val.strip()
            if v == "" or v not in lab_set:
                return np.nan
            return labs.index(v)

        matrix = [[to_index(r[n]) for r in items] for n in names]

        agree = []
        for r in items:
            given = [r[n].strip() for n in names if r[n].strip() in lab_set]
            if len(given) >= 2:
                agree.append(Counter(given).most_common(1)[0][1] / len(given))

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
    ap.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "For 'make': overwrite existing annotation files even if they already exist. "
            "Use only after confirming no completed judgments will be lost."
        ),
    )
    args = ap.parse_args(argv)
    if args.step == "make":
        make(force=args.force)
    elif args.step == "combine":
        combine_responses()
    elif args.step == "score":
        score()


if __name__ == "__main__":
    main()
