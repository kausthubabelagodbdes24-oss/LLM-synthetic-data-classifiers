"""The three data contracts from the master plan.

If you need to change a field, open a `shared/<topic>` pull request — every
workstream reads these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synthtext import config
from synthtext.io import iter_jsonl

REAL_FIELDS = ("id", "text", "label")
SYNTHETIC_FIELDS = ("id", "text", "label", "setting", "source_ids", "model", "prompt_version")
SETTINGS = ("zero_shot", "few_shot", "auggpt")
RUN_COLUMNS = [
    "task", "condition", "seed", "n_train", "macro_f1", "accuracy", "git_sha", "timestamp",
]


def check_real(record: dict[str, Any], task_id: str) -> list[str]:
    errors = [f"missing field '{f}'" for f in REAL_FIELDS if f not in record]
    if errors:
        return errors
    if not isinstance(record["text"], str) or not record["text"].strip():
        errors.append("text must be a non-empty string")
    if record["label"] not in config.labels(task_id):
        errors.append(f"label '{record['label']}' is not a {task_id} label")
    return errors


def check_synthetic(record: dict[str, Any], task_id: str) -> list[str]:
    errors = [f"missing field '{f}'" for f in SYNTHETIC_FIELDS if f not in record]
    if errors:
        return errors
    if not isinstance(record["text"], str) or not record["text"].strip():
        errors.append("text must be a non-empty string")
    if record["label"] not in config.labels(task_id):
        errors.append(f"label '{record['label']}' is not a {task_id} label")
    if record["setting"] not in SETTINGS:
        errors.append(f"setting '{record['setting']}' not in {SETTINGS}")
    if not isinstance(record["source_ids"], list):
        errors.append("source_ids must be a list")
    return errors


def check_file(path: str | Path, kind: str, task_id: str) -> list[str]:
    """Return 'line N: problem' strings; an empty list means the file is valid."""
    checker = check_real if kind == "real" else check_synthetic
    problems: list[str] = []
    seen_ids: set[str] = set()
    for i, record in enumerate(iter_jsonl(path), start=1):
        for err in checker(record, task_id):
            problems.append(f"{path}:{i}: {err}")
        rid = record.get("id")
        if rid in seen_ids:
            problems.append(f"{path}:{i}: duplicate id '{rid}'")
        seen_ids.add(rid)
    return problems
