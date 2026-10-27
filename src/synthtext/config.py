"""Load the three YAML configs in configs/."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from synthtext.paths import configs_dir


@lru_cache(maxsize=None)
def _load(name: str) -> dict[str, Any]:
    with open(configs_dir() / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def tasks() -> dict[str, dict[str, Any]]:
    """All tasks, keyed by task id (trec, rotten_tomatoes, sst5, emotion)."""
    return _load("tasks")["tasks"]


def task(task_id: str) -> dict[str, Any]:
    all_tasks = tasks()
    if task_id not in all_tasks:
        raise KeyError(f"Unknown task '{task_id}'. Known: {', '.join(all_tasks)}")
    return all_tasks[task_id]


def labels(task_id: str) -> list[str]:
    """Label names for a task, in config order."""
    return list(task(task_id)["labels"])


def generation() -> dict[str, Any]:
    return _load("generation")


def experiments() -> dict[str, Any]:
    return _load("experiments")


def task_ids(selection: str | None = None) -> list[str]:
    """Resolve a CLI --task value: 'all' (default) or a comma-separated list."""
    if selection in (None, "", "all"):
        return list(experiments()["tasks"])
    chosen = [t.strip() for t in selection.split(",") if t.strip()]
    for t in chosen:
        task(t)  # raises on unknown ids
    return chosen
