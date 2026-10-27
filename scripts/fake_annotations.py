"""Smoke-test helper: fill the annotation sheet and a survey export with FAKE answers.

Only used by scripts/smoke_test.sh. Annotator agreement is made to fall with the
task's expected subjectivity so the analysis has something to draw.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synthtext import config  # noqa: E402
from synthtext.io import read_csv, write_csv  # noqa: E402
from synthtext.paths import annotation_dir  # noqa: E402

FLIP = {"low": 0.05, "medium": 0.2, "high": 0.4}


def main() -> None:
    folder = annotation_dir()
    sheet = read_csv(folder / "annotation_sheet.csv")
    gold = {r["item_id"]: r["gold"] for r in read_csv(folder / "annotation_key.csv")}
    rng = random.Random(0)
    ann_cols = [c for c in sheet[0] if c.startswith("ann_")]
    for row in sheet:
        labels = config.labels(row["task"])
        p = FLIP[config.task(row["task"])["expected_subjectivity"]]
        for c in ann_cols:
            row[c] = rng.choice(labels) if rng.random() < p else gold[row["item_id"]]
    write_csv(folder / "annotation_filled.csv", sheet, list(sheet[0]))

    tasks = config.task_ids("all")
    order = sorted(tasks, key=lambda t: FLIP[config.task(t)["expected_subjectivity"]])
    pairs = [(a, b) for i, a in enumerate(tasks) for b in tasks[i + 1:]]
    with open(folder / "survey.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Timestamp"] + [f"Q{i + 1}" for i in range(len(pairs))])
        for k in range(8):
            answers = []
            for a, b in pairs:
                winner = a if order.index(a) < order.index(b) else b
                if rng.random() < 0.15:
                    winner = b if winner == a else a
                answers.append(config.task(winner)["title"])
            w.writerow([f"2026-10-30 10:0{k}"] + answers)
    print(f"fake annotations and survey written to {folder}")


if __name__ == "__main__":
    main()
