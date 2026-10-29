"""WS3 (Member C): check how subjective each task is, using our own labels.

1. python -m ws3_evaluation.annotate make
   -> results/annotation_sheet.csv (30 test items per task). Upload it to Google Sheets;
      each of us fills our own column with one of the listed labels, WITHOUT looking up the answer.
2. Download the filled sheet as results/annotation_filled.csv, then:
   python -m ws3_evaluation.annotate score
   -> results/subjectivity.csv: per task, mean agreement and Krippendorff's alpha (lower alpha = more subjective)
"""

import argparse
import csv
from collections import Counter

import numpy as np

from common import CFG, TASKS, labels, per_label_sample, read_jsonl, real_path, results_dir

NAMES = ["kausthuba", "member_b", "member_c"]


def make():
    rows = []
    for task in TASKS:
        test = read_jsonl(real_path(task, "test"))
        per = max(1, CFG["annotate_per_task"] // len(labels(task)))
        for r in per_label_sample(test, per, seed=0):
            rows.append({"item_id": r["id"], "task": task, "text": r["text"],
                         "choose_one_of": " | ".join(labels(task)), **{n: "" for n in NAMES}})
    path = results_dir() / "annotation_sheet.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} items to {path}")


def score():
    import krippendorff
    rows = list(csv.DictReader(open(results_dir() / "annotation_filled.csv", encoding="utf-8")))
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
        alpha = krippendorff.alpha(reliability_data=np.array(matrix, dtype=float), level_of_measurement="nominal")
        out.append({"task": task, "items": len(agree), "annotators": len(names),
                    "mean_agreement": round(float(np.mean(agree)), 3), "alpha": round(float(alpha), 3)})
        print(out[-1])
    with open(results_dir() / "subjectivity.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "score"])
    {"make": make, "score": score}[ap.parse_args().step]()
