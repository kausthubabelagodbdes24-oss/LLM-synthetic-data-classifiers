"""WS3 (Member C): the results table and the two report charts.

    python -m ws3_evaluation.analyze

Reads results/runs.csv. Writes results/summary.md (macro-F1 table + gaps),
results/f1_by_condition.png and results/gap.png.
Gap = (F1 synthetic - F1 real) / F1 real. The hypothesis predicts a bigger (more negative) gap for emotion.
"""

import pandas as pd

from common import CFG, TASKS, results_dir

COLOURS = {"trec": "#2a78d6", "emotion": "#eb6834"}   # colour-blind-safe pair
NAMES = {"C1": "Real", "C2": "Zero-shot", "C3": "Few-shot", "C4": "50/50 mix",
         "C5a": "Real ×10", "C5b": "Real ×10\n+ AugGPT"}


def main():
    out = results_dir()
    runs = pd.read_csv(out / "runs.csv")
    table = runs.groupby(["task", "condition"])["macro_f1"].agg(["mean", "std"]).round(3)
    mean = table["mean"].unstack().reindex(index=TASKS, columns=[c for c in CFG["conditions"]])

    gaps = pd.DataFrame({name: (mean[c] - mean["C1"]) / mean["C1"]
                         for name, c in (("zero-shot gap", "C2"), ("few-shot gap", "C3")) if c in mean})
    text = ["Macro-F1 on the real test set, mean ± std over seeds.\n",
            "| Task | " + " | ".join(NAMES[c].replace("\n", " ") for c in mean.columns) + " |",
            "| --- |" + " --- |" * len(mean.columns)]
    for task in mean.index:
        cells = [f"{table.loc[(task, c), 'mean']:.3f} ± {table.loc[(task, c), 'std']:.3f}"
                 if (task, c) in table.index else "—" for c in mean.columns]
        text.append(f"| {task} | " + " | ".join(cells) + " |")
    text += ["", "Relative gap vs real-only (C1):", "", "| Task | " + " | ".join(gaps.columns) + " |",
             "| --- |" + " --- |" * len(gaps.columns)]
    text += [f"| {t} | " + " | ".join(f"{v:+.1%}" for v in gaps.loc[t]) + " |" for t in gaps.index]
    (out / "summary.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                         "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True})

    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.38
    for i, task in enumerate(mean.index):
        xs = [j + (i - 0.5) * width for j in range(len(mean.columns))]
        stds = [table.loc[(task, c), "std"] if (task, c) in table.index else 0 for c in mean.columns]
        ax.bar(xs, mean.loc[task], width - 0.04, yerr=stds, color=COLOURS.get(task), label=task,
               error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    ax.set_xticks(range(len(mean.columns)), [NAMES[c] for c in mean.columns])
    ax.set_ylabel("Macro-F1 (real test set)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Macro-F1 by training data", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "f1_by_condition.png", dpi=200)

    if not gaps.empty:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, task in enumerate(gaps.index):
            xs = [j + (i - 0.5) * width for j in range(len(gaps.columns))]
            ax.bar(xs, gaps.loc[task], width - 0.04, color=COLOURS.get(task), label=task)
        ax.set_xticks(range(len(gaps.columns)), list(gaps.columns))
        ax.axhline(0, color="#52514e", linewidth=1)
        ax.set_ylabel("Relative macro-F1 gap vs real data")
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        ax.set_title("Is the gap bigger on the subjective task?", loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "gap.png", dpi=200)
    print(f"charts in {out}")


if __name__ == "__main__":
    main()
