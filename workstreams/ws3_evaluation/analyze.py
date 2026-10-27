"""Turn results into tables, the H1-H3 numbers and the report figures.

    python -m workstreams.ws3_evaluation.analyze

Reads results/runs.csv (+ subjectivity.csv, item_agreement.csv, predictions/ when present).
Writes:
    results/summary.csv          mean and std of macro-F1 / accuracy per task x condition
    results/summary.md           the same as a markdown table for the report
    results/gaps.csv             relative gap of C2 / C3 vs C1 per task, with subjectivity
    results/correlation.json     Spearman rho between subjectivity and gap (H1)
    reports/figures/*.png        f1_by_condition, subjectivity_vs_gap, mixing_curve, agreement_threshold
If subjectivity.csv is missing, the scatter falls back to the *expected* subjectivity
from tasks.yaml and says so in its title. Do not report that version.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from synthtext import config
from synthtext.io import read_csv, read_jsonl, write_csv
from synthtext.paths import figures_dir, real_dir, results_root

# Validated categorical palette (fixed order, light mode) — one colour per task.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EXPECTED = {"low": 0.2, "medium": 0.5, "high": 0.8}
MIX_ORDER = [("C1", 0.0), ("C4a", 0.25), ("C4b", 0.5), ("C4c", 0.75), ("C3", 1.0)]


def _float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarise(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"])].append(r)
    cond_order = list(config.experiments()["conditions"])
    rows = []
    for (task, cond), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], cond_order.index(kv[0][1]))):
        f1 = np.array([float(r["macro_f1"]) for r in rs])
        acc = np.array([float(r["accuracy"]) for r in rs])
        rows.append({
            "task": task, "condition": cond, "seeds": len(rs),
            "n_train": int(np.mean([int(r["n_train"]) for r in rs])),
            "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std()), 4),
            "accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(float(acc.std()), 4),
        })
    return rows


def to_markdown(summary: list[dict]) -> str:
    conds = [c for c in config.experiments()["conditions"] if any(s["condition"] == c for s in summary)]
    tasks = [t for t in config.task_ids("all") if any(s["task"] == t for s in summary)]
    cell = {(s["task"], s["condition"]): s for s in summary}
    lines = ["| Task | " + " | ".join(conds) + " |", "| --- |" + " --- |" * len(conds)]
    for t in tasks:
        vals = []
        for c in conds:
            s = cell.get((t, c))
            vals.append(f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}" if s else "—")
        lines.append(f"| {t} | " + " | ".join(vals) + " |")
    return "Macro-F1 on the real test set, mean ± std over seeds.\n\n" + "\n".join(lines) + "\n"


def gaps(summary: list[dict]) -> list[dict]:
    cell = {(s["task"], s["condition"]): s["macro_f1_mean"] for s in summary}
    subj = {r["task"]: r for r in read_csv(results_root() / "subjectivity.csv")}
    rows = []
    for t in config.task_ids("all"):
        real = cell.get((t, "C1"))
        if not real:
            continue
        measured = _float(subj.get(t, {}).get("subjectivity"))
        row = {
            "task": t,
            "subjectivity": measured if measured is not None else "",
            "expected_subjectivity": config.task(t)["expected_subjectivity"],
            "f1_real": real,
        }
        for cond, name in (("C2", "zero_shot"), ("C3", "few_shot")):
            syn = cell.get((t, cond))
            row[f"f1_{name}"] = syn if syn is not None else ""
            row[f"gap_{name}"] = round((syn - real) / real, 4) if syn is not None else ""
        rows.append(row)
    return rows


def correlation(gap_rows: list[dict]) -> dict:
    from scipy.stats import spearmanr

    out = {}
    for name in ("zero_shot", "few_shot"):
        pairs = [(r["subjectivity"], r[f"gap_{name}"]) for r in gap_rows
                 if r["subjectivity"] != "" and r[f"gap_{name}"] != ""]
        if len(pairs) >= 3:
            rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            out[name] = {"n_tasks": len(pairs), "spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4)}
        else:
            out[name] = {"n_tasks": len(pairs), "note": "need measured subjectivity for >= 3 tasks"}
    out["reading"] = "H1 predicts a negative rho: more subjective tasks lose more F1 with synthetic data."
    return out


# ---------------- figures ----------------

def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    })


def fig_f1_by_condition(plt, summary: list[dict], out) -> None:
    tasks = [t for t in config.task_ids("all") if any(s["task"] == t for s in summary)]
    if not tasks:
        return
    conds = [c for c in config.experiments()["conditions"] if any(s["condition"] == c for s in summary)]
    cell = {(s["task"], s["condition"]): s for s in summary}
    cols = 2 if len(tasks) > 1 else 1
    rows = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.0 * rows), sharey=True, squeeze=False)
    for ax, t in zip(axes.flat, tasks):
        means = [cell[(t, c)]["macro_f1_mean"] if (t, c) in cell else np.nan for c in conds]
        stds = [cell[(t, c)]["macro_f1_std"] if (t, c) in cell else 0 for c in conds]
        ax.bar(range(len(conds)), means, width=0.6, color=PALETTE[0], edgecolor=SURFACE, linewidth=2,
               yerr=stds, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
        if (t, "C1") in cell:
            ax.axhline(cell[(t, "C1")]["macro_f1_mean"], color=INK_2, linestyle="--", linewidth=1)
        ax.set_title(config.task(t)["title"], loc="left")
        ax.set_xticks(range(len(conds)), conds, fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(tasks):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1 (real test set)")
    fig.suptitle("Macro-F1 by training condition, mean ± std over seeds (dashed line = real only, C1)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "f1_by_condition.png", dpi=200)
    plt.close(fig)


def fig_subjectivity_vs_gap(plt, gap_rows: list[dict], out) -> None:
    measured = all(r["subjectivity"] != "" for r in gap_rows)
    xs = {r["task"]: (r["subjectivity"] if measured else EXPECTED[r["expected_subjectivity"]]) for r in gap_rows}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, colour, marker in (("zero_shot", PALETTE[1], "o"), ("few_shot", PALETTE[0], "s")):
        pts = [(xs[r["task"]], r[f"gap_{name}"], r["task"]) for r in gap_rows if r[f"gap_{name}"] != ""]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=56, color=colour, marker=marker,
                   edgecolor=SURFACE, linewidth=2, label=name.replace("_", "-"), zorder=3)
        for x, y, t in pts:
            ax.annotate(t, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Task subjectivity (1 − Krippendorff's α)" if measured
                  else "EXPECTED subjectivity (placeholder: not measured yet)")
    ax.set_ylabel("Relative macro-F1 gap vs real-only")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("H1: does the synthetic-data gap widen with subjectivity?", loc="left")
    fig.tight_layout()
    fig.savefig(out / "subjectivity_vs_gap.png", dpi=200)
    plt.close(fig)


def fig_mixing_curve(plt, summary: list[dict], out) -> None:
    cell = {(s["task"], s["condition"]): s for s in summary}
    tasks = [t for t in config.task_ids("all") if any((t, c) in cell for c, _ in MIX_ORDER)]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, t in enumerate(tasks):
        pts = [(share, cell[(t, c)]["macro_f1_mean"]) for c, share in MIX_ORDER if (t, c) in cell]
        colour = PALETTE[i % len(PALETTE)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linewidth=2, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=t)
        ax.annotate(t, pts[-1], xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_xticks([s for _, s in MIX_ORDER], ["0%\n(C1)", "25%", "50%", "75%", "100%\n(C3)"])
    ax.set_xlim(-0.05, 1.18)
    ax.set_xlabel("Share of few-shot synthetic data in a 1,000-example training set")
    ax.set_ylabel("Macro-F1 (real test set)")
    ax.legend(loc="lower left")
    ax.set_title("H3: how much synthetic data can replace real data?", loc="left")
    fig.tight_layout()
    fig.savefig(out / "mixing_curve.png", dpi=200)
    plt.close(fig)


def fig_agreement_threshold(plt, out) -> None:
    items = read_csv(results_root() / "item_agreement.csv")
    if not items:
        return
    seeds = config.experiments()["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    thresholds = [0.0, 0.5, 0.67, 1.0]
    for ax, (cond, name) in zip(axes, (("C2", "zero-shot"), ("C3", "few-shot"))):
        for i, t in enumerate(config.task_ids("all")):
            gold = {r["id"]: r["label"] for r in read_jsonl(real_dir(t) / "test.jsonl")}
            agree = {r["item_id"]: float(r["agreement"]) for r in items if r["task"] == t}
            preds = [{r["id"]: r["pred"] for r in read_jsonl(
                results_root() / "predictions" / t / f"{cond}_seed{s}.jsonl")} for s in seeds]
            preds = [p for p in preds if p]
            if not agree or not preds:
                continue
            ys = []
            for th in thresholds:
                ids = [k for k, a in agree.items() if a >= th and k in gold]
                accs = [np.mean([p.get(k) == gold[k] for k in ids]) for p in preds] if ids else [np.nan]
                ys.append(float(np.mean(accs)))
            ax.plot(thresholds, ys, color=PALETTE[i % len(PALETTE)], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=t)
        ax.set_title(f"Trained on {name} synthetic data", loc="left")
        ax.set_xlabel("Keep test items with annotator agreement ≥")
        ax.set_xticks(thresholds, ["all", "50%", "67%", "100%"])
    axes[0].set_ylabel("Accuracy on kept items")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "agreement_threshold.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    runs = read_csv(results_root() / "runs.csv")
    if not runs:
        raise SystemExit(f"No runs yet in {results_root() / 'runs.csv'}")
    summary = summarise(runs)
    write_csv(results_root() / "summary.csv", summary, list(summary[0]))
    (results_root() / "summary.md").write_text(to_markdown(summary))
    gap_rows = gaps(summary)
    if gap_rows:
        write_csv(results_root() / "gaps.csv", gap_rows, list(gap_rows[0]))
    corr = correlation(gap_rows)
    (results_root() / "correlation.json").write_text(json.dumps(corr, indent=2))
    print(to_markdown(summary))
    print(json.dumps(corr, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out = figures_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig_f1_by_condition(plt, summary, out)
    if gap_rows:
        fig_subjectivity_vs_gap(plt, gap_rows, out)
    fig_mixing_curve(plt, summary, out)
    fig_agreement_threshold(plt, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
