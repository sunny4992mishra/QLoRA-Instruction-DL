"""Results table and plots for the rank and data-size sweeps (issue #6).

Needs only matplotlib (no torch, no GPU):
    pip install matplotlib
    python src/analyze.py

Inputs (any subset may be missing; the script reports what it found):
    runs/<run>/run_info.json      written by train.py (rank, rows, params, memory, time, losses)
    results/<run>.jsonl           written by generate.py (answers to the 150 test prompts)
    results/judge/<run>.jsonl     LLM-judge scores, one line per answer (format below)

Judge score format (proposed for #9), one JSON object per line:
    {"id": "aya-hi-test-...", "correctness": 4, "helpfulness": 3, "fluency": 5, ...}
Scores are integers 1-5, or null if the judge failed on that answer. "hindi_fluency" is
accepted as a synonym for "fluency". Other fields (judge model, raw judge text) are ignored.
Judge ids are checked against results/<run>.jsonl: lines with an unknown id are ignored, a
repeated id keeps its first line, and both are reported as warnings, as are answers with no
judge line.

The run name joins the three sources: runs/<run>/, results/<run>.jsonl, results/judge/<run>.jsonl.
The zero-shot run has no run_info.json; it is drawn as a dashed reference line.

Outputs:
    results/summary.csv, results/summary.md     one row per run
    figures/quality_vs_rank.png                 rank sweep (all data)
    figures/quality_vs_data_size.png            data-size sweep (r = 16)
    figures/quality_vs_memory.png               judge score against peak GPU memory
    figures/loss_curves.png                     training and validation loss per run

Until judge scores exist, the quality plots show validation loss and the share of
Devanagari letters in the answers instead, so the script is useful from the first run on.
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

CRITERIA = ["correctness", "helpfulness", "fluency"]
ALIASES = {"hindi_fluency": "fluency"}
FULL_N = 775          # all training rows; the rank sweep uses these
SWEEP_RANK = 16       # the data-size sweep uses this rank

# Colours: validated categorical slots 1-3 (dataviz reference palette, light mode);
# the mean is drawn in primary ink so it reads as the summary line.
COLORS = {"correctness": "#2a78d6", "helpfulness": "#eb6834", "fluency": "#1baf7a",
          "mean": "#0b0b0b", "loss": "#2a78d6", "eval": "#eb6834", "deva": "#2a78d6"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[^\W\d_]")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_name(name):
    """Rank and training rows from a run name like Qwen2.5-1.5B-Instruct_r16_n775."""
    m = re.search(r"_r(\d+)_n(\d+)$", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def mean_ci(values):
    """Mean and 95% normal-approximation half-width; None for no values."""
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, None
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    return mu, 1.96 * sd / math.sqrt(n)


def answer_stats(rows):
    """Judge-free checks on generate.py output."""
    outs = [r.get("output") or "" for r in rows]
    # Both counts use the same letters (Unicode letters only), so vowel signs and other
    # combining marks are left out of numerator and denominator alike.
    found = [ch for o in outs for ch in LETTER.findall(o)]
    letters = len(found)
    deva = sum(1 for ch in found if DEVANAGARI.match(ch))
    lengths = [len(o) for o in outs]
    return {
        "n_answers": len(rows),
        "devanagari_pct": round(100 * deva / letters, 1) if letters else None,
        "median_answer_chars": statistics.median(lengths) if lengths else None,
        "empty_answers": sum(1 for o in outs if not o.strip()),
    }


def check_judge_ids(name, rows, answer_ids):
    """Keep one judge line per answer id and warn about anything that doesn't line up.

    A judge file made from the wrong results file, or one with repeated lines, would otherwise
    be averaged without any sign of a problem. answer_ids is None when the results file is
    missing; then only duplicates can be checked.
    """
    seen, kept, duplicates, unknown = set(), [], 0, 0
    for r in rows:
        rid = r.get("id")
        if answer_ids is not None and rid not in answer_ids:
            unknown += 1
        elif rid in seen:
            duplicates += 1
        else:
            seen.add(rid)
            kept.append(r)
    problems = []
    if unknown:
        problems.append(f"{unknown} line(s) with an id that is not in results/{name}.jsonl (ignored)")
    if duplicates:
        problems.append(f"{duplicates} duplicate id(s) (first line kept)")
    if answer_ids is None:
        problems.append(f"no results/{name}.jsonl to check the ids against")
    elif len(seen) < len(answer_ids):
        problems.append(f"{len(answer_ids) - len(seen)} answer(s) have no judge line")
    for msg in problems:
        print(f"WARNING judge/{name}.jsonl: {msg}", file=sys.stderr)
    return kept


def is_score(v):
    # bool is a subclass of int in Python, so True would otherwise count as a score of 1.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= v <= 5


def judge_stats(rows):
    out = {"n_judged": 0}
    per_answer_mean = []
    scores = {c: [] for c in CRITERIA}
    for r in rows:
        r = {ALIASES.get(k, k): v for k, v in r.items()}
        vals = []
        for c in CRITERIA:
            v = r.get(c)
            if is_score(v):
                scores[c].append(float(v))
                vals.append(float(v))
        if len(vals) == len(CRITERIA):
            per_answer_mean.append(sum(vals) / len(vals))
    out["n_judged"] = len(per_answer_mean)
    for c in CRITERIA:
        out[c], out[c + "_ci"] = mean_ci(scores[c])
    out["judge_mean"], out["judge_mean_ci"] = mean_ci(per_answer_mean)
    return out


def collect(runs_dir, results_dir, judge_dir):
    names = {p.parent.name for p in runs_dir.glob("*/run_info.json")}
    names |= {p.stem for p in results_dir.glob("*.jsonl")}
    names |= {p.stem for p in judge_dir.glob("*.jsonl")}
    table = []
    for name in sorted(names):
        row = {"run": name, "zeroshot": name.endswith("_zeroshot")}
        rank, n = parse_name(name)
        info_path = runs_dir / name / "run_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            args = info.get("args", {})
            peak = info.get("gpu_memory_peak_training") or {}
            loaded = info.get("gpu_memory_after_load") or {}
            rank = args.get("rank", rank)
            n = info.get("train_tokens", {}).get("rows", n)
            row.update({
                "trainable_params": info.get("trainable_params"),
                "trainable_pct": info.get("trainable_pct"),
                "mem_after_load_gb": loaded.get("peak_allocated_gb"),
                "peak_mem_gb": peak.get("peak_allocated_gb"),
                "peak_reserved_gb": peak.get("peak_reserved_gb"),
                "gpu": peak.get("gpu"),
                "train_min": round(info["train_seconds"] / 60, 1) if info.get("train_seconds") else None,
                "steps": info.get("global_steps"),
                "final_train_loss": info.get("final_train_loss"),
                "final_eval_loss": info.get("final_eval_loss"),
                "log_history": info.get("log_history", []),
            })
        row["rank"], row["train_rows"] = rank, n
        res = results_dir / f"{name}.jsonl"
        answer_ids = None
        if res.exists():
            answers = load_jsonl(res)
            answer_ids = {a.get("id") for a in answers}
            row.update(answer_stats(answers))
        jud = judge_dir / f"{name}.jsonl"
        if jud.exists():
            row.update(judge_stats(check_judge_ids(name, load_jsonl(jud), answer_ids)))
        table.append(row)
    return table


# ---------------------------------------------------------------- table

COLUMNS = [
    ("run", "Run", None), ("rank", "Rank", None), ("train_rows", "Train rows", None),
    ("trainable_params", "Trainable params", "{:,}"), ("trainable_pct", "Trainable %", "{:.3f}"),
    ("peak_mem_gb", "Peak GPU mem (GB)", "{:.2f}"), ("train_min", "Train time (min)", "{:.1f}"),
    ("final_eval_loss", "Val loss", "{:.3f}"), ("n_answers", "Answers", None),
    ("devanagari_pct", "Devanagari %", "{:.1f}"), ("median_answer_chars", "Median chars", "{:g}"),
    ("n_judged", "Judged", None), ("correctness", "Correctness", "{:.2f}"),
    ("helpfulness", "Helpfulness", "{:.2f}"), ("fluency", "Fluency", "{:.2f}"),
    ("judge_mean", "Judge mean", "{:.2f}"),
]


def sort_key(r):
    return (not r["zeroshot"], r["rank"] or 0, r["train_rows"] or 0, r["run"])


def write_tables(table, results_dir):
    table = sorted(table, key=sort_key)
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        keys = [k for k, _, _ in COLUMNS] + [c + "_ci" for c in CRITERIA] + ["judge_mean_ci", "gpu"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)

    def fmt(v, spec):
        if v is None or v == "":
            return "–"
        return spec.format(v) if spec else str(v)

    lines = ["| " + " | ".join(h for _, h, _ in COLUMNS) + " |",
             "|" + "|".join("---" if k == "run" else "---:" for k, _, _ in COLUMNS) + "|"]
    for r in table:
        lines.append("| " + " | ".join(fmt(r.get(k), s) for k, _, s in COLUMNS) + " |")
    note = ("\nPeak GPU memory is `torch.cuda.max_memory_allocated` during training (GPU 0). "
            "Judge scores are means over answers on a 1–5 scale; the judge mean averages the "
            "three criteria per answer. 95% intervals are in summary.csv. Median characters is "
            "the mean of the two middle values when the number of answers is even "
            "(`statistics.median`).\n")
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")


# ---------------------------------------------------------------- plots

def style(ax, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK_2, fontsize=9, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def line(ax, xs, ys, cis, key, label, width=2.0):
    pts = [(x, y, c) for x, y, c in zip(xs, ys, cis) if y is not None]
    if not pts:
        return
    x, y, c = zip(*pts)
    ax.plot(x, y, color=COLORS[key], linewidth=width, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
    if any(ci is not None for ci in c):
        ax.errorbar(x, y, yerr=[ci or 0 for ci in c], fmt="none", ecolor=COLORS[key],
                    elinewidth=1, capsize=3, alpha=0.6, zorder=2)


def zeroshot_line(ax, zs, key):
    if zs and zs.get(key) is not None:
        ax.axhline(zs[key], color=INK_2, linestyle="--", linewidth=1, zorder=1)
        ax.annotate("zero-shot", (0, zs[key]), xycoords=("axes fraction", "data"),
                    xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK_2)


def sweep_figure(rows, zs, xkey, xlabel, title, path, log2=False):
    rows = sorted(rows, key=lambda r: r[xkey])
    xs = [r[xkey] for r in rows]
    judged = any(r.get("judge_mean") is not None for r in rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")

    if not rows:
        empty(a1, "no runs yet")
        empty(a2, "no runs yet")
    else:
        if judged:
            for c in CRITERIA:
                line(a1, xs, [r.get(c) for r in rows], [r.get(c + "_ci") for r in rows], c, c)
            line(a1, xs, [r.get("judge_mean") for r in rows],
                 [r.get("judge_mean_ci") for r in rows], "mean", "mean", width=2.5)
            zeroshot_line(a1, zs, "judge_mean")
            a1.set_ylim(1, 5)
            style(a1, xlabel, "LLM-judge score (1–5)")
            a1.legend(frameon=False, fontsize=8, loc="lower right")
        else:
            line(a1, xs, [r.get("devanagari_pct") for r in rows], [None] * len(rows), "deva", "")
            zeroshot_line(a1, zs, "devanagari_pct")
            style(a1, xlabel, "Devanagari letters in answers (%)")
            a1.set_title("no judge scores yet", fontsize=8, color=INK_2, loc="left")
        line(a2, xs, [r.get("final_eval_loss") for r in rows], [None] * len(rows), "eval", "")
        style(a2, xlabel, "Validation loss after training")
        for ax in (a1, a2):
            if log2:
                ax.set_xscale("log", base=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def memory_figure(trained, zs, path):
    pts = [r for r in trained if r.get("peak_mem_gb") is not None]
    judged = any(r.get("judge_mean") is not None for r in pts)
    ykey, ylabel = ("judge_mean", "LLM-judge mean score (1–5)") if judged else \
                   ("final_eval_loss", "Validation loss (no judge scores yet)")
    pts = [r for r in pts if r.get(ykey) is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Answer quality against peak GPU memory", color=INK, fontsize=11, x=0.02, ha="left")
    if not pts:
        empty(ax, "no runs with memory numbers yet")
    else:
        for r in pts:
            on_rank = r["train_rows"] == FULL_N
            ax.scatter(r["peak_mem_gb"], r[ykey], s=60, zorder=3, edgecolors="white", linewidths=1.5,
                       color=COLORS["correctness"] if on_rank else COLORS["helpfulness"])
            if r.get(ykey + "_ci"):
                ax.errorbar(r["peak_mem_gb"], r[ykey], yerr=r[ykey + "_ci"], fmt="none",
                            ecolor=INK_2, elinewidth=1, capsize=3, alpha=0.5, zorder=2)
        if judged:
            zeroshot_line(ax, zs, "judge_mean")
        # Runs that use about the same memory would have overlapping labels. Group points whose
        # x is within one label width of the previous point, then spread each group's labels
        # at least one text line apart, measured on the final axes.
        lo, hi = ax.get_ylim()
        gap = 0.055 * (hi - lo)
        x0, x1 = ax.get_xlim()
        width = 0.16 * (x1 - x0)
        groups, label_y = [], {}
        for r in sorted(pts, key=lambda r: r["peak_mem_gb"]):
            if groups and r["peak_mem_gb"] - groups[-1][-1]["peak_mem_gb"] < width:
                groups[-1].append(r)
            else:
                groups.append([r])
        for group in groups:
            prev = None
            for r in sorted(group, key=lambda r: r[ykey]):
                y = r[ykey] if prev is None else max(r[ykey], prev + gap)
                label_y[r["run"]], prev = y, y
        for r in pts:
            ax.annotate(f"r{r['rank']}, n{r['train_rows']}", (r["peak_mem_gb"], label_y[r["run"]]),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_ylim(lo, max(hi, max(label_y.values()) + gap))
        ax.scatter([], [], color=COLORS["correctness"], label=f"rank sweep (n={FULL_N})")
        ax.scatter([], [], color=COLORS["helpfulness"], label=f"data-size sweep (r={SWEEP_RANK})")
        ax.legend(frameon=False, fontsize=8)
        style(ax, "Peak GPU memory during training (GB)", ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def loss_figure(trained, path):
    runs = [r for r in sorted(trained, key=sort_key) if r.get("log_history")]
    cols = min(3, max(1, len(runs)))
    rows_n = max(1, math.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3 * rows_n), squeeze=False, sharey=True)
    fig.suptitle("Training and validation loss", color=INK, fontsize=11, x=0.02, ha="left")
    if not runs:
        empty(axes[0][0], "no run_info.json yet")
    for ax, r in zip(axes.flat, runs):
        hist = r["log_history"]
        tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
        ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        if tr:
            ax.plot(*zip(*tr), color=COLORS["loss"], linewidth=1.5, label="train")
        if ev:
            ax.plot(*zip(*ev), color=COLORS["eval"], linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5, label="validation")
        ax.set_title(f"r={r['rank']}, n={r['train_rows']}", fontsize=9, color=INK, loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        style(ax, "step", "loss")
    for ax in list(axes.flat)[len(runs):]:
        if runs:
            ax.set_visible(False)
    if runs:
        axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--judge-dir", default=None, help="Default: <results-dir>/judge")
    p.add_argument("--figures-dir", default="figures")
    args = p.parse_args()

    runs_dir, results_dir = Path(args.runs_dir), Path(args.results_dir)
    judge_dir = Path(args.judge_dir) if args.judge_dir else results_dir / "judge"
    fig_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    table = collect(runs_dir, results_dir, judge_dir)
    if not table:
        raise SystemExit(f"Nothing found in {runs_dir}/, {results_dir}/ or {judge_dir}/")
    for r in sorted(table, key=sort_key):
        have = [s for s, k in (("run_info", "final_eval_loss"), ("answers", "n_answers"),
                               ("judge", "n_judged")) if r.get(k) is not None]
        print(f"{r['run']:40s} {', '.join(have)}")

    zs = next((r for r in table if r["zeroshot"]), None)
    trained = [r for r in table if not r["zeroshot"] and r["rank"] is not None]
    write_tables(table, results_dir)
    sweep_figure([r for r in trained if r["train_rows"] == FULL_N], zs, "rank", "LoRA rank r",
                 f"Rank sweep (all {FULL_N} training rows)", fig_dir / "quality_vs_rank.png", log2=True)
    sweep_figure([r for r in trained if r["rank"] == SWEEP_RANK], zs, "train_rows", "Training rows N",
                 f"Data-size sweep (r = {SWEEP_RANK})", fig_dir / "quality_vs_data_size.png")
    memory_figure(trained, zs, fig_dir / "quality_vs_memory.png")
    loss_figure(trained, fig_dir / "loss_curves.png")
    print(f"wrote {results_dir}/summary.csv, {results_dir}/summary.md and 4 figures in {fig_dir}/")


if __name__ == "__main__":
    main()
