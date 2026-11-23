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

Optional input: data/test_near_duplicates.json (from find_near_duplicates.py), the test
prompts that have a near-copy in the training data.

Outputs:
    results/summary.csv, results/summary.md     one row per run; summary.md also has paired
                                                differences in judge scores between runs, and
                                                judge scores with and without the near-duplicate
                                                test prompts
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

# Agg draws to image files without a screen, so the script runs on a server or in CI with no
# display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

CRITERIA = ["correctness", "helpfulness", "fluency"]
# Accept an older name for the fluency score, so judge files written before the name was fixed
# still load.
ALIASES = {"hindi_fluency": "fluency"}
# These two numbers define the two sweeps: runs with 775 rows form the rank sweep (r = 4 to 64),
# and runs with r = 16 form the data-size sweep (N = 100 to 775).
FULL_N = 775          # all training rows; the rank sweep uses these
SWEEP_RANK = 16       # the data-size sweep uses this rank

# Colours: validated categorical slots 1-3 (dataviz reference palette, light mode);
# the mean is drawn in primary ink so it reads as the summary line.
COLORS = {"correctness": "#2a78d6", "helpfulness": "#eb6834", "fluency": "#1baf7a",
          "mean": "#0b0b0b", "loss": "#2a78d6", "eval": "#eb6834", "deva": "#2a78d6"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Devanagari block, and 'any Unicode letter' ([^\W\d_] = word characters that are not digits or
# underscore). Used for the share of Hindi letters in answers.
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[^\W\d_]")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# The run name carries rank and rows (..._r16_n775), so a run is identified even if run_info.json
# is missing. The zero-shot run has no _r/_n part and gets (None, None).
def parse_name(name):
    """Rank and training rows from a run name like Qwen2.5-1.5B-Instruct_r16_n775."""
    m = re.search(r"_r(\d+)_n(\d+)$", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


# Used for every ± value in the report. Example: r16_n775 correctness 1.72 ± 0.19 over 150 answers.
# n - 1 in the standard deviation is the sample (unbiased) version.
def mean_ci(values):
    """Mean and 95% normal-approximation half-width; None for no values."""
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, None
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    # 95% interval: 1.96 x standard error (sd / sqrt(n)), assuming an approximately normal mean.
    return mu, 1.96 * sd / math.sqrt(n)


# Cheap checks that need no judge: share of Devanagari letters, median answer length in characters
# (541 for zero-shot, about 190 to 240 for fine-tuned runs except N = 250 at 436) and empty
# answers.
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


# Sanity check that each judge line belongs to an answer of this run, so a judge file made from the
# wrong results file cannot be averaged silently.
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


# Mean and 95% interval of each criterion over a run's answers, plus the 'judge mean': first
# average the three criteria per answer, then average over answers (Table 9: zero-shot 1.63,
# r16_n775 2.06, best r32 2.12).
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
        # Why complete answers only: a missing criterion would otherwise change what this mean
        # measures.
        if len(vals) == len(CRITERIA):
            per_answer_mean.append(sum(vals) / len(vals))
    out["n_judged"] = len(per_answer_mean)
    for c in CRITERIA:
        out[c], out[c + "_ci"] = mean_ci(scores[c])
    out["judge_mean"], out["judge_mean_ci"] = mean_ci(per_answer_mean)
    return out


# Build one row per run by joining three sources on the run name: runs/<name>/run_info.json (cost,
# losses), results/<name>.jsonl (answers) and results/judge/<name>.jsonl (scores). Any of them may
# be missing; the row just has fewer fields.
def collect(runs_dir, results_dir, judge_dir):
    # Every run name that appears in any of the three folders.
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
                # Memory numbers are peak allocated (torch.cuda.max_memory_allocated), in GB of
                # 1024^3 bytes: 5.83 GB at r = 4 to 6.43 GB at r = 64; 5.94 GB for every r = 16
                # run.
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
            kept = check_judge_ids(name, load_jsonl(jud), answer_ids)
            row.update(judge_stats(kept))
            # Scores per prompt id, kept for the paired differences below.
            row["judge_by_id"] = {r.get("id"): {ALIASES.get(k, k): v for k, v in r.items()} for r in kept}
        table.append(row)
    return table


def short(name):
    """Qwen2.5-1.5B-Instruct_r16_n775 -> r16_n775."""
    return name.split("_", 1)[1] if "_" in name else name


# Why paired: prompt difficulty varies a lot, and comparing two runs on the same prompts removes
# that variation. Example: r64 - r4 = +0.05 ± 0.14, much tighter than the two separate intervals (±
# 0.17 each). If the interval contains 0, the difference is not distinguishable from zero.
def paired_difference(a, b, criterion=None, ids=None):
    """Mean of (a - b) over the prompts both runs have scores for, with a 95% half-width.

    All runs answer the same test prompts, so runs are compared prompt by prompt. This gives a
    much tighter interval for a difference than two separate per-run intervals. With
    criterion=None the mean of the three criteria is compared. ids restricts the prompts.
    """
    ja, jb = a.get("judge_by_id") or {}, b.get("judge_by_id") or {}
    crit = [criterion] if criterion else CRITERIA
    diffs = []
    # Only prompts both runs have scores for.
    for rid in ja.keys() & jb.keys():
        if ids is not None and rid not in ids:
            continue
        va, vb = [ja[rid].get(c) for c in crit], [jb[rid].get(c) for c in crit]
        if all(is_score(v) for v in va + vb):
            # Same prompt on both sides, so prompt difficulty cancels out of the difference.
            diffs.append(sum(va) / len(va) - sum(vb) / len(vb))
    return mean_ci(diffs) + (len(diffs),)


# Judge mean over a chosen set of prompts, e.g. the 115 without a near-copy (zero-shot 1.70) or the
# 35 with one (zero-shot 1.42).
def subset_mean(row, ids):
    """Judge mean (of the three criteria) over the answers whose id is in ids."""
    vals = []
    for rid, r in (row.get("judge_by_id") or {}).items():
        scores = [r.get(c) for c in CRITERIA]
        if rid in ids and all(is_score(v) for v in scores):
            vals.append(sum(scores) / len(scores))
    return mean_ci(vals) + (len(vals),)


# Which differences to report (Table 10): every fine-tuned run against zero-shot (all 8 exclude 0,
# +0.18 to +0.49), each adjacent step in both sweeps, and first against last (r64 - r4 = +0.05 ±
# 0.14; only 250 -> 500 = +0.27 ± 0.14 is a clear step among fine-tuned runs).
def comparisons(table):
    """Pairs (a, b) to report as a - b: every run against zero-shot, then steps along each sweep."""
    judged = [r for r in table if r.get("judge_by_id")]
    zs = next((r for r in judged if r["zeroshot"]), None)
    trained = sorted((r for r in judged if not r["zeroshot"] and r["rank"] is not None), key=sort_key)
    pairs = [(r, zs) for r in trained] if zs else []
    for sweep in (sorted((r for r in trained if r["train_rows"] == FULL_N), key=lambda r: r["rank"]),
                  sorted((r for r in trained if r["rank"] == SWEEP_RANK), key=lambda r: r["train_rows"])):
        # Why adjacent steps: they show where along a sweep the score actually changes.
        pairs += [(hi, lo) for lo, hi in zip(sweep, sweep[1:])]
        if len(sweep) > 2:
            # Why first against last: small steps can each be unclear while the whole range is not.
            pairs.append((sweep[-1], sweep[0]))
    return pairs


# ---------------------------------------------------------------- table

# Columns of results/summary.csv and summary.md: (key, header, number format).
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


# Order: zero-shot first, then by rank, then by number of rows.
def sort_key(r):
    return (not r["zeroshot"], r["rank"] or 0, r["train_rows"] or 0, r["run"])


# Write summary.csv and summary.md: the main table, the paired differences, the near-duplicate
# split and the seen/unseen table. flagged = the 35 test prompt ids with a near-copy in training;
# first_row = for each, the position of its earliest near-copy in train.jsonl.
def write_tables(table, results_dir, flagged=None, first_row=None):
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
    pairs = comparisons(table)
    if pairs:
        # * means abs(mean) > half-width, so the 95% interval excludes 0.
        def cell(a, b, c):
            mu, ci, _ = paired_difference(a, b, c)
            if mu is None:
                return "–"
            return f"{mu:+.2f} ± {ci:.2f}" + (" *" if ci is not None and abs(mu) > ci else "") if ci is not None else f"{mu:+.2f}"
        note += ("\n## Paired differences in judge scores\n\n"
                 "Each row is A − B over the test prompts both runs were judged on, with a 95% interval. "
                 "\\* marks an interval that excludes 0.\n\n"
                 "| A − B | Correctness | Helpfulness | Fluency | Judge mean | Prompts |\n|---|---:|---:|---:|---:|---:|\n")
        for a, b in pairs:
            n = paired_difference(a, b)[2]
            note += (f"| {short(a['run'])} − {short(b['run'])} | " + " | ".join(cell(a, b, c) for c in CRITERIA)
                     + f" | {cell(a, b, None)} | {n} |\n")
    judged = [r for r in table if r.get("judge_by_id")]
    # Near-duplicate analysis: full-data runs gain +0.73 to +0.90 over zero-shot on the 35 flagged
    # prompts and +0.20 to +0.40 on the other 115. rest = the 115 prompts without a near-copy.
    if flagged and judged:
        all_ids = set().union(*(r["judge_by_id"].keys() for r in judged))
        rest = all_ids - flagged
        zs = next((r for r in judged if r["zeroshot"]), None)

        def pm(mu, ci, star=False):
            if mu is None:
                return "–"
            text = f"{mu:+.2f}" if star else f"{mu:.2f}"
            if ci is not None:
                text += f" ± {ci:.2f}" + (" *" if star and abs(mu) > ci else "")
            return text
        note += ("\n## Judge mean with and without near-duplicate test prompts\n\n"
                 f"{len(flagged & all_ids)} test prompts have a near-copy in the training data "
                 "(`data/test_near_duplicates.json`). Means are over the judged answers in each group, "
                 "with a 95% interval. The last two columns are the paired difference from zero-shot "
                 "within each group; \\* marks an interval that excludes 0.\n\n"
                 "| Run | All prompts | Without near-duplicates | Near-duplicates only | "
                 "− zero-shot, without | − zero-shot, near-duplicates only |\n|---|---:|---:|---:|---:|---:|\n")
        for r in judged:
            cells = [pm(*subset_mean(r, ids)[:2]) for ids in (all_ids, rest, flagged)]
            if zs is not None and r is not zs:
                cells += [pm(*paired_difference(r, zs, ids=ids)[:2], star=True) for ids in (rest, flagged)]
            else:
                cells += ["–", "–"]
            note += f"| {short(r['run'])} | " + " | ".join(cells) + " |\n"
        if pairs:
            # The sweep steps again on the 115 clean prompts, to check the findings are not caused
            # by near-copies (r64 - r4 = +0.06 ± 0.16; 250 -> 500 = +0.27 ± 0.15).
            sweep = [(a, b) for a, b in pairs if b is not zs]
            note += ("\nSteps along the sweeps on the prompts without a near-copy in training "
                     "(paired difference in the judge mean, A − B):\n\n| A − B | Without near-duplicates |\n|---|---:|\n")
            for a, b in sweep:
                note += f"| {short(a['run'])} − {short(b['run'])} | {pm(*paired_difference(a, b, ids=rest)[:2], star=True)} |\n"
        # Memorisation test: subsets are nested, so a run with N rows has seen the near-copy of a
        # flagged prompt only if first_train_row < N (10, 17, 26 of the 35 for N = 100, 250, 500).
        # If memorisation drove the gain, 'seen' should gain more than 'unseen'. It is the other
        # way round (r16_n100: -0.03 seen vs +0.56 unseen), so the gap is mostly not memorisation.
        if first_row and zs is not None:
            note += ("\n## Near-copy in the run's own training rows\n\n"
                     "Training subsets are nested, so each data-size run has seen only some near-copies. "
                     "Gain in the judge mean over zero-shot, by whether the run's first N rows contain a "
                     "near-copy of the test prompt (`first_train_row`). Number of prompts in brackets.\n\n"
                     "| Run | Flagged, copy in its rows | Flagged, copy not in its rows | Not flagged | "
                     "No copy in its rows (last two together) |\n|---|---:|---:|---:|---:|\n")
            for r in sorted((x for x in judged if not x["zeroshot"] and x["rank"] == SWEEP_RANK
                             and x["train_rows"] < FULL_N), key=lambda x: x["train_rows"]):
                # Why split seen and unseen: memorisation can only help prompts whose near-copy
                # this run trained on.
                seen = {i for i, f in first_row.items() if f < r["train_rows"]}
                unseen = flagged - seen
                groups = (seen, unseen, rest, unseen | rest)
                cells = []
                for ids in groups:
                    mu, ci, n = paired_difference(r, zs, ids=ids)
                    cells.append(f"{pm(mu, ci, star=True)} ({n})")
                note += f"| {short(r['run'])} | " + " | ".join(cells) + " |\n"
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")


# ---------------------------------------------------------------- plots

# Plot styling only: no top/right border, light grid, grey labels.
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


# One line with dots and 95% error bars; points without a value are skipped.
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


def zeroshot_line(ax, zs, key, label=None):
    """Dashed reference line. With a label it goes in the legend; otherwise it is named on the line."""
    if zs and zs.get(key) is not None:
        ax.axhline(zs[key], color=INK_2, linestyle="--", linewidth=1, zorder=1, label=label)
        if not label:
            ax.annotate("zero-shot", (0, zs[key]), xycoords=("axes fraction", "data"),
                        xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK_2)


# Figures 3 and 4 of the report: left panel judge scores (three criteria and their mean, with
# zero-shot as a dashed line), right panel validation loss. The rank axis is log base 2 because the
# ranks double (4, 8, 16, 32, 64). Before judge scores existed, the left panel showed the
# Devanagari share instead.
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
            # The lines can sit close to the zero-shot level, so it is named in the legend,
            # and the legend goes where the scores are not.
            zeroshot_line(a1, zs, "judge_mean", label="zero-shot (mean)")
            a1.set_ylim(1, 5)
            style(a1, xlabel, "LLM-judge score (1–5)")
            top = max((r.get(c) or 0) + (r.get(c + "_ci") or 0) for r in rows for c in CRITERIA + ["judge_mean"])
            a1.legend(frameon=False, fontsize=8, loc="upper right" if top < 3.6 else "lower right")
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


# Figure 5: judge mean against peak GPU memory. The five ranks spread over only 0.60 GB and their
# scores overlap; the data sizes all sit at 5.94 GB.
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


# Figure 2: training loss (logged every 5 steps) and validation loss (after each epoch) for every
# run. Shows the epoch-3 rise for the higher ranks.
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
    p.add_argument("--near-duplicates", default="data/test_near_duplicates.json",
                   help="Test prompts with a near-copy in training (from find_near_duplicates.py). "
                        "If the file exists, judge scores are also reported with and without them.")
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
    nd_path = Path(args.near_duplicates)
    # Optional input from find_near_duplicates.py. Without it the near-duplicate tables are
    # skipped.
    near = json.loads(nd_path.read_text(encoding="utf-8")) if nd_path.exists() else None
    flagged = set(near["ids"]) if near else None
    first_row = {q["id"]: q["first_train_row"] for q in near["pairs"] if "first_train_row" in q} if near else None
    write_tables(table, results_dir, flagged, first_row)
    sweep_figure([r for r in trained if r["train_rows"] == FULL_N], zs, "rank", "LoRA rank r",
                 f"Rank sweep (all {FULL_N} training rows)", fig_dir / "quality_vs_rank.png", log2=True)
    sweep_figure([r for r in trained if r["rank"] == SWEEP_RANK], zs, "train_rows", "Training rows N",
                 f"Data-size sweep (r = {SWEEP_RANK})", fig_dir / "quality_vs_data_size.png")
    memory_figure(trained, zs, fig_dir / "quality_vs_memory.png")
    loss_figure(trained, fig_dir / "loss_curves.png")
    print(f"wrote {results_dir}/summary.csv, {results_dir}/summary.md and 4 figures in {fig_dir}/")


if __name__ == "__main__":
    main()
