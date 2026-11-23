"""Read the filled human rating sheets, match them to the key, and summarize (issue #12).

    pip install openpyxl
    python src/collect_ratings.py

Inputs (in --dir, default results/human/), as written by make_rating_sheets.py:
    key.json                    item number -> test prompt id and system
    rating_sheet_<rater>.xlsx   one filled sheet per rater; a missing sheet is skipped

Rows are matched to the key by the item number in column A, so a rater who sorted or
filtered the sheet is still read correctly. Scores must be whole numbers 1-5. Blank cells
count as missing. Any other value stops the script with the sheet, row and value.

Outputs (in --dir):
    ratings.csv      one row per rater x item x criterion
    comments.csv     every non-empty comment, with its item, prompt id and system
    scores.jsonl     one line per rated answer: the mean over raters for each criterion.
                     Same keys as the judge files (id, correctness, helpfulness, fluency),
                     plus system and n_raters, so judge and human scores can be joined on
                     (system, id) for the judge-vs-human agreement in #10.
    summary.json, summary.md
                     mean score per system and criterion with a 95% interval, each rater's
                     means, and agreement between raters

Per-system means average the raters' scores for each answer first, then average over
the 30 answers. The 95% interval is the normal approximation over those answers, the
same as mean_ci in analyze.py.

Agreement between raters, per criterion:
    Krippendorff's alpha (interval)  all raters at once, handles missing scores;
                                     1 = perfect, 0 = chance level
    weighted kappa                   Cohen's kappa with quadratic weights, mean over all
                                     pairs of raters
    exact / within 1                 share of rater pairs giving the same score, or scores
                                     at most 1 apart, on the same answer
"""

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

CRITERIA = ["correctness", "helpfulness", "fluency"]
# Sheet layout written by make_rating_sheets.py: A item, B instruction, C reference, D answer, E-G
# the three scores, H comment.
SCORE_COLS = {"correctness": 4, "helpfulness": 5, "fluency": 6}  # 0-based: columns E, F, G
COMMENT_COL = 7                                                  # column H
LABELS = {"correctness": "Correctness", "helpfulness": "Helpfulness", "fluency": "Hindi fluency"}


# Short system name for tables.
def short(system):
    """Qwen2.5-1.5B-Instruct_r16_n775 -> r16_n775."""
    return system.split("_", 1)[1] if "_" in system else system


# Read one score cell. Blank -> None (missing, e.g. fluency not applicable). Excel can store 3 as
# 3.0, so the value is read as a number and must be a whole number from 1 to 5. 'where' names the
# sheet, row and criterion in the error message.
def parse_score(value, where):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        sys.exit(f"{where}: '{value}' is not a score (whole number 1-5)")
    # Why stop on a bad value: a typed N/A or 0 would otherwise corrupt the means without warning.
    if number != int(number) or not 1 <= number <= 5:
        sys.exit(f"{where}: '{value}' is not a score (whole number 1-5)")
    return int(number)


# Read one rater's filled sheet. data_only=True reads cell values, not formulas (three reference
# cells once started with '=' and were fixed in #29). read_only=True is faster for large sheets.
def read_sheet(path, key_items):
    """{item: {criterion: score or None, "comment": str}} for one rater's sheet."""
    wb = load_workbook(path, read_only=True, data_only=True)
    if "Ratings" not in wb.sheetnames:
        sys.exit(f"{path}: no 'Ratings' tab")
    rows = {}
    # min_row=2 skips the header row. r is the Excel row number, for error messages.
    for r, row in enumerate(wb["Ratings"].iter_rows(min_row=2, values_only=True), start=2):
        if not row or row[0] is None:
            continue
        try:
            # Rows are matched to key.json by the item number in column A, not by position, so a
            # rater who sorted or filtered the sheet is still read correctly.
            item = int(row[0])
        except (TypeError, ValueError):
            sys.exit(f"{path.name}, row {r}: item number '{row[0]}' is not a number")
        if item not in key_items:
            sys.exit(f"{path.name}, row {r}: item {item} is not in key.json")
        # Stop on an item that appears twice, or on items missing from the sheet (below): either
        # would silently change the means.
        if item in rows:
            sys.exit(f"{path.name}, row {r}: item {item} appears twice")
        rows[item] = {c: parse_score(row[SCORE_COLS[c]] if len(row) > SCORE_COLS[c] else None,
                                     f"{path.name}, row {r}, {LABELS[c]}")
                      for c in CRITERIA}
        comment = row[COMMENT_COL] if len(row) > COMMENT_COL else None
        rows[item]["comment"] = str(comment).strip() if comment is not None else ""
    missing_items = sorted(set(key_items) - set(rows))
    if missing_items:
        sys.exit(f"{path.name}: items missing from the sheet: {missing_items[:10]}"
                 f"{' ...' if len(missing_items) > 10 else ''}")
    wb.close()
    return rows


# Same 95% interval as analyze.py, so human and judge intervals can be compared directly. Example:
# r16_n775 human correctness 1.93 ± 0.49 over 30 answers.
def mean_ci(values):
    """Mean and 95% normal-approximation half-width (as in analyze.py)."""
    if not values:
        return None, None
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, None
    sd = math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))
    return mu, 1.96 * sd / math.sqrt(n)


# Krippendorff's alpha (Krippendorff, 2011), interval version: all raters at once, and it handles
# missing scores. Result: 0.80 correctness, 0.78 helpfulness, 0.78 fluency.
def krippendorff_alpha_interval(units):
    """Krippendorff's alpha with the interval metric.

    units: list of lists, the scores given to one answer by the raters who scored it.
    Answers with fewer than 2 scores are not pairable and are left out.
    """
    # An answer scored by only one rater has nothing to agree with, so it is left out.
    units = [u for u in units if len(u) >= 2]
    values = [v for u in units for v in u]
    n = len(values)
    if n < 2:
        return None
    # Observed disagreement: for every answer, squared differences between all ordered pairs of its
    # scores, normalised by (raters on that answer - 1), averaged over all scores. Two raters
    # giving 2 and 4 add (2-4)^2 twice.
    observed = sum(sum((a - b) ** 2 for a, b in itertools.permutations(u, 2)) / (len(u) - 1)
                   for u in units) / n
    # Sum of squared differences over all ordered pairs of the pooled values:
    # sum_{i,j} (x_i - x_j)^2 = 2 * n * sum (x - mean)^2.
    mu = sum(values) / n
    # alpha = 1 - observed / expected disagreement: 1 is perfect agreement and 0 is chance.
    expected = 2 * n * sum((v - mu) ** 2 for v in values) / (n * (n - 1))
    if expected == 0:
        return None  # every score is the same; alpha is undefined
    return 1 - observed / expected


# Cohen's weighted kappa (Cohen, 1968) for two raters (or the judge and one rater) on the same
# answers. kappa = 1 - observed / expected disagreement; 1 = perfect agreement, 0 = no better than
# chance, below 0 = worse than chance.
def weighted_kappa(a, b, categories=(1, 2, 3, 4, 5)):
    """Cohen's kappa with quadratic weights for two raters' paired scores."""
    n = len(a)
    if n == 0:
        return None
    k = len(categories)
    # Map the scores 1..5 to positions 0..4.
    index = {c: i for i, c in enumerate(categories)}
    # Why quadratic: scores are ordered, so a 1 vs 5 disagreement costs 16 times a 1 vs 2.
    weight = lambda i, j: (i - j) ** 2 / (k - 1) ** 2
    # Observed: average weighted disagreement over the answers both scored.
    observed = sum(weight(index[x], index[y]) for x, y in zip(a, b)) / n
    # pa and pb: how often each rater gave each score (their own score distributions).
    pa = [sum(index[x] == i for x in a) / n for i in range(k)]
    pb = [sum(index[y] == j for y in b) / n for j in range(k)]
    # Expected disagreement if both raters scored at random with their own score distributions.
    expected = sum(pa[i] * pb[j] * weight(i, j) for i in range(k) for j in range(k))
    if expected == 0:
        return None  # both raters gave one and the same score throughout
    return 1 - observed / expected


# Two decimals, or a dash for a missing value.
def fmt(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/human", help="Folder with key.json and the filled sheets")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the summary prints α; Windows consoles default to cp1252

    d = Path(args.dir)
    # key.json maps each of the 90 item numbers to its test prompt id and system. It is opened only
    # here, after all four sheets are filled, so the raters stay blind while rating.
    key = json.load(open(d / "key.json", encoding="utf-8"))
    key_items = {it["item"]: it for it in key["items"]}
    systems = key["systems"]

    sheets, absent = {}, []
    # A missing sheet is skipped and reported, so the script also works while raters are still
    # finishing.
    for rater in key["raters"]:
        path = d / f"rating_sheet_{rater}.xlsx"
        if path.exists():
            sheets[rater] = read_sheet(path, key_items)
        else:
            absent.append(rater)
    if not sheets:
        sys.exit(f"no filled sheets found in {d}")
    raters = list(sheets)

    # Completeness per rater.
    # How many of the 270 scores (90 answers x 3 criteria) each rater filled. Pranjal and Sunny
    # have 268: fluency left blank on items 2 and 9, where the correct answer is not Hindi.
    completeness = {}
    for rater, rows in sheets.items():
        filled = sum(rows[i][c] is not None for i in rows for c in CRITERIA)
        completeness[rater] = {"scores_given": filled, "scores_expected": len(rows) * len(CRITERIA)}

    # Long-format ratings and comments.
    # ratings.csv: one row per rater x item x criterion (4 x 90 x 3 = 1,080 rows).
    # judge_vs_human.py reads this file.
    with open(d / "ratings.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "item", "id", "system", "criterion", "score"])
        for rater, rows in sheets.items():
            for item in sorted(rows):
                for c in CRITERIA:
                    w.writerow([rater, item, key_items[item]["id"], key_items[item]["system"], c,
                                "" if rows[item][c] is None else rows[item][c]])
    n_comments = 0
    # comments.csv: every non-empty comment. The error analysis used these to check its categories.
    with open(d / "comments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "item", "id", "system", "comment"])
        for rater, rows in sheets.items():
            for item in sorted(rows):
                if rows[item]["comment"]:
                    w.writerow([rater, item, key_items[item]["id"], key_items[item]["system"],
                                rows[item]["comment"]])
                    n_comments += 1

    # Per-answer mean over raters (judge-compatible), and per-system summaries.
    # scores.jsonl: one line per rated answer with the mean over raters for each criterion. Same
    # keys as the judge files, so human and judge scores join on (system, id).
    per_answer = {}
    with open(d / "scores.jsonl", "w", encoding="utf-8") as f:
        for item in sorted(key_items):
            it = key_items[item]
            rec = {"id": it["id"], "system": it["system"], "item": item}
            n_raters = 0
            for c in CRITERIA:
                vals = [sheets[r][item][c] for r in raters if sheets[r][item][c] is not None]
                # Why average only given scores: blank means not applicable, not zero. Fluency on
                # items 2 and 9 uses two raters.
                rec[c] = sum(vals) / len(vals) if vals else None
                n_raters = max(n_raters, len(vals))
            rec["n_raters"] = n_raters
            per_answer[item] = rec
            rounded = {k: round(v, 4) if k in CRITERIA and v is not None else v for k, v in rec.items()}
            f.write(json.dumps(rounded, ensure_ascii=False) + "\n")

    # Mean per system: average the raters for each answer first, then average over that system's 30
    # answers (Table 13: zero-shot 1.68, r16_n100 2.12, r16_n775 2.26 for the mean of the three
    # criteria).
    by_system = {}
    for s in systems:
        items = [i for i in key_items if key_items[i]["system"] == s]
        by_system[s] = {}
        for c in CRITERIA:
            mu, ci = mean_ci([per_answer[i][c] for i in items if per_answer[i][c] is not None])
            by_system[s][c] = {"mean": mu, "ci95": ci}
        answer_means = [sum(per_answer[i][c] for c in CRITERIA) / len(CRITERIA)
                        for i in items if all(per_answer[i][c] is not None for c in CRITERIA)]
        mu, ci = mean_ci(answer_means)
        by_system[s]["mean_of_criteria"] = {"mean": mu, "ci95": ci}

    # Each rater's own means, to see whether one rater is systematically harsher or more lenient.
    by_rater = {r: {s: {c: mean_ci([sheets[r][i][c] for i in key_items
                                    if key_items[i]["system"] == s and sheets[r][i][c] is not None])[0]
                        for c in CRITERIA} for s in systems} for r in raters}

    # Agreement between raters.
    # Agreement between raters, per criterion.
    agreement = {}
    for c in CRITERIA:
        # For alpha: the list of scores each answer received.
        units = [[sheets[r][i][c] for r in raters if sheets[r][i][c] is not None] for i in key_items]
        kappas, exact, within1, n_pairs = [], 0, 0, 0
        # For kappa and exact/within-1: every pair of raters (6 pairs for 4 raters), on the answers
        # both scored. Pairwise kappa ranges from 0.68 to 0.96; one pair is at 0.96 on all three
        # criteria, so the mean without that pair is 0.77 / 0.74 / 0.74.
        for r1, r2 in itertools.combinations(raters, 2):
            pairs = [(sheets[r1][i][c], sheets[r2][i][c]) for i in key_items
                     if sheets[r1][i][c] is not None and sheets[r2][i][c] is not None]
            if pairs:
                kappa = weighted_kappa([a for a, _ in pairs], [b for _, b in pairs])
                if kappa is not None:
                    kappas.append(kappa)
                exact += sum(a == b for a, b in pairs)
                within1 += sum(abs(a - b) <= 1 for a, b in pairs)
                n_pairs += len(pairs)
        agreement[c] = {
            "krippendorff_alpha_interval": krippendorff_alpha_interval(units),
            "weighted_kappa_mean_pairwise": sum(kappas) / len(kappas) if kappas else None,
            # Exact = same score; within 1 = at most one point apart. Result: 70% / 60% / 73%
            # exact, 95% / 98% / 96% within 1.
            "exact_agreement": exact / n_pairs if n_pairs else None,
            "within_1_agreement": within1 / n_pairs if n_pairs else None,
            "rater_pairs_x_answers": n_pairs,
        }

    summary = {"raters": raters, "raters_missing": absent, "completeness": completeness,
               "systems": systems, "by_system": by_system, "by_rater": by_rater,
               "agreement": agreement, "n_comments": n_comments}
    with open(d / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # Markdown summary.
    lines = [f"# Human ratings: {len(raters)} rater(s), {len(key_items)} answers", ""]
    if absent:
        lines += [f"**No sheet yet from:** {', '.join(absent)}.", ""]
    incomplete = {r: v for r, v in completeness.items() if v["scores_given"] < v["scores_expected"]}
    if incomplete:
        lines += ["**Incomplete sheets:** " + ", ".join(
            f"{r} ({v['scores_given']} of {v['scores_expected']} scores)" for r, v in incomplete.items()), ""]
    lines += ["## Mean score per system (1–5, ± 95% interval over answers)", "",
              "| System | " + " | ".join(LABELS[c] for c in CRITERIA) + " | Mean of the three |",
              "|---|" + "---|" * (len(CRITERIA) + 1)]
    for s in systems:
        cells = [f"{fmt(by_system[s][c]['mean'])} ± {fmt(by_system[s][c]['ci95'])}"
                 for c in CRITERIA + ["mean_of_criteria"]]
        lines.append(f"| {short(s)} | " + " | ".join(cells) + " |")
    lines += ["", "## Agreement between raters", "",
              "| Criterion | Krippendorff's α (interval) | Weighted κ (mean over pairs) | Exact | Within 1 |",
              "|---|---|---|---|---|"]
    for c in CRITERIA:
        a = agreement[c]
        pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"
        lines.append(f"| {LABELS[c]} | {fmt(a['krippendorff_alpha_interval'])} | "
                     f"{fmt(a['weighted_kappa_mean_pairwise'])} | {pct(a['exact_agreement'])} | "
                     f"{pct(a['within_1_agreement'])} |")
    lines += ["", "## Each rater's mean score", "",
              "| Rater | " + " | ".join(f"{short(s)} {c[:4]}." for s in systems for c in CRITERIA) + " |",
              "|---|" + "---|" * (len(systems) * len(CRITERIA))]
    for r in raters:
        lines.append(f"| {r} | " + " | ".join(fmt(by_rater[r][s][c]) for s in systems for c in CRITERIA) + " |")
    lines += ["", f"{n_comments} comments in `comments.csv`."]
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nwrote ratings.csv, comments.csv, scores.jsonl, summary.json, summary.md in {d}")


if __name__ == "__main__":
    main()
