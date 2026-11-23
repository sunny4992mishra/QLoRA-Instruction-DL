"""Agreement between the LLM judge and the human raters (issue #10).

    pip install openpyxl
    python src/collect_ratings.py          # first: writes results/human/ratings.csv
    python src/judge_vs_human.py

Inputs:
    results/human/ratings.csv       one row per rater x answer x criterion (collect_ratings.py)
    results/judge/<system>.jsonl    judge scores (judge.py), joined on the test prompt id

Only the answers the humans rated (30 prompts x 3 systems) are compared. Per criterion:
    weighted kappa, judge vs rater   Cohen's kappa with quadratic weights between the judge and
                                     each rater, averaged over raters. Compare it with the same
                                     number between raters. The judge counts as agreeing at
                                     least as well as a human if its mean kappa is at least the
                                     lowest of the rater-rater kappas (the 'Judge κ ≥ lowest pair'
                                     column); a higher kappa than every rater pair passes too.
    exact / within 1                 share of (judge, rater) pairs with equal or at most 1 apart
    Spearman rho                     rank correlation between the judge score and the mean human
                                     score of each answer
    mean difference                  judge minus mean human score (positive: judge is more lenient)
    system order                     systems sorted by mean score, judge and humans side by side

Outputs (in --dir): judge_vs_human.json, judge_vs_human.md
"""

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make the other scripts in src/ importable, so this file reuses collect_ratings.py's kappa and
# labels instead of a second copy that could drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_ratings import CRITERIA, LABELS, fmt, short, weighted_kappa  # noqa: E402


# Ranks for Spearman's rho. Example: scores [1, 3, 3, 5] get ranks [1, 2.5, 2.5, 4]; the two tied
# 3s share the mean of ranks 2 and 3.
def ranks(values):
    """Average ranks (ties share the mean rank)."""
    # Positions of the values from smallest to largest.
    order = sorted(range(len(values)), key=lambda i: values[i])
    out, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        # Extend j over every value equal to the one at i: that run of positions is one group of
        # ties.
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            # Why mean ranks: 1-5 scores have many ties, and tied answers must get the same rank.
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


# Pearson correlation: covariance divided by the product of the standard deviations. Returns None
# when one side has no spread (all values equal), because the correlation is then undefined.
def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


# Judge scores keyed by (system, prompt id), so they can be joined with the human scores, which use
# the same key. 'hindi_fluency' is accepted as an old name for 'fluency'.
def load_judge(judge_dir, systems):
    scores = {}
    for s in systems:
        path = judge_dir / f"{s}.jsonl"
        if not path.exists():
            sys.exit(f"no judge file for {s}: {path}")
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                scores[(s, r["id"])] = {c: r.get(c, r.get("hindi_fluency") if c == "fluency" else None)
                                        for c in CRITERIA}
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/human")
    p.add_argument("--judge-dir", default="results/judge")
    args = p.parse_args()
    d = Path(args.dir)

    # Human scores from ratings.csv (written by collect_ratings.py). A blank score means the rater
    # left that cell empty (for example fluency on items 2 and 9) and is skipped, so it counts as
    # missing, not as 0.
    human = defaultdict(dict)  # (system, id, criterion) -> {rater: score}
    with open(d / "ratings.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["score"] != "":
                human[(r["system"], r["id"], r["criterion"])][r["rater"]] = int(r["score"])
    # Only the three rated systems (zero-shot, r16_n100, r16_n775) are compared, because the humans
    # rated nothing else.
    systems = sorted({k[0] for k in human})
    judge = load_judge(Path(args.judge_dir), systems)
    raters = sorted({r for v in human.values() for r in v})

    result = {"raters": raters, "systems": systems, "criteria": {}}
    for c in CRITERIA:
        # The rated answers for this criterion that also have a judge score. For each criterion
        # this is all 90 answers.
        keys = [k for k in human if k[2] == c and judge.get((k[0], k[1]), {}).get(c) is not None]
        # j_pairs[rater] = list of (judge score, that rater's score) on the same answers.
        # rr_pairs[(rater1, rater2)] = list of their two scores on the same answers. means_h is the
        # mean human score of each answer and scores_j the judge's score, in the same order.
        j_pairs, rr_pairs = defaultdict(list), defaultdict(list)
        means_h, scores_j = [], []
        for s, i, _ in keys:
            j = judge[(s, i)][c]
            h = human[(s, i, c)]
            for r, v in h.items():
                j_pairs[r].append((j, v))
            # Every pair of raters who both scored this answer. With 4 raters that is 6 pairs.
            for r1, r2 in itertools.combinations(sorted(h), 2):
                rr_pairs[(r1, r2)].append((h[r1], h[r2]))
            means_h.append(sum(h.values()) / len(h))
            scores_j.append(j)
        # One kappa per rater (judge vs that rater), then the mean over raters below. None means
        # kappa is undefined (one side gave a single score throughout) and is left out.
        jk = [weighted_kappa([a for a, _ in v], [b for _, b in v]) for v in j_pairs.values()]
        jk = [k for k in jk if k is not None]
        # One kappa per rater pair (6 pairs). Their mean, min and max are the human baseline the
        # judge is compared with. Result: mean 0.80 / 0.78 / 0.78 for correctness / helpfulness /
        # fluency, range 0.68 to 0.96.
        rk = [weighted_kappa([a for a, _ in v], [b for _, b in v]) for v in rr_pairs.values()]
        rk = [k for k in rk if k is not None]
        # All (judge, rater) score pairs, for the exact and within-1 agreement shares (71% / 66% /
        # 45% exact).
        flat = [pair for v in j_pairs.values() for pair in v]
        # Spearman is Pearson on ranks, so it compares ordering and ignores the judge's scale
        # offset.
        rho = pearson(ranks(scores_j), ranks(means_h)) if len(keys) > 2 else None
        # Mean judge score and mean human score per system on the same answers (Table E3 / Table 13
        # in the report).
        by_system = {}
        for s in systems:
            idx = [n for n, k in enumerate(keys) if k[0] == s]
            by_system[s] = {"judge": sum(scores_j[n] for n in idx) / len(idx) if idx else None,
                            "human": sum(means_h[n] for n in idx) / len(idx) if idx else None}
        # Systems sorted from highest to lowest mean, for the judge or for the humans.
        order = lambda who: sorted(systems, key=lambda s: -(by_system[s][who] or 0))
        result["criteria"][c] = {
            "n_answers": len(keys),
            # Main agreement number: judge vs each rater, averaged. Result 0.82 / 0.76 / 0.71.
            "kappa_judge_vs_rater": sum(jk) / len(jk) if jk else None,
            "kappa_rater_vs_rater": sum(rk) / len(rk) if rk else None,
            "kappa_rater_vs_rater_min": min(rk) if rk else None,
            "kappa_rater_vs_rater_max": max(rk) if rk else None,
            # Why the lowest pair only: a judge that agrees better than every rater pair should
            # still pass.
            "judge_at_least_lowest_pair": bool(jk and rk and sum(jk) / len(jk) >= min(rk)),
            "exact": sum(a == b for a, b in flat) / len(flat) if flat else None,
            "within_1": sum(abs(a - b) <= 1 for a, b in flat) / len(flat) if flat else None,
            "spearman_rho": rho,
            # Negative means the judge is harsher than the humans. Fluency is -0.35: the judge
            # counts looping against fluency, the human sheet does not (0.64 lower on the 42
            # looping answers, 0.10 on the other 48).
            "mean_difference_judge_minus_human": (sum(scores_j) - sum(means_h)) / len(keys) if keys else None,
            "by_system": by_system,
            # Compares exact means, so a tiny gap can flip it. Read it with the intervals.
            "same_system_order": order("judge") == order("human"),
        }

    # Write the full result as JSON (for other scripts) and as a Markdown table (for the report).
    with open(d / "judge_vs_human.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"
    lines = [f"# Judge vs human: {len(raters)} rater(s), {len(systems)} systems", "",
             "| Criterion | Answers | κ judge–rater | κ rater–rater (min–max) | Judge κ ≥ lowest pair | Exact | Within 1 | Spearman ρ | Judge − human | Same system order |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for c in CRITERIA:
        a = result["criteria"][c]
        lines.append(f"| {LABELS[c]} | {a['n_answers']} | {fmt(a['kappa_judge_vs_rater'])} | "
                     f"{fmt(a['kappa_rater_vs_rater'])} ({fmt(a['kappa_rater_vs_rater_min'])} to {fmt(a['kappa_rater_vs_rater_max'])}) | "
                     f"{'yes' if a['judge_at_least_lowest_pair'] else 'no'} | {pct(a['exact'])} | {pct(a['within_1'])} | "
                     f"{fmt(a['spearman_rho'])} | {fmt(a['mean_difference_judge_minus_human'])} | "
                     f"{'yes' if a['same_system_order'] else 'no'} |")
    lines += ["", "## Mean score per system, judge and humans", "",
              "| System | " + " | ".join(f"{LABELS[c]} judge / human" for c in CRITERIA) + " |",
              "|---|" + "---|" * len(CRITERIA)]
    for s in systems:
        lines.append(f"| {short(s)} | " + " | ".join(
            f"{fmt(result['criteria'][c]['by_system'][s]['judge'])} / "
            f"{fmt(result['criteria'][c]['by_system'][s]['human'])}" for c in CRITERIA) + " |")
    (d / "judge_vs_human.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
