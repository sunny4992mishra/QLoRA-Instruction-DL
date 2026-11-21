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
                                     number between raters: the judge agrees as well as a human
                                     does if the two are close.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_ratings import CRITERIA, LABELS, fmt, short, weighted_kappa  # noqa: E402


def ranks(values):
    """Average ranks (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


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

    human = defaultdict(dict)  # (system, id, criterion) -> {rater: score}
    with open(d / "ratings.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["score"] != "":
                human[(r["system"], r["id"], r["criterion"])][r["rater"]] = int(r["score"])
    systems = sorted({k[0] for k in human})
    judge = load_judge(Path(args.judge_dir), systems)
    raters = sorted({r for v in human.values() for r in v})

    result = {"raters": raters, "systems": systems, "criteria": {}}
    for c in CRITERIA:
        keys = [k for k in human if k[2] == c and judge.get((k[0], k[1]), {}).get(c) is not None]
        j_pairs, rr_pairs = defaultdict(list), defaultdict(list)
        means_h, scores_j = [], []
        for s, i, _ in keys:
            j = judge[(s, i)][c]
            h = human[(s, i, c)]
            for r, v in h.items():
                j_pairs[r].append((j, v))
            for r1, r2 in itertools.combinations(sorted(h), 2):
                rr_pairs[(r1, r2)].append((h[r1], h[r2]))
            means_h.append(sum(h.values()) / len(h))
            scores_j.append(j)
        jk = [weighted_kappa([a for a, _ in v], [b for _, b in v]) for v in j_pairs.values()]
        jk = [k for k in jk if k is not None]
        rk = [weighted_kappa([a for a, _ in v], [b for _, b in v]) for v in rr_pairs.values()]
        rk = [k for k in rk if k is not None]
        flat = [pair for v in j_pairs.values() for pair in v]
        rho = pearson(ranks(scores_j), ranks(means_h)) if len(keys) > 2 else None
        by_system = {}
        for s in systems:
            idx = [n for n, k in enumerate(keys) if k[0] == s]
            by_system[s] = {"judge": sum(scores_j[n] for n in idx) / len(idx) if idx else None,
                            "human": sum(means_h[n] for n in idx) / len(idx) if idx else None}
        order = lambda who: sorted(systems, key=lambda s: -(by_system[s][who] or 0))
        result["criteria"][c] = {
            "n_answers": len(keys),
            "kappa_judge_vs_rater": sum(jk) / len(jk) if jk else None,
            "kappa_rater_vs_rater": sum(rk) / len(rk) if rk else None,
            "exact": sum(a == b for a, b in flat) / len(flat) if flat else None,
            "within_1": sum(abs(a - b) <= 1 for a, b in flat) / len(flat) if flat else None,
            "spearman_rho": rho,
            "mean_difference_judge_minus_human": (sum(scores_j) - sum(means_h)) / len(keys) if keys else None,
            "by_system": by_system,
            "same_system_order": order("judge") == order("human"),
        }

    with open(d / "judge_vs_human.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"
    lines = [f"# Judge vs human: {len(raters)} rater(s), {len(systems)} systems", "",
             "| Criterion | Answers | κ judge–rater | κ rater–rater | Exact | Within 1 | Spearman ρ | Judge − human | Same system order |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in CRITERIA:
        a = result["criteria"][c]
        lines.append(f"| {LABELS[c]} | {a['n_answers']} | {fmt(a['kappa_judge_vs_rater'])} | "
                     f"{fmt(a['kappa_rater_vs_rater'])} | {pct(a['exact'])} | {pct(a['within_1'])} | "
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
