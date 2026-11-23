"""Make the blind human rating sheets: 30 test prompts x 3 systems = 90 answers.

    pip install openpyxl
    python src/make_rating_sheets.py

The same 30 prompts are sampled for every system (fixed seed), the 90 answers are shuffled
into one order, and each answer gets an item number. The sheets show only the item number,
the instruction, the reference answer and the answer, so nothing says which system wrote it.

Outputs (in --out-dir, default results/human/):
    rating_sheet_<rater>.xlsx   one per rater, same items in the same order
    key.json                    item number -> test prompt id and system

Don't open key.json until all four raters are done (#14). collect_ratings.py uses it to
match the filled sheets back to the systems.

The criteria are the ones the LLM judge uses (#9): correctness, helpfulness and Hindi
fluency, each 1-5. Like the judge, raters see the reference answer.

Re-running with the same arguments gives the same sample and order. It refuses to
overwrite existing sheets unless --force is given, so filled-in sheets aren't lost.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

TAG = "Qwen2.5-1.5B-Instruct"
# Fixed in advance (#12): base model, r=16 on all data, r=16 on 100 rows.
# Why these three: they cover the base model, the default setting and the smallest data size.
SYSTEMS = [f"{TAG}_zeroshot", f"{TAG}_r16_n775", f"{TAG}_r16_n100"]
RATERS = ["Pranjal", "Sunny", "Shahid", "Ompal"]
CRITERIA = {
    "correctness": ("Correctness",
                    "Are the facts and the content right? Compare with the reference, but the reference "
                    "is one good answer, not the only one.",
                    "1 = wrong or unrelated · 3 = partly right, with clear errors · 5 = fully correct"),
    "helpfulness": ("Helpfulness",
                    "Does it do what the instruction asks, completely and at a sensible length? "
                    "An answer that is cut off, loops, or ignores part of the instruction scores lower.",
                    "1 = ignores the instruction or is unusable · 3 = does part of it · 5 = does all of it well"),
    "fluency": ("Hindi fluency",
                "Is it natural, grammatical Hindi in Devanagari? Judge the language only, not the facts.",
                "1 = not Hindi or unreadable · 3 = understandable, with errors or mixed language · "
                "5 = natural Hindi"),
}
HEADERS = ["item", "instruction", "reference answer", "answer to rate",
           *[label for label, _, _ in CRITERIA.values()], "comment (optional)"]
WIDTHS = [6, 45, 55, 70, 12, 12, 12, 30]
SCORE_COLS = "EFG"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_items(results_dir, systems, test_ids, n_prompts, seed):
    answers = {}
    for name in systems:
        path = Path(results_dir) / f"{name}.jsonl"
        if not path.exists():
            sys.exit(f"missing {path}: that run's answers must be in results/ first")
        rows = {r["id"]: r for r in load_jsonl(path)}
        missing = [i for i in test_ids if i not in rows]
        if missing:
            sys.exit(f"{path} has no answer for {len(missing)} of the {len(test_ids)} test prompts")
        answers[name] = rows

    rng = random.Random(seed)
    # Why the same prompts for all systems: human scores can then be compared prompt by prompt.
    prompt_ids = rng.sample(test_ids, n_prompts)
    items = [(pid, name) for pid in prompt_ids for name in systems]
    # Why shuffle: an answer's position can't reveal which system wrote it.
    rng.shuffle(items)
    # Excel rejects control characters, which a model can emit; drop them from the sheet text.
    clean = lambda text: ILLEGAL_CHARACTERS_RE.sub("", text or "")
    return [{"item": k, "id": pid, "system": name,
             "instruction": clean(answers[name][pid]["instruction"]),
             "reference": clean(answers[name][pid]["reference"]),
             "output": clean(answers[name][pid]["output"])}
            for k, (pid, name) in enumerate(items, start=1)]


def write_sheet(path, rater, items):
    wb = Workbook()
    guide = wb.active
    guide.title = "Instructions"
    lines = [
        (f"Blind rating sheet for {rater}", Font(bold=True, size=14)),
        (f"Rate each of the {len(items)} answers on the 'Ratings' tab: one whole number from 1 to 5 "
         "per criterion. Rate each criterion on its own.", None),
        ("The answers come from different systems in random order. Don't try to guess which one wrote "
         "an answer, and don't compare ratings with teammates until all four are done.", None),
        ("", None),
    ]
    for label, question, scale in CRITERIA.values():
        lines += [(label, Font(bold=True)), (question, None), (scale, None), ("", None)]
    lines.append(("Some answers and references are long. If a cell is cut off, click it and read the full "
                  "text in the formula bar.", None))
    lines.append(("", None))
    lines.append(("Save the file under the same name when you're done.", None))
    for r, (text, font) in enumerate(lines, start=1):
        cell = guide.cell(row=r, column=1, value=text)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if font:
            cell.font = font
    guide.column_dimensions["A"].width = 110

    ws = wb.create_sheet("Ratings")
    ws.append(HEADERS)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for it in items:
        ws.append([it["item"], it["instruction"], it["reference"], it["output"], None, None, None, None])
        # openpyxl stores a string that starts with "=" as a formula, and Excel then shows an
        # error instead of the text. One reference starts with "== ... ==". Force plain text.
        for c in ws[ws.max_row][1:4]:
            c.data_type = "s"
    last = len(items) + 1
    for i, w in enumerate(WIDTHS):
        ws.column_dimensions[chr(ord("A") + i)].width = w
    for row in ws.iter_rows(min_row=2, max_row=last):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    score_fill = PatternFill("solid", fgColor="FFF2CC")
    for col in SCORE_COLS:
        for r in range(2, last + 1):
            ws[f"{col}{r}"].fill = score_fill
            ws[f"{col}{r}"].alignment = Alignment(horizontal="center", vertical="top")
    # Why allow blanks: a criterion can be not applicable, and collect_ratings.py counts a blank as
    # missing.
    dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                        showErrorMessage=True, errorTitle="Score 1-5", error="Enter a whole number from 1 to 5.")
    dv.add(f"{SCORE_COLS[0]}2:{SCORE_COLS[-1]}{last}")
    ws.add_data_validation(dv)
    ws.freeze_panes = "B2"
    wb.active = wb.index(guide)
    wb.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--test-file", default="data/splits/test.jsonl")
    p.add_argument("--systems", nargs="+", default=SYSTEMS, help="Run names; their answers are in results/")
    p.add_argument("--raters", nargs="+", default=RATERS)
    p.add_argument("--n-prompts", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/human")
    p.add_argument("--force", action="store_true", help="Overwrite existing sheets and key")
    args = p.parse_args()

    out = Path(args.out_dir)
    targets = [out / "key.json"] + [out / f"rating_sheet_{r}.xlsx" for r in args.raters]
    existing = [str(t) for t in targets if t.exists()]
    if existing and not args.force:
        sys.exit(f"already exists, not overwriting (use --force): {', '.join(existing)}")

    test_ids = [r["id"] for r in load_jsonl(args.test_file)]
    items = build_items(args.results_dir, args.systems, test_ids, args.n_prompts, args.seed)

    out.mkdir(parents=True, exist_ok=True)
    key = {"seed": args.seed, "systems": args.systems, "raters": args.raters,
           "criteria": list(CRITERIA),
           "items": [{"item": it["item"], "id": it["id"], "system": it["system"]} for it in items]}
    with open(out / "key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    for rater in args.raters:
        write_sheet(out / f"rating_sheet_{rater}.xlsx", rater, items)

    print(f"{len(items)} items ({args.n_prompts} prompts x {len(args.systems)} systems), seed {args.seed}")
    print(f"wrote {out / 'key.json'} and {len(args.raters)} sheets: {', '.join(args.raters)}")


if __name__ == "__main__":
    main()
