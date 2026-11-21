"""Error analysis of the generated answers (issue #12).

    pip install tokenizers huggingface_hub     # or transformers; only the tokenizer is used
    python src/error_analysis.py

Reads results/<run>.jsonl and results/judge/<run>.jsonl for every run, and gives each answer
one category: the first rule below that matches.

    not_hindi      under half of the answer's letters are Devanagari and it has 3+ Latin-script
                   words, although the reference is in Hindi (so "translate into English"
                   prompts and one-word answers such as "NPTEL" or an emoji don't count)
    loops          reaches the 512-token generation limit and repeats an 8-word phrase 3+ times
    ok             the judge gives correctness >= 3
    cut_off        reaches the limit without such a loop (still talking when generation stopped)
    wrong_fluent   the judge gives correctness <= 2 but fluency >= 4: fluent, but wrong
    copies_prompt  correctness <= 2, half or more of the answer's word 3-grams come from the
                   instruction, and the answer repeats 60%+ of the instruction's 3-grams
    incoherent     correctness <= 2 and fluency <= 2: Hindi that makes little sense
    weak           correctness <= 2 otherwise: vague or only partly right

not_hindi, loops and the cut_off test use only the text. A loop counts as an error even when
the judge accepts the start of the answer; a cut-off answer the judge accepts counts as ok.
A wrong answer that restates the question ("In 2001, X took charge as ...") is wrong_fluent,
not copies_prompt, because the rule for fluent answers comes first. An answer "reaches
the limit" if re-tokenizing it gives at least 510 tokens: no answer stops between 505 and 509,
and the ones below 505 all end on a complete sentence. A cut-off answer is not always a
mistake, because some references are longer than 512 tokens too; the summary counts those.

Outputs:
    results/error_analysis/answers.csv    one row per answer: run, id, category, flags, scores
    results/error_analysis/summary.md     category counts per run, flag counts per run
    results/error_analysis/examples.md    short examples of each category

Examples come only from test prompts that are NOT in the human-rating sample
(results/human/key.json), so reading them doesn't unblind any rater.
"""

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 512
LIMIT_TOKENS = 510            # re-tokenized length that counts as reaching the limit
LOOP_N, LOOP_K = 8, 3         # an 8-word phrase repeated 3 or more times
COPY_SHARE = 0.5              # share of answer 3-grams found in the instruction
COVER_SHARE = 0.6             # share of instruction 3-grams repeated in the answer
CATEGORIES = ["not_hindi", "loops", "cut_off", "wrong_fluent", "copies_prompt",
              "incoherent", "weak", "ok"]
LABELS = {
    "not_hindi": "Not in Hindi",
    "loops": "Doesn't stop: loops",
    "cut_off": "Doesn't stop: cut off",
    "copies_prompt": "Copies the instruction",
    "wrong_fluent": "Fluent but wrong",
    "incoherent": "Incoherent Hindi",
    "weak": "Vague or partly right",
    "ok": "Correct enough (judge >= 3)",
}

EXAMPLE_RUNS = [f"Qwen2.5-1.5B-Instruct_{s}" for s in ("r16_n775", "zeroshot", "r16_n100")]

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[^\W\d_]")
LATIN_WORD = re.compile(r"[A-Za-z]{3,}")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tokenizer():
    """Token counter for the base model: the tokenizers library if installed, else transformers."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
        return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    except ImportError:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def devanagari_share(text):
    letters = LETTER.findall(text)
    if not letters:
        return 0.0
    return sum(1 for ch in letters if DEVANAGARI.match(ch)) / len(letters)


def ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def max_repeat(text, n=LOOP_N):
    counts = collections.Counter(ngrams(text.split(), n))
    return max(counts.values()) if counts else 0


def share_in(text, source):
    """Share of the word 3-grams of text that also occur in source."""
    grams = ngrams(text.split(), 3)
    if not grams:
        return 0.0
    known = set(ngrams(source.split(), 3))
    return sum(1 for g in grams if g in known) / len(grams)


def classify(a):
    if (a["devanagari_share"] < 0.5 and a["latin_words"] >= 3
            and a["ref_devanagari_share"] >= 0.5):
        return "not_hindi"
    if a["hits_limit"] and a["max_repeat"] >= LOOP_K:
        return "loops"
    c, f = a["correctness"], a["fluency"]
    if c is not None and c >= 3:
        return "ok"
    if a["hits_limit"]:
        return "cut_off"
    if c is None:      # judge failed on this answer
        return "weak"
    if f is not None and f >= 4:
        return "wrong_fluent"
    if a["copy_share"] >= COPY_SHARE and a["instruction_covered"] >= COVER_SHARE:
        return "copies_prompt"
    if f is not None and f <= 2:
        return "incoherent"
    return "weak"


def run_names(results_dir):
    return sorted(p.stem for p in results_dir.glob("*.jsonl"))


def sort_key(name):
    m = re.search(r"_r(\d+)_n(\d+)$", name)
    return (0, 0, 0) if not m else (1, int(m.group(2)) != 775, int(m.group(1)) + int(m.group(2)))


def short(name):
    return name.split("_", 1)[1] if "_" in name else name


def analyze(results_dir, count_tokens):
    answers = []
    ref_tokens = {}
    for run in sorted(run_names(results_dir), key=sort_key):
        rows = load_jsonl(results_dir / f"{run}.jsonl")
        judge_path = results_dir / "judge" / f"{run}.jsonl"
        judge = {j["id"]: j for j in load_jsonl(judge_path)} if judge_path.exists() else {}
        if not judge:
            print(f"warning: no judge scores for {run}; judge-based categories become 'weak'",
                  file=sys.stderr)
        for r in rows:
            out = r.get("output") or ""
            if r["id"] not in ref_tokens:
                ref_tokens[r["id"]] = count_tokens(r.get("reference") or "")
            j = judge.get(r["id"], {})
            a = {
                "run": run, "id": r["id"],
                "tokens": count_tokens(out),
                "ref_tokens": ref_tokens[r["id"]],
                "devanagari_share": round(devanagari_share(out), 3),
                "ref_devanagari_share": round(devanagari_share(r.get("reference") or ""), 3),
                "latin_words": len(LATIN_WORD.findall(out)),
                "max_repeat": max_repeat(out),
                "copy_share": round(share_in(out, r.get("instruction") or ""), 3),
                "instruction_covered": round(share_in(r.get("instruction") or "", out), 3),
                "correctness": j.get("correctness"),
                "helpfulness": j.get("helpfulness"),
                "fluency": j.get("fluency"),
            }
            a["hits_limit"] = a["tokens"] >= LIMIT_TOKENS
            a["category"] = classify(a)
            a["_instruction"], a["_output"] = r.get("instruction") or "", out
            answers.append(a)
    return answers


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|---|" + "---:|" * (len(header) - 1)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_outputs(answers, out_dir, rated_ids):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run", "id", "category", "tokens", "ref_tokens", "hits_limit", "max_repeat",
              "copy_share", "instruction_covered", "devanagari_share", "correctness", "helpfulness", "fluency"]
    with open(out_dir / "answers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(answers)

    runs = list(dict.fromkeys(a["run"] for a in answers))
    by_run = {r: [a for a in answers if a["run"] == r] for r in runs}

    cat_rows = []
    for c in CATEGORIES:
        cat_rows.append([LABELS[c]] + [sum(a["category"] == c for a in by_run[r]) for r in runs])
    cat_rows.append(["Answers"] + [len(by_run[r]) for r in runs])

    def flag_row(label, pred):
        return [label] + [sum(1 for a in by_run[r] if pred(a)) for r in runs]

    flag_rows = [
        flag_row("Reaches the 512-token limit", lambda a: a["hits_limit"]),
        flag_row("... and the reference is also over 512 tokens",
                 lambda a: a["hits_limit"] and a["ref_tokens"] >= MAX_NEW_TOKENS),
        flag_row(f"Repeats an {LOOP_N}-word phrase {LOOP_K}+ times",
                 lambda a: a["max_repeat"] >= LOOP_K),
        flag_row("Copies the instruction (both 3-gram shares)",
                 lambda a: a["copy_share"] >= COPY_SHARE
                 and a["instruction_covered"] >= COVER_SHARE),
        flag_row("Under half of the letters Devanagari, 3+ Latin words",
                 lambda a: a["devanagari_share"] < 0.5 and a["latin_words"] >= 3),
        flag_row("Judge correctness <= 2",
                 lambda a: a["correctness"] is not None and a["correctness"] <= 2),
    ]

    header = ["Category"] + [short(r) for r in runs]
    md = [
        "# Error analysis (issue #12)",
        "",
        "Written by `src/error_analysis.py`. Each answer gets one category; the rules and "
        "their order are in the script's docstring. Not in Hindi, loops and cut off use only "
        "the answer text; the other categories use the LLM judge's scores.",
        "",
        "## Answers per category (out of 150 per run)",
        "",
        table(header, cat_rows),
        "",
        "## Flags (an answer can have several)",
        "",
        table(["Flag"] + [short(r) for r in runs], flag_rows),
        "",
        f"References over {MAX_NEW_TOKENS} tokens: "
        f"{sum(1 for a in by_run[runs[0]] if a['ref_tokens'] >= MAX_NEW_TOKENS)} of "
        f"{len(by_run[runs[0]])} test prompts.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    ex = ["# Examples per category", "",
          "Up to three answers per category, from test prompts outside the human-rating "
          "sample. Text is cut to 300 characters.", ""]
    for c in CATEGORIES[:-1]:
        ex += [f"## {LABELS[c]}", ""]
        picked = [a for a in answers if a["category"] == c and a["id"] not in rated_ids]
        # one answer per run where possible, starting with the main fine-tuned run
        picked.sort(key=lambda a: (a["run"] not in EXAMPLE_RUNS,
                                   EXAMPLE_RUNS.index(a["run"]) if a["run"] in EXAMPLE_RUNS else 0))
        seen, used_runs = set(), set()
        for a in picked + picked:
            if a["id"] in seen or (a["run"] in used_runs and len(used_runs) < 3
                                   and any(b["run"] not in used_runs for b in picked)):
                continue
            seen.add(a["id"])
            used_runs.add(a["run"])
            ex += [f"**{short(a['run'])} · {a['id']}** · judge c/h/f = "
                   f"{a['correctness']}/{a['helpfulness']}/{a['fluency']} · {a['tokens']} tokens",
                   "", "> Instruction: " + a["_instruction"][:300].replace("\n", " "), ">",
                   "> Answer: " + a["_output"][:300].replace("\n", " "), ""]
            if len(seen) == 3:
                break
    (out_dir / "examples.md").write_text("\n".join(ex), encoding="utf-8")
    return header, cat_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "error_analysis"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    results_dir = Path(args.results)
    key = results_dir / "human" / "key.json"
    rated = set()
    if key.exists():
        rated = {i["id"] for i in json.loads(key.read_text(encoding="utf-8"))["items"]}
    answers = analyze(results_dir, load_tokenizer())
    header, rows = write_outputs(answers, Path(args.out), rated)
    print(table(header, rows))
    print(f"\nwrote {args.out}/answers.csv, summary.md, examples.md")


if __name__ == "__main__":
    main()
