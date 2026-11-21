"""Judge-free statistics on the generated answers: length, cut-offs and looping.

Needs only the tokenizer (no model weights, no GPU, no torch):
    pip install transformers==5.22.2 jinja2
    python src/answer_stats.py

For every results/<run>.jsonl it reports, over the 150 answers:
  median_tokens     median answer length in tokens (statistics.median)
  at_limit          answers that ran to the generation limit of 512 new tokens
  looping           answers that repeat the same 8-word phrase 3 or more times
  mostly_devanagari answers in which more than half of the letters are Devanagari

"At the limit" is counted as 510 tokens or more. generate.py stores the decoded, stripped
text, and tokenizing that text again can give one or two tokens fewer than were generated.
The same statistics are computed for the reference answers, for comparison.

Output: results/answer_stats.json, and a Markdown table on stdout.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def load_jsonl(path):
    # Same as utils.load_jsonl; utils imports torch, which this script doesn't need.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def loops(text, words=8, times=3):
    w = text.split()
    counts = collections.Counter(tuple(w[i:i + words]) for i in range(len(w) - words + 1))
    return bool(counts) and max(counts.values()) >= times


def devanagari_share(text):
    letters = [c for c in text if c.isalpha() or "ऀ" <= c <= "ॿ"]
    return sum("ऀ" <= c <= "ॿ" for c in letters) / max(1, len(letters))


def stats(texts, tok, limit):
    n_tokens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    return {
        "answers": len(texts),
        "median_tokens": statistics.median(n_tokens),
        "at_limit": sum(n >= limit - 2 for n in n_tokens),
        "looping": sum(loops(t) for t in texts),
        "mostly_devanagari": sum(devanagari_share(t) > 0.5 for t in texts),
    }


def sort_key(name):
    # zero-shot first, then by rank, then by training rows
    if name.endswith("_zeroshot"):
        return (0, 0, 0)
    rank, rows = name.rsplit("_r", 1)[1].split("_n")
    return (1, int(rank), int(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Tokenizer used for the token counts")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--test-file", default="data/splits/test.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=512, help="The limit used in generate.py")
    p.add_argument("--out", default="results/answer_stats.json")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    out = {"tokenizer": args.model, "max_new_tokens": args.max_new_tokens, "runs": {}}
    out["reference"] = stats([r["reference"] for r in load_jsonl(args.test_file)], tok, args.max_new_tokens)
    for path in sorted(Path(args.results_dir).glob("*.jsonl"), key=lambda q: sort_key(q.stem)):
        out["runs"][path.stem] = stats([r["output"] for r in load_jsonl(path)], tok, args.max_new_tokens)

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("| Run | Median tokens | At the limit | Looping | Mostly Devanagari |\n|---|---:|---:|---:|---:|")
    for name, s in [("reference answers", out["reference"])] + list(out["runs"].items()):
        print(f"| {name} | {s['median_tokens']:g} | {s['at_limit']} | {s['looping']} | {s['mostly_devanagari']} |")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
