"""Tokenizer statistics for the Hindi split: characters per token, sequence lengths, truncation.

Needs only the tokenizers (no model weights, no GPU, no torch):
    pip install transformers==5.22.2 jinja2 matplotlib
    python src/tokenizer_stats.py

Sequence lengths, truncation and drops follow the same rules as tokenize() in train.py:
prompt and response are built with prompts.build_example and tokenized separately; a row is
dropped if the prompt alone has >= max_len tokens, and truncated if prompt + response > max_len.
For each tokenizer, train_tokens and val_tokens use the same keys and definitions as the
"train_tokens" and "val_tokens" fields of run_info.json, so the two can be compared directly.

"Characters" are Unicode code points (Python len()), so a Devanagari vowel sign counts as one
character. Characters per token is measured on the raw instruction and response text, with no
chat template or system prompt.

Outputs:
    data/tokenizer_stats.json      numbers for each tokenizer
    figures/token_lengths.png      histogram of training-example lengths
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from prompts import build_example


def load_jsonl(path):
    # Same as utils.load_jsonl; utils imports torch, which this script doesn't need.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def length_stats(rows, tok, max_len):
    """Same keys and rules as tokenize() in train.py, plus response tokens lost to the limit."""
    lengths, n_truncated, n_dropped = [], 0, 0
    response_tokens, lost_per_row = 0, []
    for r in rows:
        prompt, response = build_example(tok, r["instruction"], r["response"])
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        n_response = len(tok(response, add_special_tokens=False)["input_ids"])
        lengths.append(n_prompt + n_response)
        response_tokens += n_response
        if n_prompt >= max_len:
            n_dropped += 1
            lost_per_row.append(n_response)
        elif n_prompt + n_response > max_len:
            n_truncated += 1
            lost_per_row.append(n_prompt + n_response - max_len)
    sorted_lengths = sorted(lengths)
    pct = lambda q: sorted_lengths[min(len(sorted_lengths) - 1, int(q * len(sorted_lengths)))]
    lost = sum(lost_per_row)
    top25 = sum(sorted(lost_per_row, reverse=True)[:25])
    info = {"rows": len(rows), "kept": len(rows) - n_dropped, "truncated": n_truncated,
            "dropped_prompt_too_long": n_dropped,
            "token_len_p50": pct(0.5), "token_len_p95": pct(0.95), "token_len_max": sorted_lengths[-1],
            "response_tokens": response_tokens, "response_tokens_lost": lost,
            "response_tokens_lost_pct": round(100 * lost / response_tokens, 2),
            # How concentrated the loss is: share of the lost tokens that come from the 25 rows losing most.
            "top25_rows_share_of_lost_pct": round(100 * top25 / lost, 1) if lost else None}
    return info, lengths


def chars_per_token(rows, tok):
    """Corpus-level ratio: total characters / total tokens over instructions and responses."""
    texts = [r["instruction"] for r in rows] + [r["response"] for r in rows]
    n_chars = sum(len(t) for t in texts)
    n_tokens = sum(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    return {"chars": n_chars, "tokens": n_tokens, "chars_per_token": round(n_chars / n_tokens, 2)}


def plot(lengths_by_tok, max_len, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A few rows are 10-20k tokens long. Cap the axis at 3x max_len and put longer rows in the last bin.
    cap, width = 3 * max_len, 64
    bins = list(range(0, cap + width + 1, width))
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, lengths in lengths_by_tok.items():
        over = sum(n > max_len for n in lengths)
        ax.hist([min(n, cap + width // 2) for n in lengths], bins=bins, histtype="step", linewidth=1.5,
                label=f"{name} ({over} of {len(lengths)} rows > {max_len})")
    ax.axvline(max_len, color="black", linestyle="--", linewidth=1, label=f"max_seq_len = {max_len}")
    ax.set_xlim(0, cap + width)
    ticks = list(range(0, cap, max_len))  # 0, 1024, 2048; the last bin gets its own ">3072" label
    ax.set_xticks(ticks + [cap + width // 2], [str(t) for t in ticks] + [f">{cap}"])
    ax.set_xlabel("Tokens per training example (system prompt + instruction + response)")
    ax.set_ylabel("Training rows")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizers", nargs="+", default=["Qwen/Qwen2.5-1.5B-Instruct", "sarvamai/sarvam-1"])
    p.add_argument("--train-file", default="data/splits/train.jsonl")
    p.add_argument("--val-file", default="data/splits/val.jsonl")
    p.add_argument("--max-seq-len", type=int, default=1024, help="Same default as train.py")
    p.add_argument("--out", default="data/tokenizer_stats.json")
    p.add_argument("--plot", default="figures/token_lengths.png")
    args = p.parse_args()

    train_rows, val_rows = load_jsonl(args.train_file), load_jsonl(args.val_file)
    results, lengths_by_tok = {}, {}
    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(name)
        train_info, train_lengths = length_stats(train_rows, tok, args.max_seq_len)
        val_info, _ = length_stats(val_rows, tok, args.max_seq_len)
        results[name] = {"vocab_size": len(tok), "train_text": chars_per_token(train_rows, tok),
                         "train_tokens": train_info, "val_tokens": val_info}
        lengths_by_tok[name.split("/")[-1]] = train_lengths
        print(name, json.dumps(results[name], ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"max_seq_len": args.max_seq_len, "tokenizers": results}, f, indent=2)
        f.write("\n")
    plot(lengths_by_tok, args.max_seq_len, Path(args.plot))
    print(f"Wrote {out} and {args.plot}")


if __name__ == "__main__":
    main()
