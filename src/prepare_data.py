"""Build the fixed Hindi train/val/test split from the Aya Dataset.

The Aya test split has no Hindi rows, so we hold out our own test set.
Run once; the output files are committed so every team member uses the same split.

    python src/prepare_data.py

Cleaning steps (counts are written to data/split_stats.json):
  1. Normalise newlines, strip whitespace, remove one pair of quotes wrapping the whole text.
  2. Drop rows with an empty prompt or response.
  3. Drop prompts with no Devanagari text. In this subset these are English prompts whose
     "response" is just a Hindi translation of the prompt, which would teach the model to
     translate instructions instead of following them.
  4. Drop exact duplicate (prompt, response) pairs.

Splitting is done by prompt: rows that share a prompt (original + re-annotations) always
land in the same split, so no test prompt is ever seen in training.

train.jsonl is written in a fixed random order. Taking the first N rows gives nested subsets
(100 ⊂ 250 ⊂ 500 ...) for the data-size experiments.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path

from datasets import load_dataset

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
QUOTE_PAIRS = [('"', '"'), ("“", "”"), ("'", "'")]


def clean(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    for left, right in QUOTE_PAIRS:
        inner = text[1:-1]
        if (
            len(text) >= 2
            and text.startswith(left)
            and text.endswith(right)
            and left not in inner
            and right not in inner
        ):
            text = inner.strip()
            break
    return text


def prompt_key(text):
    return " ".join(text.split())


def short_hash(*parts):
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:10]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="CohereLabs/aya_dataset")
    p.add_argument("--language-code", default="hin")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--test-prompts", type=int, default=150)
    p.add_argument("--val-prompts", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-non-hindi-prompts", action="store_true",
                   help="Keep prompts that contain no Devanagari (dropped by default, see step 3).")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split="train")
    ds = ds.filter(lambda r: r["language_code"] == args.language_code)
    stats = Counter(raw_rows=len(ds))

    rows, seen_pairs = [], set()
    for r in ds:
        instruction, response = clean(r["inputs"]), clean(r["targets"])
        if not instruction or not response:
            stats["dropped_empty"] += 1
            continue
        if not args.keep_non_hindi_prompts and not DEVANAGARI.search(instruction):
            stats["dropped_non_hindi_prompt"] += 1
            continue
        pair = (prompt_key(instruction), response)
        if pair in seen_pairs:
            stats["dropped_duplicate_pair"] += 1
            continue
        seen_pairs.add(pair)
        rows.append({
            "id": "aya-hi-" + short_hash(*pair),
            "instruction": instruction,
            "response": response,
            "annotation_type": r["annotation_type"],
        })
    stats["clean_rows"] = len(rows)

    # Group rows by prompt; original annotations first so they become the main test reference.
    groups = OrderedDict()
    for row in sorted(rows, key=lambda r: r["annotation_type"] != "original-annotations"):
        groups.setdefault(prompt_key(row["instruction"]), []).append(row)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    stats["unique_prompts"] = len(keys)

    n_test, n_val = args.test_prompts, args.val_prompts
    if n_test + n_val >= len(keys):
        raise SystemExit(f"Only {len(keys)} prompts; test+val sizes are too large.")
    test_keys, val_keys, train_keys = keys[:n_test], keys[n_test:n_test + n_val], keys[n_test + n_val:]

    test = []
    for k in test_keys:
        g = groups[k]
        test.append({
            "id": "aya-hi-test-" + short_hash(k),
            "instruction": g[0]["instruction"],
            "reference": g[0]["response"],
            "references": [r["response"] for r in g],
        })
    val = [r for k in val_keys for r in groups[k]]
    train = [r for k in train_keys for r in groups[k]]
    random.Random(args.seed).shuffle(train)  # fixed order -> nested data-size subsets

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    stats.update({
        "train_rows": len(train), "train_prompts": len(train_keys),
        "val_rows": len(val), "val_prompts": len(val_keys),
        "test_prompts": len(test),
    })
    summary = {"dataset": args.dataset, "language_code": args.language_code,
               "seed": args.seed, **dict(stats)}
    with open(out.parent / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
