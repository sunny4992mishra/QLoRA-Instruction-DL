"""List test prompts that are near-duplicates of training rows.

prepare_data.py splits by exact prompt (after whitespace normalisation), so no test prompt
appears in train word for word. Re-annotations in the Aya Dataset often rephrase the prompt
slightly ("इस प्रश्न का उत्तर दें: ..." vs "प्रश्न का उत्तर दें: ..."), so a near-copy of a
test example can still be in the training split. This script finds those cases. It does not
change the split.

A test prompt is flagged when either holds:
  prompt     its prompt has character 4-gram Jaccard similarity >= --prompt-threshold (0.8)
             with some training prompt (both whitespace-normalised)
  reference  its reference answer has difflib similarity >= --reference-threshold (0.9)
             with some training response

Needs only the standard library:
    python src/find_near_duplicates.py

Output: data/test_near_duplicates.json with the flagged ids, the counts and, for each
flagged prompt, its closest training row.
"""

import argparse
import difflib
import json
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text):
    return " ".join(text.split())


def grams(text, n=4):
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def text_similarity(a, b, threshold):
    """difflib ratio, or 0 when the cheap upper bounds already rule out the threshold."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
        return 0.0
    return sm.ratio()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", default="data/splits/train.jsonl")
    p.add_argument("--test-file", default="data/splits/test.jsonl")
    p.add_argument("--prompt-threshold", type=float, default=0.8)
    p.add_argument("--reference-threshold", type=float, default=0.9)
    p.add_argument("--out", default="data/test_near_duplicates.json")
    args = p.parse_args()

    train = load_jsonl(args.train_file)
    test = load_jsonl(args.test_file)
    train_grams = [grams(norm(r["instruction"])) for r in train]

    flagged = []
    for t in test:
        g = grams(norm(t["instruction"]))
        p_sim, p_idx = max((jaccard(g, tg), i) for i, tg in enumerate(train_grams))
        r_sim, r_idx = max((text_similarity(t["reference"], r["response"], args.reference_threshold), i)
                           for i, r in enumerate(train))
        reasons = [name for name, hit in (("prompt", p_sim >= args.prompt_threshold),
                                          ("reference", r_sim >= args.reference_threshold)) if hit]
        if reasons:
            match = train[r_idx if "reference" in reasons else p_idx]
            flagged.append({"id": t["id"], "reasons": reasons,
                            "prompt_similarity": round(p_sim, 3), "reference_similarity": round(r_sim, 3),
                            "test_instruction": t["instruction"], "train_id": match["id"],
                            "train_instruction": match["instruction"]})

    by_prompt = [f["id"] for f in flagged if "prompt" in f["reasons"]]
    by_reference = [f["id"] for f in flagged if "reference" in f["reasons"]]
    out = {
        "prompt_threshold": args.prompt_threshold,
        "reference_threshold": args.reference_threshold,
        "test_prompts": len(test),
        "flagged_by_prompt": len(by_prompt),
        "flagged_by_reference": len(by_reference),
        "flagged_by_both": len(set(by_prompt) & set(by_reference)),
        "flagged_by_either": len(flagged),
        "ids": [f["id"] for f in flagged],
        "ids_by_reference": by_reference,
        "pairs": flagged,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(test)} test prompts: {len(by_prompt)} flagged by prompt, {len(by_reference)} by reference, "
          f"{out['flagged_by_both']} by both, {len(flagged)} by either -> {args.out}")


if __name__ == "__main__":
    main()
