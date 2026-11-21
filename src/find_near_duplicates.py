"""List test and validation prompts that are near-duplicates of training rows.

prepare_data.py splits by exact prompt (after whitespace normalisation), so no test or
validation prompt appears in train word for word. Re-annotations in the Aya Dataset often
rephrase the prompt slightly ("इस प्रश्न का उत्तर दें: ..." vs "प्रश्न का उत्तर दें: ..."),
so a near-copy of a held-out example can still be in the training split. This script finds
those cases. It does not change the split.

A held-out prompt is flagged when either holds:
  prompt     its prompt has character 4-gram Jaccard similarity >= --prompt-threshold (0.8)
             with some training prompt (both whitespace-normalised)
  reference  one of its answers has difflib similarity >= --reference-threshold (0.9) with
             some training response: any of the test prompt's references, or the response of
             any validation row with that prompt

Needs only the standard library:
    python src/find_near_duplicates.py

Output: data/test_near_duplicates.json with the flagged test ids, the counts and, for each
flagged test prompt, its closest training row; plus the same counts for the validation split.
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


def closest(prompt, answers, train, train_grams, args):
    """(prompt similarity, its train index, answer similarity, its train index) for one prompt."""
    g = grams(norm(prompt))
    p_sim, p_idx = max((jaccard(g, tg), i) for i, tg in enumerate(train_grams))
    r_sim, r_idx = max((text_similarity(a, r["response"], args.reference_threshold), i)
                       for a in answers for i, r in enumerate(train))
    return p_sim, p_idx, r_sim, r_idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-file", default="data/splits/train.jsonl")
    p.add_argument("--val-file", default="data/splits/val.jsonl")
    p.add_argument("--test-file", default="data/splits/test.jsonl")
    p.add_argument("--prompt-threshold", type=float, default=0.8)
    p.add_argument("--reference-threshold", type=float, default=0.9)
    p.add_argument("--out", default="data/test_near_duplicates.json")
    args = p.parse_args()

    train = load_jsonl(args.train_file)
    test = load_jsonl(args.test_file)
    train_grams = [grams(norm(r["instruction"])) for r in train]

    def reasons_for(p_sim, r_sim):
        return [name for name, hit in (("prompt", p_sim >= args.prompt_threshold),
                                       ("reference", r_sim >= args.reference_threshold)) if hit]

    flagged = []
    for t in test:
        answers = t.get("references") or [t["reference"]]
        p_sim, p_idx, r_sim, r_idx = closest(t["instruction"], answers, train, train_grams, args)
        reasons = reasons_for(p_sim, r_sim)
        if reasons:
            match = train[r_idx if "reference" in reasons else p_idx]
            # Earliest training row (in the stored order) that triggers either criterion. Subsets
            # for the data-size sweep are the first N rows, so this says from which N on a
            # near-copy of this test prompt is in the training data.
            g = grams(norm(t["instruction"]))
            first = next(i for i, r in enumerate(train)
                         if jaccard(g, train_grams[i]) >= args.prompt_threshold
                         or any(text_similarity(a, r["response"], args.reference_threshold)
                                >= args.reference_threshold for a in answers))
            flagged.append({"id": t["id"], "reasons": reasons,
                            "prompt_similarity": round(p_sim, 3), "reference_similarity": round(r_sim, 3),
                            "test_instruction": t["instruction"], "train_id": match["id"],
                            "train_instruction": match["instruction"], "first_train_row": first})

    # Validation: one entry per distinct prompt, with the responses of all its rows.
    val_groups = {}
    for r in load_jsonl(args.val_file):
        val_groups.setdefault(norm(r["instruction"]), []).append(r)
    val_flagged = {}
    for prompt, rows in val_groups.items():
        p_sim, _, r_sim, _ = closest(prompt, [r["response"] for r in rows], train, train_grams, args)
        reasons = reasons_for(p_sim, r_sim)
        if reasons:
            val_flagged[prompt] = {"ids": [r["id"] for r in rows], "reasons": reasons,
                                   "prompt_similarity": round(p_sim, 3), "reference_similarity": round(r_sim, 3)}

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
        "flagged_with_copy_in_first_n_rows": {str(n): sum(f["first_train_row"] < n for f in flagged)
                                              for n in (100, 250, 500, len(train))},
        "pairs": flagged,
        "validation": {
            "prompts": len(val_groups),
            "rows": sum(len(v) for v in val_groups.values()),
            "flagged_by_prompt": sum("prompt" in v["reasons"] for v in val_flagged.values()),
            "flagged_by_reference": sum("reference" in v["reasons"] for v in val_flagged.values()),
            "flagged_prompts": len(val_flagged),
            "flagged_rows": sum(len(v["ids"]) for v in val_flagged.values()),
            "row_ids": [i for v in val_flagged.values() for i in v["ids"]],
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    v = out["validation"]
    print(f"{len(test)} test prompts: {len(by_prompt)} flagged by prompt, {len(by_reference)} by reference, "
          f"{out['flagged_by_both']} by both, {len(flagged)} by either")
    print(f"{v['prompts']} validation prompts: {v['flagged_prompts']} flagged "
          f"({v['flagged_rows']} of {v['rows']} rows) -> {args.out}")


if __name__ == "__main__":
    main()
