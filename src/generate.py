"""Generate answers for the held-out test prompts, with or without a LoRA adapter.

Zero-shot baseline:
    python src/generate.py --out results/Qwen2.5-1.5B-Instruct_zeroshot.jsonl
Fine-tuned:
    python src/generate.py --adapter runs/Qwen2.5-1.5B-Instruct_r16_n775/adapter \
        --out results/Qwen2.5-1.5B-Instruct_r16_n775.jsonl

The base model is loaded in 4-bit, as in training. Decoding is greedy so runs are
reproducible. If --out already exists, prompts already in it are skipped, so a run
cut off by a Colab disconnect can be resumed with the same command.
"""

import os

# Use only the first GPU, as in train.py. This must run before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import GenerationConfig

from prompts import build_prompt
from utils import gpu_memory_gb, load_base_model, load_jsonl, load_tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default=None, help="Path to a LoRA adapter dir (omit for zero-shot)")
    p.add_argument("--input", default="data/splits/test.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0, help="Only the first N prompts (0 = all)")
    p.add_argument("--batch-size", type=int, default=8)
    # 512 new tokens is the "length limit" in the report: an answer that reaches it was cut off.
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume support: read the ids already in the output file and answer only the rest.
    # After a disconnect, running the same command continues where it stopped.
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        return

    # Load the tokenizer from the adapter folder when there is one (train.py saved a copy
    # there), otherwise from the base model.
    tok = load_tokenizer(args.adapter or args.model)
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    # Left padding puts every prompt's last token at the same position, so the new tokens
    # follow each prompt directly instead of coming after padding.
    model, quantized = load_base_model(args.model, quantize=not args.no_4bit)
    if args.adapter:
        # Put the trained LoRA adapter on top of the 4-bit base model. Without --adapter this
        # is the zero-shot baseline: the same base model with no fine-tuning.
        model = PeftModel.from_pretrained(model, args.adapter)
    # Evaluation mode turns dropout off, so the outputs don't change randomly.
    model.eval()

    # Fresh greedy config; keep only the model's stop tokens (e.g. <|im_end|> for Qwen).
    # Replacing model.generation_config stops the model's sampling defaults being merged in.
    model.generation_config = GenerationConfig(
        # Greedy decoding: always pick the most likely next token. No sampling, so the same
        # model and prompt always give the same answer.
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        # Values above 1 make tokens that already appeared less likely, to discourage
        # repetition. 1.1 is mild; many zero-shot answers still loop.
        repetition_penalty=args.repetition_penalty,
        # Stop when the model writes its end-of-turn token.
        eos_token_id=model.generation_config.eos_token_id or tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    start = time.time()
    # Open in append mode, so answers from an earlier, interrupted run are kept.
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i: i + args.batch_size]
            # The same prompt format as in training: system prompt + instruction + the opening
            # of the assistant turn (see prompts.py).
            prompts = [build_prompt(tok, r["instruction"]) for r in batch]
            # Tokenize the batch, padding to the longest prompt. add_special_tokens=False
            # because the chat template already added them.
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            # No gradients are needed for generation; this saves memory and time.
            with torch.no_grad():
                out = model.generate(**enc)
            # model.generate returns prompt + answer. With left padding, every prompt has the
            # same length in the batch, so slicing it off leaves only the new tokens.
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                # One line per test prompt. The judge, analyze.py and answer_stats.py read these.
                f.write(json.dumps({
                    "id": r["id"],
                    "instruction": r["instruction"],
                    "reference": r.get("reference", r.get("response")),
                    "output": text.strip(),
                    "model": args.model,
                    "adapter": args.adapter,
                    "quantized_4bit": quantized,
                }, ensure_ascii=False) + "\n")
            # Write to disk after every batch, so a disconnect loses at most one batch.
            f.flush()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}")
    print(f"done in {time.time() - start:.0f}s -> {out_path}", gpu_memory_gb())


if __name__ == "__main__":
    main()
