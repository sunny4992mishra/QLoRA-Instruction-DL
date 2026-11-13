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
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts, {len(done)} already done, {len(todo)} to generate")
    if not todo:
        return

    tok = load_tokenizer(args.adapter or args.model)
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    model, quantized = load_base_model(args.model, quantize=not args.no_4bit)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Fresh greedy config; keep only the model's stop tokens (e.g. <|im_end|> for Qwen).
    # Replacing model.generation_config stops the model's sampling defaults being merged in.
    model.generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=model.generation_config.eos_token_id or tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i: i + args.batch_size]
            prompts = [build_prompt(tok, r["instruction"]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                f.write(json.dumps({
                    "id": r["id"],
                    "instruction": r["instruction"],
                    "reference": r.get("reference", r.get("response")),
                    "output": text.strip(),
                    "model": args.model,
                    "adapter": args.adapter,
                    "quantized_4bit": quantized,
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}")
    print(f"done in {time.time() - start:.0f}s -> {out_path}", gpu_memory_gb())


if __name__ == "__main__":
    main()
