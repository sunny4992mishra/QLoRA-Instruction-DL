"""QLoRA fine-tuning on the Hindi Aya split.

Example (free Colab T4):
    python src/train.py --rank 16 --train-size 0          # all training rows
    python src/train.py --rank 8  --train-size 250

Outputs in --output-dir (default runs/<model>_r<rank>_n<size>):
    adapter/         final LoRA adapter + tokenizer (load with generate.py --adapter)
    checkpoint-*/    adapter after each epoch, for comparing epochs
    run_info.json    settings, trainable params, peak GPU memory, time, losses
"""

import os

# Use a single GPU (Kaggle gives 2x T4). Keeps memory numbers comparable across runs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

from prompts import build_example
from utils import compute_dtype, gpu_memory_gb, load_base_model, load_jsonl, load_tokenizer


def tokenize(rows, tok, max_len):
    """Tokenize (prompt, response) pairs; loss only on response tokens."""
    examples, n_truncated, n_dropped, lengths = [], 0, 0, []
    for r in rows:
        prompt, response = build_example(tok, r["instruction"], r["response"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(response, add_special_tokens=False)["input_ids"]
        lengths.append(len(p_ids) + len(r_ids))
        if len(p_ids) >= max_len:
            n_dropped += 1
            continue
        ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids
        if len(ids) > max_len:
            n_truncated += 1
            ids, labels = ids[:max_len], labels[:max_len]
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    lengths.sort()
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    info = {"rows": len(rows), "kept": len(examples), "truncated": n_truncated,
            "dropped_prompt_too_long": n_dropped,
            "token_len_p50": pct(0.5), "token_len_p95": pct(0.95), "token_len_max": lengths[-1]}
    return Dataset.from_list(examples), info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", default="data/splits/train.jsonl")
    p.add_argument("--val-file", default="data/splits/val.jsonl")
    p.add_argument("--train-size", type=int, default=0, help="First N training rows (0 = all). Subsets are nested.")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default 2 * rank)")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    # Effective batch = 2 * 8 = 16. Small per-step batch keeps 1.5B-3B models within a 16 GB T4.
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs (for quick tests)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true", help="Plain LoRA without quantization")
    args = p.parse_args()

    set_seed(args.seed)
    alpha = args.alpha or 2 * args.rank
    train_rows = load_jsonl(args.train_file)
    if args.train_size:
        if args.train_size > len(train_rows):
            raise SystemExit(f"--train-size {args.train_size} > {len(train_rows)} available rows")
        train_rows = train_rows[: args.train_size]
    val_rows = load_jsonl(args.val_file)

    out_dir = Path(args.output_dir or
                   f"runs/{args.model.split('/')[-1]}_r{args.rank}_n{args.train_size or len(train_rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    tok.padding_side = "right"
    train_ds, train_tok_info = tokenize(train_rows, tok, args.max_seq_len)
    val_ds, val_tok_info = tokenize(val_rows, tok, args.max_seq_len)
    print("train tokens:", train_tok_info)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, quantized = load_base_model(args.model, quantize=not args.no_4bit)
    mem_after_load = gpu_memory_gb()
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout,
        target_modules=args.target_modules, task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    dtype = compute_dtype()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(round(args.warmup_ratio * total_steps)),
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        processing_class=tok,
    )

    start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - start
    mem_after_train = gpu_memory_gb()
    # Validation runs after every epoch; only evaluate again if training stopped mid-epoch.
    eval_logs = [h for h in trainer.state.log_history if "eval_loss" in h]
    if eval_logs and eval_logs[-1]["step"] == train_result.global_step:
        final_eval_loss = eval_logs[-1]["eval_loss"]
    else:
        final_eval_loss = trainer.evaluate()["eval_loss"]

    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    run_info = {
        "args": vars(args) | {"alpha": alpha},
        "quantized_4bit": quantized,
        "compute_dtype": str(dtype),
        "train_tokens": train_tok_info,
        "val_tokens": val_tok_info,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "gpu_memory_after_load": mem_after_load,
        "gpu_memory_peak_training": mem_after_train,
        "train_seconds": round(train_seconds, 1),
        "global_steps": train_result.global_step,
        "final_train_loss": train_result.training_loss,
        "final_eval_loss": final_eval_loss,
        "log_history": trainer.state.log_history,
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in run_info.items() if k != "log_history"}, indent=2, ensure_ascii=False))
    print(f"Adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
