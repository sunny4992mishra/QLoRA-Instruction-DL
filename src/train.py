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
# This must run before torch is imported, because CUDA reads the variable once at startup.
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
        # Split each row into the prompt text and the response text (see prompts.py).
        prompt, response = build_example(tok, r["instruction"], r["response"])
        # Tokenize the two parts separately so we know exactly where the response starts.
        # add_special_tokens=False because the chat template already added the special tokens.
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(response, add_special_tokens=False)["input_ids"]
        lengths.append(len(p_ids) + len(r_ids))
        # Drop the row if the prompt alone fills the 1,024-token limit: no room would be left
        # for even one response token, so the row could teach nothing. 9 of 775 rows.
        if len(p_ids) >= max_len:
            n_dropped += 1
            continue
        ids = p_ids + r_ids
        # Labels are what the model is trained to predict. -100 is the value PyTorch's
        # cross-entropy loss ignores, so prompt tokens add nothing to the loss. The model
        # learns to write answers, not to predict the instruction.
        labels = [-100] * len(p_ids) + r_ids
        # Cut rows longer than the limit. They keep the prompt and the start of the answer but
        # lose its end, including the end-of-turn token. 103 of 775 rows, losing 34.5% of all
        # response tokens.
        if len(ids) > max_len:
            n_truncated += 1
            ids, labels = ids[:max_len], labels[:max_len]
        # attention_mask is all 1s here. The collator adds 0s for padding when it builds batches.
        examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    # Length statistics (before cutting) for run_info.json: median, 95th percentile and maximum.
    lengths.sort()
    pct = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    info = {"rows": len(rows), "kept": len(examples), "truncated": n_truncated,
            "dropped_prompt_too_long": n_dropped,
            "token_len_p50": pct(0.5), "token_len_p95": pct(0.95), "token_len_max": lengths[-1]}
    return Dataset.from_list(examples), info


def main():
    # Command-line options. The defaults are the settings used for every run in the report;
    # only --rank and --train-size change between runs.
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-file", default="data/splits/train.jsonl")
    p.add_argument("--val-file", default="data/splits/val.jsonl")
    p.add_argument("--train-size", type=int, default=0, help="First N training rows (0 = all). Subsets are nested.")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default 2 * rank)")
    p.add_argument("--dropout", type=float, default=0.05)
    # "all-linear" puts adapters on every linear layer of every transformer block (attention
    # q, k, v, o and MLP gate, up, down), as the QLoRA paper recommends.
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

    # Fix the random seeds (Python, NumPy, PyTorch) so a rerun gives the same adapter
    # initialisation, dropout masks and data order.
    set_seed(args.seed)
    # LoRA scales its update by alpha / r. With alpha = 2r that scale is always 2, so changing
    # the rank changes the adapter's capacity but not the size of its update.
    alpha = args.alpha or 2 * args.rank
    train_rows = load_jsonl(args.train_file)
    # --train-size N takes the first N rows. train.jsonl is in a fixed shuffled order, so the
    # subsets are nested (100 ⊂ 250 ⊂ 500 ⊂ 775).
    if args.train_size:
        if args.train_size > len(train_rows):
            raise SystemExit(f"--train-size {args.train_size} > {len(train_rows)} available rows")
        train_rows = train_rows[: args.train_size]
    val_rows = load_jsonl(args.val_file)

    # Default output folder, for example runs/Qwen2.5-1.5B-Instruct_r16_n775.
    out_dir = Path(args.output_dir or
                   f"runs/{args.model.split('/')[-1]}_r{args.rank}_n{args.train_size or len(train_rows)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(args.model)
    # In training, padding goes on the right (after the text). Padded positions get
    # attention mask 0 and label -100, so they don't affect the model or the loss.
    tok.padding_side = "right"
    train_ds, train_tok_info = tokenize(train_rows, tok, args.max_seq_len)
    val_ds, val_tok_info = tokenize(val_rows, tok, args.max_seq_len)
    print("train tokens:", train_tok_info)

    # Reset the peak-memory counter just before loading, so the peak recorded in
    # run_info.json covers loading the model plus training and nothing earlier.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, quantized = load_base_model(args.model, quantize=not args.no_4bit)
    mem_after_load = gpu_memory_gb()
    # The key/value cache only speeds up generation. It is not needed in training and doesn't
    # work with gradient checkpointing, so turn it off.
    model.config.use_cache = False
    if quantized:
        # Prepares a 4-bit model for training: freezes all base weights, casts the small
        # non-quantized layers (such as layer norms) to fp32 for numerical stability, and turns
        # on gradient checkpointing. Gradient checkpointing stores fewer activations and
        # recomputes them in the backward pass, trading some time for a lot of memory.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # Without quantization: with frozen embeddings and gradient checkpointing, the input
        # embeddings must require gradients, or no gradient would flow back to the adapters.
        model.enable_input_require_grads()

    # Wrap the frozen model with LoRA adapters. Each targeted weight matrix W gets a trainable
    # low-rank update B·A of rank r (A is r x k, B is d x r). Only A and B are trained.
    # lora_dropout applies dropout to the adapter input, for regularisation.
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=alpha, lora_dropout=args.dropout,
        target_modules=args.target_modules, task_type="CAUSAL_LM"))
    # For Qwen2.5-1.5B this is r x 1,154,048, so 18,464,768 (1.18% of all parameters) at r = 16.
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # Optimizer steps: one step per 16 examples (batch 2 x 8 accumulation), rounded up.
    # With 766 kept rows: 48 steps per epoch, 144 over 3 epochs. With 98 kept rows (N = 100): 21.
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    dtype = compute_dtype()
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Gradients from 8 small batches are added up before each optimizer step, which acts
        # like a batch of 16 without the memory of holding 16 examples at once.
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        # The learning rate rises linearly during warmup (3% of steps, 4 of 144), then falls
        # along a cosine curve towards 0.
        lr_scheduler_type="cosine",
        warmup_steps=int(round(args.warmup_ratio * total_steps)),
        # Paged AdamW 8-bit: stores optimizer state in 8 bits and can page it to CPU memory
        # during memory spikes (part of the QLoRA recipe).
        optim="paged_adamw_8bit" if quantized else "adamw_torch",
        # Mixed precision in the type the GPU supports: fp16 on the T4.
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Log the training loss every 5 steps; these logs end up in run_info.json.
        logging_steps=5,
        # Compute the validation loss and save an adapter checkpoint after every epoch.
        eval_strategy="epoch",
        save_strategy="epoch",
        # Save only the adapter weights, not the optimizer state, to keep checkpoints small.
        save_only_model=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # Builds batches: pads input_ids with the pad token and labels with -100, so padding
        # is never learned.
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        processing_class=tok,
    )

    # Train and time it. The time measured here is the "train time" in the report.
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

    # Save the final adapter (the end of epoch 3, the one evaluated in every run) with a copy
    # of the tokenizer. A PEFT model saves only the small adapter weights, not the base model.
    trainer.save_model(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    # Everything needed to report and check the run. src/check_run.py and src/analyze.py read
    # this file. It is the only file under runs/ that is committed to git.
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
        # Every logged training loss and every per-epoch validation loss (for the loss curves).
        "log_history": trainer.state.log_history,
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in run_info.items() if k != "log_history"}, indent=2, ensure_ascii=False))
    print(f"Adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
