"""Check a finished run's files before they are committed or merged.

    python src/check_run.py Qwen2.5-1.5B-Instruct_zeroshot Qwen2.5-1.5B-Instruct_r16_n775

For each run name it checks results/<name>.jsonl and, for fine-tuned runs,
runs/<name>/run_info.json. Each check prints PASS or FAIL; the exit code is 1 if any fails.

What must match exactly (fixed by the data split and the default settings):
  - every setting in run_info.json except rank and train size is the default
  - 4-bit, fp16 and a T4 GPU, so memory and time are comparable across runs
  - rows kept after truncation, trainable parameters and training steps
  - 150 answers, one per test prompt, from the right model and adapter
Losses and answer text are not compared: they differ slightly between GPUs.

Needs no torch or GPU.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TAG = MODEL.split("/")[-1]
FULL_TRAIN = 775

# Rows left after the 1024-token limit (Qwen tokenizer, train.py rules; see data/tokenizer_stats.json).
KEPT = {100: 98, 250: 248, 500: 496, 775: 766}
# LoRA on all linear layers of Qwen2.5-1.5B, computed from the model config with peft.
TRAINABLE = {4: 4_616_192, 8: 9_232_384, 16: 18_464_768, 32: 36_929_536, 64: 73_859_072}
# Defaults of src/train.py that every run must keep.
DEFAULTS = {"model": MODEL, "train_file": "data/splits/train.jsonl", "val_file": "data/splits/val.jsonl",
            "dropout": 0.05, "target_modules": "all-linear", "epochs": 3, "lr": 2e-4, "batch_size": 2,
            "grad_accum": 8, "max_seq_len": 1024, "warmup_ratio": 0.03, "max_steps": -1, "seed": 42,
            "no_4bit": False}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def devanagari_share(text):
    letters = [c for c in text if c.isalpha() or "\u0900" <= c <= "\u097F"]
    return sum("\u0900" <= c <= "\u097F" for c in letters) / max(1, len(letters))


def check_run_info(info, rank, n):
    args = info["args"]
    kept = KEPT[n]
    batches = math.ceil(kept / DEFAULTS["batch_size"])
    steps = math.ceil(batches / DEFAULTS["grad_accum"]) * DEFAULTS["epochs"]
    changed = {k: args.get(k) for k, v in DEFAULTS.items() if args.get(k) != v}
    evals = [h["eval_loss"] for h in info["log_history"] if "eval_loss" in h]
    gpu = info["gpu_memory_peak_training"]
    yield f"rank {rank}, alpha {2 * rank}", args["rank"] == rank and args["alpha"] == 2 * rank
    yield f"train size {n}", args["train_size"] == (0 if n == FULL_TRAIN else n)
    yield "all other settings are the defaults" + (f" (changed: {changed})" if changed else ""), not changed
    yield "4-bit, fp16", info["quantized_4bit"] and info["compute_dtype"] == "torch.float16"
    yield f"GPU is a T4 (got {gpu.get('gpu')})", "T4" in gpu.get("gpu", "")
    yield f"{kept} rows kept after truncation (got {info['train_tokens']['kept']})", info["train_tokens"]["kept"] == kept
    yield f"{TRAINABLE[rank]:,} trainable params (got {info['trainable_params']:,})", info["trainable_params"] == TRAINABLE[rank]
    yield f"{steps} training steps (got {info['global_steps']})", info["global_steps"] == steps
    yield f"validation loss after each of 3 epochs (got {len(evals)})", len(evals) == 3
    yield "losses are finite", all(math.isfinite(x) for x in evals + [info["final_train_loss"]])


def check_results(rows, test_ids, name, adapter_run):
    ids = [r["id"] for r in rows]
    adapters = {r["adapter"] for r in rows}
    yield f"{len(test_ids)} answers (got {len(rows)})", len(rows) == len(test_ids)
    yield "one answer per test prompt, no duplicates", len(set(ids)) == len(ids) and set(ids) == test_ids
    yield "model, 4-bit", all(r["model"] == MODEL and r["quantized_4bit"] for r in rows)
    if adapter_run:
        ok = len(adapters) == 1 and Path(str(next(iter(adapters)))).parts[-2:] == (name, "adapter")
    else:
        ok = adapters == {None}
    yield f"adapter is {name + '/adapter' if adapter_run else 'none'} (got {sorted(map(str, adapters))})", ok
    yield "no empty answers", all(r["output"].strip() for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help=f"Run names, e.g. {TAG}_zeroshot or {TAG}_r16_n775")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--test-file", default="data/splits/test.jsonl")
    args = p.parse_args()

    test_ids = {r["id"] for r in load_jsonl(args.test_file)}
    failed = 0
    for name in args.names:
        print(f"== {name}")
        m = re.fullmatch(re.escape(TAG) + r"_r(\d+)_n(\d+)", name)
        checks = []
        if m:
            rank, n = int(m[1]), int(m[2])
            if rank not in TRAINABLE or n not in KEPT:
                checks.append((f"rank in {sorted(TRAINABLE)} and size in {sorted(KEPT)}", False))
            else:
                path = Path(args.runs_dir, name, "run_info.json")
                if path.exists():
                    checks += check_run_info(json.loads(path.read_text(encoding="utf-8")), rank, n)
                else:
                    checks.append((f"{path} exists", False))
        elif name != f"{TAG}_zeroshot":
            checks.append((f"name is {TAG}_zeroshot or {TAG}_r<rank>_n<rows>", False))

        path = Path(args.results_dir, f"{name}.jsonl")
        if path.exists():
            rows = load_jsonl(path)
            checks += check_results(rows, test_ids, name, adapter_run=bool(m))
        else:
            checks.append((f"{path} exists", False))
            rows = []

        for label, ok in checks:
            print("PASS" if ok else "FAIL", label)
            failed += not ok
        if rows:
            hindi = sum(devanagari_share(r["output"]) > 0.5 for r in rows)
            words = sorted(len(r["output"].split()) for r in rows)
            print(f"info: {hindi} of {len(rows)} answers mostly Devanagari, median {words[len(words) // 2]} words")
    print(f"\n{failed} check(s) failed" if failed else "\nall checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
