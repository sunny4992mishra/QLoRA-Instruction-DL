"""Helpers shared by train.py and generate.py."""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4/P100 -> fp16


def load_tokenizer(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name, quantize=True):
    """Load the base model in 4-bit NF4 (QLoRA setup). Falls back to fp32 on CPU for smoke tests."""
    cuda = torch.cuda.is_available()
    if quantize and not cuda:
        print("WARNING: no CUDA GPU found, loading without 4-bit quantization (smoke test only).")
        quantize = False

    dtype = compute_dtype()
    kwargs = {"dtype": dtype}
    if cuda:
        kwargs["device_map"] = {"": 0}  # whole model on one GPU
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    return model, quantize


def gpu_memory_gb():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
