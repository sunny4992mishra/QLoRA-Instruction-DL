"""Helpers shared by train.py and generate.py."""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path):
    # A .jsonl file has one JSON object per line. Blank lines (for example a trailing
    # newline at the end of the file) are skipped.
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_dtype():
    # The data type used for the actual matrix multiplications. The 4-bit weights are
    # dequantized to this type on the fly during each forward and backward pass.
    if not torch.cuda.is_available():
        # CPU fallback for smoke tests: plain fp32.
        return torch.float32
    # Native bf16 only (Ampere+). With emulation allowed, the T4 reports bf16 support and would
    # train in slow emulated bf16 instead of fp16.
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    return torch.bfloat16 if native_bf16 else torch.float16  # T4/P100 -> fp16


def load_tokenizer(name_or_path):
    # name_or_path is a Hugging Face model id, or an adapter folder (generate.py passes the
    # adapter folder, where train.py saved a copy of the tokenizer).
    tok = AutoTokenizer.from_pretrained(name_or_path)
    # Batching needs a padding token. If the tokenizer has none, reuse the end-of-sequence
    # token. Padded positions are masked out anyway (attention mask 0, label -100).
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(name, quantize=True):
    """Load the base model in 4-bit NF4 (QLoRA setup). Falls back to fp32 on CPU for smoke tests."""
    cuda = torch.cuda.is_available()
    # bitsandbytes 4-bit quantization needs a CUDA GPU, so without one we load the model
    # unquantized. This is only for checking that the code runs, not for real runs.
    if quantize and not cuda:
        print("WARNING: no CUDA GPU found, loading without 4-bit quantization (smoke test only).")
        quantize = False

    dtype = compute_dtype()
    # dtype: the type for the weights that are not quantized (embeddings, layer norms) and
    # for activations.
    kwargs = {"dtype": dtype}
    if cuda:
        kwargs["device_map"] = {"": 0}  # whole model on one GPU
    if quantize:
        # The QLoRA recipe (Dettmers et al., 2023):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            # Store the frozen base weights in 4 bits instead of 16, about 4x less memory.
            load_in_4bit=True,
            # NF4 ("4-bit NormalFloat"): the 16 levels are placed for normally distributed
            # weights, which pretrained weights roughly are. More accurate than plain 4-bit floats.
            bnb_4bit_quant_type="nf4",
            # Double quantization: weights are quantized in small blocks, each with its own
            # scaling constant. This also quantizes those constants, saving a little more memory.
            bnb_4bit_use_double_quant=True,
            # Weights are stored in 4 bits but computed in fp16 (on the T4).
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    # Return whether 4-bit was really used, because the fallback above may have turned it off.
    # train.py and generate.py record it, and train.py picks the optimizer from it.
    return model, quantize


def gpu_memory_gb():
    # GPU memory figures for run_info.json. 1024**3 bytes = 1 GB, as the report states.
    if not torch.cuda.is_available():
        return {}
    return {
        # Memory held by tensors right now.
        "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        # Highest memory held by tensors since the last reset. train.py resets the counter just
        # before loading the model, so this is the peak of loading plus training. It is the
        # "peak GPU memory" number in the report.
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        # Highest memory PyTorch's allocator kept reserved, including its cache. Always at least
        # the allocated peak, and it varies more between runs.
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "gpu": torch.cuda.get_device_name(0),
    }
