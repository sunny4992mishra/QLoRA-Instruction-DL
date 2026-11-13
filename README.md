# QLoRA Instruction-Tuning of a Small LLM for Hindi (P16)

Deep Learning course project P16, team 16, section-b:
Pranjal Dubey, Sunny Kumar, Shahid Ansari, Ompal Yadav.

**Goal.** Make a small open LLM (1–3B parameters) follow instructions in Hindi, using QLoRA on a free Colab or Kaggle T4 GPU.

**Research question.** How much Hindi instruction data and what LoRA rank is enough? We measure answer quality against GPU memory, training time and trainable parameters.

**References**
1. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. https://arxiv.org/abs/2305.14314
2. Singh et al., *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning*, ACL 2024. https://arxiv.org/abs/2402.06619

## Repository layout

```
data/
  splits/train.jsonl      775 rows (675 prompts), fixed random order
  splits/val.jsonl         61 rows (50 prompts), used for validation loss
  splits/test.jsonl       150 held-out prompts, never trained on
  split_stats.json        counts from each cleaning step
notebooks/
  p16_qlora_hindi.ipynb   Colab/Kaggle notebook: baseline, training, generation, grid
src/
  prepare_data.py         builds the split from the Aya Dataset (already run)
  prompts.py              chat-template formatting shared by training and generation
  train.py                QLoRA fine-tuning, logs memory/time/loss to run_info.json
  generate.py             answers the test prompts, with or without an adapter
  utils.py                model/tokenizer loading, GPU memory stats
requirements.txt
```

## Quick start

Open `notebooks/p16_qlora_hindi.ipynb` in Colab (T4 GPU) or Kaggle (GPU T4, Internet on), then run the cells from top to bottom. On Colab, the notebook saves adapters and results to Google Drive under `MyDrive/p16`.

To run the scripts directly, start from the repo root:

```bash
pip install -r requirements.txt

# zero-shot baseline
python src/generate.py --out results/Qwen2.5-1.5B-Instruct_zeroshot.jsonl

# QLoRA fine-tuning: rank 16, all training rows
python src/train.py --rank 16
# -> runs/Qwen2.5-1.5B-Instruct_r16_n775/{adapter, checkpoint-*, run_info.json}

# answers from the fine-tuned model
python src/generate.py --adapter runs/Qwen2.5-1.5B-Instruct_r16_n775/adapter \
    --out results/Qwen2.5-1.5B-Instruct_r16_n775.jsonl
```

Run any script with `--help` to see all options.

## Data

The source is the Hindi subset (`language_code == "hin"`) of [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), licensed Apache-2.0. The subset has 1,153 rows in its train split and **none in its test split**, so we hold out our own test set.

`src/prepare_data.py` cleans the data as follows (counts are in `data/split_stats.json`):

| Step | Rows |
|---|---|
| Hindi rows in Aya train split | 1,153 |
| Dropped: prompt has no Devanagari text | −115 |
| Dropped: exact duplicate (prompt, response) pairs | −22 |
| Clean rows | 1,016 (875 unique prompts) |

- The 115 dropped prompts are English instructions whose "response" is only a Hindi translation of the instruction. Keeping them would teach the model to translate instructions instead of following them.
- Text is stripped, and a pair of quotes wrapping the whole text is removed.
- Splits are made **by prompt**. Some prompts appear more than once (an original annotation plus re-annotations), and all copies of a prompt go to the same split. No test prompt appears in train or val. 25 test prompts have more than one reference answer, and all of them are kept in `references`.
- `train.jsonl` is stored in a fixed shuffled order, so `--train-size N` takes the first N rows. This makes the data-size subsets nested (100 ⊂ 250 ⊂ 500 ⊂ 775).

The split is committed. **Don't regenerate it** unless the whole team agrees, because changing it makes earlier results incomparable.

## Training setup

These are the defaults in `src/train.py`:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, not gated) |
| Quantization | 4-bit NF4, double quantization, fp16 compute on T4 |
| LoRA | all linear layers, r = 16, alpha = 2r, dropout 0.05 |
| Optimizer | paged AdamW 8-bit, lr 2e-4, cosine schedule, 3% warmup |
| Batch | 2 per step × 8 accumulation = 16 |
| Epochs | 3, with validation loss and an adapter checkpoint after each |
| Max sequence length | 1024 tokens |
| Loss | response tokens only; prompt tokens are masked |

`run_info.json` records the settings, trainable parameters, peak GPU memory after loading and after training (`torch.cuda.max_memory_allocated`), training time, and the full loss history. Kaggle provides 2 GPUs, but the scripts use only GPU 0 so that memory numbers are comparable across runs.

Generation uses greedy decoding (repetition penalty 1.1, up to 512 new tokens), so outputs are reproducible. If an output file already exists, `generate.py` skips prompts that are already in it. After a disconnect, re-run the same command to continue.

### Tokenizer note

On our training data, the Qwen2.5 tokenizer averages **1.09 characters per token** on Hindi. Sarvam-1, a 2B model trained for Indian languages, averages 3.66. At 1024 tokens with Qwen2.5, 103 of the 775 training rows are truncated and 9 are dropped because the prompt alone is too long. The median example is 363 tokens. This is worth discussing in the report, and Sarvam-1 is a possible second model to compare.

## Experiment plan

1. Run the zero-shot baseline with the base model.
2. Rank sweep at full data: r ∈ {4, 8, 16, 32, 64}.
3. Data-size sweep at a fixed rank: N ∈ {100, 250, 500, 775}.
4. Optionally, repeat the best setting with a second base model.
5. Evaluate with an LLM judge (1–5 scores for correctness, helpfulness and Hindi fluency) and blind human ratings on a sample.
6. Plot quality against rank, data size and peak GPU memory.

Section 5 of the notebook runs steps 2 and 3. The notebook writes answers to `SAVE/results/`. Copy those `*.jsonl` files into the repo's `results/` folder and commit them, so everyone evaluates the same outputs. Adapters stay out of git: only `run_info.json` is tracked under `runs/`.

## Still to do

- [ ] LLM-judge script and fixed judge prompt
- [ ] Human rating sheet (blind, randomised order)
- [ ] Plotting and results table
