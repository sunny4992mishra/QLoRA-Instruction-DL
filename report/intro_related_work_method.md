# P16 report: Introduction, Related work, Training method

Draft for issue #13 (Ompal). Section numbers and cross-references to the data section (#4), the
evaluation section (#10) and the results (#7) are placeholders until the report is assembled.

---

## 1. Introduction

Hindi is one of the most widely spoken languages in the world, but it is a small part of the
data that open large language models (LLMs) are trained on. Small open models, the kind that run
on a single free GPU, often answer Hindi instructions poorly: they repeat themselves, or keep
going until they hit the length limit. In our zero-shot baseline, 103 of 150 answers run to the
512-token limit (Section [Results]). Instruction tuning on Hindi data is the obvious fix. Full
fine-tuning does not fit on a single free 16 GB GPU, even for a model with 1.5 billion
parameters. In fp32, the weights and the gradients need about 6.2 GB each and the AdamW
optimizer state another 12.3 GB, about 25 GB before any activations are stored.

QLoRA (Dettmers et al., 2023) removes most of this cost. The base model is stored in 4-bit
precision and frozen, and only small low-rank adapter matrices are trained (Hu et al., 2022).
This makes instruction tuning possible on a free Colab or Kaggle T4 GPU. QLoRA still leaves two
practical choices that decide both the cost and the quality of the result: the rank of the
adapters and the amount of training data. For a low-resource language, human-written
instruction data is scarce, so it matters how far a few hundred examples can go.

This project asks: **how much Hindi instruction data and what LoRA rank is enough?** We
fine-tune Qwen2.5-1.5B-Instruct (Qwen Team, 2024) with QLoRA on the Hindi part of the Aya
Dataset (Singh et al., 2024), which was written or edited by fluent speakers rather than
machine-translated. We
run two sweeps on a single T4 GPU, changing one factor at a time and keeping every other setting
fixed:

- a **rank sweep** with r ∈ {4, 8, 16, 32, 64} on all 775 training rows, and
- a **data-size sweep** with N ∈ {100, 250, 500, 775} training rows at r = 16.

We compare each fine-tuned model with the zero-shot base model on 150 held-out Hindi prompts.
Answer quality is measured with an LLM judge and with blind ratings from all four team members.
Cost is measured as peak GPU memory, training time and the number of trainable parameters.

Our contributions are:

1. A cleaned Hindi instruction split from the Aya Dataset, split by prompt so that no test
   prompt is seen in training, with an analysis of how the Qwen2.5 tokenizer handles Hindi
   (Section [Data]).
2. A reproducible QLoRA pipeline that fits on a free 16 GB T4 GPU, with eight fine-tuning runs
   that use identical settings apart from rank or data size, plus a zero-shot baseline
   (Section 3).
3. An evaluation of answer quality against cost along both sweeps, using an LLM judge and
   blind human ratings (Sections [Evaluation] and [Results]).

## 2. Related work

**Parameter-efficient fine-tuning.** LoRA (Hu et al., 2022) freezes the pretrained weights
and learns a low-rank update ΔW = BA for chosen weight matrices, where B ∈ ℝ^{d×r},
A ∈ ℝ^{r×k} and the rank r is much smaller than d and k. Only A and B are trained, which
cuts the number of trainable parameters by orders of magnitude, and the update can be merged
into the weights after training. QLoRA (Dettmers et al., 2023) combines LoRA with a frozen base
model quantized to 4 bits. It introduces the 4-bit NormalFloat (NF4) data type, double
quantization of the quantization constants, and paged optimizers that move optimizer state to
CPU memory during memory spikes. With these, the authors fine-tuned a 65B model on a single
48 GB GPU. Two of their findings shape our setup. First, adapting all linear layers was
necessary to match full 16-bit fine-tuning, and once all layers were adapted the rank r had
little effect on the result. Second, a small high-quality dataset (about 9,000 examples from
OASST1) gave a better chatbot than a much larger one (FLAN v2). Biderman et al. (2024) compare
LoRA with full fine-tuning and find that LoRA learns less on the target domain but forgets
less of the base model's abilities, and that full fine-tuning learns weight updates of much
higher rank than typical LoRA settings. Our rank sweep tests whether higher rank helps in a
small-data, low-resource setting.

**Instruction tuning with little data.** Self-Instruct (Wang et al., 2023) and Stanford
Alpaca (Taori et al., 2023) showed that tens of thousands of model-generated instruction
examples can turn a pretrained model into an instruction follower. LIMA (Zhou et al., 2023)
went further: a 65B model fine-tuned on only 1,000 carefully chosen examples produced strong
answers. The authors argue that a model learns almost all of its knowledge in pretraining, and
that instruction tuning mainly teaches the format and style of answers. If this holds for a
small model in a language that is rare in its pretraining data, a few hundred Hindi examples
should be enough to fix the format of the answers, but not the facts in them. Our data-size
sweep tests this.

**Multilingual and Hindi instruction tuning.** BLOOMZ and mT0 (Muennighoff et al., 2023) were
fine-tuned on a multitask prompt collection in 46 languages and generalize to tasks in other
languages. Several later works machine-translate English instruction data instead.
Bactrian-X (Li et al., 2023) translated the Alpaca and Dolly instructions into 52 languages,
generated responses with gpt-3.5-turbo, and trained LoRA adapters on LLaMA and BLOOM. Chen et
al. (2024) fine-tuned on Alpaca and its machine translations with LoRA and full fine-tuning.
Under a fixed compute budget, they found that multilingual tuning matches or beats separate
models per language. For Hindi, Airavata (Gala et al., 2024) fine-tuned OpenHathi, a Llama 2
7B model extended for Hindi, on IndicInstruct, mostly English instruction datasets translated
into Hindi with IndicTrans2. They chose LoRA over full fine-tuning after comparing the two
(Gala et al., 2024, Section 3.1). The Aya initiative instead collected instruction data from
fluent speakers. The Aya Dataset (Singh et al., 2024) has about 204,000 human-annotated
prompt–completion pairs in 65 languages, and the Aya model (Üstün et al., 2024) was
instruction-tuned on 101 languages. Our work differs from these in scale and focus. We use
only the human-annotated Hindi part of the Aya Dataset (775 training rows after
cleaning), a 1.5B model, and a single free GPU, and we measure how quality changes with LoRA
rank and data size, not only the best result.

**Tokenization of Hindi.** Petrov et al. (2023) show that tokenizers of multilingual models
can need many times more tokens for the same text in some languages than in English, which
raises cost and shortens the usable context. We see this effect in our data: the Qwen2.5
tokenizer averages 1.09 characters per Hindi token, compared with 3.67 for Sarvam-1, a model
built for Indian languages. As a result, some training rows do not fit in our 1,024-token
limit (Section [Data]).

## 3. Training method

All runs use the same code (`src/train.py`, `src/generate.py`) and the same settings. Only the
LoRA rank or the number of training rows changes between runs.

### 3.1 Base model

We use **Qwen2.5-1.5B-Instruct** (Qwen Team, 2024), a decoder-only transformer with 1.54
billion parameters and 28 layers, released under the Apache-2.0 license. It is already
instruction-tuned and supports over 29 languages, but Hindi is not among those named in its
model card. It has a chat template and follows instructions in principle. This lets us test whether a small amount
of Hindi data improves an existing instruction follower in a language it handles poorly,
instead of teaching instruction following from scratch. At 1.5B parameters, the model in 4-bit
precision leaves enough of the T4's 16 GB of memory for training with sequences of up to 1,024
tokens.

### 3.2 Quantization

Following QLoRA, the base model is loaded with bitsandbytes in 4-bit NF4 with double
quantization and kept frozen. Computation runs in 16-bit floating point. The T4 has no native
bfloat16 support, so all runs use fp16. The model is prepared for k-bit training with peft,
and gradient checkpointing (Chen et al., 2016) is turned on to save activation memory.

### 3.3 LoRA adapters

LoRA adapters are added to **all linear layers** of every transformer block: the query, key,
value and output projections of the attention layer, and the gate, up and down projections of
the MLP. This follows the QLoRA finding that adapting all linear layers is needed to match full
fine-tuning. The output layer and the embeddings are not adapted. The scaling factor is
α = 2r, so the ratio α/r is the same for every rank, and changing r changes the capacity
of the update but not its scale. LoRA dropout is 0.05.

For Qwen2.5-1.5B, each adapted layer adds r × (d_in + d_out) parameters. Summed over the 7
matrices in each of the 28 blocks, the adapter has **r × 1,154,048** trainable parameters
(Table 1).

**Table 1.** Trainable parameters per rank, as reported by peft in each run.

| Rank r | α | Trainable parameters | Share of all parameters |
|---|---|---|---|
| 4 | 8 | 4,616,192 | 0.30% |
| 8 | 16 | 9,232,384 | 0.59% |
| 16 | 32 | 18,464,768 | 1.18% |
| 32 | 64 | 36,929,536 | 2.34% |
| 64 | 128 | 73,859,072 | 4.57% |

### 3.4 Prompt format and loss masking

Each example is formatted with the model's own chat template, with a fixed Hindi system
prompt: "आप एक सहायक हैं। उपयोगकर्ता के निर्देश का पालन करें और हिंदी में स्पष्ट और सही उत्तर दें।"
("You are an assistant. Follow the user's instruction and give a clear and correct answer in
Hindi."). The instruction is the user turn and the reference answer is the assistant turn.
Training and generation build the prompt with the same function (`src/prompts.py`), so the
model sees exactly the same format at test time.

The loss is computed on the **response tokens only**. Prompt tokens (system prompt, instruction
and template markers) get the label −100 and are ignored by the cross-entropy loss. The
response includes the end-of-turn token `<|im_end|>`, so the model also learns when to stop.
For the 103 training rows that are cut at the length limit (see below and Section [Data]), the
end-of-turn token is cut off as well, so these rows show no stopping point. Answers that run to
the generation limit are a common failure in our results (Section [Results]), so this matters.

Sequences are limited to **1,024 tokens**. If the prompt alone is 1,024 tokens or longer, the
row is dropped. If the prompt fits but prompt plus response is longer, the response is cut at
the limit. This removes 9 of the 775 training rows and truncates 103. The effect on the data is
discussed in Section [Data].

### 3.5 Optimization

| Setting | Value |
|---|---|
| Optimizer | paged AdamW, 8-bit (Dettmers et al., 2022) |
| Learning rate | 2 × 10⁻⁴, cosine schedule, linear warmup over 3% of steps |
| Batch size | 2 per step × 8 gradient accumulation steps = 16 |
| Epochs | 3 |
| Max sequence length | 1,024 tokens |
| Precision | 4-bit NF4 base model, fp16 compute |
| Random seed | 42 |

Validation loss is computed after every epoch on 61 validation rows from 50 prompts that do not
appear in training or testing, and the adapter is saved after each epoch. We evaluate the
adapter from the end of the third epoch in every run. We do not pick the epoch with the lowest
validation loss, so that all runs are treated the same way.

Because the number of epochs is fixed, the number of optimizer steps grows with the data size:
21 steps for N = 100, 48 for N = 250, 93 for N = 500 and 144 for all 775 rows (98, 248, 496 and
766 rows are left after the length limit). Runs with less data therefore also get fewer updates.
This is part of what the data-size sweep measures.

### 3.6 Experiment grid

**Table 2.** The zero-shot baseline and the eight fine-tuning runs. Run names follow `Qwen2.5-1.5B-Instruct_r<rank>_n<rows>`.

| Run | Rank r | Training rows N | Sweep |
|---|---|---|---|
| zero-shot | – | 0 | baseline |
| r4_n775 | 4 | 775 | rank |
| r8_n775 | 8 | 775 | rank |
| r16_n775 | 16 | 775 | rank and data size |
| r32_n775 | 32 | 775 | rank |
| r64_n775 | 64 | 775 | rank |
| r16_n100 | 16 | 100 | data size |
| r16_n250 | 16 | 250 | data size |
| r16_n500 | 16 | 500 | data size |

The training file is stored in a fixed random order, and a run with N rows uses the first N,
so the data subsets are nested (100 ⊂ 250 ⊂ 500 ⊂ 775). A larger subset therefore only adds
examples and never swaps them, so differences between data sizes are not caused by a different
sample. Each setting is run once, with the same seed.

### 3.7 Hardware and measurements

Every run uses a single NVIDIA Tesla T4 (16 GB) on Kaggle or Colab. Kaggle provides two T4s,
but the scripts use only the first, so memory numbers are comparable across runs. For each run,
`run_info.json` records:

- trainable and total parameters,
- **peak GPU memory** during training (`torch.cuda.max_memory_allocated`, plus the peak
  reserved by PyTorch's allocator),
- **training time** (wall-clock time of the training loop, including the validation after each
  epoch),
- the training and validation loss history.

### 3.8 Generating answers

Each model answers the 150 held-out test prompts with the same 4-bit base model, with the
adapter loaded for fine-tuned runs and without it for the zero-shot baseline. Decoding is
greedy, so the outputs are reproducible, with a repetition penalty of 1.1 and at most 512 new
tokens. The prompt format is the same as in training (Section 3.4). The answers are evaluated
as described in Section [Evaluation].

---

## References

- Biderman, D., et al. (2024). LoRA learns less and forgets less. *Transactions on Machine
  Learning Research*. arXiv:2405.09673.
- Chen, P., Ji, S., Bogoychev, N., Kutuzov, A., Haddow, B., and Heafield, K. (2024).
  Monolingual or multilingual instruction tuning: Which makes a better Alpaca? In *Findings of
  EACL 2024*, pages 1347–1356. https://aclanthology.org/2024.findings-eacl.90
- Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016). Training deep nets with sublinear
  memory cost. arXiv:1604.06174.
- Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. (2022). 8-bit optimizers via
  block-wise quantization. In *ICLR 2022*. arXiv:2110.02861.
- Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient
  finetuning of quantized LLMs. In *NeurIPS 2023*. arXiv:2305.14314.
- Gala, J., et al. (2024). Airavata: Introducing Hindi instruction-tuned LLM. arXiv:2401.15006.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W.
  (2022). LoRA: Low-rank adaptation of large language models. In *ICLR 2022*.
  arXiv:2106.09685.
- Li, H., Koto, F., et al. (2023). Bactrian-X: Multilingual replicable instruction-following
  models with low-rank adaptation. arXiv:2305.15011.
- Muennighoff, N., et al. (2023). Crosslingual generalization through multitask finetuning. In
  *ACL 2023*. arXiv:2211.01786.
- Petrov, A., La Malfa, E., Torr, P. H. S., and Bibi, A. (2023). Language model tokenizers
  introduce unfairness between languages. In *NeurIPS 2023*. arXiv:2305.15425.
- Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115.
- Singh, S., et al. (2024). Aya Dataset: An open-access collection for multilingual instruction
  tuning. In *ACL 2024*. arXiv:2402.06619.
- Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and
  Hashimoto, T. B. (2023). Stanford Alpaca: An instruction-following LLaMA model.
  https://github.com/tatsu-lab/stanford_alpaca
- Üstün, A., et al. (2024). Aya model: An instruction finetuned open-access multilingual
  language model. In *ACL 2024*. arXiv:2402.07827.
- Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H.
  (2023). Self-Instruct: Aligning language models with self-generated instructions. In *ACL
  2023*. arXiv:2212.10560.
- Zhou, C., et al. (2023). LIMA: Less is more for alignment. In *NeurIPS 2023*.
  arXiv:2305.11206.
