# P16 report: Results, Conclusion

Draft for issue #7 (Sunny). Section, table and figure numbers are placeholders until the report
is assembled; tables and figures here are numbered R1, R2, …. Cross-references to other sections
are marked like `Section [Evaluation]`.

**Status.** Everything that needs only the run files is written: cost, validation loss, and
answer length and looping. The parts that need the LLM-judge scores (#9) or the human ratings
(#14) are marked **[PENDING]** and are not filled with guesses.

All numbers come from `runs/*/run_info.json` and `results/*.jsonl` on `main` at `4b43bd2`. They
are reproduced by `python src/analyze.py` (Tables R1 and R2, Figures R1–R4) and
`python src/answer_stats.py` (Table R3).

---

## 4. Results

We report cost first, then validation loss, then what the answers look like, and last the
quality scores from the LLM judge and the human raters. Every fine-tuned model is the adapter
from the end of epoch 3 (Section 3.5), and each setting was run once with the same seed.

### 4.1 Cost: memory, time and trainable parameters

**Table R1.** Cost of the eight fine-tuning runs on a Tesla T4. Peak GPU memory is
`torch.cuda.max_memory_allocated` during training. Training time includes the validation pass
after each epoch.

| Run | Rank r | Training rows N | Trainable parameters | Peak GPU memory | Training time | Steps |
|---|---:|---:|---:|---:|---:|---:|
| r4_n775 | 4 | 775 | 4,616,192 (0.30%) | 5.83 GB | 30.3 min | 144 |
| r8_n775 | 8 | 775 | 9,232,384 (0.59%) | 5.87 GB | 30.4 min | 144 |
| r16_n775 | 16 | 775 | 18,464,768 (1.18%) | 5.94 GB | 30.2 min | 144 |
| r32_n775 | 32 | 775 | 36,929,536 (2.34%) | 6.13 GB | 31.0 min | 144 |
| r64_n775 | 64 | 775 | 73,859,072 (4.57%) | 6.43 GB | 28.8 min | 144 |
| r16_n100 | 16 | 100 | 18,464,768 (1.18%) | 5.94 GB | 4.0 min | 21 |
| r16_n250 | 16 | 250 | 18,464,768 (1.18%) | 5.94 GB | 10.7 min | 48 |
| r16_n500 | 16 | 500 | 18,464,768 (1.18%) | 5.94 GB | 20.8 min | 93 |

**Rank is cheap.** Going from r = 4 to r = 64 multiplies the trainable parameters by 16 but
raises peak memory by only 0.60 GB, from 5.83 to 6.43 GB (about 10%). The 4-bit base model takes
1.10 GB once loaded, so most of the peak is activations and gradients for the 1,024-token
sequences, which do not depend on the rank. Training time does not depend on the rank either:
all five full-data runs take 29 to 31 minutes, or 12 to 13 seconds per optimizer step.

**Data size sets the time, not the memory.** Peak memory is 5.94 GB for every r = 16 run,
whatever the number of rows, because it is set by the longest sequence in a batch and not by
the number of batches. Training time grows in proportion to the number of steps, from 4 minutes
for 100 rows to 30 minutes for 775.

**Everything fits with room to spare.** The largest run peaks at 6.43 GB allocated, and
PyTorch's allocator reserved at most 9.6 GB in any run, well under the T4's 16 GB. Generating
the zero-shot answers peaks at 1.98 GB (reported in #18; the baseline has no `run_info.json`). GPU memory is therefore not the binding limit for this model and data
on a free GPU.

### 4.2 Validation loss

**Table R2.** Validation loss after each epoch (61 validation rows), and the training loss
averaged over the whole run. The last column is how far the epoch-3 loss is above the lowest of the three.

| Run | Epoch 1 | Epoch 2 | Epoch 3 (evaluated) | Train loss | Epoch 3 − best |
|---|---:|---:|---:|---:|---:|
| r4_n775 | 0.976 | 0.945 | 0.946 | 0.905 | +0.002 |
| r8_n775 | 0.964 | 0.936 | 0.942 | 0.873 | +0.006 |
| r16_n775 | 0.951 | 0.925 | 0.937 | 0.838 | +0.012 |
| r32_n775 | 0.935 | 0.910 | 0.932 | 0.793 | +0.023 |
| r64_n775 | 0.922 | 0.892 | 0.928 | 0.736 | +0.036 |
| r16_n100 | 1.041 | 1.042 | 1.041 | 0.915 | +0.001 |
| r16_n250 | 1.003 | 0.990 | 0.988 | 0.896 | 0.000 |
| r16_n500 | 0.973 | 0.950 | 0.957 | 0.852 | +0.007 |

**Rank sweep (Figure R1, right panel).** Validation loss falls steadily as the rank rises, but
the effect is small: 0.946 at r = 4 and 0.928 at r = 64, a difference of 0.019 for 16 times the
parameters.

**Higher rank overfits more in the third epoch.** In every full-data run the validation loss is
lowest after epoch 2 and rises in epoch 3, and the rise grows with the rank, from +0.002 at
r = 4 to +0.036 at r = 64. Over the same range the training loss falls from 0.905 to 0.736, so
the higher ranks fit the training rows better without a matching gain on held-out prompts. This
hides part of the benefit of rank: after epoch 2 the ranks are 0.053 apart (0.945 at r = 4,
0.892 at r = 64), almost three times the gap after epoch 3. We evaluate the epoch-3 adapter in
every run (Section 3.5), so our quality scores describe a slightly over-trained r = 32 and
r = 64. Figure R4 shows the curves.

**Data-size sweep (Figure R2, right panel).** Data size has a much larger effect than rank.
Validation loss falls from 1.041 with 100 rows to 0.988 with 250, 0.957 with 500 and 0.937 with
775, a difference of 0.104 across the sweep, more than five times the effect of rank. Each added
row helps less than the one before, but the curve has not flattened at 775 rows: the last 275
rows still lower the loss by 0.020. Our data ends there (Section [Data]), so we can say that
more Hindi data would probably still help, but not where the gains stop.

**With 100 rows, validation loss does not move.** It is 1.041 after every epoch while the
training loss keeps falling, from 1.04 in the first logged steps to 0.81 in the last. The run has only 21 optimizer steps, so the
model changes little and what it learns from 98 rows does not transfer to the validation
prompts. As Section 3.5 notes, smaller subsets also get fewer updates, so this sweep measures
data and training steps together.

Two limits apply to this subsection. We did not measure the validation loss of the base model
without an adapter, so Table R2 compares fine-tuned runs with each other and not with zero-shot.
And validation loss is not answer quality: it scores the probability of one reference answer
per prompt.

### 4.3 What the answers look like

**Table R3.** Length, cut-offs and looping in the 150 test answers per system, counted with the
Qwen2.5 tokenizer. "At the limit" means the answer ran to the 512-token generation limit.
"Looping" means the same 8-word phrase appears three or more times. For the reference answers,
"at the limit" counts those that are themselves 510 tokens or longer.

| System | Median tokens | At the limit | Looping | Mostly Devanagari |
|---|---:|---:|---:|---:|
| Reference answers | 201.5 | 26 | 2 | 149 |
| Zero-shot | 512 | 103 | 79 | 148 |
| r4_n775 | 197 | 46 | 43 | 147 |
| r8_n775 | 225 | 47 | 36 | 148 |
| r16_n775 | 187 | 49 | 35 | 148 |
| r32_n775 | 192.5 | 40 | 30 | 148 |
| r64_n775 | 194.5 | 34 | 15 | 147 |
| r16_n100 | 224.5 | 41 | 46 | 148 |
| r16_n250 | 417 | 70 | 71 | 147 |
| r16_n500 | 178.5 | 32 | 28 | 148 |

**The base model's problem is stopping, not language.** Zero-shot, 148 of 150 answers are
already mostly in Devanagari, but 103 run to the length limit and 79 loop. The median answer is
512 tokens, the limit itself, against 201.5 for the references.

**Fine-tuning teaches the model to stop.** Every fine-tuned run except N = 250 has a median
length between 178 and 225 tokens, close to the references, and cuts the number of answers at
the limit from 103 to between 32 and 49. Some long answers are legitimate: 26 of the references
are themselves 510 tokens or longer. That still leaves 6 to 23 more answers at the limit than
there are references of that length. Section 3.4 gives one likely cause:
the 103 training rows that are cut at 1,024 tokens show no stopping point.

**100 rows are enough for most of this change.** The N = 100 run already brings the median to
224.5 tokens and the answers at the limit to 41, about the same as the full-data runs, even
though its validation loss barely moves (Section 4.2). This fits the view in Section 2 that a
small amount of instruction data mainly teaches the format of an answer.

**Looping falls with rank.** At full data, the number of looping answers goes down as the rank
goes up: 43, 36, 35, 30 and 15 for r = 4 to 64. The r = 64 model loops in 15 answers, against
79 zero-shot.

**N = 250 is an outlier.** Its answers have a median of 417 tokens, 70 run to the limit and 71
loop, which is worse than N = 100 and close to zero-shot, although its validation loss lies
between N = 100 and N = 500 as expected. The run passed every check in `check_run.py`, so the
files are valid. With one seed per setting we cannot tell whether this is a property of 250
rows or chance, and repeating the run with other seeds would settle it. **[PENDING: do the judge
and human scores for N = 250 also fall out of line?]** The human ratings do not cover this run
(they rate zero-shot, r16_n775 and r16_n100), so only the judge can answer this.

These counts describe the form of the answers, not whether they are right.

### 4.4 Answer quality: LLM judge

**[PENDING: judge scores, #9.]** To fill once `results/judge/` exists:

- Table: mean correctness, helpfulness and Hindi fluency per system, with 95% intervals
  (`results/summary.md`).
- Rank sweep (Figure R1, left panel) and data-size sweep (Figure R2, left panel), with the
  zero-shot score as the reference line.
- Questions the scores should answer:
  1. How much does fine-tuning improve on zero-shot, and on which criterion most? Section 4.3
     predicts the largest gain in helpfulness, which is the criterion that penalises answers
     that loop or are cut off.
  2. Does quality follow validation loss across ranks (small, steady gain), or is it flat?
  3. Does quality keep rising with data size up to 775 rows?
  4. Is N = 250 below N = 100, as Table R3 suggests?
  5. Does correctness improve at all? Section 2 predicts that a few hundred rows fix the format
     but not the facts.

### 4.5 Answer quality: human ratings

**[PENDING: human ratings, #14.]** Mean scores per criterion for zero-shot, r16_n775 and
r16_n100 from the four raters, and whether they rank the three systems in the same order as the
judge. Agreement between raters and between judge and humans is reported in
Section [Evaluation].

### 4.6 Quality against cost

**[PENDING: judge scores.]** Figure R3 plots quality against peak GPU memory. With validation
loss on the vertical axis, it already shows the shape of the trade-off: the five ranks spread
over 0.60 GB of memory and 0.019 of loss, while the four data sizes sit at the same memory and
spread over 0.104 of loss. Memory buys little here. The paragraph needs the judge scores to say
whether the same holds for answer quality.

### Figures

- **Figure R1.** Rank sweep on all 775 rows (`figures/quality_vs_rank.png`).
- **Figure R2.** Data-size sweep at r = 16 (`figures/quality_vs_data_size.png`).
- **Figure R3.** Quality against peak GPU memory (`figures/quality_vs_memory.png`).
- **Figure R4.** Training and validation loss for each run (`figures/loss_curves.png`).

Until the judge scores exist, the left panels of Figures R1 and R2 show the share of Devanagari
letters, and Figure R3 shows validation loss. They are regenerated with `python src/analyze.py`.

---

## 5. Conclusion

We asked how much Hindi instruction data and what LoRA rank is enough to make a 1.5B model
follow Hindi instructions on a free GPU. What the run files already show:

1. **Memory is not the constraint.** Every run fits in 6.5 GB on a 16 GB T4, and a full run
   takes about 30 minutes. Raising the rank from 4 to 64 costs 0.6 GB and no extra time.
2. **Data matters more than rank.** Across our sweeps, going from 100 to 775 rows lowers the
   validation loss more than five times as much as going from r = 4 to r = 64. The loss is still
   falling at 775 rows, which is all the cleaned Hindi data the Aya Dataset gives us.
3. **Higher rank needs fewer epochs.** With three epochs, r = 32 and r = 64 overfit in the last
   one. Choosing the epoch by validation loss, or training for two epochs, would probably serve
   the higher ranks better than our fixed three.
4. **A little data fixes the form of the answers.** The base model writes Hindi but does not
   stop: 103 of 150 answers run to the length limit. One hundred rows are enough to bring
   answer length close to the references.

**[PENDING: judge scores and human ratings.]** Whether these gains in loss and in form are also
gains in answer quality, which rank and data size are "enough", and whether correctness improves
or only format and fluency. The answer to the research question goes here, in two or three
sentences.

**Limits of these results.** Each setting was run once, so differences as small as those
between neighbouring ranks may be noise, and the N = 250 outlier cannot be explained. The
data-size sweep changes the number of training steps together with the number of rows. All runs
use one base model and one language, and at most 775 training rows. Section [Evaluation]
discusses the limits of the judge and the human ratings.

**Future work.** Repeat each setting with several seeds. Add a run with the epoch chosen by
validation loss. Raise the 1,024-token limit or use a tokenizer that is more efficient for Hindi
such as Sarvam-1's, so that fewer training rows lose their ending (Section [Data]). Add more
Hindi instruction data from other sources to find where the data-size curve flattens.
