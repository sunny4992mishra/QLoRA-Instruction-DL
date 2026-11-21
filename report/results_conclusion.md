# P16 report: Results, Conclusion

Draft for issue #7 (Sunny). Section, table and figure numbers are placeholders until the report
is assembled; tables and figures here are numbered R1, R2, …. Cross-references to other sections
are marked like `Section [Evaluation]`.

**Status.** Cost, validation loss, the form of the answers and the LLM-judge scores are written.
The human ratings (#14) are still marked **[PENDING]** and are not filled with guesses.

All numbers come from `runs/*/run_info.json`, `results/*.jsonl` and `results/judge/*.jsonl`.
They are reproduced by `python src/analyze.py` (Tables R1, R2, R4 and R5, Figures R1–R4) and
`python src/answer_stats.py` (Table R3). Memory is reported in GB as PyTorch counts it, with
1 GB = 1024³ bytes.

---

## 4. Results

We report cost first, then validation loss, then what the answers look like, and last the
quality scores from the LLM judge and the human raters. Every fine-tuned model is the adapter
from the end of epoch 3 (Section 3.5), and each setting was run once with the same seed.

### 4.1 Cost: memory, time and trainable parameters

**Table R1.** Cost of the eight fine-tuning runs on a Tesla T4. Peak GPU memory is the peak
allocated memory (`torch.cuda.max_memory_allocated`) during training. Training time includes the
validation pass after each epoch.

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
raises peak memory by only 0.60 GB, from 5.83 to 6.43 GB (about 10%). Loading the 4-bit base
model peaks at 1.10 GB, of which 1.07 GB stays allocated. Most of the training peak is
activations for the 1,024-token sequences, which do not depend on the rank. The 0.60 GB that
r = 64 adds over r = 4 is about what its 69 million extra adapter parameters need for fp32
weights, fp32 gradients and 8-bit optimizer state, roughly 10 bytes per parameter. Training time
does not depend on the rank: all five full-data runs take 29 to 31 minutes, or 12 to 13 seconds
per optimizer step.

**Data size sets the time, not the memory.** Peak allocated memory is 5.94 GB for every r = 16
run, whatever the number of rows, because it is set by the longest sequence in a batch and not
by the number of batches. Training time grows in proportion to the number of steps, from 4
minutes for 100 rows to 30 minutes for 775.

**Everything fits with room to spare.** The largest run peaks at 6.43 GB allocated. The memory
that PyTorch's allocator reserved varies more between runs, from 7.3 to 9.6 GB, and is still
well under the T4's 16 GB. Generating the zero-shot answers peaks at 1.98 GB (reported in #18;
the baseline has no `run_info.json`). GPU memory is therefore not the binding limit for this
model and data on a free GPU.

### 4.2 Validation loss

**Table R2.** Validation loss after each epoch (61 validation rows), and the training loss
averaged over the whole run. The last column is how far the epoch-3 loss is above the lowest of
the three.

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

**Higher ranks overfit more in the third epoch.** In every full-data run the validation loss is
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
rows still lower the loss by 0.020. Our training split ends there (Section [Data]), so we can
say that more Hindi data would probably still lower the loss, but not where the gains stop.

**With 100 rows, validation loss does not move.** It stays between 1.040 and 1.042 across the
three epochs while the training loss keeps falling, from 1.04 in the first logged steps to 0.81
in the last. The run has only 21 optimizer steps, so the model changes little and what it
learns from 98 rows does not transfer to the validation prompts. As Section 3.5 notes, smaller
subsets also get fewer updates, so this sweep measures data and training steps together.

Two limits apply to this subsection. We did not measure the validation loss of the base model
without an adapter, so Table R2 compares fine-tuned runs with each other and not with zero-shot.
And validation loss is not answer quality: it scores the probability of one reference answer
per prompt. Section 4.4 shows that the two do not always move together.

### 4.3 What the answers look like

**Table R3.** Length, cut-offs and looping in the 150 test answers per system, counted with the
Qwen2.5 tokenizer. "At the limit" means the answer ran to the 512-token generation limit.
"Looping" means the same 8-word phrase appears three or more times. For the reference answers,
"at the limit" counts those that are themselves 510 tokens or longer.

| System | Median tokens | At the limit | Looping | Mostly Devanagari |
|---|---:|---:|---:|---:|
| Reference answers | 201.5 | 26 | 2 | 149 |
| Zero-shot | 512.0 | 103 | 79 | 148 |
| r4_n775 | 197.0 | 46 | 43 | 147 |
| r8_n775 | 225.0 | 47 | 36 | 148 |
| r16_n775 | 187.0 | 49 | 35 | 148 |
| r32_n775 | 192.5 | 40 | 30 | 148 |
| r64_n775 | 194.5 | 34 | 15 | 147 |
| r16_n100 | 224.5 | 41 | 46 | 148 |
| r16_n250 | 417.0 | 70 | 71 | 147 |
| r16_n500 | 178.5 | 32 | 28 | 148 |

**The base model's problem is stopping, not language.** Zero-shot, 148 of 150 answers are
already mostly in Devanagari, but 103 run to the length limit and 79 loop. The median answer is
512 tokens, the limit itself, against 201.5 for the references.

**Fine-tuning teaches the model to stop.** Every fine-tuned run except N = 250 has a median
length between 178.5 and 225 tokens, close to the references, and cuts the number of answers at
the limit from 103 to between 32 and 49. Some long answers are legitimate: 26 of the references
are themselves 510 tokens or longer. That still leaves 6 to 23 more answers at the limit than
there are references of that length. Section 3.4 gives one likely cause: the 103 training rows
that are cut at 1,024 tokens show no stopping point.

**100 rows are enough to fix the stopping, but not the looping.** The N = 100 run already
brings the median to 224.5 tokens and the answers at the limit to 41, about the same as the
full-data runs, even though its validation loss barely moves (Section 4.2). This fits the view
in Section 2 that a small amount of instruction data mainly teaches the format of an answer. It
still loops in 46 answers, though, more than any full-data run.

**Looping falls with rank.** At full data, the number of looping answers goes down as the rank
goes up: 43, 36, 35, 30 and 15 for r = 4 to 64. The r = 64 model loops in 15 answers, against
79 zero-shot.

**N = 250 is an outlier.** Its answers have a median of 417 tokens, 70 run to the limit and 71
loop. That puts it between N = 100 and zero-shot, although its validation loss lies between
N = 100 and N = 500 as expected. The run passed every check in `check_run.py`, so the files are
valid. With one seed per setting we cannot tell whether this is a property of 250 rows or
chance, and repeating the run with other seeds would settle it. Section 4.4 shows how the judge
scores it.

These counts describe the form of the answers, not whether they are right.

### 4.4 Answer quality: LLM judge

The judge (`gemini-3.5-flash-lite`, Section [Evaluation]) scores every answer from 1 to 5 for
correctness, helpfulness and Hindi fluency.

**Table R4.** Mean judge score per system over the 150 test answers, with a 95% interval. "Mean"
averages the three criteria for each answer first.

| System | Correctness | Helpfulness | Fluency | Mean |
|---|---:|---:|---:|---:|
| Zero-shot | 1.46 ± 0.13 | 1.39 ± 0.12 | 2.05 ± 0.17 | 1.63 ± 0.12 |
| r4_n775 | 1.71 ± 0.19 | 1.70 ± 0.18 | 2.54 ± 0.20 | 1.98 ± 0.17 |
| r8_n775 | 1.69 ± 0.20 | 1.74 ± 0.19 | 2.65 ± 0.22 | 2.03 ± 0.18 |
| r16_n775 | 1.72 ± 0.19 | 1.79 ± 0.19 | 2.66 ± 0.21 | 2.06 ± 0.18 |
| r32_n775 | 1.81 ± 0.20 | 1.90 ± 0.21 | 2.67 ± 0.20 | 2.12 ± 0.19 |
| r64_n775 | 1.69 ± 0.18 | 1.77 ± 0.19 | 2.63 ± 0.19 | 2.03 ± 0.17 |
| r16_n100 | 1.55 ± 0.17 | 1.60 ± 0.16 | 2.35 ± 0.21 | 1.83 ± 0.16 |
| r16_n250 | 1.58 ± 0.18 | 1.60 ± 0.18 | 2.25 ± 0.22 | 1.81 ± 0.18 |
| r16_n500 | 1.73 ± 0.19 | 1.74 ± 0.18 | 2.76 ± 0.19 | 2.08 ± 0.17 |

All systems answer the same 150 prompts, so we compare them prompt by prompt. This gives a
tighter interval for a difference than the separate intervals in Table R4.

**Table R5.** Paired differences in the judge mean, A − B, with a 95% interval. \* marks an
interval that excludes 0. The full table, with each criterion, is in `results/summary.md`.

| A − B | Difference in the mean |
|---|---:|
| r16_n100 − zero-shot | +0.20 ± 0.14 * |
| r16_n250 − zero-shot | +0.18 ± 0.17 * |
| r16_n500 − zero-shot | +0.44 ± 0.16 * |
| r16_n775 − zero-shot | +0.42 ± 0.15 * |
| r4_n775 − zero-shot | +0.35 ± 0.17 * |
| r64_n775 − zero-shot | +0.40 ± 0.17 * |
| r16_n250 − r16_n100 | −0.02 ± 0.16 |
| r16_n500 − r16_n250 | +0.27 ± 0.14 * |
| r16_n775 − r16_n500 | −0.02 ± 0.14 |
| r8_n775 − r4_n775 | +0.05 ± 0.12 |
| r16_n775 − r8_n775 | +0.03 ± 0.14 |
| r32_n775 − r16_n775 | +0.07 ± 0.12 |
| r64_n775 − r32_n775 | −0.10 ± 0.15 |
| r64_n775 − r4_n775 | +0.05 ± 0.14 |

**All scores are low.** The best system averages 2.12 out of 5, and no system reaches 2 for
correctness. Over all 1,350 judged answers, 65% get the lowest score for correctness and 62% for
helpfulness. Fine-tuning improves the answers, but it does not make this model a reliable Hindi
assistant.

**Every fine-tuned run beats zero-shot.** All eight differences from zero-shot exclude 0, from
+0.18 for N = 250 to +0.49 for r = 32. For r16_n775 the gain is largest in fluency
(+0.61 ± 0.21), then helpfulness (+0.39 ± 0.17), and smallest in correctness (+0.26 ± 0.18). The
judge's instructions count looping and repeated phrases against fluency, so the fluency gain
partly reflects the drop in looping in Table R3.

**Rank makes no measurable difference (Figure R1, left panel).** No step from one rank to the
next, and not the whole range from r = 4 to r = 64 (+0.05 ± 0.14), is distinguishable from
zero. The small, steady fall in validation loss with rank (Section 4.2) does not show up in the
judge scores. Neither does the drop in looping: r = 64 loops in 15 answers and r = 16 in 35, and
the judge scores them the same.

**Data size helps in one step, between 250 and 500 rows (Figure R2, left panel).** The mean
goes 1.83, 1.81, 2.08 and 2.06 for N = 100, 250, 500 and 775. The step from 250 to 500 rows is
+0.27 ± 0.14, and the steps on either side of it are indistinguishable from zero. So the judge
sees no gain from the last 275 rows, although the validation loss still falls there.

**The judge does not single out N = 250.** Its mean equals that of N = 100 (−0.02 ± 0.16),
although it loops in 71 answers against 46. Its fluency is the lowest of the fine-tuned runs
(2.25), but not distinguishably lower than N = 100 (2.35).

**Correctness improves only with enough data.** With 500 or 775 rows the gain in correctness
over zero-shot is +0.23 to +0.35 and excludes 0. With 100 or 250 rows it does not (+0.09 ± 0.14
and +0.12 ± 0.18), while helpfulness already improves. This fits the view in Section 2 that a
little instruction data fixes the form of an answer before its content.

These scores come from one judge model, which is not fully consistent with itself, and from one
seed per setting. Section [Evaluation] discusses how far they can be trusted.

### 4.5 Answer quality: human ratings

**[PENDING: human ratings, #14.]** Mean scores per criterion for zero-shot, r16_n775 and
r16_n100 from the four raters (`results/human/summary.md`), and whether they rank the three
systems in the same order as the judge (zero-shot 1.63, r16_n100 1.83, r16_n775 2.06). Agreement
between raters and between judge and humans is reported in Section [Evaluation].

### 4.6 Quality against cost

Figure R3 plots the judge mean against peak GPU memory. The five ranks spread over 0.60 GB of
memory and their scores overlap: r = 4 at 5.83 GB scores 1.98 and r = 64 at 6.43 GB scores 2.03.
The four data sizes sit at the same memory, 5.94 GB, and differ by up to 0.27 in score. On this
model and data, extra memory spent on rank buys no measurable quality. What costs is data, and
it costs training time, not memory: 21 minutes for 500 rows against 4 for 100.

### Figures

- **Figure R1.** Rank sweep on all 775 rows: judge scores (left) and validation loss (right)
  (`figures/quality_vs_rank.png`).
- **Figure R2.** Data-size sweep at r = 16: judge scores (left) and validation loss (right)
  (`figures/quality_vs_data_size.png`).
- **Figure R3.** Judge mean against peak GPU memory (`figures/quality_vs_memory.png`).
- **Figure R4.** Training and validation loss for each run (`figures/loss_curves.png`).

---

## 5. Conclusion

We asked how much Hindi instruction data and what LoRA rank is enough to make a 1.5B model
follow Hindi instructions on a free GPU.

**The answer from our runs:** about 500 training rows and the lowest rank we tried, r = 4. With
fewer rows the judge sees a smaller gain; with more rows or a higher rank it sees none. "Enough"
here means enough to get what this setup can give. It does not mean good: the best system
averages about 2 out of 5 from the judge.

1. **Memory is not the constraint.** Every run peaks below 6.5 GB allocated, and at most 9.6 GB
   reserved, on a 16 GB T4, and a full run takes about 30 minutes. Raising the rank from 4 to 64
   costs 0.6 GB and no extra time.
2. **Data matters more than rank.** Going from 100 to 775 rows lowers the validation loss more
   than five times as much as going from r = 4 to r = 64, and only data size moves the judge
   scores. The loss is still falling at 775 rows, which is all of our training split; the other
   241 clean rows are held out for validation and testing.
3. **Higher ranks overfit sooner.** With three epochs, the validation loss of r = 32 and r = 64
   rises in the last one. Choosing the epoch by validation loss, or training for two epochs,
   might serve the higher ranks better than our fixed three.
4. **Fine-tuning fixes the form of the answers more than their content.** The base model writes
   Hindi but does not stop: 103 of 150 answers run to the length limit. One hundred rows bring
   the answer length close to the references. Correctness improves only from 500 rows on, and
   stays below 2 out of 5.
5. **Loss and form are not quality.** Validation loss falls steadily with rank and with the
   last 275 rows, and looping falls with rank, but the judge scores do not follow either.

**[PENDING: human ratings.]** Whether the four raters rank zero-shot, r16_n100 and r16_n775 in
the same order as the judge, in one or two sentences.

**Limits of these results.** Each setting was run once, so small differences may be noise and
the N = 250 outlier cannot be explained. The data-size sweep changes the number of training
steps together with the number of rows. All runs use one base model, one language and at most
775 training rows, and the quality scores come from one judge model. Section [Evaluation]
discusses the limits of the judge and the human ratings.

**Future work.** Repeat each setting with several seeds. Add a run with the epoch chosen by
validation loss. Raise the 1,024-token limit, or use a base model with a more efficient Hindi
tokenizer such as Sarvam-1, so that fewer training rows lose their ending (Section [Data]). Add
more Hindi instruction data from other sources to see whether quality rises again beyond 775
rows.
