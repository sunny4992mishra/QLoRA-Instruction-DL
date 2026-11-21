# P16 report: Evaluation method, judge vs human agreement, limitations

Draft for issue #10 (Shahid). Section, table and figure numbers are placeholders until the report
is assembled; tables here are numbered E1, E2, … Cross-references to other sections are marked
like `Section [Data]`, and to results tables as in `report/results_conclusion.md` (R1, R2, …).

**Status.** Complete. Every number comes from files on `main`. Tables E1 to E3 are the outputs of
`python src/collect_ratings.py` and `python src/judge_vs_human.py`, run on the four filled rating
sheets. Table E4 is worked out separately from `results/human/scores.jsonl` and `results/judge/`,
without the prompts of items 2 and 9.

---

## Evaluation

We judge every system on the same 150 held-out test prompts. No test prompt is an exact copy of a training
prompt, but 35 of the 150 have a near-copy in training (see E.6). Validation loss and the form of the answers (length, cut-offs, looping; Section 4.3)
are cheap to compute but do not say whether an answer is right. For that we use two
measurements of answer quality: an LLM judge on all 150 answers of all nine systems, and human
ratings on a sample. Both score the same three criteria on a 1 to 5 scale.

### E.1 Test prompts and generated answers

The test set has 150 prompts from the Hindi part of the Aya Dataset. The train, validation and
test splits are made by prompt, so no test prompt appears in training word for word (Section [Data]). Near-copies exist (E.6). 25 of the
test prompts have more than one reference answer. The judge and the human raters see only the
first reference of each prompt.

All systems answer with the same settings: greedy decoding, repetition penalty 1.1 and at most
512 new tokens (Section 3.8). Each fine-tuned system is the adapter from the end of epoch 3
(Section 3.5). The zero-shot baseline is the same base model without an adapter. There is one
answer per prompt and system.

### E.2 Criteria

Every answer gets three whole-number scores from 1 to 5:

- **Correctness.** Are the facts and the content right? The reference is one good answer, not the
  only one. 1 is wrong or invented, 3 is partly right, 5 is fully correct.
- **Helpfulness.** Does the answer do what the instruction asks, completely and at a sensible
  length? An answer that is cut off, loops, or ignores part of the instruction scores lower.
- **Hindi fluency.** Is it natural, grammatical Hindi in Devanagari? Fluency is judged on the
  language only, not on the facts.

The judge prompt and the rating sheets use the same three criteria and closely matching 1 to 5 anchors (the wording differs slightly),
with one difference that matters for looping and cut-off answers. The rating sheet counts a
cut-off or looping answer against **helpfulness** and asks for fluency on the language only. The
judge prompt counts repeated phrases and looping text against **fluency**, says a looping answer
can still be correct in its first part (correctness), and says nothing about looping under
helpfulness. 42 of the 90 rated answers loop (the 8-word repeat rule of `answer_stats.py`), so
this difference can lower judge–human agreement on helpfulness and fluency for reasons that are
not about the quality of the judge (E.5). The judging was finished before this was noticed, and
we did not change either text afterwards.

### E.3 LLM judge

**Model.** `gemini-3.5-flash-lite`, called through the Gemini API on its free tier on
2026-11-21. The same model, prompt and settings are used for all nine files, so scores are
comparable between systems. The judge is not one of the systems being evaluated, and it is not a
Qwen model, so it does not score its own output.

**Prompt.** The prompt is fixed and is stored as `PROMPT` in `src/judge.py`. It tells the judge
to be strict and impartial, to score the answer and not the reference, to use the whole range,
not to reward length, and to count looping text and unwanted switches to another language against
fluency. For each item the judge sees the question, the reference answer and the answer. It does
not see which system wrote the answer. It replies with JSON only, one object of three scores per
item. Decoding is at temperature 0.

**Batching.** To keep the number of requests low on a free tier, the script sends 10 answers in
one request. The prompt asks the judge to score each item on its own. The 10 items are in test-set order and
all come from the same system. Every run sends the same 10 prompts in the same order in each
batch, so the position of each prompt in a batch is the same for all runs. The neighbouring answers differ between runs, so neighbour effects do not cancel (E.6). The whole job took about 135 requests for 1,350 answers. A batch of 10 is a design
choice that can affect scores (Section E.6).

**Inputs and failures.** Fields are clipped at 3,000 characters. This touched the reference of 3
prompts and no question or answer. All 1,350 answers received three scores, with none missing
(`results/judge/<system>.jsonl`, one line per answer, with the judge model and date).

**How the model was chosen.** The judge had to be free to use, not Qwen2.5-1.5B and the same for
every run. We first tried `gemma-4-31b-it:free` on OpenRouter. It was throttled heavily by the
provider, and it gave the top score 5/5/5 to a one-line answer that was awkward and thin next to
a long reference, also when that answer was scored alone, so batching was not the cause. On the
same five zero-shot answers `gemini-3.5-flash-lite` gave that answer 3/2/3 and the others 1 or 2.
Larger Gemini Flash models (`3.5-flash`, `3.6-flash`, `3.8-flash`) did not respond or returned
errors on the day, so we used a Flash-Lite model. We did not compare judges on a larger sample
and did not check the choice against human ratings before running it.

### E.4 Human ratings

**Sample.** Four raters, the four authors, rate the same 90 answers: 30 test prompts times three
systems. The 30 prompts are drawn at random from the 150 test prompts with a fixed seed (42). The
three systems were fixed in advance, before any judge scores existed: the zero-shot baseline,
`r16_n775` (r = 16 on all data) and `r16_n100` (r = 16 on 100 rows). They cover the base model,
the default setting and the smallest data size. They do not cover the rank sweep or N = 250 and 500.

**Blinding.** The 90 answers are shuffled into one order, and every rater gets the same order in
an Excel sheet. The sheet shows the item number, the instruction, the reference answer and the
answer, and nothing that names the system. The key that maps items to systems is kept in a
separate file (`results/human/key.json`), and raters were asked not to open it or to compare
ratings until all four sheets were done. All four raters confirmed that they did not
open the key until they had finished rating. Pranjal then ran `collect_ratings.py` on his own sheet
to check it, which loads the key, so he saw his own scores per system after he had finished. All four raters
are authors, and some had worked with aggregate results for the systems before or while rating, so
their scores may reflect those expectations. Like the judge, raters see the reference.

**Scoring.** For each answer the rater enters three whole numbers from 1 to 5 in cells that reject
other values, with an optional comment. A blank cell counts as missing. Pranjal and Sunny left Hindi fluency blank on items 2 and 9, where
the correct output is not Hindi (an English translation and the single word "NPTEL"), and Ompal and
Shahid gave a fluency score on both. So fluency on those two answers comes from two raters. Fluency on the other 88 answers, and
correctness and helpfulness on all 90, come from four raters. `src/collect_ratings.py`
reads the sheets, matches rows to the key by item number, and writes per-system means and the
agreement between raters. Per-answer scores are the mean over raters.

**Raters.** Hindi is the first language of all four raters. Hindi fluency is one of the criteria,
so this matters. The raters did not calibrate: we held no session
and rated no examples together, and each rater worked from the Instructions tab of the sheet
alone. The rater–rater agreement in Table E1 therefore measures how well the written scale works
on its own.

### E.5 Agreement

**Between raters.** For each criterion we report Krippendorff's alpha (interval) over all raters,
the mean pairwise weighted kappa (quadratic weights), and the share of rater pairs that give the
same score or scores at most 1 apart on the same answer.

**Table E1.** Agreement between the four raters, per criterion (output of
`python src/collect_ratings.py`: `results/human/summary.md`).

| Criterion | Krippendorff's α | Weighted κ (mean over pairs) | Exact | Within 1 |
|---|---:|---:|---:|---:|
| Correctness | 0.80 | 0.80 | 70% | 95% |
| Helpfulness | 0.78 | 0.78 | 60% | 98% |
| Hindi fluency | 0.78 | 0.78 | 73% | 96% |

The raters agree well on all three criteria, although they worked from the written scale alone.
The pairwise κ values range from 0.68 to 0.96. One pair of raters has κ = 0.96 on all three
criteria (89 to 90% exact), and the other five pairs have κ between 0.68 and 0.84. The mean over
pairs is therefore higher than most pairs: without the highest pair it is 0.77 for correctness and 0.74
for helpfulness and for fluency.

**Between the judge and the humans.** The judge and the humans are compared on the 90 rated
answers only. Per criterion we report:

- the weighted kappa between the judge and each rater, averaged over raters, next to the
  same statistic between raters. We read the judge as agreeing at least as well as a human does if its mean
  κ with the raters is at least the lowest of the six rater–rater κ values. A judge κ above the
  highest rater–rater value also passes: a judge close to the raters' consensus can agree with each
  rater more than they agree with one another. We chose this criterion before combining or
  comparing any ratings;
- the share of (judge, rater) pairs with the same score, and within 1;
- the Spearman correlation between the judge score and the mean human score of each answer;
- the mean difference, judge minus humans, which shows whether the judge is more lenient or harsher;
- whether the judge and the humans put the three systems in the same order.

**Table E2.** Judge against humans, per criterion (output of
`python src/judge_vs_human.py`: `results/human/judge_vs_human.md`).

| Criterion | κ judge–rater | κ rater–rater (min to max) | Judge κ ≥ lowest pair | Exact | Within 1 | Spearman ρ | Judge − human | Same system order |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Correctness | 0.82 | 0.80 (0.71 to 0.96) | yes | 71% | 96% | 0.73 | −0.02 | yes |
| Helpfulness | 0.76 | 0.78 (0.68 to 0.96) | yes | 66% | 97% | 0.64 | −0.01 | yes |
| Hindi fluency | 0.71 | 0.78 (0.70 to 0.96) | yes | 45% | 92% | 0.80 | −0.35 | no |

**Table E3.** Mean score per system on the 90 rated answers, judge and humans (same output file).

| System | Correctness judge / human | Helpfulness judge / human | Hindi fluency judge / human |
|---|---:|---:|---:|
| zero-shot | 1.47 / 1.46 | 1.33 / 1.41 | 1.80 / 2.18 |
| r16_n100 | 1.63 / 1.66 | 1.70 / 1.64 | 2.60 / 3.06 |
| r16_n775 | 1.90 / 1.93 | 1.93 / 1.96 | 2.67 / 2.88 |

**What the agreement means.** By the rule fixed in advance (the judge's mean κ with the raters is at
least the lowest of the six rater–rater κ values), the judge passes on all three criteria. The
margins differ. Compared with the rater mean without the highest pair (0.77, 0.74 and 0.74), the
judge (0.82, 0.76 and 0.71) is above the typical pair for correctness and helpfulness, and below it
for fluency.

- **Correctness.** The judge agrees with the raters about as well as they agree with each other
  (κ 0.82 against a mean of 0.80). The mean difference is −0.02. The judge and the humans
  put the three systems in the same order, but see the check below.
- **Helpfulness.** The judge is a little below the rater mean (0.76 against 0.78) and well inside
  the range. The mean difference is −0.01. Its rank correlation with the
  mean human score is the lowest of the three (ρ = 0.64).
- **Hindi fluency.** This agrees least. The judge's κ is 0.71 against 0.78, only just above the
  lowest rater pair (0.70). Exact agreement is 45%, and the judge scores 0.35 lower than the
  humans on average, for all three systems. It also orders r16_n100 and r16_n775 the other way
  round: humans 3.06 and 2.88, judge 2.60 and 2.67. Both gaps are small (the human intervals are
  about ±0.4 for 30 prompts), so this is a weak disagreement. The judge's prompt counts repeated
  phrases and looping text against fluency, and the rating sheet asks for the language only
  (E.2). On the 42 rated answers that loop (the 8-word rule), the judge's fluency is 0.64 lower
  than the humans', against 0.10 lower on the other 48, so the looping difference explains most of
  the gap. For all three systems the judge is 0.60 to 0.68 lower on looping answers. The order of
  r16_n100 and r16_n775 comes from the answers that do not loop, where the judge scores r16_n100
  lower than the humans do (3.26 against 3.62) and r16_n775 about the same (3.59 against 3.51).

**The order of the two fine-tuned systems depends on two answers.** Items 2 and 9 are `r16_n775`
answers that every rater scored 5 for correctness and helpfulness (an English translation and the
single word "NPTEL"). Without those two prompts (28 prompts), the order of the two fine-tuned
systems no longer shows agreement on correctness or helpfulness (Table E4). What holds in every
version is that the judge and the humans both put zero-shot lowest. We therefore do not read the
order of r16_n100 and r16_n775 as agreement on any criterion. On fluency the order of the two
matches in the 28-prompt version (humans 3.10 and 2.73, judge 2.57 and 2.50), so that
disagreement in Table E2 also depends on these two answers.

**Table E4.** Mean score on the 28 rated prompts without items 2 and 9, humans / judge.

| System | Correctness | Helpfulness | Hindi fluency |
|---|---:|---:|---:|
| zero-shot | 1.47 / 1.46 | 1.42 / 1.32 | 2.19 / 1.64 |
| r16_n100 | 1.70 / 1.68 | 1.67 / 1.75 | 3.10 / 2.57 |
| r16_n775 | 1.71 / 1.68 | 1.74 / 1.71 | 2.73 / 2.50 |

On correctness the two fine-tuned systems are nearly tied for both the humans (1.70, 1.71) and the
judge (1.68, 1.68). On helpfulness the humans put r16_n775 higher (1.67, 1.74) and the judge puts
r16_n100 higher (1.75, 1.71).

The judge and the humans agree on the comparison with zero-shot, and the judge scores for
correctness and helpfulness can be used to compare each system with zero-shot, including the six
that the humans did not rate. The 30-prompt sample cannot check the judge's comparisons between
fine-tuned systems, so the rank and data-size findings in Section [Results] rest on the judge
alone, with the limits in E.6. For fluency, the judge and the humans
agree that zero-shot is lowest, but the judge's level is lower than the humans' and its order of the
two fine-tuned systems is not reliable. We therefore do not read small fluency differences between
fine-tuned runs from the judge. Everything here rests on three systems and 30 prompts, with
four raters who are the authors (E.6).

### E.6 Limitations

**One seed per setting.** Every run is trained once with seed 42, and every answer is one greedy
sample. We cannot separate the effect of rank or data size from run-to-run noise, and differences
between neighbouring ranks may be noise. The N = 250 run has far more long and looping answers
than N = 100 (Section 4.3) and one seed cannot say whether that is a property of 250 rows or
chance. The data-size sweep also changes the number of training steps together with the number of
rows (Section 3.5).

**Small data and small test sets.** Training uses at most 775 rows, drawn from the 1,016 clean Hindi rows of one dataset (the
rest are in validation and test).
The test set has 150 prompts. For the judge's scores this gives a 95% interval of about ±0.2 on
the mean of one run (for example ±0.19 for the correctness of `r16_n775`). Paired over the same
150 prompts, `r64_n775` and `r4_n775` differ in correctness by −0.01 ± 0.15, so rank effects on
the judge's scores are not distinguishable from zero. The difference between `r16_n775` and the
zero-shot baseline is +0.26 ± 0.18 in correctness and +0.61 ± 0.21 in fluency. The human sample
is 30 prompts and 3 systems, so human results exist only for those three systems and the
intervals are wider still.

**Near-duplicates between test and training.** The split is by prompt, so no prompt is shared,
but 35 of the 150 test prompts have a near-copy in the training data, 8 of them in the human
sample. 10 of the 50 validation prompts do too. A fine-tuned model may answer these better
because it saw a close variant, which can flatter the fine-tuned systems against the zero-shot
baseline. See Section [Data] and Section 4.5 for the check.

**Judge bias.** An LLM judge is not a human.

- *One judge, one model size.* We used a Flash-Lite model, chosen after one failed attempt and
  because larger models were not available on the day, without validating the choice against
  humans first. Its agreement with the raters is reported in Table E2, for three systems only.
- *Floor effect.* The fine-tuned models are weak, so most scores sit at the bottom of the scale.
  For `r16_n775`, 63% of correctness scores and 59% of helpfulness scores are 1, and mean
  correctness over the nine systems is only 1.46 to 1.81. A judge can barely tell a bad answer
  from a worse one at the bottom of the scale, so small differences between systems are hard to
  read.
- *Batching.* The judge sees 10 answers of one system in one request. A judge can compare items
  within a request, so one answer's score may depend on its neighbours. We did not test a batch
  size of 1 on the full set (we tried it for one answer only, with another model).
- *Consistency.* The test set has 258 pairs of identical answers (43 prompts), found when
  checking the committed scores. The judge gave different scores to the two copies in 45% of the pairs, and the two sets of
  scores correlate at 0.80. This is the clearest single number for the noise in the judge, even at
  temperature 0.
- *Reference and length.* The judge sees one reference and may favour answers that resemble it,
  and 25 prompts have more than one acceptable reference. Judges can also favour longer answers
  in general. Our prompt tells the judge not to reward length. We checked it on the committed scores.
  Among the 847 answers that neither loop nor reach the token limit, the correlation between
  length and the mean judge score is −0.31, so longer answers tend to score lower. Answers that
  reach the limit average 1.34, against 2.27 for the rest. Those answers are
  also the looping and cut-off ones, so this is not a clean test of length alone.
- *Free-tier model.* The model version behind a hosted free-tier name can change, so the judging
  may not be reproducible later. The scores we used are committed, and each row records the model
  name and date.
- *Facts.* A small judge cannot check facts it does not know. The reference helps, but for
  open-ended prompts with a different valid answer, a correct answer may be marked down.

**Human ratings.** The raters are the authors, who know the research question and may have an
expectation about which system is better, although the sheets hide the system. Agreement depends on how each rater read the scale (Table E1), and the raters were not calibrated (E.4). One pair of raters agrees far more with each other than the other pairs do (E.5), so the mean rater agreement may overstate how well four independent readers would agree. The sample is small, and the raters see the reference, which can anchor their scores.

**Truncated training rows.** Training is limited to 1,024 tokens, and the Qwen2.5 tokenizer
splits Hindi into very many tokens (1.09 characters per token on our data). 103 of 775 training
rows are cut at the limit, and 9 more are dropped because the prompt alone is too long. Together
this loses 34% of all response tokens (Section [Data]). For the cut rows, the response is cut off
before the end-of-turn token, so these rows show the model an answer that never ends. This is a
likely cause of answers that run to the 512-token limit (Section 4.3), and it affects every
fine-tuned system in the same way. It limits what any rank or data size can reach with this model
and tokenizer, so our results describe this setup and not QLoRA on Hindi in general.

**Scope.** All runs use one base model (Qwen2.5-1.5B-Instruct), one language and one training
recipe, except for rank and data size. The base model's documentation does not name Hindi among
its supported languages (Section 3.1). We evaluate epoch-3 adapters, which are slightly
over-trained for the higher ranks (Section 4.2). We did not measure the base model's validation
loss, safety or other abilities.
