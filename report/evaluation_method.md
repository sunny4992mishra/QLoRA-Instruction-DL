# P16 report: Evaluation method, judge vs human agreement, limitations

Draft for issue #10 (Shahid). Section, table and figure numbers are placeholders until the report
is assembled; tables here are numbered E1, E2, … Cross-references to other sections are marked
like `Section [Data]`, and to results tables as in `report/results_conclusion.md` (R1, R2, …).

**Status.** The judge setup, the human rating setup and the limitations are written, and every
number in them comes from files on `main` or from PR #28 (judge scores). The parts that need the
filled human rating sheets (#14) are marked **[PENDING]**. They are produced by
`python src/collect_ratings.py` and `python src/judge_vs_human.py`, and are not filled with guesses.

---

## Evaluation

We judge every system on the same 150 held-out test prompts, which the models never saw in
training. Validation loss and the form of the answers (length, cut-offs, looping; Section 4.3)
are cheap to compute but do not say whether an answer is right. For that we use two
measurements of answer quality: an LLM judge on all 150 answers of all nine systems, and human
ratings on a sample. Both score the same three criteria on a 1 to 5 scale.

### E.1 Test prompts and generated answers

The test set has 150 prompts from the Hindi part of the Aya Dataset. The train, validation and
test splits are made by prompt, so no test prompt appears in training (Section [Data]). 25 of the
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

The judge prompt and the rating sheets use the same three criteria and, in substance, the same
scale descriptions.

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
one request. The 10 items are scored independently, in test-set order, and all come from the same
system. The whole job took about 135 requests for 1,350 answers. A batch of 10 is a design
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
ratings until all four sheets were done. Like the judge, raters see the reference.

**Scoring.** For each answer the rater enters three whole numbers from 1 to 5 in cells that reject
other values, with an optional comment. Rating takes about 2.5 hours per rater. `src/collect_ratings.py`
reads the sheets, matches rows to the key by item number, and writes per-system means and the
agreement between raters. Per-answer scores are the mean over raters.

**[CHECK with the team: who the raters are.]** State whether each rater reads Hindi as a first or
fluent second language, since Hindi fluency is one of the criteria. Not claimed here.

### E.5 Agreement

**Between raters.** For each criterion we report Krippendorff's alpha (interval) over all raters,
the mean pairwise weighted kappa (quadratic weights), and the share of rater pairs that give the
same score or scores at most 1 apart on the same answer.

**Table E1 [PENDING: #14].** Agreement between the four raters, per criterion (output of
`python src/collect_ratings.py`: `results/human/summary.md`).

| Criterion | Krippendorff's α | Weighted κ (mean over pairs) | Exact | Within 1 |
|---|---:|---:|---:|---:|
| Correctness | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Helpfulness | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Hindi fluency | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

**Between the judge and the humans.** The judge and the humans are compared on the 90 rated
answers only. Per criterion we report:

- the weighted kappa between the judge and each rater, averaged over raters, next to the
  same statistic between raters. We read the judge as agreeing about as well as a human does if
  the two numbers are close. We chose this criterion before seeing any human ratings;
- the share of (judge, rater) pairs with the same score, and within 1;
- the Spearman correlation between the judge score and the mean human score of each answer;
- the mean difference, judge minus humans, which shows whether the judge is more lenient or harsher;
- whether the judge and the humans put the three systems in the same order.

**Table E2 [PENDING: #14].** Judge against humans, per criterion (output of
`python src/judge_vs_human.py`: `results/human/judge_vs_human.md`).

| Criterion | κ judge–rater | κ rater–rater | Exact | Within 1 | Spearman ρ | Judge − human | Same system order |
|---|---:|---:|---:|---:|---:|---:|---|
| Correctness | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Helpfulness | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Hindi fluency | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

**[PENDING: what the agreement means for the results.]** Two outcomes to write up. If the judge
agrees about as well as the raters agree with each other and gives the same system order, the
judge scores can be used for all nine systems, including the ones the humans did not rate. If
not, the judge scores are reported as indicative only and the conclusions rest on the human
ratings of the three systems. Either way, say which criterion agrees least, read from our numbers.

### E.6 Limitations

**One seed per setting.** Every run is trained once with seed 42, and every answer is one greedy
sample. We cannot separate the effect of rank or data size from run-to-run noise, and differences
between neighbouring ranks may be noise. The N = 250 run has far more long and looping answers
than N = 100 (Section 4.3) and one seed cannot say whether that is a property of 250 rows or
chance. The data-size sweep also changes the number of training steps together with the number of
rows (Section 3.5).

**Small data and small test sets.** Training uses at most 775 rows, the Hindi part of one dataset.
The test set has 150 prompts. For the judge's scores this gives a 95% interval of about ±0.2 on
the mean of one run (for example ±0.19 for the correctness of `r16_n775`). Paired over the same
150 prompts, `r64_n775` and `r4_n775` differ in correctness by −0.01 ± 0.15, so rank effects on
the judge's scores are not distinguishable from zero. The difference between `r16_n775` and the
zero-shot baseline is +0.26 ± 0.18 in correctness and +0.61 ± 0.21 in fluency. The human sample
is 30 prompts and 3 systems, so human results exist only for those three systems and the
intervals are wider still.

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
- *Reference and length.* The judge sees one reference and may favour answers that resemble it,
  and 25 prompts have more than one acceptable reference. Judges can also favour longer answers
  in general. Our prompt tells the judge not to reward length, but we did not measure the effect.
- *Free-tier model.* The model version behind a hosted free-tier name can change, so the judging
  may not be reproducible later. The scores we used are committed, and each row records the model
  name and date.
- *Facts.* A small judge cannot check facts it does not know. The reference helps, but for
  open-ended prompts with a different valid answer, a correct answer may be marked down.

**Human ratings.** The raters are the authors, who know the research question and may have an
expectation about which system is better, although the sheets hide the system. Agreement depends on how each rater read the scale (Table E1). The sample is small, and the raters see the reference, which can anchor their scores. **[CHECK:
whether the raters agreed on how to read the scale beforehand.]**

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
