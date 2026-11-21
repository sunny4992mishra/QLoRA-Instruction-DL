# Error analysis

*Draft (Ompal). The numbers come from `src/error_analysis.py`, which writes `results/error_analysis/`.*

The mean scores show how much fine-tuning helps but not what goes wrong. Here we sort every answer into one type of error.

## Method

We gave each of the 1,350 answers (nine systems × 150 test prompts) one category. Each answer gets the first category below that applies:

1. **Not in Hindi.** Under half of the answer's letters are Devanagari, and the answer contains at least three Latin-script words. Prompts whose reference answer is itself in English (one prompt asks for an English translation) are exempt. So are one-word answers such as "NPTEL" or a lone emoji.
2. **Doesn't stop: loops.** The answer reaches the 512-token generation limit and repeats some 8-word phrase three or more times.
3. **Correct enough.** The LLM judge gives correctness 3 or more.
4. **Doesn't stop: cut off.** The answer reaches the limit without such a loop.
5. **Fluent but wrong.** Judge correctness is 2 or less, and fluency is 4 or more.
6. **Restates the question.** Judge correctness is 2 or less, and the answer is mostly the instruction copied back. At least half of the answer's word 3-grams come from the instruction, and the answer repeats at least 60% of the instruction's 3-grams.
7. **Incoherent Hindi.** Judge correctness is 2 or less, and fluency is 2 or less.
8. **Vague or partly right.** Any other answer with judge correctness 2 or less.

Categories 1, 2 and 4 use only the answer text. We count the tokens of each answer again with the Qwen2.5 tokenizer. An answer that has 510 or more tokens "reaches the limit". Every answer has either 510 or more tokens or 504 or fewer. The ones near 504 end on a complete sentence, so they stopped on their own. The other categories depend on the LLM judge's scores (Section [Evaluation]), so they carry its errors.

The order of the rules matters in two places. First, "Correct enough" is not the same as judge correctness 3 or more. A looping answer counts as a loop even when the judge accepts its start: 4 zero-shot answers, and 1 each for r = 4 and r = 8, have judge correctness 3 or more but are counted as loops. Second, rule 3 comes before rule 4, so the 5 "Correct enough" answers that reach the limit are not counted as cut off.

## Results

**Table X1.** Answers per category, out of 150 per system.

| Category | Zero-shot | r=4 | r=8 | r=16 | r=32 | r=64 | N=100 | N=250 | N=500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Not in Hindi | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| Doesn't stop: loops | 79 | 38 | 31 | 31 | 27 | 14 | 36 | 64 | 22 |
| Doesn't stop: cut off | 22 | 7 | 15 | 17 | 13 | 20 | 5 | 6 | 10 |
| Fluent but wrong | 9 | 9 | 13 | 11 | 12 | 13 | 10 | 10 | 14 |
| Restates the question | 3 | 1 | 3 | 3 | 1 | 2 | 2 | 2 | 2 |
| Incoherent Hindi | 9 | 44 | 29 | 36 | 45 | 55 | 56 | 33 | 43 |
| Vague or partly right | 13 | 28 | 35 | 25 | 21 | 21 | 21 | 12 | 34 |
| Correct enough | 13 | 23 | 24 | 27 | 31 | 24 | 19 | 23 | 25 |

The rank columns use all 775 training rows. The N columns use r = 16, and the r=16 column is also the N=775 point of the data-size sweep.

**Not stopping is the main failure of the base model.** 103 of the 150 zero-shot answers run to the 512-token limit, and 79 of them repeat a phrase until they are cut off. Fine-tuning roughly halves this in every run except N=250: 32 to 49 answers reach the limit, against 70 for N=250. Reaching the limit is not always a mistake, because 26 of the 150 references are themselves longer than 512 tokens. But in every system, most of the answers that reach the limit are on prompts with shorter references (for zero-shot, 80 of 103).

- **Rank.** Looping is about the same from r = 4 to r = 16 and lower at r = 32 and r = 64: 38, 31, 31, 27 and 14 answers for r = 4 to 64. The drop at r = 32 and r = 64 holds at every phrase length we tried. With a 5-word phrase instead of 8, the counts are 40, 41, 37, 30 and 18.
- **Data size.** Looping does not fall steadily. N=250 loops on 64 answers, almost as many as zero-shot, while N=500 loops on only 22. With a single seed per setting, we can't tell whether N=250 is a real effect or run-to-run variation.

**English answers are rare.** Only two zero-shot answers and at most one answer per fine-tuned run are mostly in English. The base model already answers Hindi prompts in Hindi, so this failure barely matters on our test set.

**Fine-tuning trades loops for short answers the judge scores as incoherent.** Fine-tuning shortens the answers. The median answer drops from 541 characters to between 192 and 239 characters in every fine-tuned run except N=250, which is at 436 (`results/summary.md`). That matches the mostly short references in the Aya data. But many of the short answers make little sense. 9 zero-shot answers fall into the incoherent category, against 29 to 56 for every fine-tuned run. The zero-shot count is low partly because most of its failures are already counted as loops. Rank shows the same trade-off: r = 64 has the fewest loops (14) but the most incoherent answers of the five ranks (55). So a higher rank removes loops without making the answers correct. The judge's fluency score decides this category, and the human ratings (Section [Evaluation]) only partly agree with it. 11 of the 90 rated answers fall in this category. The raters also score them as wrong (mean human correctness 1.27), but they give them a mean fluency of 2.5, against 1.6 from the judge. Only 5 of the 11 have a mean human fluency of 2 or less. In the raters' comments, 5 of the 11 are wrong facts in readable Hindi, such as a wrong year or a misread emoji. Another 4 repeat one sentence but stop before the token limit, so the loop rule misses them. So this category mixes truly incoherent answers with fluent wrong answers and short loops.

**Wrong facts stated fluently** make up 9 to 14 answers in every system, and fine-tuning does not reduce them. These answers read well, which makes them the most misleading kind of error for a user.

**Restating the question** is rare as an error, at 1 to 3 answers per system. More answers meet the copy rule, 4 to 10 per system, but 54 of these 73 answers fall under an earlier rule. Most of them (40) the judge scores as correct enough, so repeating the question's words is often part of a correct answer rather than an error.

**Most answers are still wrong.** Even the best system (r = 32) gets judge correctness 3 or more on only 31 of 150 answers, against 17 for zero-shot. Part of this gain comes from the 35 test prompts that have a near-copy in training (Section [Data]). On those, zero-shot gets judge correctness 3 or more on 2 answers and the full-data runs on 7 or 8. On the other 115 prompts, the gain is smaller: 15 for zero-shot against 15 to 23 for the fine-tuned runs. Section [Results], 4.5, shows that the larger gain on the 35 is mostly not because of memorisation.

## Examples

The examples come only from test prompts that are not in the human-rating sample, so they don't unblind the raters. The English glosses are ours.

- **Loops (r=16, N=775).** *ताज महल कहाँ स्थित है?* (Where is the Taj Mahal?) The model says it was built by "Chandragupta". It then repeats the sentence *इस चंद्रगुप्त के अभिवादन के रूप में ताज महल भारतीय अभिवादन का एक छोटा सा खंड है* ("as this Chandragupta's greeting, the Taj Mahal is a small part of the Indian greeting") until the 512-token limit.
- **Loops (r=16, N=100).** *ग्लोबल वार्मिंग क्या है?* (What is global warming?) The model gives one confused definition ("a state … between modern mathematics, science and history") and repeats it four times.
- **Fluent but wrong (r=16, N=775).** *कोरोना वायरस 2019 के पहले मामले की पहचान कहां हुई थी?* (Where was the first case of COVID-19 identified?) The model answers "अमेरिका" (America); the reference says Wuhan, China. In another answer, it says Delaware has more electoral votes than Florida.
- **Fluent but wrong (r=16, N=100).** Asked on which date in 1945 the United Nations came into being, the model answers "26 January" instead of 24 October.
- **Restates the question (r=16, N=100).** *जितने का अर्थ और पैमाना क्या है?* (What is the meaning and scale of "jitne"?) The answer is *वह कि जितने का अर्थ और पैमाना है* ("that which is the meaning and scale of jitne"): the question turned into a statement.
- **Restates the question (r=16, N=775).** Asked who took charge of the Insolvency and Bankruptcy Board of India in 2001, the model repeats the whole question and inserts an invented name, "James Goodson". The question itself has a false premise, because the board was only set up in 2016, so any name would be wrong. The example still shows the restating.
- **Incoherent (r=16, N=775).** Asked to finish the story of the hare and the tortoise, the model writes that the hare "made his race his friend and won it".
- **Not in Hindi (r=64).** Asked what a Hindi essay is and how to write one, the model points to a WolframAlpha URL instead of answering.

## Limitations of this analysis

- Categories 3 and 5–8 depend on the LLM judge, and the judge makes mistakes. It gives fluency 5 to an answer that is only the emoji "😊". It also gives helpfulness 5 to the wrong Delaware answer.
- The judge is not consistent with itself. When two systems gave the identical answer, it scored 45% of the 258 pairs differently (correlation 0.80; Section [Evaluation], E.6). Categories 3 and 5–8 depend on these scores, so counts that differ by a few answers between neighbouring runs, such as 29, 33 and 36 incoherent answers, are not reliable differences.
- The thresholds (8-word phrase, 3 repeats, 50% and 60% 3-gram overlap) were picked by reading answers, not tuned. The loop counts depend on the phrase length. With a 5-word phrase instead of 8, zero-shot goes from 79 to 95 looping answers and r = 8 from 31 to 41. Only a sample of the extra matches was read, so not all of them are necessarily true loops. The overall pattern holds at 5, 6 and 8 words: zero-shot answers mostly loop, r = 64 loops least, and fine-tuned answers tend to be short but incoherent or wrong.
- Every setting was run once with seed 42. Differences of a few answers between two runs, such as r = 8 against r = 16, are within what a second seed could change.
- The human ratings (Section [Evaluation]) support the categories that do not depend on the judge's fluency score. None of the 38 rated answers counted as loops gets a mean human correctness of 3 or more, while 9 of the 12 "Correct enough" answers do. The 9 "Fluent but wrong" answers get a mean human fluency of 4.0 and correctness of 1.2. "Incoherent Hindi" agrees less well, as described in the results. The rated sample has only 90 answers from three systems (zero-shot, r = 16 and N = 100), so these are rough checks.
