# P16 report: Data

Draft for issue #4 (Pranjal). Section, table and figure numbers are placeholders until the report
is assembled; tables and figures here are numbered D1, D2, …. Other drafts refer to this section
as `Section [Data]`.

Numbers come from `data/split_stats.json`, `data/tokenizer_stats.json`,
`data/test_near_duplicates.json` and the split files in `data/splits/`, unless stated otherwise.

---

## Data

### Source

We use the Hindi part (`language_code == "hin"`) of the Aya Dataset (Singh et al., 2024),
released on Hugging Face as `CohereLabs/aya_dataset` under the Apache-2.0 license. It has 1,153
Hindi prompt–completion pairs, all in the dataset's train split. Its test split has no Hindi rows,
so we hold out our own test set.

Each row is marked with one of two annotation types. The dataset card describes *original
annotations* as "brand new prompts and completions written by annotators" and *re-annotations* as
"human edits of automatically generated prompts and completions". The two types often share a
prompt, as in this pair from our training split:

| Prompt | Original annotation | Re-annotation |
|---|---|---|
| खट्टे फलों में कौन सा एसिड होता है? | उत्तर: साइट्रिक एसिड | उत्तर: खट्टे फलों में सिट्रिक एसिड होता है। |

### Cleaning

`src/prepare_data.py` normalises line endings, strips whitespace and removes one pair of quotes
that wraps the whole text. It then drops rows in three steps (Table D1).

**Table D1.** Cleaning steps.

| Step | Rows |
|---|---:|
| Hindi rows in the Aya train split | 1,153 |
| Empty prompt or response | −0 |
| Prompt contains no Devanagari | −115 |
| Exact duplicate (prompt, response) pair | −22 |
| **Clean rows** | **1,016** (875 distinct prompts) |

The 115 prompts without Devanagari are not Hindi instructions. We checked them in the raw
dataset. 113 have a Hindi "completion" that translates the prompt instead of answering it:
"Create a dialogue between two friends…" is completed with "दो दोस्तों के बीच एक संवाद
बनाएं…". In all 113 the completion is about as long as the prompt (median length ratio 1.00,
range 0.50–1.67), which fits a translation and not an answer. We read the 11 with the most
extreme ratios and a sample of the rest, and each was a translation. 112 of these prompts are
English instructions. The other one explicitly asks for a Vietnamese sentence to be translated into
Hindi, a genuine translation task that is lost with the rest. Training on these rows would teach
the model to translate instructions instead of following them. The last 2 of the 115 are short
questions in romanised Hindi ("Bharat ki rajdhani kya h"). They are also lost as a side effect of
the Devanagari filter.

### Splitting

The split is made **by prompt**, not by row. All rows that share a prompt, after whitespace is
normalised, go to the same split, so no test prompt appears word for word in training. The 875
prompts are shuffled with seed 42. The first 150 become the test set, the next 50 the
validation set, and the remaining 675 the training set (Table D2).

**Table D2.** The split.

| Split | Prompts | Rows | Original annotations | Re-annotations |
|---|---:|---:|---:|---:|
| Train | 675 | 775 | 498 | 277 |
| Validation | 50 | 61 | 39 | 22 |
| Test | 150 | 150 prompts, 180 reference answers | – | – |

A prompt can have up to four rows. In the training split, 80 of the 675 prompts have more than
one: 64 have two, 12 have three and 4 have four. These are kept as separate training rows.

In the test set, each prompt has one entry with all its answers as references. Original
annotations come first, so the first reference is an original annotation where one exists. The
human raters see only this first reference. 25 test prompts have more than one reference: 22
have two, 1 has three and 2 have four.

The training file is stored in a fixed random order. A run with N training rows uses the first N,
so the data-size subsets are nested (100 ⊂ 250 ⊂ 500 ⊂ 775).

### What the data looks like

The prompts are short and the answers vary widely in length. In characters, the median training
instruction is 77 and the median response 189. The 90th percentile response is 1,081 characters
and the longest is 23,077. The test set is similar: median instruction 73.5 characters, median
reference 215.5.

The tasks are mixed: factual questions, Hindi grammar exercises, riddles, multiple choice,
fill-in-the-blank, short stories, letters and essays. Many answers are very short. 102 of the 775
training responses are under 50 characters, and 50 begin with "उत्तर:" ("Answer:"), a format
the fine-tuned models pick up (Section [Results]). Some open-ended prompts get a one-sentence
answer. "एक स्वच्छता अभियान की कहानी।" ("A story of a cleanliness campaign.") is answered with a
single sentence that summarises the campaign.

At the other end are long essays (निबंध) and Wikipedia-style sections whose prompt is only a
topic, such as "वाराणसी शिक्षा" ("Varanasi education"). The longest training response
(`aya-hi-4a5b4eb6dd`) is not an answer at all. The prompt asks for a letter thanking a friend, and
the response is a copied web page of CBSE Class 10 question papers. These long rows are the ones
most affected by the length limit described next.

### Tokenization and the 1,024-token limit

The Qwen2.5 tokenizer splits Hindi into short pieces. On the training text, it averages **1.09
characters per token**. Sarvam-1, a 2B model built for Indian languages, averages 3.67, so Qwen2.5
needs 3.35 times as many tokens for the same text (413,967 against 123,511 tokens). The fixed
Hindi system prompt and the chat template add 93 tokens to every example.

We train with sequences of at most 1,024 tokens, using the rules in Section 3.4: a row is dropped
if its prompt alone is 1,024 tokens or longer, and its response is cut at the limit if prompt and
response together are longer. Table D3 shows the effect, and Figure D1 the length distribution.

**Table D3.** Training examples at the 1,024-token limit (system prompt, instruction and response,
in the chat template). Lost tokens are response tokens that the model never trains on.

| | Qwen2.5-1.5B-Instruct | Sarvam-1 |
|---|---:|---:|
| Characters per token | 1.09 | 3.67 |
| Median / 95th percentile / longest example (tokens) | 363 / 1,913 / 21,037 | 115 / 616 / 7,288 |
| Rows dropped (prompt ≥ 1,024 tokens) | 9 of 775 | 0 |
| Rows truncated | 103 of 775 | 6 |
| Rows that fit completely | 663 of 775 | 769 |
| Response tokens lost | 112,736 of 327,008 (34.5%) | 9,814 of 98,710 (9.9%) |

**Figure D1.** Tokens per training example for both tokenizers, with the 1,024-token limit
(`figures/token_lengths.png`). Examples longer than 3,072 tokens are counted in the last bar.

Of the 9 dropped rows, 8 are reading-comprehension tasks: a long passage followed by one or more
questions ("निम्नलिखित गद्यांश को ध्यानपूर्वक पढ़िए…", "Read the following passage
carefully…"). The ninth asks for a summary of a news article. Each prompt alone is over 1,024
tokens.

A third of the response tokens is lost, but the loss is concentrated. The 25 rows that lose the
most account for 69.2% of the lost tokens. 22 of them are the long essays, Wikipedia-style
sections and copied web page described above. The truncated rows still teach
the beginning of their answer. What they lose besides the length is the end-of-turn token: 103
training rows never show the model where to stop (Section 3.4). The validation split is affected
in the same way: 8 of its 61 rows are truncated and 24.1% of its response tokens are lost, so
validation loss is also computed on cut answers.

With Sarvam-1's tokenizer, all but 6 rows would fit. A model with a more efficient Hindi tokenizer
would avoid most of the truncation at the same sequence length.

### Near-duplicates between test and training data

Splitting by exact prompt does not catch prompts that differ only slightly. Re-annotations often
rephrase a prompt in small ways: a prefix such as "इस प्रश्न का उत्तर दें:" ("Answer this
question:"), different punctuation, or a spelling variant such as पडती for पड़ती. So a near-copy of
a test example can be in the training split. `src/find_near_duplicates.py` flags a test prompt
when its prompt has character 4-gram Jaccard similarity of at least 0.8 with a training prompt,
or its reference has a `difflib` similarity of at least 0.9 with a training response.

- **24** of the 150 test prompts have a near-identical training prompt. For example, the test
  prompt "इस प्रश्न का उत्तर दें: प्रश्न: भारत की सबसे ऊची पर्वतश्रेणी कौन सी है?" is in training
  without its first word.
- **28** have a reference answer that is nearly identical to a training response.
- **17** are flagged both ways, so **35** test prompts (23%) are flagged in total.

We did not change the split, because all runs already use it, and changing it would make them
incomparable. The 35 flagged ids are listed in `data/test_near_duplicates.json`. They can be left
out of the scores as a check: if fine-tuned models score higher on these prompts than on the
rest, part of their gain is memorisation and not instruction following. The human rating sample
includes 8 of the 35 (6 flagged by their reference).

---

## References

- Singh, S., et al. (2024). Aya Dataset: An open-access collection for multilingual instruction
  tuning. In *ACL 2024*. arXiv:2402.06619.
