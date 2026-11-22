# Abstract

We ask how much Hindi instruction data and what LoRA rank are enough to make a small open model
follow instructions in Hindi on a free GPU. We fine-tune Qwen2.5-1.5B-Instruct with QLoRA on up to
775 human-annotated Hindi rows from the Aya Dataset, sweeping the rank from 4 to 64 and the data
size from 100 to 775 rows, and compare the runs with the zero-shot model on 150 held-out prompts.
Every run fits on a free 16 GB T4 and takes at most about 30 minutes. An LLM judge scores all
answers, and the four authors rate a blind sample of 90. Fine-tuning helps: rank 16 with all the
data scores 2.06 out of 5 against 1.63 for zero-shot, and the human raters also score the
fine-tuned systems above zero-shot. The gain is mostly in the form of the answers, and correctness
stays below 2 for every system. By the judge, we could not measure a benefit from any rank above 4,
and among the fine-tuned runs the only clear data step is from 250 to 500 rows, starting from an
unexplained outlier at 250. The Qwen2.5 tokenizer splits Hindi finely, so 34.5% of response tokens
are lost at our 1,024-token limit.
