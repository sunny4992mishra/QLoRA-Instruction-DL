# Error analysis (issue #12)

Written by `src/error_analysis.py`. Each answer gets one category; the rules and their order are in the script's docstring. Not in Hindi, loops and cut off use only the answer text; the other categories use the LLM judge's scores.

## Answers per category (out of 150 per run)

| Category | zeroshot | r4_n775 | r8_n775 | r16_n775 | r32_n775 | r64_n775 | r16_n100 | r16_n250 | r16_n500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Not in Hindi | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| Doesn't stop: loops | 79 | 38 | 31 | 31 | 27 | 14 | 36 | 64 | 22 |
| Doesn't stop: cut off | 22 | 7 | 15 | 17 | 13 | 20 | 5 | 6 | 10 |
| Fluent but wrong | 9 | 9 | 13 | 11 | 12 | 13 | 10 | 10 | 14 |
| Copies the instruction | 3 | 1 | 3 | 3 | 1 | 2 | 2 | 2 | 2 |
| Incoherent Hindi | 9 | 44 | 29 | 36 | 45 | 55 | 56 | 33 | 43 |
| Vague or partly right | 13 | 28 | 35 | 25 | 21 | 21 | 21 | 12 | 34 |
| Correct enough (judge >= 3) | 13 | 23 | 24 | 27 | 31 | 24 | 19 | 23 | 25 |
| Answers | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 |

## Flags (an answer can have several)

| Flag | zeroshot | r4_n775 | r8_n775 | r16_n775 | r32_n775 | r64_n775 | r16_n100 | r16_n250 | r16_n500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reaches the 512-token limit | 103 | 46 | 47 | 49 | 40 | 34 | 41 | 70 | 32 |
| ... and the reference is also over 512 tokens | 23 | 10 | 16 | 18 | 14 | 13 | 12 | 18 | 9 |
| Repeats an 8-word phrase 3+ times | 79 | 43 | 36 | 35 | 30 | 15 | 46 | 71 | 28 |
| Copies the instruction (both 3-gram shares) | 4 | 9 | 10 | 8 | 8 | 6 | 8 | 10 | 10 |
| Under half of the letters Devanagari, 3+ Latin words | 3 | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 |
| Judge correctness <= 2 | 133 | 126 | 125 | 123 | 119 | 126 | 131 | 127 | 125 |

References over 512 tokens: 26 of 150 test prompts.
