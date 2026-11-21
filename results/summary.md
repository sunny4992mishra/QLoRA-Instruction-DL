| Run | Rank | Train rows | Trainable params | Trainable % | Peak GPU mem (GB) | Train time (min) | Val loss | Answers | Devanagari % | Median chars | Judged | Correctness | Helpfulness | Fluency | Judge mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct_zeroshot | – | – | – | – | – | – | – | 150 | 94.8 | 541 | 150 | 1.46 | 1.39 | 2.05 | 1.63 |
| Qwen2.5-1.5B-Instruct_r4_n775 | 4 | 775 | 4,616,192 | 0.298 | 5.83 | 30.3 | 0.946 | 150 | 99.6 | 209 | 150 | 1.71 | 1.70 | 2.54 | 1.98 |
| Qwen2.5-1.5B-Instruct_r8_n775 | 8 | 775 | 9,232,384 | 0.595 | 5.87 | 30.4 | 0.942 | 150 | 99.2 | 236 | 150 | 1.69 | 1.74 | 2.65 | 2.03 |
| Qwen2.5-1.5B-Instruct_r16_n100 | 16 | 100 | 18,464,768 | 1.182 | 5.94 | 4.0 | 1.041 | 150 | 98.9 | 239 | 150 | 1.55 | 1.60 | 2.35 | 1.83 |
| Qwen2.5-1.5B-Instruct_r16_n250 | 16 | 250 | 18,464,768 | 1.182 | 5.94 | 10.7 | 0.988 | 150 | 99.4 | 436 | 150 | 1.58 | 1.60 | 2.25 | 1.81 |
| Qwen2.5-1.5B-Instruct_r16_n500 | 16 | 500 | 18,464,768 | 1.182 | 5.94 | 20.8 | 0.957 | 150 | 98.7 | 192.5 | 150 | 1.73 | 1.74 | 2.76 | 2.08 |
| Qwen2.5-1.5B-Instruct_r16_n775 | 16 | 775 | 18,464,768 | 1.182 | 5.94 | 30.2 | 0.937 | 150 | 99.2 | 197 | 150 | 1.72 | 1.79 | 2.66 | 2.06 |
| Qwen2.5-1.5B-Instruct_r32_n775 | 32 | 775 | 36,929,536 | 2.336 | 6.13 | 31.0 | 0.932 | 150 | 99.5 | 205.5 | 150 | 1.81 | 1.90 | 2.67 | 2.12 |
| Qwen2.5-1.5B-Instruct_r64_n775 | 64 | 775 | 73,859,072 | 4.566 | 6.43 | 28.8 | 0.928 | 150 | 98.1 | 205.5 | 150 | 1.69 | 1.77 | 2.63 | 2.03 |

Peak GPU memory is `torch.cuda.max_memory_allocated` during training (GPU 0). Judge scores are means over answers on a 1–5 scale; the judge mean averages the three criteria per answer. 95% intervals are in summary.csv. Median characters is the mean of the two middle values when the number of answers is even (`statistics.median`).

## Paired differences in judge scores

Each row is A − B over the test prompts both runs were judged on, with a 95% interval. \* marks an interval that excludes 0.

| A − B | Correctness | Helpfulness | Fluency | Judge mean | Prompts |
|---|---:|---:|---:|---:|---:|
| r4_n775 − zeroshot | +0.25 ± 0.18 * | +0.31 ± 0.19 * | +0.49 ± 0.22 * | +0.35 ± 0.17 * | 150 |
| r8_n775 − zeroshot | +0.23 ± 0.19 * | +0.35 ± 0.18 * | +0.61 ± 0.24 * | +0.40 ± 0.17 * | 150 |
| r16_n100 − zeroshot | +0.09 ± 0.14 | +0.21 ± 0.14 * | +0.30 ± 0.20 * | +0.20 ± 0.14 * | 150 |
| r16_n250 − zeroshot | +0.12 ± 0.18 | +0.21 ± 0.18 * | +0.20 ± 0.22 | +0.18 ± 0.17 * | 150 |
| r16_n500 − zeroshot | +0.27 ± 0.18 * | +0.35 ± 0.17 * | +0.71 ± 0.20 * | +0.44 ± 0.16 * | 150 |
| r16_n775 − zeroshot | +0.26 ± 0.18 * | +0.39 ± 0.17 * | +0.61 ± 0.21 * | +0.42 ± 0.15 * | 150 |
| r32_n775 − zeroshot | +0.35 ± 0.18 * | +0.51 ± 0.18 * | +0.62 ± 0.20 * | +0.49 ± 0.16 * | 150 |
| r64_n775 − zeroshot | +0.23 ± 0.19 * | +0.37 ± 0.19 * | +0.58 ± 0.21 * | +0.40 ± 0.17 * | 150 |
| r8_n775 − r4_n775 | -0.01 ± 0.12 | +0.04 ± 0.12 | +0.11 ± 0.18 | +0.05 ± 0.12 | 150 |
| r16_n775 − r8_n775 | +0.03 ± 0.14 | +0.05 ± 0.15 | +0.01 ± 0.18 | +0.03 ± 0.14 | 150 |
| r32_n775 − r16_n775 | +0.09 ± 0.13 | +0.11 ± 0.13 | +0.01 ± 0.17 | +0.07 ± 0.12 | 150 |
| r64_n775 − r32_n775 | -0.11 ± 0.17 | -0.13 ± 0.17 | -0.04 ± 0.18 | -0.10 ± 0.15 | 150 |
| r64_n775 − r4_n775 | -0.01 ± 0.15 | +0.07 ± 0.15 | +0.09 ± 0.19 | +0.05 ± 0.14 | 150 |
| r16_n250 − r16_n100 | +0.03 ± 0.17 | +0.00 ± 0.17 | -0.10 ± 0.21 | -0.02 ± 0.16 | 150 |
| r16_n500 − r16_n250 | +0.15 ± 0.15 | +0.14 ± 0.14 | +0.51 ± 0.20 * | +0.27 ± 0.14 * | 150 |
| r16_n775 − r16_n500 | -0.01 ± 0.15 | +0.05 ± 0.15 | -0.10 ± 0.18 | -0.02 ± 0.14 | 150 |
| r16_n775 − r16_n100 | +0.17 ± 0.21 | +0.19 ± 0.19 | +0.31 ± 0.22 * | +0.22 ± 0.18 * | 150 |

## Judge mean with and without near-duplicate test prompts

35 test prompts have a near-copy in the training data (`data/test_near_duplicates.json`). Means are over the judged answers in each group, with a 95% interval. The last two columns are the paired difference from zero-shot within each group; \* marks an interval that excludes 0.

| Run | All prompts | Without near-duplicates | Near-duplicates only | − zero-shot, without | − zero-shot, near-duplicates only |
|---|---:|---:|---:|---:|---:|
| zeroshot | 1.63 ± 0.12 | 1.70 ± 0.14 | 1.42 ± 0.17 | – | – |
| r4_n775 | 1.98 ± 0.17 | 1.90 ± 0.17 | 2.26 ± 0.46 | +0.20 ± 0.18 * | +0.84 ± 0.38 * |
| r8_n775 | 2.03 ± 0.18 | 1.94 ± 0.18 | 2.32 ± 0.49 | +0.24 ± 0.18 * | +0.90 ± 0.42 * |
| r16_n100 | 1.83 ± 0.16 | 1.84 ± 0.19 | 1.81 ± 0.29 | +0.14 ± 0.16 | +0.39 ± 0.26 * |
| r16_n250 | 1.81 ± 0.18 | 1.69 ± 0.17 | 2.19 ± 0.49 | -0.01 ± 0.16 | +0.77 ± 0.46 * |
| r16_n500 | 2.08 ± 0.17 | 1.96 ± 0.17 | 2.46 ± 0.45 | +0.26 ± 0.16 * | +1.04 ± 0.37 * |
| r16_n775 | 2.06 ± 0.18 | 2.03 ± 0.19 | 2.15 ± 0.44 | +0.33 ± 0.16 * | +0.73 ± 0.36 * |
| r32_n775 | 2.12 ± 0.19 | 2.10 ± 0.20 | 2.22 ± 0.48 | +0.40 ± 0.16 * | +0.80 ± 0.39 * |
| r64_n775 | 2.03 ± 0.17 | 1.96 ± 0.17 | 2.26 ± 0.46 | +0.26 ± 0.18 * | +0.84 ± 0.40 * |

Steps along the sweeps on the prompts without a near-copy in training (paired difference in the judge mean, A − B):

| A − B | Without near-duplicates |
|---|---:|
| r8_n775 − r4_n775 | +0.04 ± 0.14 |
| r16_n775 − r8_n775 | +0.09 ± 0.16 |
| r32_n775 − r16_n775 | +0.07 ± 0.14 |
| r64_n775 − r32_n775 | -0.14 ± 0.16 |
| r64_n775 − r4_n775 | +0.06 ± 0.16 |
| r16_n250 − r16_n100 | -0.14 ± 0.16 |
| r16_n500 − r16_n250 | +0.27 ± 0.15 * |
| r16_n775 − r16_n500 | +0.07 ± 0.17 |
| r16_n775 − r16_n100 | +0.19 ± 0.19 |
