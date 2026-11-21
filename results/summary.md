| Run | Rank | Train rows | Trainable params | Trainable % | Peak GPU mem (GB) | Train time (min) | Val loss | Answers | Devanagari % | Median chars | Judged | Correctness | Helpfulness | Fluency | Judge mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct_zeroshot | – | – | – | – | – | – | – | 150 | 94.8 | 541 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r4_n775 | 4 | 775 | 4,616,192 | 0.298 | 5.83 | 30.3 | 0.946 | 150 | 99.6 | 209 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r8_n775 | 8 | 775 | 9,232,384 | 0.595 | 5.87 | 30.4 | 0.942 | 150 | 99.2 | 236 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r16_n100 | 16 | 100 | 18,464,768 | 1.182 | 5.94 | 4.0 | 1.041 | 150 | 98.9 | 239 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r16_n250 | 16 | 250 | 18,464,768 | 1.182 | 5.94 | 10.7 | 0.988 | 150 | 99.4 | 436 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r16_n500 | 16 | 500 | 18,464,768 | 1.182 | 5.94 | 20.8 | 0.957 | 150 | 98.7 | 192.5 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r16_n775 | 16 | 775 | 18,464,768 | 1.182 | 5.94 | 30.2 | 0.937 | 150 | 99.2 | 197 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r32_n775 | 32 | 775 | 36,929,536 | 2.336 | 6.13 | 31.0 | 0.932 | 150 | 99.5 | 205.5 | – | – | – | – | – |
| Qwen2.5-1.5B-Instruct_r64_n775 | 64 | 775 | 73,859,072 | 4.566 | 6.43 | 28.8 | 0.928 | 150 | 98.1 | 205.5 | – | – | – | – | – |

Peak GPU memory is `torch.cuda.max_memory_allocated` during training (GPU 0). Judge scores are means over answers on a 1–5 scale; the judge mean averages the three criteria per answer. 95% intervals are in summary.csv. Median characters is the mean of the two middle values when the number of answers is even (`statistics.median`).
