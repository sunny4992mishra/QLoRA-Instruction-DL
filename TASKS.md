# P16 task split

Deadline: **Sun 22 Nov 2026**. Submit the report and the repo link.

Each member has 2 training runs (Pranjal also runs the zero-shot baseline), one main task and part of the report. Each share is about 16–17 hours of work (estimated). Every task has a GitHub issue under the [P16 submission](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/milestone/1) milestone. Tick the checklist in the issue as you go.

## Who does what

| Member | Training runs | Main task | Report sections |
|---|---|---|---|
| **Pranjal** ([@Iampranjaldubey](https://github.com/Iampranjaldubey)) | [#2](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/2): zero-shot baseline, r=16, r=4 | [#1](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/1): smoke test. [#3](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/3): coordination, PR merges, README | [#4](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/4): abstract, data and tokenizer analysis, final assembly |
| **Sunny** ([@sunny4992mishra](https://github.com/sunny4992mishra)) | [#5](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/5): r=8, r=32 | [#6](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/6): analysis script, results table and plots | [#7](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/7): results, conclusion |
| **Shahid** ([@shahidthisside](https://github.com/shahidthisside)) | [#8](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/8): r=64, r=16 with N=100 | [#9](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/9): LLM judge script, judge all outputs | [#10](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/10): evaluation method, judge vs human agreement, limitations |
| **Ompal** ([@Ompalyadav-beep](https://github.com/Ompalyadav-beep)) | [#11](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/11): r=16 with N=250 and N=500 | [#12](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/12): blind human rating sheet, ratings script, error analysis | [#13](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/13): introduction, related work, training method |
| **All four** | | [#14](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/14): rate the same 90 blind answers (~2.5 h each). Review at least one teammate's PR. | Proofread the full report |

## Runs

All runs use `Qwen/Qwen2.5-1.5B-Instruct` with the default settings in `src/train.py`. Only the rank or the data size changes.

| Run name | Rank | Training rows | Owner |
|---|---|---|---|
| `Qwen2.5-1.5B-Instruct_zeroshot` | none | none | Pranjal |
| `Qwen2.5-1.5B-Instruct_r16_n775` | 16 | 775 (all) | Pranjal |
| `Qwen2.5-1.5B-Instruct_r4_n775` | 4 | 775 | Pranjal |
| `Qwen2.5-1.5B-Instruct_r8_n775` | 8 | 775 | Sunny |
| `Qwen2.5-1.5B-Instruct_r32_n775` | 32 | 775 | Sunny |
| `Qwen2.5-1.5B-Instruct_r64_n775` | 64 | 775 | Shahid |
| `Qwen2.5-1.5B-Instruct_r16_n100` | 16 | 100 | Shahid |
| `Qwen2.5-1.5B-Instruct_r16_n250` | 16 | 250 | Ompal |
| `Qwen2.5-1.5B-Instruct_r16_n500` | 16 | 500 | Ompal |

The rank sweep is r = 4, 8, 16, 32, 64 on all data. The data-size sweep is N = 100, 250, 500, 775 at r=16. The r16_n775 run belongs to both sweeps.

**Wait for the go-ahead comment on [#2](https://github.com/sunny4992mishra/QLoRA-Instruction-DL/issues/2) before starting your runs.** It confirms that the pipeline works and the first fine-tuned outputs look sensible.

## Schedule

| Date | Work |
|---|---|
| Sun 15 Nov | Smoke test (Pranjal). Sunny, Shahid and Ompal start their scripts, which don't need a GPU. |
| Mon 16 Nov | Baseline and r=16 (Pranjal), output check, go-ahead on #2 |
| Tue 17 – Wed 18 Nov | Everyone's training runs. Draft the sections that don't need results: introduction, related work, data, method, evaluation method. |
| Thu 19 Nov | Judge all outputs (Shahid). Human ratings (all four). |
| Fri 20 Nov | Plots and table (Sunny). Results-based sections. |
| Sat 21 Nov | Report assembly (Pranjal). Everyone proofreads. |
| Sun 22 Nov | Buffer and submission |

## How we work on GitHub

1. Don't commit to `main`. Create a branch per task, named `<name>/<task>`, e.g. `sunny/runs-r8-r32` or `shahid/judge-script`.
2. Open a pull request into `main`. Write `Closes #<issue>` in its description so the issue closes when it is merged.
3. One teammate reviews the PR, then Pranjal merges it.
4. For a training run, the PR adds two files:
   - `results/<run name>.jsonl` (the answers to the 150 test prompts)
   - `runs/<run name>/run_info.json` (memory, time, losses)

   Adapters and checkpoints stay out of git. `.gitignore` handles this.
5. Don't regenerate the data split and don't change default hyperparameters. Either change makes the runs incomparable.
6. Never commit API keys. Read them from an environment variable.
