# Committee run log

One row per (chunk, judge) run. Append a new row after every future run —
do not edit existing rows except to fix a genuine transcription error.
"not recorded" means the value was not retrievable after the fact, not a
guess. Generation seconds is the model's own generation-loop elapsed time
(`elapsed:` in run_judge.py's Final summary), not total Kaggle session
wall-clock (which also includes git clone / pip install / model load).

| Date | Chunk | Model | Repo ID | Revision | Quantization | max_model_len | Seed | Prompt file | Processed | Parse failures | Generation (s) | Kernel version |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-06 | 1 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 1 | 2568.6 | not recorded |
| 2026-08-06 | 1 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 1 | 2233.4 | 30 |
| 2026-08-07 | 1 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 3249.7 | not recorded |
| 2026-08-07 | 1 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 4301.5 | 32 |
| 2026-08-07 | 1 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 9748.4 | 33 |
| 2026-08-07 | 2 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 2520.8 | 34 |
| 2026-08-07 | 2 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 2 | 2283.7 | 35 |
| 2026-08-07 | 2 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 1 | 3113.8 | 36 |
| 2026-08-07 | 2 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 4055.6 | 37 |
| 2026-08-07 | 2 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 9453.3 | 38 |
| 2026-08-07 | 3 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 2550.7 | 39 |
| 2026-08-07 | 3 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 1 | 2303.5 | 40 |
