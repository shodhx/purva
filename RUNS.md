# Committee run log

One row per (chunk, judge) run. Append a new row after every future run —
do not edit existing rows except to fix a genuine transcription error.
"not recorded" means the value was not retrievable after the fact, not a
guess. Generation seconds is the model's own generation-loop elapsed time
(`elapsed:` in run_judge.py's Final summary), not total Kaggle session
wall-clock (which also includes git clone / pip install / model load).
Hardware is the Kaggle machine shape every run used; which of the
project's Kaggle accounts a given run happened under is not tracked here
— it has no bearing on reproducibility, only GPU-quota bookkeeping. Rows
are appended in the order runs actually happened, so position in the
table (within a chunk) is what orders them — no date column is kept.

| Chunk | Model | Repo ID | Revision | Quantization | max_model_len | Seed | Prompt file | Processed | Parse failures | Generation (s) | Hardware |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 1 | 2568.6 | Kaggle T4x2 (free tier) |
| 1 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 1 | 2233.4 | Kaggle T4x2 (free tier) |
| 1 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 3249.7 | Kaggle T4x2 (free tier) |
| 1 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 4301.5 | Kaggle T4x2 (free tier) |
| 1 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 9748.4 | Kaggle T4x2 (free tier) |
| 2 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 2520.8 | Kaggle T4x2 (free tier) |
| 2 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 2 | 2283.7 | Kaggle T4x2 (free tier) |
| 2 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 1 | 3113.8 | Kaggle T4x2 (free tier) |
| 2 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 4055.6 | Kaggle T4x2 (free tier) |
| 2 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10026 | 0 | 9453.3 | Kaggle T4x2 (free tier) |
| 3 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 2550.7 | Kaggle T4x2 (free tier) |
| 3 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 1 | 2303.5 | Kaggle T4x2 (free tier) |
| 3 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 3144.4 | Kaggle T4x2 (free tier) |
| 3 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 4012.7 | Kaggle T4x2 (free tier) |
| 3 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 10403.9 | Kaggle T4x2 (free tier) |
| 4 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 0 | 2637.7 | Kaggle T4x2 (free tier) |
| 4 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 3 | 2159.3 | Kaggle T4x2 (free tier) |
| 4 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 0 | 2906.0 | Kaggle T4x2 (free tier) |
| 4 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 1 | 4748.9 | Kaggle T4x2 (free tier) |
| 4 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 1 | 9563.3 | Kaggle T4x2 (free tier) |
| 5 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 2671.3 | Kaggle T4x2 (free tier) |
| 5 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 4 | 2163.1 | Kaggle T4x2 (free tier) |
| 5 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 1 | 3470.7 | Kaggle T4x2 (free tier) |
| 5 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 2 | 4041.9 | Kaggle T4x2 (free tier) |
| 5 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10021 | 0 | 9385.2 | Kaggle T4x2 (free tier) |
| 6 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 0 | 2684.7 | Kaggle T4x2 (free tier) |
| 6 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 1 | 2094.4 | Kaggle T4x2 (free tier) |
| 6 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 0 | 3236.8 | Kaggle T4x2 (free tier) |
| 6 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 2 | 4003.8 | Kaggle T4x2 (free tier) |
| 7 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 2427.3 | Kaggle T4x2 (free tier) |
| 7 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 1 | 2186.9 | Kaggle T4x2 (free tier) |
