# Committee run log

Judge short name → HuggingFace repo ID (used as the key in `purva_master.jsonl`'s
`judges` object and as the `judge_<short>_*` column prefix in the Parquet
export): `qwen` = Qwen/Qwen2.5-14B-Instruct-AWQ, `gemma` =
hugging-quants/gemma-2-9b-it-AWQ-INT4, `llama` =
hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4, `mistral` =
casperhansen/mistral-nemo-instruct-2407-awq, `aya` =
Orion-zhen/aya-expanse-8b-AWQ.

Every run below specified revision `main`, not a pinned commit SHA — a
real gap against PROTOCOL.md §4's commitment to pinned revisions (see
CHANGELOG v1.6). The SHAs below were resolved post-hoc via
`HfApi().model_info(repo_id)`, not pinned before the fact; `main` is what
was actually requested at run time and is kept as-is in the "Revision"
column of the table further down, unchanged. Every judge repo's
`lastModified` predates our run period, so no drift was detected between
the run and this resolution (recheck if these ever diverge from
`data/purva_master.meta.json`, which is regenerated on every run of
`build_master.py` and is the source of truth for this table).

| Judge | Revision requested | Revision resolved | Repo lastModified | Resolution |
|---|---|---|---|---|
| aya | main | ff52d61b71c613180581c3a4c6b3b3f636ce79e5 | 2024-10-26T23:26:54+00:00 | resolved post-hoc; runs specified 'main' rather than a pinned SHA |
| gemma | main | 6e62725da8e92309167814dad7aacc0ed8cb2484 | 2024-10-17T08:31:37+00:00 | resolved post-hoc; runs specified 'main' rather than a pinned SHA |
| llama | main | db1f81ad4b8c7e39777509fac66c652eb0a52f91 | 2024-08-07T07:29:21+00:00 | resolved post-hoc; runs specified 'main' rather than a pinned SHA |
| mistral | main | c83b6438e13051ad1c0f5683635705ee83bb8772 | 2024-09-27T07:14:03+00:00 | resolved post-hoc; runs specified 'main' rather than a pinned SHA |
| qwen | main | 539535859b135b0244c91f3e59816150c8056698 | 2024-10-09T12:26:42+00:00 | resolved post-hoc; runs specified 'main' rather than a pinned SHA |

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
| 6 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10024 | 0 | 9090.7 | Kaggle T4x2 (free tier) |
| 7 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 3479.8 | Kaggle T4x2 (free tier) |
| 7 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 1 | 4163.0 | Kaggle T4x2 (free tier) |
| 8 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10022 | 0 | 2681.1 | Kaggle T4x2 (free tier) |
| 8 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10022 | 0 | 2251.1 | Kaggle T4x2 (free tier) |
| 7 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 9464.7 | Kaggle T4x2 (free tier) |
| 8 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10022 | 1 | 3372.0 | Kaggle T4x2 (free tier) |
| 8 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10022 | 3 | 4251.8 | Kaggle T4x2 (free tier) |
| 9 | aya-expanse-8b | Orion-zhen/aya-expanse-8b-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 1 | 2926.3 | Kaggle T4x2 (free tier) |
| 9 | llama-3.1-8b | hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 3 | 2219.3 | Kaggle T4x2 (free tier) |
| 8 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10022 | 0 | 9047.8 | Kaggle T4x2 (free tier) |
| 9 | mistral-nemo-12b | casperhansen/mistral-nemo-instruct-2407-awq | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 3257.5 | Kaggle T4x2 (free tier) |
| 9 | qwen2.5-14b | Qwen/Qwen2.5-14B-Instruct-AWQ | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 2 | 4479.8 | Kaggle T4x2 (free tier) |
| 9 | gemma-2-9b | hugging-quants/gemma-2-9b-it-AWQ-INT4 | main | awq | 1536 | 42 | prompts/judge_prompt_v1.txt | 10023 | 0 | 8701.7 | Kaggle T4x2 (free tier) |
