# Judge prompts

- `judge_prompt_v1.txt` is the **frozen full-corpus prompt** — every judge
  model in the committee (PROTOCOL.md §4) labels the entire corpus with this
  prompt.
- `judge_prompt_v2.txt` and `judge_prompt_v3.txt` are **sensitivity
  variants** — semantically equivalent paraphrases of v1 (same label scheme,
  same JSON schema, same worked examples, different wording/ordering). They
  are used only on the 10% stratified subsample for the prompt-sensitivity
  study described in PROTOCOL.md §4, never on the full corpus.

All three prompts expect a single placeholder, `{{SENTENCE}}`, to be
substituted with the sentence being labeled.

**Prompts are immutable once the pilot starts.** Any change to the wording,
schema, or examples in any of these files after the 1,000-sentence pilot
(PROTOCOL.md §4) begins requires a new PROTOCOL.md CHANGELOG entry recording
the date and reason before the change is made.
