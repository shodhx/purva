# PURVA Layer 2: Consensus Statistics Summary

**Evaluated Corpus Scale**: 91,446 authentic Bhojpuri sentences across 3 diverse model families (Sarvam AI 2B, Google Gemma 2 9B, Alibaba Qwen 2.5 3B).

---

## 1. Consensus Routing & Workload Distribution

| Routing Status | Sentence Count | Percentage | Downstream Action |
| :--- | :---: | :---: | :--- |
| **RESOLVED (Agreed Consensus)** | `84,225` | **92.10%** | Directly incorporated into training & evaluation benchmarks |
| **DISAGREED (Active Learning Queue)** | `7,221` | **7.90%** | Routed to Layer 3 human annotation experts for adjudication |
| **PIPELINE PROCESSING FAILURES** | `0` | **0.00%** | Zero crashed or unparseable model responses (engineering reliability metric; distinct from annotation error rate, which is pending Layer 3 human evaluation) |
| **Total Evaluated Corpus** | **`91,446`** | **100.00%** | Complete Bhojpuri repository analysis |

---

## 2. Information-Theoretic & Agreement Metrics

* **Mean Shannon Entropy H(X) (Resolved Items)**: `0.5887 bits`
* **Mean Local Fleiss' Kappa (Resolved Items)**: `-0.1516` (Reflects local divergence caused by the 2B model's class collapse before Dawid-Skene EM weighting resolves the consensus).

---

## 3. Affective Label Distribution (3-Class Taxonomy)

| Sentiment Category | Sentence Count | Percentage of Corpus |
| :--- | :---: | :---: |
| `neutral_factual` | 62,648 | 68.51% |
| `negative` | 11,088 | 12.13% |
| `positive` | 10,489 | 11.47% |
| `disagreed` | 7,221 | 7.90% |
