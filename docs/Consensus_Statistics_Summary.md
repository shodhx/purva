# PURVA L2: Consensus Statistics Summary

**Target Venue**: ACL Rolling Review (ARR) — Resources & Findings Track  
**Dataset Scale**: 91,446 sentences evaluated across 3 AI families (Sarvam AI 2B, Google Gemma 2 9B, Alibaba Qwen 2.5 3B).

---

## 1. Consensus Routing Summary

| Routing Status | Sentence Count | Percentage | Downstream Action |
|:---|:---:|:---:|:---|
| **RESOLVED (Agreed Consensus)** | `76,103` | **83.22%** | Directly incorporated into training & evaluation benchmarks |
| **DISAGREED (Active Learning Queue)** | `15,343` | **16.78%** | Routed to L3 native human experts for adjudication |
| **ERROR (Pipeline Failures)** | `0` | **0.00%** | Zero-fault engineering guarantee |
| **Total Evaluated Corpus** | **`91,446`** | **100.00%** | Complete Bhojpuri repository analysis |

---

## 2. Information-Theoretic & Agreement Metrics
* **Mean Shannon Entropy H(X) (Resolved Items)**: `0.6278 bits`
* **Mean Local Fleiss' Kappa (Resolved Items)**: `-0.1525`

---

## 3. Label Distribution (3-Class Taxonomy)

| Sentiment Category | Sentence Count | Percentage of Corpus |
|:---|:---:|:---:|
| `neutral_factual` | 54,903 | 60.04% |
| `disagreed` | 15,343 | 16.78% |
| `negative` | 10,866 | 11.88% |
| `positive` | 10,334 | 11.30% |
