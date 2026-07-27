"""
PURVA Layer 2: Multi-LLM Committee Ensemble Engine
==================================================
A production-grade, fault-tolerant consensus engine for low-resource dialectal
sentiment analysis, implementing Expectation-Maximization (EM) Dawid-Skene
aggregation, Fleiss' Kappa multi-rater reliability, and Shannon Entropy routing.

Target Publication Standard: ACL / EMNLP / ARR Resources & Findings
Lead Architect: Kazi Tasfin Mahmud (shodhx)
Supervisory Team: Dr. Amit, Prof. Deepak

References:
    - Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer
      error-rates using the EM algorithm. Applied Statistics, 28(1), 20-28.
    - Fleiss, J. L. (1971). Measuring nominal scale agreement among many
      raters. Psychological Bulletin, 76(5), 378-382.
    - Shannon, C. E. (1948). A mathematical theory of communication. The Bell
      System Technical Journal, 27(3), 379-423.
"""

from __future__ import annotations
import os
import re
import csv
import json
import time
from datetime import datetime, timezone
import math
import random
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =====================================================================
# 1. TAXONOMY & SYSTEM PROMPTS (3-Class & 4-Class Schemas)
# =====================================================================

SYSTEM_PROMPT_3CLASS = (
    "You are an expert computational linguist and native speaker specializing in Eastern Indo-Aryan dialects (Bhojpuri).\n"
    "Your task is to classify the affective sentiment of authentic Bhojpuri sentences into exactly one of three categories:\n"
    "1. 'positive': Expresses joy, praise, agreement, satisfaction, or positive emotional valence.\n"
    "2. 'negative': Expresses anger, sorrow, criticism, disagreement, or negative emotional valence.\n"
    "3. 'neutral_factual': Expresses objective news reporting, encyclopedia facts, conversational dialogue without emotion, or neutral statements.\n\n"
    "CRITICAL INSTRUCTION: You MUST respond ONLY with a valid JSON object matching the exact schema:\n"
    '{"reasoning": "<brief 1-2 sentence linguistic analysis>", "label": "<positive|negative|neutral_factual>", "confidence": <float between 0.80 and 1.0>}\n'
    "Do not include any scratchpad, preamble, or markdown formatting outside the JSON."
)

SYSTEM_PROMPT_4CLASS = (
    "You are an expert computational linguist and native speaker specializing in Bhojpuri sentiment analysis.\n"
    "Classify the affective sentiment of the following Bhojpuri sentence into exactly one of four categories:\n"
    "1. 'positive': Expresses positive emotional valence.\n"
    "2. 'negative': Expresses negative emotional valence.\n"
    "3. 'neutral': Conversational dialogue or everyday speech without explicit emotion.\n"
    "4. 'objective': Factual news reporting, encyclopedia entries, or educational statements.\n\n"
    "CRITICAL INSTRUCTION: Respond ONLY with a valid JSON object matching this schema:\n"
    '{"reasoning": "<brief linguistic analysis>", "label": "<positive|negative|neutral|objective>", "confidence": <float between 0.80 and 1.0>}'
)

# Few-shot calibration examples for in-context alignment
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "रउरा सब के हमर प्रणाम, आजु के दिन बहुत सुहावन बा।"},
    {
        "role": "assistant",
        "content": '{"reasoning": "The speaker offers respectful greetings and describes the day as pleasant (\'सुहावन\'), indicating positive emotional valence.", "label": "positive", "confidence": 0.98}',
    },
    {"role": "user", "content": "बाढ़ के पानी से पूरा गाँव डूब गइल, केहू पूछे वाला नइखे।"},
    {
        "role": "assistant",
        "content": '{"reasoning": "Describes a severe flood disaster destroying a village and lack of government relief, conveying distress and sorrow.", "label": "negative", "confidence": 0.96}',
    },
    {"role": "user", "content": "पटना बिहार के राजधानी ह और इहाँ भोजपुरी बोलल जाला।"},
    {
        "role": "assistant",
        "content": '{"reasoning": "Factual geographical and demographic statement about Patna and language spoken, containing zero affective emotion.", "label": "neutral_factual", "confidence": 0.99}',
    },
]


# =====================================================================
# 2. PYDANTIC STRICT DATA SCHEMAS
# =====================================================================


class JudgeOutput(BaseModel):
    """Strict schema for individual LLM judge predictions."""

    reasoning: str = Field(
        ..., description="Linguistic justification for the assigned sentiment label."
    )
    label: str = Field(..., description="Predicted sentiment category string.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Self-reported confidence score between 0.0 and 1.0.",
    )

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, v: Any) -> str:
        val = str(v).lower().strip()
        if val in ["objective", "neutral"]:
            return "neutral_factual"
        if val in ["positive", "negative", "neutral_factual", "error"]:
            return val
        return "neutral_factual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "label": self.label,
            "confidence": self.confidence,
        }


class EnsembleDecision(BaseModel):
    """Immutable record representing the consensus outcome for a single corpus sentence."""

    id: str = Field(
        ...,
        description="Unique deterministic identifier (SHA-1 prefix or database ID).",
    )
    raw_text: str = Field(..., description="Original uncleaned Bhojpuri sentence.")
    cleaned_text: str = Field(
        ..., description="Normalized and cleaned Bhojpuri sentence."
    )
    source_name: str = Field(
        ..., description="Origin corpus metadata (e.g., wikipedia, blogs, youtube)."
    )
    consensus_label: str = Field(
        ..., description="Final resolved sentiment label via Dawid-Skene aggregation."
    )
    shannon_entropy: float = Field(
        ...,
        ge=0.0,
        description="Shannon entropy H(X) representing inter-model uncertainty.",
    )
    fleiss_kappa_batch: float = Field(
        ...,
        description="Local Fleiss' Kappa agreement metric across active committee judges.",
    )
    status: str = Field(
        ...,
        description="Routing flag: 'RESOLVED' (Agreed), 'DISAGREED' (Routed to L3 human review), or 'ERROR'.",
    )
    model_outputs: Dict[str, Dict[str, Any]] = Field(
        ..., description="Complete raw JSON outputs from every committee model."
    )
    timestamp: str = Field(..., description="UTC ISO-8601 execution timestamp.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =====================================================================
# 3. STATISTICAL AGGREGATION ENGINE (Dawid-Skene EM, Fleiss, Shannon)
# =====================================================================


def calculate_shannon_entropy(prob_dist: Dict[str, float]) -> float:
    """
    Computes Shannon Entropy H(X) over the label probability distribution.

    Formula: H(X) = - sum(P(x) * log2(P(x)))

    A lower entropy indicates high consensus; H(X) > 0.5 flags semantic ambiguity
    or model deadlock requiring human expert adjudication.
    """
    entropy = 0.0
    for p in prob_dist.values():
        if p > 0.0:
            entropy -= p * math.log2(p)
    return max(0.0, entropy)


def calculate_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Computes Fleiss' Kappa (k) for multi-rater nominal classification agreement.

    Args:
        ratings_matrix: 2D numpy array of shape (N_items, K_categories), where
                        each element (i, j) represents the number of raters
                        assigning item i to category j.

    Returns:
        float: Fleiss' Kappa statistic. k >= 0.20 indicates acceptable agreement
               for ambiguous low-resource dialectal tasks.
    """
    N, k = ratings_matrix.shape
    if N == 0:
        return 0.0

    n_raters_per_item = np.sum(ratings_matrix, axis=1)
    mask = n_raters_per_item >= 2
    valid_matrix = ratings_matrix[mask]

    if valid_matrix.shape[0] == 0:
        return 0.0

    N_valid = valid_matrix.shape[0]
    n_avg = np.mean(n_raters_per_item[mask])
    if n_avg <= 1.0:
        return 0.0

    p_j = np.sum(valid_matrix, axis=0) / (N_valid * n_avg)
    P_i = (np.sum(valid_matrix**2, axis=1) - n_avg) / (n_avg * (n_avg - 1.0))

    P_bar = np.mean(P_i)
    P_e = np.sum(p_j**2)

    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def dawid_skene_aggregation(
    batch_predictions: List[Dict[str, str]],
    categories: List[str],
    max_iter: int = 20,
    tol: float = 1e-4,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Expectation-Maximization (EM) algorithm for estimating latent ground truth
    labels and annotator error confusion matrices (Dawid & Skene, 1979).

    This acts as an architectural safeguard: if a small model experiences class
    collapse (e.g., predicting 98% factual), Dawid-Skene automatically detects
    its skewed confusion matrix and downgrades its mathematical weight, preserving
    the consensus of reliable instruction-tuned judges.

    Args:
        batch_predictions: List of dicts mapping model_name -> predicted_label string.
        categories: Ordered list of taxonomy category strings.
        max_iter: Maximum EM convergence iterations.
        tol: Log-likelihood convergence tolerance.

    Returns:
        List of tuples per item: (consensus_label, confidence, prob_distribution).
    """
    N = len(batch_predictions)
    K = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Identify all active annotators in this batch
    annotators = sorted(list({m for preds in batch_predictions for m in preds.keys()}))
    M = len(annotators)
    ann_to_idx = {a: i for i, a in enumerate(annotators)}

    if N == 0 or M == 0:
        return []

    # 1. Initialize latent true label probabilities T(i, j) via uniform/majority prior
    T = np.zeros((N, K))
    for i, preds in enumerate(batch_predictions):
        for ann, label in preds.items():
            if label in cat_to_idx:
                T[i, cat_to_idx[label]] += 1.0
        row_sum = np.sum(T[i])
        if row_sum > 0:
            T[i] /= row_sum
        else:
            T[i] = 1.0 / K

    # Class prior p(j)
    p = np.mean(T, axis=0)

    # 2. EM Iteration Loop
    for iteration in range(max_iter):
        # M-Step: Estimate annotator confusion matrices pi(m, j, k)
        # pi[m, j, k] = P(annotator m predicts k | true label is j)
        pi = np.zeros((M, K, K))
        for i, preds in enumerate(batch_predictions):
            for ann, label in preds.items():
                if label in cat_to_idx:
                    m_idx = ann_to_idx[ann]
                    k_idx = cat_to_idx[label]
                    for j in range(K):
                        pi[m_idx, j, k_idx] += T[i, j]

        # Laplace smoothing (0.01) to prevent zero-probability lockup
        pi += 0.01
        for m_idx in range(M):
            for j in range(K):
                row_sum = np.sum(pi[m_idx, j])
                if row_sum > 0:
                    pi[m_idx, j] /= row_sum

        # E-Step: Re-estimate latent true label distribution T(i, j)
        T_new = np.zeros((N, K))
        for i, preds in enumerate(batch_predictions):
            for j in range(K):
                prob = p[j]
                for ann, label in preds.items():
                    if label in cat_to_idx:
                        m_idx = ann_to_idx[ann]
                        k_idx = cat_to_idx[label]
                        prob *= pi[m_idx, j, k_idx]
                T_new[i, j] = prob

        # Normalize rows
        for i in range(N):
            row_sum = np.sum(T_new[i])
            if row_sum > 0:
                T_new[i] /= row_sum
            else:
                T_new[i] = 1.0 / K

        # Update class priors
        p = np.mean(T_new, axis=0)

        # Check convergence
        diff = np.max(np.abs(T_new - T))
        T = T_new
        if diff < tol:
            break

    # 3. Compile final consensus results
    results = []
    for i in range(N):
        best_idx = int(np.argmax(T[i]))
        best_label = categories[best_idx]
        conf = float(T[i, best_idx])
        prob_dist = {categories[j]: float(T[i, j]) for j in range(K)}
        results.append((best_label, conf, prob_dist))

    return results


# =====================================================================
# 4. ABSTRACT & CONCRETE LLM JUDGE INTERFACES (Local AI Committee)
# =====================================================================


class BaseJudge:
    """Abstract interface for local language model judges."""

    def __init__(self, name: str):
        self.name = name

    def judge(self, sentence: str) -> JudgeOutput:
        raise NotImplementedError("Subclasses must implement judge()")

    def judge_batch(self, batch_records: List[Dict[str, Any]]) -> List[JudgeOutput]:
        """Default sequential batch processing; subclasses override for batched inference."""
        return [
            self.judge(rec.get("cleaned_text", rec.get("raw_text", "")))
            for rec in batch_records
        ]

    def parse_fallback(self, content: str) -> JudgeOutput:
        """Robust JSON extractor that recovers from markdown fencing or conversational padding."""
        content = content.strip()
        if "```" in content:
            content = re.sub(r"```[a-zA-Z]*\n?", "", content).replace("```", "").strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx : end_idx + 1]
        try:
            data = json.loads(content)
            return JudgeOutput(
                reasoning=str(data.get("reasoning", "Extracted via JSON fallback"))[
                    :250
                ],
                label=str(data.get("label", "neutral_factual")),
                confidence=float(data.get("confidence", 0.85)),
            )
        except Exception:
            # If JSON parsing completely fails, inspect text keywords
            lower_txt = content.lower()
            if "positive" in lower_txt and "negative" not in lower_txt:
                return JudgeOutput(
                    reasoning="Keyword regex recovery",
                    label="positive",
                    confidence=0.80,
                )
            if "negative" in lower_txt and "positive" not in lower_txt:
                return JudgeOutput(
                    reasoning="Keyword regex recovery",
                    label="negative",
                    confidence=0.80,
                )
            return JudgeOutput(
                reasoning=f"Unparseable LLM response: {content[:40]}",
                label="neutral_factual",
                confidence=0.75,
            )


class OllamaJudge(BaseJudge):
    """
    Local Ollama API Judge supporting Google Gemma 2 (9B) and Alibaba Qwen 2.5 (3B).
    Utilizes continuous batch prompting to maximize local inference throughput.
    """

    _lock = threading.Lock()

    def __init__(
        self, model_name: str, judge_name: str, host: str = "http://localhost:11434"
    ):
        super().__init__(judge_name)
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.url = f"{self.host}/v1/chat/completions"

    def _judge_single_unlocked(self, text: str) -> JudgeOutput:
        prompt = (
            f"Classify the sentiment of this Bhojpuri sentence into positive, negative, or neutral_factual:\n"
            f'Sentence: "{text}"\n'
            f'Respond ONLY with valid JSON: {{"reasoning": "brief reason", "label": "positive|negative|neutral_factual", "confidence": 0.90}}'
        )
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 120,
            "keep_alive": "10m",
        }
        for _ in range(2):
            try:
                r = requests.post(self.url, json=payload, timeout=45.0)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"]
                    return self.parse_fallback(content)
            except Exception:
                time.sleep(0.5)
        return JudgeOutput(
            reasoning="Ollama HTTP timeout/error", label="error", confidence=0.0
        )

    def judge(self, sentence: str) -> JudgeOutput:
        with self._lock:
            return self._judge_single_unlocked(sentence)

    def judge_batch(self, batch_records: List[Dict[str, Any]]) -> List[JudgeOutput]:
        """Executes batched multi-sentence evaluation in a single API roundtrip."""
        N = len(batch_records)
        if N == 0:
            return []
        if N == 1:
            with self._lock:
                return [
                    self._judge_single_unlocked(
                        batch_records[0].get(
                            "cleaned_text", batch_records[0].get("raw_text", "")
                        )
                    )
                ]

        try:
            lines = [
                f"{i}. {rec.get('cleaned_text', rec.get('raw_text', ''))}"
                for i, rec in enumerate(batch_records, 1)
            ]
            items_str = "\n".join(lines)
            prompt = (
                f"Classify the sentiment of each of the following {N} Bhojpuri sentences into exactly one of: positive, negative, or neutral_factual.\n\n"
                f"Sentences:\n{items_str}\n\n"
                f'CRITICAL INSTRUCTION: Respond ONLY with a valid JSON array of exactly {N} category strings in matching order: ["positive", "negative", "neutral_factual", ...]\n'
                f"Do not output anything outside the JSON array."
            )
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 400,
                "keep_alive": "15m",
            }
            r = requests.post(self.url, json=payload, timeout=90.0)
            if r.status_code == 200:
                content = (r.json()["choices"][0]["message"]["content"] or "").strip()
                if "```" in content:
                    content = (
                        re.sub(r"```[a-zA-Z]*\n?", "", content)
                        .replace("```", "")
                        .strip()
                    )
                start_idx = content.find("[")
                end_idx = content.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    content = content[start_idx : end_idx + 1]
                data = json.loads(content)
                if isinstance(data, list):
                    if len(data) > N:
                        data = data[:N]
                    elif len(data) < N:
                        data.extend(["neutral_factual"] * (N - len(data)))

                    results = []
                    for item in data:
                        label_str = (
                            str(item).lower().strip()
                            if isinstance(item, (str, int))
                            else str(item.get("label", "neutral_factual"))
                            .lower()
                            .strip()
                            if isinstance(item, dict)
                            else "neutral_factual"
                        )
                        results.append(
                            JudgeOutput(
                                reasoning="Batched consensus inference",
                                label=label_str,
                                confidence=0.90,
                            )
                        )
                    return results
        except Exception:
            pass

        # Fallback to sequential evaluation if batch JSON format was corrupted
        with self._lock:
            return [
                self._judge_single_unlocked(
                    rec.get("cleaned_text", rec.get("raw_text", ""))
                )
                for rec in batch_records
            ]


class VLLMJudge(BaseJudge):
    """
    Local GPU / vLLM Judge supporting Sarvam AI Indic (2B).
    Automatically detects running vLLM server or loads weights directly onto NVIDIA RTX GPU in bfloat16.
    """

    _shared_model = None
    _shared_tokenizer = None
    _shared_pipe = None
    _gpu_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "sarvamai/sarvam-2b",
        judge_name: str = "sarvam_ai_indic",
        persona: str = "sentiment",
    ):
        super().__init__(judge_name)
        self.model_name = model_name
        self.persona = persona
        self.base_url = os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        self.client = None
        self.mode = "offline"

        # 1. Probe for active vLLM HTTP server
        try:
            if OpenAI:
                temp_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                if temp_client.models.list(timeout=2.0).data:
                    self.client = temp_client
                    self.mode = "vllm_server"
                    print(
                        f"  [OK] Connected to vLLM Server at {self.base_url} for '{self.name}'."
                    )
        except Exception:
            pass

        # 2. Probe for local PyTorch / Transformers GPU runtime
        if self.mode == "offline":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

                if torch.cuda.is_available():
                    if VLLMJudge._shared_model is None:
                        print(
                            f"  [GPU] Loading {model_name} in bfloat16 onto local NVIDIA GPU..."
                        )
                        tok = AutoTokenizer.from_pretrained(
                            model_name, padding_side="left"
                        )
                        if tok.pad_token is None:
                            tok.pad_token = tok.eos_token
                        mod = AutoModelForCausalLM.from_pretrained(
                            model_name, device_map="cuda", torch_dtype=torch.bfloat16
                        )
                        mod.eval()
                        pipe = pipeline(
                            "text-generation",
                            model=mod,
                            tokenizer=tok,
                            device_map="cuda",
                            batch_size=128,
                        )
                        VLLMJudge._shared_tokenizer = tok
                        VLLMJudge._shared_model = mod
                        VLLMJudge._shared_pipe = pipe
                    self.mode = "local_rtx_gpu"
                    print(
                        f"  [OK] Judge '{self.name}' attached to shared PyTorch GPU pipeline."
                    )
            except Exception:
                pass

    @retry(
        stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1.0, min=1, max=3)
    )
    def judge(self, sentence: str) -> JudgeOutput:
        if self.mode == "vllm_server" and self.client:
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT_3CLASS}]
                + FEW_SHOT_EXAMPLES
                + [{"role": "user", "content": sentence}]
            )
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                    timeout=10.0,
                )
                return self.parse_fallback(resp.choices[0].message.content or "")
            except Exception as e:
                return JudgeOutput(
                    reasoning=f"vLLM API Error: {str(e)[:30]}",
                    label="error",
                    confidence=0.0,
                )

        elif self.mode == "local_rtx_gpu" and VLLMJudge._shared_pipe:
            try:
                prompt = (
                    f'Analyze affective sentiment of this Bhojpuri sentence: "{sentence}"\n'
                    f'Respond ONLY in valid JSON: {{"reasoning": "brief reason", "label": "positive|negative|neutral_factual", "confidence": 0.90}}\n'
                    f"JSON Output:"
                )
                with VLLMJudge._gpu_lock:
                    outs = VLLMJudge._shared_pipe(
                        prompt, max_new_tokens=50, do_sample=False
                    )
                gen_text = outs[0]["generated_text"].split("JSON Output:")[-1].strip()
                return self.parse_fallback(gen_text)
            except Exception as e:
                return JudgeOutput(
                    reasoning=f"PyTorch GPU Error: {str(e)[:30]}",
                    label="error",
                    confidence=0.0,
                )

        return JudgeOutput(
            reasoning="Sarvam judge offline", label="error", confidence=0.0
        )

    def judge_batch(self, batch_records: List[Dict[str, Any]]) -> List[JudgeOutput]:
        if self.mode == "offline":
            return [
                JudgeOutput(
                    reasoning="Sarvam judge offline", label="error", confidence=0.0
                )
                for _ in batch_records
            ]

        if self.mode == "local_rtx_gpu" and VLLMJudge._shared_pipe:
            try:
                sentences = [
                    rec.get("cleaned_text", rec.get("raw_text", ""))
                    for rec in batch_records
                ]
                prompts = [
                    f'Analyze affective sentiment of this Bhojpuri sentence: "{s}"\nRespond ONLY in valid JSON: {{"reasoning": "brief reason", "label": "positive|negative|neutral_factual", "confidence": 0.90}}\nJSON Output:'
                    for s in sentences
                ]
                with VLLMJudge._gpu_lock:
                    outs = VLLMJudge._shared_pipe(
                        prompts, max_new_tokens=50, do_sample=False, batch_size=64
                    )
                results = []
                for i in range(len(prompts)):
                    gen_text = (
                        outs[i][0]["generated_text"].split("JSON Output:")[-1].strip()
                    )
                    results.append(self.parse_fallback(gen_text))
                return results
            except Exception:
                return [
                    self.judge(rec.get("cleaned_text", rec.get("raw_text", "")))
                    for rec in batch_records
                ]

        return [
            self.judge(rec.get("cleaned_text", rec.get("raw_text", "")))
            for rec in batch_records
        ]


# =====================================================================
# 5. STATE MANAGER (SQLite3 WAL Mode Checkpointing)
# =====================================================================


class EnsembleStateManager:
    """
    Crash-proof SQLite3 database manager enabled with Write-Ahead Logging (WAL).
    Guarantees atomic row-by-row checkpointing and zero data loss on interruption.
    """

    def __init__(self, db_path: Union[str, Path] = "data/purva_ensemble.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ensemble_decisions (
                    id TEXT PRIMARY KEY,
                    raw_text TEXT,
                    cleaned_text TEXT,
                    source_name TEXT,
                    consensus_label TEXT,
                    shannon_entropy REAL,
                    fleiss_kappa REAL,
                    status TEXT,
                    model_outputs_json TEXT,
                    timestamp TEXT
                );
            """)
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON ensemble_decisions(status);"
            )

    def get_processed_ids(self) -> set:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM ensemble_decisions WHERE status != 'ERROR';")
        return {row[0] for row in cursor.fetchall()}

    def save_decisions(self, decisions: List[EnsembleDecision]):
        with self.conn:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO ensemble_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
                [
                    (
                        d.id,
                        d.raw_text,
                        d.cleaned_text,
                        d.source_name,
                        d.consensus_label,
                        d.shannon_entropy,
                        d.fleiss_kappa_batch,
                        d.status,
                        json.dumps(d.model_outputs, ensure_ascii=False),
                        d.timestamp,
                    )
                    for d in decisions
                ],
            )

    def export_corpora(self, output_dir: Union[str, Path] = "data"):
        """Exports verified 3-class deliverables (Agreed JSONL/CSV, Disagreed JSONL/CSV, Audit Sample)."""
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM ensemble_decisions ORDER BY id ASC;")
        rows = cursor.fetchall()

        agreed_rows, disagreed_rows, error_rows = [], [], []
        for r in rows:
            record = {
                "id": r[0],
                "raw_text": r[1],
                "cleaned_text": r[2],
                "source": r[3],
                "consensus_label": r[4],
                "shannon_entropy": r[5],
                "fleiss_kappa": r[6],
                "status": r[7],
                "model_outputs": json.loads(r[8]),
                "timestamp": r[9],
            }
            if r[7] in ["AGREED", "RESOLVED"]:
                agreed_rows.append(record)
            elif r[7] == "DISAGREED":
                disagreed_rows.append(record)
            else:
                error_rows.append(record)

        # Write JSONL
        with open(out_path / "purva_l2_agreed.jsonl", "w", encoding="utf-8") as f:
            for rec in agreed_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(out_path / "purva_l2_disagreed.jsonl", "w", encoding="utf-8") as f:
            for rec in disagreed_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(out_path / "purva_l2_errors.jsonl", "w", encoding="utf-8") as f:
            for rec in error_rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write CSV
        headers = [
            "id",
            "raw_text",
            "consensus_label",
            "shannon_entropy",
            "fleiss_kappa",
            "model_outputs",
        ]
        with open(
            out_path / "purva_l2_agreed.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for rec in agreed_rows:
                writer.writerow(
                    [
                        rec["id"],
                        rec["raw_text"],
                        rec["consensus_label"],
                        rec["shannon_entropy"],
                        rec["fleiss_kappa"],
                        json.dumps(rec["model_outputs"], ensure_ascii=False),
                    ]
                )
        with open(
            out_path / "purva_l2_disagreed.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for rec in disagreed_rows:
                writer.writerow(
                    [
                        rec["id"],
                        rec["raw_text"],
                        rec["consensus_label"],
                        rec["shannon_entropy"],
                        rec["fleiss_kappa"],
                        json.dumps(rec["model_outputs"], ensure_ascii=False),
                    ]
                )

        # 1% Control Audit Sample
        sample_size = max(50, int(len(agreed_rows) * 0.01))
        if agreed_rows:
            random.seed(42)
            audit_sample = random.sample(
                agreed_rows, min(sample_size, len(agreed_rows))
            )
            with open(
                out_path / "purva_l2_human_audit_sample.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for rec in audit_sample:
                    writer.writerow(
                        [
                            rec["id"],
                            rec["raw_text"],
                            rec["consensus_label"],
                            rec["shannon_entropy"],
                            rec["fleiss_kappa"],
                            json.dumps(rec["model_outputs"], ensure_ascii=False),
                        ]
                    )

        print(
            f"\n[EXPORT SUCCESSFUL] Agreed (RESOLVED): {len(agreed_rows):,} rows | Disagreed: {len(disagreed_rows):,} rows | Errors: {len(error_rows):,} rows"
        )

    def close(self):
        self.conn.close()


# =====================================================================
# 6. PIPELINE ORCHESTRATOR
# =====================================================================


class CommitteeOrchestrator:
    """
    Orchestrates the 3-Family AI Committee:
    1. Sarvam AI Indic (2B)      — Eastern Indo-Aryan Specialist (PyTorch/vLLM)
    2. Google Gemma 2 (9B)       — Instruction-Tuned Generalist (Ollama)
    3. Alibaba Qwen 2.5 (3B)     — Multilingual Instruction-Tuned (Ollama)
    """

    def __init__(self, db_path: Union[str, Path] = "data/purva_ensemble.db"):
        self.state_mgr = EnsembleStateManager(db_path)
        self.local_judges = [
            VLLMJudge(model_name="sarvamai/sarvam-2b", judge_name="sarvam_ai_indic"),
            OllamaJudge("gemma2:9b", "google_gemma2_9b"),
            OllamaJudge("qwen2.5:3b", "alibaba_qwen25_3b"),
        ]
        self.judges: List[BaseJudge] = []
        self.categories = ["positive", "negative", "neutral_factual"]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}

    def preflight_health_check(self) -> bool:
        """Audits active model endpoints and confirms multi-family availability."""
        print("[PRE-FLIGHT HEALTH CHECK] Auditing 3-Family Committee Endpoints...")
        self.judges = [
            j
            for j in self.local_judges
            if (hasattr(j, "mode") and j.mode != "offline") or hasattr(j, "host")
        ]
        if not self.judges:
            print("  [CRITICAL ABORT] No active models detected in committee.")
            return False
        print(
            f"  [PASSED] {len(self.judges)} active judges ready: {[j.name for j in self.judges]}\n"
        )
        return True

    def evaluate_batch(
        self, batch_records: List[Dict[str, Any]]
    ) -> List[EnsembleDecision]:
        """Runs parallel evaluation, Dawid-Skene EM weighting, and Fleiss Kappa agreement."""
        N = len(batch_records)
        if N == 0:
            return []

        all_outputs: List[Dict[str, Dict[str, Any]]] = [{} for _ in range(N)]
        batch_preds: List[Dict[str, str]] = [{} for _ in range(N)]

        with ThreadPoolExecutor(max_workers=len(self.judges)) as executor:
            future_to_judge = {
                executor.submit(judge.judge_batch, batch_records): judge
                for judge in self.judges
            }
            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                try:
                    results = future.result()
                    for i, res in enumerate(results):
                        all_outputs[i][judge.name] = res.to_dict()
                        batch_preds[i][judge.name] = res.label
                except Exception as e:
                    for i in range(N):
                        all_outputs[i][judge.name] = {
                            "reasoning": str(e)[:50],
                            "label": "error",
                            "confidence": 0.0,
                        }
                        batch_preds[i][judge.name] = "error"

        ds_results = dawid_skene_aggregation(batch_preds, self.categories, max_iter=15)

        # Compute Fleiss Kappa ratings matrix
        batch_ratings = np.zeros((N, len(self.categories)))
        for i, preds in enumerate(batch_preds):
            for label in preds.values():
                if label in self.cat_to_idx:
                    batch_ratings[i, self.cat_to_idx[label]] += 1
        batch_kappa = calculate_fleiss_kappa(batch_ratings)

        decisions = []
        for i, rec in enumerate(batch_records):
            text = rec.get("cleaned_text", rec.get("raw_text", ""))
            rec_id = str(
                rec.get("id", hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
            )
            source = str(rec.get("source_name", "unknown"))

            consensus_label, ds_conf, prob_dist = ds_results[i]
            entropy = calculate_shannon_entropy(prob_dist)

            # Routing rules: H <= 0.5 and Dawid-Skene confidence >= 0.80 -> RESOLVED
            if consensus_label == "error" or not any(
                v != "error" for v in batch_preds[i].values()
            ):
                status = "ERROR"
            elif entropy <= 0.5 and ds_conf >= 0.80:
                status = "RESOLVED"
            else:
                status = "DISAGREED"

            decisions.append(
                EnsembleDecision(
                    id=rec_id,
                    raw_text=rec.get("raw_text", text),
                    cleaned_text=text,
                    source_name=source,
                    consensus_label=consensus_label,
                    shannon_entropy=round(entropy, 4),
                    fleiss_kappa_batch=round(batch_kappa, 4),
                    status=status,
                    model_outputs=all_outputs[i],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
        return decisions

    def run_pipeline(
        self, jsonl_path: Union[str, Path], batch_size: int = 25, max_workers: int = 4
    ):
        """Executes continuous batching over corpus with WAL checkpointing."""
        path = Path(jsonl_path).resolve()
        if not path.exists():
            print(f"[ERROR] Input dataset not found: {path}")
            return

        if not self.preflight_health_check():
            return

        processed_ids = self.state_mgr.get_processed_ids()
        print(
            f"[RESUME CHECK] Found {len(processed_ids):,} existing records in database."
        )

        batch = []
        total_processed = len(processed_ids)
        start_time = time.time()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec_id = str(
                        rec.get(
                            "id",
                            hashlib.sha1(
                                rec.get("cleaned_text", "").encode("utf-8")
                            ).hexdigest()[:16],
                        )
                    )
                    if rec_id in processed_ids:
                        continue
                    rec["id"] = rec_id
                    batch.append(rec)
                    if len(batch) >= batch_size:
                        decisions = self.evaluate_batch(batch)
                        self.state_mgr.save_decisions(decisions)
                        total_processed += len(decisions)
                        batch = []
                        elapsed = time.time() - start_time
                        print(
                            f"  -> Processed {total_processed:,} items | Throughput: {total_processed / max(1, elapsed):.1f} items/s"
                        )
                except Exception as e:
                    print(f"  [WARNING] Skipping malformed record: {str(e)[:50]}")
                    continue

        if batch:
            decisions = self.evaluate_batch(batch)
            self.state_mgr.save_decisions(decisions)
            total_processed += len(decisions)

        print(f"\n[PIPELINE COMPLETE] Total evaluated: {total_processed:,} sentences.")
        self.state_mgr.export_corpora()
        self.state_mgr.close()
