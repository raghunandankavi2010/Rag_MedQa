# Retrieval-Augmented Generation for Medical Question Answering

**A Comparative Study of RAG Pipelines for Reducing Hallucination in Clinical Decision Support Systems**

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Five Pipeline Configurations](#five-pipeline-configurations)
- [Results](#results)
- [Key Findings](#key-findings)
- [Acknowledgements](#acknowledgements)

---

## Abstract

Medical question-answering systems powered by large language models (LLMs) frequently generate factually incorrect or unsupported information — a phenomenon known as **hallucination**. In healthcare, even minor inaccuracies in drug recommendations or symptom interpretations can lead to adverse patient outcomes.

This repository contains the complete implementation, evaluation framework, and results for a **controlled comparative study of five RAG pipeline configurations** designed to reduce hallucination in medical QA:

- **Vanilla LLM** (baseline, no retrieval)
- **Standard RAG** (dense semantic retrieval)
- **Multi-Query Expansion RAG**
- **Hybrid Retrieval RAG** (semantic + BM25 via Reciprocal Rank Fusion)
- **Query Reformulation + Reranking RAG**

All pipelines are evaluated using **RAGAS** metrics (faithfulness, context precision, context recall, answer relevance) and **DeepEval** metrics (hallucination rate, safety compliance) on the **MedQuAD** dataset. Each pipeline was evaluated on n=50 questions (N=250 total).

**Key Result:** Multi-Query Expansion achieves the highest faithfulness (**0.614**) — a **23-point improvement** over the vanilla baseline (0.381). Standard RAG and Query Reformulation achieve the lowest hallucination rate (**0.248**), a **22.5% relative reduction** from the vanilla baseline (0.320).

---

## Problem Statement

Current LLMs suffer from three critical limitations in clinical settings:

1. **Knowledge Currency** — Medical guidelines change continuously; static models cannot incorporate new drug approvals or revised protocols.
2. **Factual Precision** — Semantic similarity blurs clinically distinct concepts (e.g., Type 1 vs Type 2 diabetes).
3. **Explainability & Safety** — Clinicians need to trace recommendations to sources; black-box answers break norms of medical reasoning.

Most published studies evaluate a single RAG configuration with generic metrics (BLEU, BERTScore). There is **no controlled comparative evidence** guiding retrieval method selection for medical QA from the perspective of hallucination reduction and safety compliance.

---

## Research Objectives

1. Prepare a clean MedQuAD evaluation set and retrieval corpus (400-token chunks, 50-token overlap).
2. Implement five comparable pipelines using identical generator, prompt, and decoding settings.
3. Run all pipelines on the same question set with full traceability and reproducibility.
4. Measure performance using RAGAS and DeepEval metrics, reporting aggregate results.
5. Analyse results by medical question type (diagnosis, treatment, causes, symptoms).
6. Identify and document failure patterns for each pipeline.

---

## Dataset

**MedQuAD** — curated from 12 NIH sources (Ben Abacha & Demner-Fushman, 2019).

| Statistic | Value |
|-----------|-------|
| Total QA pairs (after cleaning) | 16,407 |
| Unique medical focus terms | 5,125 |
| Unique source files | 3,497 |
| Mean answer length (words) | 201 |
| Median answer length (words) | 138 |
| Mean question length (words) | 8 |

Corpus chunked at **400 tokens** with **50-token overlap** and indexed in ChromaDB.

---

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MedQuAD CSV   │────▶│  Text Splitter  │────▶│ Vector Database │
│  (16,407 Q&A)   │     │ (400t / 50ovlp) │     │    (ChromaDB)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐            │
│   GPT-4o-mini   │◀────│  Prompt Engine  │◀───────────┘
│   (Generator)   │     │  (Safety + RAG) │
└─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │RAGAS Metrics│  │DeepEval     │  │Statistical Testing      │  │
│  │Faithfulness │  │Hallucination│  │One-way ANOVA            │  │
│  │Ctx Precision│  │Relevance    │  │F=2.08, p=0.084          │  │
│  │Ctx Recall   │  │Safety       │  │                         │  │
│  │Answer Rel.  │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Five Pipeline Configurations

| Pipeline | Retrieval Strategy | Description |
|----------|-------------------|-------------|
| **P1** | Vanilla LLM | No retrieval; baseline using only parametric knowledge |
| **P2** | Standard RAG | Dense semantic retrieval with vector embeddings |
| **P3** | Multi-Query Expansion | Generates 4 query variants, fuses results (avg 8.14 docs) |
| **P4** | Hybrid Retrieval | Combines dense + BM25 via Reciprocal Rank Fusion (RRF) |
| **P5** | Query Reformulation + Reranking | Rewrites query, then reranks with cross-encoder |

**Controlled Variables (fixed across all pipelines):**
- LLM: `GPT-4o-mini` (temperature = 0.0)
- Embedding: `text-embedding-3-small`
- Chunking: 400 tokens, 50-token overlap
- Top-k: 5 documents
- Prompt: Safety-oriented clinical template

---

## Results

### Aggregate Pipeline Performance (n=50 per pipeline)

| Pipeline | Faithfulness ↑ | Ans. Relevance ↑ | Ctx Precision ↑ | Ctx Recall ↑ | Hallucination ↓ | Latency (ms) ↓ |
|----------|:--------------:|:----------------:|:---------------:|:------------:|:---------------:|:--------------:|
| P1 Vanilla LLM | 0.381 | **0.425** | 0.200 | 0.293 | 0.320 | 4,660 |
| P2 Standard RAG | 0.494 | 0.185 | **0.725** | 0.512 | **0.248** | **3,107** |
| P3 Multi-Query Expansion | **0.614** | 0.239 | 0.719 | **0.578** | 0.283 | 41,607 |
| P4 Hybrid Retrieval | 0.485 | 0.238 | 0.642 | 0.532 | 0.280 | 3,360 |
| P5 Query Reformulation | 0.512 | 0.182 | 0.656 | 0.468 | **0.248** | 3,969 |

Bold = best in column. ↑ higher is better, ↓ lower is better.

### RAGAS Faithfulness by Question Type

| Question Type | P1 Vanilla | P2 Std RAG | P3 Multi-Query | P4 Hybrid | P5 Q-Reform |
|---------------|:----------:|:----------:|:--------------:|:---------:|:-----------:|
| Causes | 0.333 | 0.744 | **0.801** | 0.714 | 0.798 |
| Diagnosis | 0.500 | 0.462 | **0.933** | 0.250 | 0.375 |
| Frequency | 0.000 | 0.333 | **0.333** | 0.000 | 0.333 |
| Information | 0.657 | 0.635 | **0.699** | 0.670 | 0.629 |
| Symptoms | 0.110 | 0.161 | **0.272** | 0.216 | 0.272 |
| Treatment | 0.321 | 0.405 | **0.638** | 0.423 | 0.381 |

### Failure Mode Analysis

| Pipeline | Missing Evidence | Noisy Evidence | Unsupported Claims | Unsafe Tone |
|----------|:----------------:|:--------------:|:------------------:|:-----------:|
| P1 Vanilla LLM | 1.000 | 0.000 | 1.000 | 0.000 |
| P2 Standard RAG | 0.100 | 0.820 | 0.000 | 0.000 |
| P3 Multi-Query Expansion | 0.020 | **0.920** | 0.000 | 0.000 |
| P4 Hybrid Retrieval | 0.100 | 0.860 | 0.000 | 0.000 |
| P5 Query Reformulation | 0.100 | 0.820 | 0.000 | 0.000 |

### Statistical Testing

One-way ANOVA on RAGAS faithfulness across all five pipelines:

- **F(4, 245) = 2.08**, **p = 0.084**

The difference in mean faithfulness across pipelines does not reach statistical significance at α=0.05, suggesting the observed ranking should be interpreted cautiously given n=50 per group.

---

## Key Findings

1. **Multi-Query Expansion achieves the highest faithfulness (0.614)** — a 23-point absolute improvement over the vanilla baseline (0.381), and the best performance across every question type.
2. **Retrieval grounding reduces hallucination** — Standard RAG and Query Reformulation both achieve 0.248 hallucination rate, a 22.5% relative reduction from the vanilla baseline (0.320).
3. **Standard RAG delivers the best context precision (0.725)** and the lowest latency among retrieval pipelines (3,107 ms), making it the most practical choice for real-time use.
4. **Multi-Query Expansion is extremely slow (41,607 ms per query)** — its faithfulness gains come at the cost of ~13× latency versus Standard RAG, making it unsuitable for interactive systems.
5. **The vanilla baseline fails entirely on missing-evidence and unsupported-claim dimensions** (both 1.00) — confirming that retrieval grounding is essential for safe medical QA.
6. **Noisy evidence is the dominant failure mode for all RAG pipelines** (0.82–0.92), indicating that retrieved context often contains irrelevant passages that the model must ignore.
7. **Diagnosis questions benefit most from retrieval** — P3 achieves 0.933 faithfulness on diagnosis versus 0.500 for the vanilla baseline.
8. **Frequency questions are consistently hardest** — all pipelines score ≤ 0.333, suggesting this category requires specialised handling.
9. **ANOVA is not significant (p = 0.084)** — differences across pipelines are directionally consistent but do not reach significance at n=50 per group; a larger evaluation set is recommended.

---

## Acknowledgements

- **Liverpool John Moores University** — for academic supervision and research guidance.
- **upGrad** — for providing the MSc AI & ML platform and resources.
- **NIH / MedQuAD** — Ben Abacha and Demner-Fushman (2019) for the dataset.
- **OpenAI** — for API access to GPT-4o-mini and embedding models.
- **Open-source community** — LangChain, ChromaDB, RAGAS, and DeepEval teams.
