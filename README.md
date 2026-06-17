# Retrieval-Augmented Generation for Medical Question Answering

**A Comparative Study of RAG Pipelines for Reducing Hallucination in Clinical Decision Support Systems**

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [System Architecture](#system-architecture)
- [Five Pipeline Configurations](#five-pipeline-configurations)
- [Usage](#usage)
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

All pipelines are evaluated using **RAGAS** metrics (faithfulness, context precision, context recall, answer relevance) and **DeepEval** metrics (hallucination rate, groundedness, correctness, safety compliance) on the **MedQuAD** dataset containing 47,457 question-answer pairs from 12 NIH sources.

**Key Result:** Hybrid retrieval achieves **0.89 faithfulness** and **0.11 hallucination rate** — a **47-point improvement** and **81% relative reduction** over the vanilla baseline.

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
5. Analyse results by medical question type (diagnosis, treatment, medication, symptoms).
6. Identify and document failure patterns for each pipeline.


## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MedQuAD CSV   │────▶│  Text Splitter  │────▶│ Vector Database │
│  (47,457 Q&A)   │     │ (400t / 50ovlp) │     │    (ChromaDB)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌─────────────────┐            │
│   GPT-4o-mini   │◀───│  Prompt Engine  │◀───────────┘
│   (Generator)   │     │  (Safety + RAG) │
└─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │RAGAS Metrics│  │DeepEval     │  │Statistical Testing      │  │
│  │Faithfulness │  │Hallucination│  │ANOVA + Tukey HSD        │  │
│  │Precision    │  │Groundedness │  │Pearson Correlation      │  │
│  │Recall       │  │Correctness  │  │                         │  │
│  │Relevance    │  │Safety       │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Five Pipeline Configurations

| Pipeline | Retrieval Strategy | Description |
|----------|-------------------|-------------|
| **P1** | Vanilla LLM | No retrieval; baseline using only parametric knowledge |
| **P2** | Standard RAG | Dense semantic retrieval with vector embeddings |
| **P3** | Multi-Query Expansion | Generates 4 query variants, fuses results |
| **P4** | Hybrid Retrieval | Combines dense + BM25 via Reciprocal Rank Fusion (RRF) |
| **P5** | Query Reformulation + Reranking | Rewrites query, then reranks with cross-encoder |

**Controlled Variables (fixed across all pipelines):**
- LLM: `GPT-4o-mini` (temperature = 0.0)
- Embedding: `text-embedding-3-small`
- Chunking: 400 tokens, 50-token overlap
- Top-k: 5 documents
- Prompt: Safety-oriented clinical template

## Results

### RAGAS Faithfulness

| Pipeline | Faithfulness | Context Precision | Context Recall |
|----------|-------------:|------------------:|---------------:|
| Vanilla LLM | 0.42 | — | — |
| Standard RAG | 0.78 | 0.74 | 0.71 |
| Multi-Query Expansion | 0.81 | 0.68 | 0.79 |
| **Hybrid Retrieval** | **0.89** | **0.86** | **0.83** |
| Query Reformulation | 0.85 | 0.82 | 0.76 |

### DeepEval Hallucination & Safety

| Pipeline | Hallucination Rate | Groundedness | Safety Compliance |
|----------|-------------------:|-------------:|------------------:|
| Vanilla LLM | 0.58 | 0.44 | 0.45 |
| Standard RAG | 0.22 | 0.76 | 0.84 |
| Multi-Query Expansion | 0.19 | 0.79 | 0.86 |
| **Hybrid Retrieval** | **0.11** | **0.88** | **0.92** |
| Query Reformulation | 0.15 | 0.84 | 0.89 |

### Statistical Validation

- **ANOVA:** Significant differences in faithfulness across pipelines (*p* < 0.001)
- **Post-hoc Tukey HSD:** Hybrid > Standard > Multi-Query > Reformulation > Vanilla
- **Pearson correlation:** Faithfulness vs Hallucination *r* = -0.97


## Key Findings

1. **Retrieval grounding dramatically reduces hallucination** — 62–81% relative improvement across all RAG variants.
2. **Hybrid retrieval is the standout performer** — highest faithfulness (0.89), lowest hallucination (0.11), highest safety (0.92).
3. **Multi-query expansion improves recall but introduces noise** — a clear precision-coverage trade-off.
4. **Query reformulation improves precision but adds latency** (3,200ms), limiting real-time viability.
5. **All RAG pipelines exceed the clinical safety threshold of 0.80**; the vanilla baseline is clinically unsafe at 0.45.
6. **Symptom questions are easiest** to answer faithfully; treatment questions remain challenging due to patient-specific factors.

## Acknowledgements

- **Liverpool John Moores University** — for academic supervision and research guidance.
- **upGrad** — for providing the MSc AI & ML platform and resources.
- **NIH / MedQuAD** — Ben Abacha and Demner-Fushman (2019) for the dataset.
- **OpenAI** — for API access to GPT-4o-mini and embedding models.
- **Open-source community** — LangChain, ChromaDB, RAGAS, and DeepEval teams.
