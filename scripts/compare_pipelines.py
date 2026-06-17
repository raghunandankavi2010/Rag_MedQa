#!/usr/bin/env python3
"""Real comparative evaluation of FIVE RAG pipeline configurations on MedQuAD.

This script reuses the components implemented in ``notebooks/rag_corrected.py``
(safety guardrail, multi-query expander, BM25 index, hybrid RRF retriever,
cross-encoder/TF-IDF re-ranker, generator) to construct the five distinct
pipelines described in the thesis and evaluate them under identical conditions:

  P1  Vanilla LLM            - no retrieval (parametric knowledge only)
  P2  Standard RAG           - dense semantic retrieval (top-k)
  P3  Multi-Query Expansion  - 3 paraphrases + dense retrieval, merged
  P4  Hybrid Retrieval       - dense + BM25 fused with Reciprocal Rank Fusion
  P5  Query Reformulation    - query rewrite + dense retrieval + re-ranking

Controlled variables (identical across all RAG pipelines): generator model,
generation prompt, embedding model, chunk size/overlap, top-k, temperature.
The dense vector store is loaded from the persisted ChromaDB created by the
main pipeline, so no re-embedding cost is incurred.

Outputs (written to reports/):
  pipeline_comparison_raw.csv       - per-question, per-pipeline metrics
  pipeline_comparison_summary.csv   - aggregate mean per pipeline
  pipeline_comparison_by_qtype.csv  - faithfulness by question type x pipeline
  pipeline_comparison_anova.txt     - one-way ANOVA on faithfulness

Usage:
    RUN_EVAL=true python scripts/compare_pipelines.py            # full metrics
    MAX_Q=50 python scripts/compare_pipelines.py
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
for p in (ROOT, NB):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

# Import the implemented components (this also loads .env and validates the key)
import rag_corrected as rc
from langchain_core.messages import SystemMessage, HumanMessage

MAX_Q = int(os.getenv("MAX_Q", "50"))
TOP_K = 5  # controlled variable: documents passed to the generator


# ─────────────────────────────────────────────────────────────────────────────
# Question-type classification (heuristic, from MedQuAD's templated questions)
# ─────────────────────────────────────────────────────────────────────────────
def classify_qtype(question: str) -> str:
    q = question.lower()
    if "symptom" in q:
        return "symptoms"
    if "treat" in q or "therap" in q:
        return "treatment"
    if "diagnos" in q:
        return "diagnosis"
    if "cause" in q or "risk" in q or "inherit" in q:
        return "causes"
    if "prevent" in q:
        return "prevention"
    if "outlook" in q or "prognos" in q:
        return "prognosis"
    if "how many" in q or "how common" in q or "frequency" in q:
        return "frequency"
    return "information"


# ─────────────────────────────────────────────────────────────────────────────
# Shared generation helpers
# ─────────────────────────────────────────────────────────────────────────────
_VANILLA_SYSTEM = (
    "You are a board-certified medical information assistant. Answer the "
    "question using your medical knowledge. Distinguish facts from uncertainty "
    "with hedges where appropriate. ALWAYS end with: 'Please consult a qualified "
    "healthcare professional before making any clinical decisions.' Never "
    "fabricate drug interactions or dosages."
)


def _ctx_string(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline definitions (each exposes .query(question) -> result dict)
# ─────────────────────────────────────────────────────────────────────────────
class VanillaPipeline:
    name = "P1_Vanilla_LLM"

    def __init__(self, gen_llm):
        self.gen_llm = gen_llm

    def query(self, question):
        t0 = time.time()
        resp = self.gen_llm.invoke([
            SystemMessage(content=_VANILLA_SYSTEM),
            HumanMessage(content=question),
        ])
        return {
            "answer": resp.content.strip(), "context": [], "num_docs": 0,
            "queries": [question], "blocked": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


class StandardRAGPipeline:
    name = "P2_Standard_RAG"

    def __init__(self, gen_llm, vectorstore, k=TOP_K):
        self.gen_llm = gen_llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def query(self, question):
        t0 = time.time()
        docs = self.retriever.invoke(question)
        ans = rc._call_generator_llm(question, _ctx_string(docs), self.gen_llm)
        return {
            "answer": ans, "context": docs, "num_docs": len(docs),
            "queries": [question], "blocked": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


class MultiQueryPipeline:
    name = "P3_Multi_Query_Expansion"

    def __init__(self, gen_llm, vectorstore, k=TOP_K):
        self.gen_llm = gen_llm
        self.vectorstore = vectorstore
        self.k = k
        self.expander = rc.MultiQueryExpander(n_variants=3)

    def query(self, question):
        t0 = time.time()
        queries = self.expander.expand(question)
        seen, merged = set(), []
        for q in queries:
            for d in self.vectorstore.similarity_search(q, k=self.k):
                key = d.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        docs = merged[: self.k * 2]
        ans = rc._call_generator_llm(question, _ctx_string(docs), self.gen_llm)
        return {
            "answer": ans, "context": docs, "num_docs": len(docs),
            "queries": queries, "blocked": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


class HybridPipeline:
    name = "P4_Hybrid_Retrieval"

    def __init__(self, gen_llm, vectorstore, bm25, final_k=TOP_K):
        self.gen_llm = gen_llm
        self.retriever = rc.HybridRetriever(
            vectorstore=vectorstore, bm25_index=bm25,
            dense_k=12, sparse_k=12, final_k=final_k,
        )

    def query(self, question):
        t0 = time.time()
        docs = self.retriever.invoke(question)
        ans = rc._call_generator_llm(question, _ctx_string(docs), self.gen_llm)
        return {
            "answer": ans, "context": docs, "num_docs": len(docs),
            "queries": [question], "blocked": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


_REFORMULATE_SYSTEM = (
    "You are a medical query-rewriting expert. Rewrite the user's question into "
    "a single clear, specific, clinically-phrased search query that preserves the "
    "original intent. Return ONLY the rewritten query, no explanation."
)


class ReformulationPipeline:
    name = "P5_Query_Reformulation"

    def __init__(self, gen_llm, aux_llm, vectorstore, reranker, k=TOP_K):
        self.gen_llm = gen_llm
        self.aux_llm = aux_llm
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.k = k

    def _reformulate(self, question):
        try:
            resp = self.aux_llm.invoke([
                SystemMessage(content=_REFORMULATE_SYSTEM),
                HumanMessage(content=question),
            ])
            rewritten = resp.content.strip().strip('"')
            return rewritten or question
        except Exception:
            return question

    def query(self, question):
        t0 = time.time()
        rewritten = self._reformulate(question)
        candidates = self.vectorstore.similarity_search(rewritten, k=self.k * 2)
        docs = self.reranker.rerank(rewritten, candidates, top_k=self.k)
        ans = rc._call_generator_llm(question, _ctx_string(docs), self.gen_llm)
        return {
            "answer": ans, "context": docs, "num_docs": len(docs),
            "queries": [question, rewritten], "blocked": False,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# ROUGE helper (per-row)
# ─────────────────────────────────────────────────────────────────────────────
def add_rouge(df):
    if not getattr(rc, "ROUGE_AVAILABLE", False):
        df["rouge1_f"] = 0.0
        df["rougeL_f"] = 0.0
        return df
    r1, rL = [], []
    for pred, ref in zip(df["generated_answer"], df["ground_truth"]):
        s = rc._ROUGE.score(str(ref), str(pred))
        r1.append(s["rouge1"].fmeasure)
        rL.append(s["rougeL"].fmeasure)
    df["rouge1_f"] = r1
    df["rougeL_f"] = rL
    return df


def main():
    print("=== DATA LOADING ===")
    df = rc.load_medquad_csv()
    from sklearn.model_selection import train_test_split
    corpus_df, eval_df = train_test_split(df, test_size=0.01, random_state=42)
    eval_df = eval_df.copy()
    eval_df["qtype"] = eval_df["question"].apply(classify_qtype)
    print(f"  Corpus: {len(corpus_df):,}  |  Eval: {len(eval_df):,}  |  Using: {MAX_Q}")
    print("  Eval qtype distribution:")
    print(eval_df.head(MAX_Q)["qtype"].value_counts().to_string())

    # Shared components -------------------------------------------------------
    print("\n=== BUILDING SHARED COMPONENTS ===")
    embeddings = rc.OpenAIEmbeddings(model=rc.CFG.EMBEDDING_MODEL)
    vectorstore = rc.Chroma(
        persist_directory=rc.CFG.VECTORDB_DIR, embedding_function=embeddings,
    )
    print(f"  Loaded persisted vector store ({vectorstore._collection.count():,} vectors)")
    docs = rc.build_documents(corpus_df)
    bm25 = rc.BM25Index(docs)
    reranker = rc.CrossEncoderReranker()
    reranker.fit(docs)
    gen_llm = rc.ChatOpenAI(model=rc.CFG.GENERATION_MODEL, temperature=0.0)
    aux_llm = rc.ChatOpenAI(model=rc.CFG.AUX_MODEL, temperature=0.3)
    print(f"  Generator: {rc.CFG.GENERATION_MODEL}  |  Aux: {rc.CFG.AUX_MODEL}")

    pipelines = [
        VanillaPipeline(gen_llm),
        StandardRAGPipeline(gen_llm, vectorstore),
        MultiQueryPipeline(gen_llm, vectorstore),
        HybridPipeline(gen_llm, vectorstore, bm25),
        ReformulationPipeline(gen_llm, aux_llm, vectorstore, reranker),
    ]

    run_ragas = rc.RAGAS_AVAILABLE
    run_deepeval = rc.DEEPEVAL_AVAILABLE
    print(f"\n  RAGAS={run_ragas}  DeepEval={run_deepeval}  ROUGE={rc.ROUGE_AVAILABLE}")

    all_frames = []
    for pipe in pipelines:
        # Resume support: skip pipelines already saved (e.g. after a crash or
        # API-quota interruption); re-running only evaluates what is missing.
        part_path = os.path.join(rc.CFG.RESULTS_DIR, f"cmp_{pipe.name}.csv")
        if os.path.exists(part_path):
            print(f"\n=== SKIPPING {pipe.name} (already saved: {part_path}) ===")
            all_frames.append(pd.read_csv(part_path))
            continue
        print(f"\n=== EVALUATING {pipe.name} ===")
        res = rc.run_evaluation(
            pipeline=pipe, eval_df=eval_df, max_questions=MAX_Q,
            run_ragas=run_ragas, run_deepeval=run_deepeval,
        )
        res = add_rouge(res)
        res.insert(0, "pipeline", pipe.name)
        # Persist immediately so progress survives interruptions.
        res.to_csv(part_path, index=False)
        print(f"  Saved {pipe.name} -> {part_path}")
        all_frames.append(res)

    combined = pd.concat(all_frames, ignore_index=True)
    os.makedirs(rc.CFG.RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(rc.CFG.RESULTS_DIR, "pipeline_comparison_raw.csv")
    combined.to_csv(raw_path, index=False)
    print(f"\nSaved raw per-question results -> {raw_path}")

    # Aggregate summary -------------------------------------------------------
    metric_cols = [
        "faithfulness_proxy", "latency_ms", "word_overlap", "safety_compliance",
        "num_docs", "ragas_faithfulness", "ragas_answer_relevance",
        "ragas_context_precision", "ragas_context_recall",
        "deepeval_hallucination", "deepeval_relevance", "rouge1_f", "rougeL_f",
        "failure_missing_evidence", "failure_noisy_evidence",
        "failure_unsupported_claims", "failure_unsafe_tone",
    ]
    metric_cols = [c for c in metric_cols if c in combined.columns]
    summary = combined.groupby("pipeline")[metric_cols].mean().round(4)
    summary["n"] = combined.groupby("pipeline").size()
    sum_path = os.path.join(rc.CFG.RESULTS_DIR, "pipeline_comparison_summary.csv")
    summary.to_csv(sum_path)
    print(f"Saved aggregate summary -> {sum_path}")
    print("\n=== AGGREGATE SUMMARY ===")
    show = [c for c in ["ragas_faithfulness", "ragas_context_precision",
            "ragas_context_recall", "ragas_answer_relevance",
            "deepeval_hallucination", "deepeval_relevance", "rouge1_f",
            "rougeL_f", "safety_compliance", "latency_ms"] if c in summary.columns]
    print(summary[show].to_string())

    # By question type (faithfulness) ----------------------------------------
    if "ragas_faithfulness" in combined.columns:
        by_qtype = combined.pivot_table(
            index="qtype", columns="pipeline",
            values="ragas_faithfulness", aggfunc="mean",
        ).round(4)
        qt_path = os.path.join(rc.CFG.RESULTS_DIR, "pipeline_comparison_by_qtype.csv")
        by_qtype.to_csv(qt_path)
        print(f"\nSaved by-qtype faithfulness -> {qt_path}")
        print(by_qtype.to_string())

    # One-way ANOVA on faithfulness ------------------------------------------
    try:
        from scipy import stats
        groups = [g["ragas_faithfulness"].dropna().values
                  for _, g in combined.groupby("pipeline")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            anova_txt = (
                f"One-way ANOVA on RAGAS faithfulness across pipelines\n"
                f"  k (groups) = {len(groups)}\n"
                f"  N (total)  = {sum(len(g) for g in groups)}\n"
                f"  F          = {f_stat:.4f}\n"
                f"  p-value    = {p_val:.3e}\n"
            )
            anova_path = os.path.join(rc.CFG.RESULTS_DIR, "pipeline_comparison_anova.txt")
            with open(anova_path, "w") as fh:
                fh.write(anova_txt)
            print(f"\n=== ANOVA ===\n{anova_txt}")
    except Exception as e:
        print(f"  [ANOVA] skipped: {e}")

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
