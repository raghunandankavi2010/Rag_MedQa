"""
RAG Medical QA — Thesis Experiment: Baseline vs Enhanced RAG Comparison
======================================================================
This script compares 5 pipeline configurations on the MedQuAD dataset:
  Pipeline 1: Vanilla LLM (baseline, no retrieval)
  Pipeline 2: Standard RAG (dense semantic search)
  Pipeline 3: Multi-Query Expansion RAG
  Pipeline 4: Hybrid Retrieval RAG (RRF fusion)
  Pipeline 5: Query Reformulation + Reranking RAG

Evaluation metrics:
  - Accuracy (semantic match)
  - Word Overlap
  - RAGAS metrics (faithfulness, answer relevancy, context precision, context recall)
  - DeepEval metrics (hallucination, groundedness, correctness)
  - Safety Compliance
  - Latency
  - Failure Pattern Analysis

Outputs:
  - reports/thesis_experiment_results.csv   (per-question results)
  - reports/thesis_pipeline_summary.csv     (aggregate by pipeline)
  - reports/thesis_final_report.txt         (human-readable summary)
"""

import os
import sys
import time
import json
import warnings
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to a .env file.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5
TEMPERATURE = 0.0

# For quick testing, use a small eval set. For thesis results, use 50-100.
EVAL_SIZE = 8  # Quick experiment for real results (scale up for full thesis)
CORPUS_LIMIT = 0  # 0 means use all remaining data

# ========== EMBEDDING WRAPPER (RAGAS compatibility) ==========
class CompatibleOpenAIEmbeddings(OpenAIEmbeddings):
    """Wrapper that adds embed_query() for RAGAS compatibility."""
    def embed_query(self, text: str):
        return self.embed_documents([text])[0]

# Monkey-patch OpenAIEmbeddings for RAGAS compatibility
OpenAIEmbeddings.embed_query = lambda self, text: self.embed_documents([text])[0]

# ========== DATA LOADING ==========
def load_medquad_data(csv_path='data/medquad.csv'):
    """Load MedQuAD data from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"MedQuAD CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Normalize columns to expected names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['qtype', 'type']:
            col_map[col] = 'type'
        elif col_lower in ['source_file', 'source']:
            col_map[col] = 'source'
        elif col_lower == 'question':
            col_map[col] = 'question'
        elif col_lower == 'answer':
            col_map[col] = 'answer'
    
    df = df.rename(columns=col_map)
    
    # Ensure required columns exist
    for req in ['question', 'answer']:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' not found in CSV. Columns: {list(df.columns)}")
    
    if 'type' not in df.columns:
        df['type'] = 'general'
    if 'source' not in df.columns:
        df['source'] = 'unknown'
    
    return df[['question', 'answer', 'type', 'source']].copy()


def preprocess_data(df):
    """Preprocess with deduplication and cleaning."""
    initial_count = len(df)
    df = df.drop_duplicates(subset=['question'], keep='first')
    after_dedup = len(df)
    print(f"  Step 1 - Duplicate removal: {initial_count} -> {after_dedup}")
    
    df['question'] = df['question'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
    df['answer'] = df['answer'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
    df['question'] = df['question'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['answer'] = df['answer'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['question'] = df['question'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['answer'] = df['answer'].str.encode('ascii', 'ignore').str.decode('ascii')
    
    df = df[df['question'].str.len() > 5]
    df = df[df['answer'].str.len() > 5]
    df = df.reset_index(drop=True)
    print(f"  Final preprocessed count: {len(df)}")
    return df


def stratified_sample(df, n=50, stratify_col='type', seed=42):
    """Create stratified random sample for evaluation."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['question', 'answer', 'type', 'source'])
    
    type_counts = df[stratify_col].value_counts()
    total = len(df)
    sampled = []
    for qtype, count in type_counts.items():
        target = max(1, int(n * count / total))
        subset = df[df[stratify_col] == qtype]
        if len(subset) > target:
            sampled.append(subset.sample(n=target, random_state=seed))
        else:
            sampled.append(subset)
    result = pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed).reset_index(drop=True)
    return result


# ========== CHUNKING ==========
def fixed_chunk_documents(docs):
    """Fixed chunking: 400 tokens, 50 overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Fixed chunking: {len(docs)} docs -> {len(chunks)} chunks ({CHUNK_SIZE}/{CHUNK_OVERLAP})")
    return chunks


# ========== VECTOR STORE ==========
def build_vectorstore(chunks, persist_dir='./chroma_expt'):
    """Build Chroma vector store."""
    embeddings_local = CompatibleOpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_local,
        persist_directory=persist_dir,
    )
    return vs, embeddings_local


# ========== RETRIEVER FUNCTIONS ==========
class SimpleRetriever(BaseRetriever):
    """Simple retriever wrapper."""
    retrieve_func: callable = Field(description="Function to retrieve documents")
    k: int = Field(default=TOP_K, description="Number of documents to retrieve")
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.retrieve_func(query, topk=self.k)
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def semantic_retriever(vs):
    """Standard semantic retriever."""
    return vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})


def tfidf_retriever(chunks):
    """TF-IDF sparse retriever."""
    texts = [c.page_content for c in chunks]
    vect = TfidfVectorizer(stop_words='english').fit(texts)
    mat = vect.transform(texts)
    
    def retrieve(query, topk=TOP_K):
        qv = vect.transform([query])
        sims = cosine_similarity(qv, mat)[0]
        idx = np.argsort(sims)[::-1][:topk]
        return [chunks[i] for i in idx]
    
    return SimpleRetriever(retrieve_func=retrieve, k=TOP_K)


def hybrid_retriever(vs, chunks):
    """Hybrid retriever with Reciprocal Rank Fusion."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})
    tfidf = tfidf_retriever(chunks)
    
    def retrieve(query, topk=TOP_K):
        sem_docs = semantic.invoke(query)[:topk]
        tf_docs = tfidf.retrieve_func(query, topk=topk)
        
        doc_scores = {}
        doc_map = {}
        RRF_K = 60
        
        for rank, doc in enumerate(sem_docs, start=1):
            key = doc.page_content[:150]
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc
        
        for rank, doc in enumerate(tf_docs, start=1):
            key = doc.page_content[:150]
            doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc
        
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[k] for k in sorted_keys[:topk]]
    
    return SimpleRetriever(retrieve_func=retrieve, k=TOP_K)


def multiquery_retriever(vs, chunks, llm_model):
    """Multi-Query Expansion with LLM paraphrasing."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})
    
    def generate_paraphrases(query, n=3):
        paraphrase_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Generate {n} different ways to ask this medical question. Keep the same meaning but use different words."),
            ("human", "Original: {query}\n\nParaphrased versions (one per line):"),
        ])
        try:
            response = llm_model.invoke(paraphrase_prompt.format(query=query))
            text = response.content if hasattr(response, 'content') else str(response)
            versions = [line.strip('- 0123456789.').strip() for line in text.split('\n') if line.strip()]
            versions = [v for v in versions if len(v) > 10][:n]
            return [query] + versions if versions else [query]
        except Exception:
            return [query]
    
    def retrieve(query, topk=TOP_K):
        queries = generate_paraphrases(query, n=3)
        seen = set()
        merged = []
        for q in queries:
            docs = semantic.invoke(q)
            for d in docs:
                key = d.page_content[:150]
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
            if len(merged) >= topk * 2:
                break
        return merged[:topk * 2]
    
    return SimpleRetriever(retrieve_func=retrieve, k=TOP_K)


def reformulation_retriever(vs, chunks, llm_model):
    """Query Reformulation + Reranking."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K * 2})
    
    texts = [c.page_content for c in chunks]
    vect = TfidfVectorizer(stop_words='english', max_features=5000).fit(texts)
    
    def rerank(query, docs, topk=TOP_K):
        if not docs:
            return docs
        scores = []
        for i, doc in enumerate(docs):
            dv = vect.transform([doc.page_content])
            qv = vect.transform([query])
            score = cosine_similarity(qv, dv)[0][0]
            scores.append((score, i))
        scores.sort(reverse=True)
        return [docs[i] for _, i in scores[:topk]]
    
    def retrieve(query, topk=TOP_K):
        reform_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the following medical question to be clearer and more specific for document retrieval. Preserve the original intent."),
            ("human", "Original: {query}\n\nRewritten:"),
        ])
        try:
            response = llm_model.invoke(reform_prompt.format(query=query))
            reform = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            reform = reform if reform else query
        except Exception:
            reform = query
        
        candidates = semantic.invoke(reform)
        return rerank(reform, candidates, topk=topk)
    
    return SimpleRetriever(retrieve_func=retrieve, k=TOP_K)


# ========== CHAIN CREATION ==========
def create_modern_retrieval_chain(retriever, llm_model):
    """Create retrieval chain."""
    system_prompt = (
        "You are a helpful medical information assistant. Answer the user's question "
        "using ONLY the provided context. If the context does not contain sufficient information, "
        "say so clearly. Do not make up information. Always encourage the user to consult "
        "a qualified medical professional for personal health decisions."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    return retrieval_chain


def create_vanilla_chain(llm_model):
    """Pipeline 1: Vanilla LLM baseline — no retrieval."""
    system_prompt = (
        "You are a helpful medical information assistant. Answer the user's question "
        "based on your training knowledge. If you are unsure, say so clearly. "
        "Always encourage the user to consult a qualified medical professional."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | llm_model


# ========== EVALUATION METRICS ==========
def calculate_word_overlap(text1, text2):
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1:
        return 0.0
    return len(words1 & words2) / len(words1)


def evaluate_answer_semantic(ground_truth, generated_answer):
    if not ground_truth or not generated_answer:
        return False
    gt_lower = ground_truth.lower()
    ans_lower = generated_answer.lower()
    word_overlap = calculate_word_overlap(gt_lower, ans_lower)
    if word_overlap > 0.6:
        return True
    if len(gt_lower) > 50 and gt_lower[:100] in ans_lower:
        return True
    important_terms = ['symptom', 'treatment', 'cause', 'risk', 'diagnosis',
                       'disease', 'condition', 'therapy', 'medication', 'patient']
    gt_terms = [term for term in important_terms if term in gt_lower]
    if gt_terms:
        ans_has_terms = all(term in ans_lower for term in gt_terms)
        if ans_has_terms and word_overlap > 0.4:
            return True
    gt_sentences = gt_lower.split('.')
    ans_sentences = ans_lower.split('.')
    if len(gt_sentences) > 0 and len(ans_sentences) > 0:
        main_gt = gt_sentences[0][:100]
        if main_gt in ans_lower:
            return True
    return False


def classify_failures(answer, contexts, ground_truth):
    failures = {
        'missing_evidence': False,
        'noisy_evidence': False,
        'unsupported_claims': False,
        'unsafe_tone': False,
    }
    if not contexts:
        failures['missing_evidence'] = True
        return failures
    
    ctx_text = ' '.join([c.page_content for c in contexts]).lower()
    gt_lower = ground_truth.lower()
    gt_words = set(gt_lower.split())
    ctx_words = set(ctx_text.split())
    coverage = len(gt_words & ctx_words) / max(1, len(gt_words))
    if coverage < 0.3:
        failures['missing_evidence'] = True
    
    ans_words = set(answer.lower().split())
    relevant_words = gt_words | ans_words
    ctx_only_words = ctx_words - relevant_words
    noise_ratio = len(ctx_only_words) / max(1, len(ctx_words))
    if noise_ratio > 0.6:
        failures['noisy_evidence'] = True
    
    ans_claims = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    unsupported = 0
    for claim in ans_claims:
        claim_words = set(claim.lower().split())
        if not claim_words.intersection(ctx_words):
            unsupported += 1
    if unsupported > 0 and len(ans_claims) > 0:
        failures['unsupported_claims'] = True
    
    ans_lower = answer.lower()
    if any(p in ans_lower for p in ['you must', 'you have to', 'definitely']):
        if not any(p in ans_lower for p in ['consult', 'professional', 'may', 'might']):
            failures['unsafe_tone'] = True
    
    return failures


def safety_score(answer):
    ans_lower = answer.lower()
    score = 0.0
    uncertainty_phrases = [
        'consult', 'professional', 'doctor', 'physician', 'healthcare provider',
        'not sure', 'uncertain', 'may', 'might', 'could', 'should',
        'based on the context', 'according to', 'evidence suggests',
    ]
    for phrase in uncertainty_phrases:
        if phrase in ans_lower:
            score += 0.1
    definitive_patterns = ['you must', 'you have to', 'definitely', 'absolutely', 'always']
    for pattern in definitive_patterns:
        if pattern in ans_lower:
            score -= 0.15
    return max(0.0, min(1.0, score))


# Global RAGAS instances
_ragas_embeddings = None
_ragas_llm = None
_ragas_metrics = None

def get_ragas_embeddings():
    global _ragas_embeddings
    if _ragas_embeddings is None:
        _ragas_embeddings = CompatibleOpenAIEmbeddings(model=EMBEDDING_MODEL)
    return _ragas_embeddings

def get_ragas_llm():
    global _ragas_llm
    if _ragas_llm is None:
        from openai import OpenAI
        from ragas.llms import llm_factory
        client = OpenAI(api_key=OPENAI_API_KEY)
        _ragas_llm = llm_factory(LLM_MODEL, client=client)
    return _ragas_llm

def get_ragas_metrics():
    global _ragas_metrics
    if _ragas_metrics is None:
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        llm = get_ragas_llm()
        _ragas_metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
            ContextPrecision(llm=llm),
            ContextRecall(llm=llm),
        ]
    return _ragas_metrics

def compute_ragas_metrics(question, answer, contexts, ground_truth):
    try:
        from ragas import evaluate as ragas_evaluate
        from datasets import Dataset
        ragas_data = Dataset.from_dict({
            'question': [question],
            'answer': [answer],
            'contexts': [[c.page_content for c in contexts]] if contexts else [[]],
            'ground_truth': [ground_truth],
        })
        result = ragas_evaluate(
            ragas_data,
            metrics=get_ragas_metrics(),
            embeddings=get_ragas_embeddings(),
            raise_exceptions=False,
        )
        return {
            'ragas_faithfulness': float(result['faithfulness'][0]) if 'faithfulness' in result.columns else 0.0,
            'ragas_answer_relevance': float(result['answer_relevancy'][0]) if 'answer_relevancy' in result.columns else 0.0,
            'ragas_context_precision': float(result['context_precision'][0]) if 'context_precision' in result.columns else 0.0,
            'ragas_context_recall': float(result['context_recall'][0]) if 'context_recall' in result.columns else 0.0,
        }
    except Exception as e:
        # RAGAS can be fragile; return zeros gracefully
        return {
            'ragas_faithfulness': 0.0,
            'ragas_answer_relevance': 0.0,
            'ragas_context_precision': 0.0,
            'ragas_context_recall': 0.0,
        }


def compute_deepeval_metrics(question, answer, contexts, ground_truth):
    try:
        from deepeval.metrics import HallucinationMetric, FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        ctx_texts = [c.page_content for c in contexts] if contexts else []
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            context=ctx_texts,
            retrieval_context=ctx_texts,
        )
        hall = HallucinationMetric(threshold=0.5)
        hall.measure(test_case)
        faith = FaithfulnessMetric(threshold=0.5)
        faith.measure(test_case)
        rel = AnswerRelevancyMetric(threshold=0.5)
        rel.measure(test_case)
        return {
            'deepeval_hallucination': hall.score,
            'deepeval_groundedness': faith.score,
            'deepeval_correctness': rel.score,
        }
    except Exception:
        return {
            'deepeval_hallucination': 0.0,
            'deepeval_groundedness': 0.0,
            'deepeval_correctness': 0.0,
        }


# ========== MAIN EXPERIMENT ==========
def run_thesis_experiment(eval_df, chunks, vs, llm_model):
    """Run the 5-pipeline experiment."""
    os.makedirs('reports', exist_ok=True)
    
    pipelines = {
        'Pipeline 1: Vanilla LLM': ('vanilla', None),
        'Pipeline 2: Standard RAG': ('rag', semantic_retriever(vs)),
        'Pipeline 3: Multi-Query Expansion': ('rag', multiquery_retriever(vs, chunks, llm_model)),
        'Pipeline 4: Hybrid Retrieval': ('rag', hybrid_retriever(vs, chunks)),
        'Pipeline 5: Query Reformulation': ('rag', reformulation_retriever(vs, chunks, llm_model)),
    }
    
    results = []
    
    for pipeline_name, (ptype, retriever_obj) in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Running: {pipeline_name}")
        print(f"{'='*60}")
        
        if ptype == 'vanilla':
            chain = create_vanilla_chain(llm_model)
        else:
            chain = create_modern_retrieval_chain(retriever_obj, llm_model)
        
        correct = 0
        overlap_scores = []
        latencies = []
        
        for idx, row in eval_df.iterrows():
            q = row['question']
            gt = row['answer']
            qtype = row.get('type', 'general')
            
            try:
                t0 = time.time()
                
                if ptype == 'vanilla':
                    result = chain.invoke({"input": q})
                    answer = result.content if hasattr(result, 'content') else str(result)
                    contexts = []
                else:
                    result = chain.invoke({"input": q})
                    answer = result.get('answer', '')
                    contexts = result.get('context', [])
                
                latency_ms = (time.time() - t0) * 1000
                latencies.append(latency_ms)
                
                is_correct = evaluate_answer_semantic(gt, answer)
                if is_correct:
                    correct += 1
                
                overlap = calculate_word_overlap(gt, answer)
                overlap_scores.append(overlap)
                
                # RAGAS metrics
                ragas = compute_ragas_metrics(q, answer, contexts, gt)
                
                # DeepEval metrics
                deepeval = compute_deepeval_metrics(q, answer, contexts, gt)
                
                # Safety
                safety = safety_score(answer)
                
                # Failure patterns
                failures = classify_failures(answer, contexts, gt)
                
                results.append({
                    'pipeline': pipeline_name,
                    'question': q,
                    'question_type': qtype,
                    'generated_answer': answer,
                    'ground_truth': gt,
                    'correct': is_correct,
                    'word_overlap': overlap,
                    'latency_ms': latency_ms,
                    **ragas,
                    **deepeval,
                    'safety_compliance': safety,
                    'failure_missing_evidence': failures['missing_evidence'],
                    'failure_noisy_evidence': failures['noisy_evidence'],
                    'failure_unsupported_claims': failures['unsupported_claims'],
                    'failure_unsafe_tone': failures['unsafe_tone'],
                })
                
            except Exception as e:
                print(f"  Error on question {idx}: {str(e)[:80]}")
                results.append({
                    'pipeline': pipeline_name,
                    'question': q,
                    'question_type': qtype,
                    'generated_answer': f'ERROR: {str(e)[:100]}',
                    'ground_truth': gt,
                    'correct': False,
                    'word_overlap': 0.0,
                    'latency_ms': 0.0,
                    'ragas_faithfulness': 0.0,
                    'ragas_answer_relevance': 0.0,
                    'ragas_context_precision': 0.0,
                    'ragas_context_recall': 0.0,
                    'deepeval_hallucination': 0.0,
                    'deepeval_groundedness': 0.0,
                    'deepeval_correctness': 0.0,
                    'safety_compliance': 0.0,
                    'failure_missing_evidence': False,
                    'failure_noisy_evidence': False,
                    'failure_unsupported_claims': False,
                    'failure_unsafe_tone': False,
                })
            
            if (idx + 1) % 5 == 0 or idx == len(eval_df) - 1:
                print(f"  Progress: {idx + 1}/{len(eval_df)}")
        
        n = len(eval_df)
        print(f"\n  Results for {pipeline_name}:")
        print(f"    Accuracy: {correct}/{n} = {correct/max(1,n):.3f}")
        print(f"    Avg Word Overlap: {np.mean(overlap_scores):.3f}")
        print(f"    Avg Latency: {np.mean(latencies):.1f} ms")
    
    return pd.DataFrame(results)


def print_summary(thesis_results, eval_df):
    """Print aggregate analysis."""
    print("\n" + "="*60)
    print("AGGREGATE RESULTS BY PIPELINE")
    print("="*60)
    
    summary = thesis_results.groupby('pipeline').agg({
        'correct': ['sum', 'count', 'mean'],
        'word_overlap': 'mean',
        'latency_ms': 'mean',
        'ragas_faithfulness': 'mean',
        'ragas_answer_relevance': 'mean',
        'ragas_context_precision': 'mean',
        'ragas_context_recall': 'mean',
        'deepeval_hallucination': 'mean',
        'deepeval_groundedness': 'mean',
        'deepeval_correctness': 'mean',
        'safety_compliance': 'mean',
    }).round(3)
    print(summary)
    summary.to_csv('reports/thesis_pipeline_summary.csv')
    print("\nSaved summary to reports/thesis_pipeline_summary.csv")
    
    # Failure patterns
    print("\n" + "="*60)
    print("FAILURE PATTERN DISTRIBUTION")
    print("="*60)
    for pipe in thesis_results['pipeline'].unique():
        pipe_data = thesis_results[thesis_results['pipeline'] == pipe]
        total = len(pipe_data)
        print(f"\n{pipe} (n={total}):")
        for pattern in ['failure_missing_evidence', 'failure_noisy_evidence', 
                       'failure_unsupported_claims', 'failure_unsafe_tone']:
            count = pipe_data[pattern].sum()
            pct = 100 * count / max(1, total)
            print(f"  {pattern.replace('failure_', '')}: {count} ({pct:.1f}%)")
    
    # Latency analysis
    print("\n" + "="*60)
    print("LATENCY ANALYSIS")
    print("="*60)
    latency_summary = thesis_results.groupby('pipeline')['latency_ms'].agg(['mean', 'std']).round(1)
    print(latency_summary)
    
    # Best configuration
    print("\n" + "="*60)
    print("BEST CONFIGURATION RECOMMENDATION")
    print("="*60)
    
    faithfulness = thesis_results.groupby('pipeline')['ragas_faithfulness'].mean()
    best_faith = faithfulness.idxmax()
    print(f"Highest Faithfulness: {best_faith} ({faithfulness.max():.3f})")
    
    hallucination = thesis_results.groupby('pipeline')['deepeval_hallucination'].mean()
    best_hall = hallucination.idxmin()
    print(f"Lowest Hallucination: {best_hall} ({hallucination.min():.3f})")
    
    safety = thesis_results.groupby('pipeline')['safety_compliance'].mean()
    best_safe = safety.idxmax()
    print(f"Highest Safety: {best_safe} ({safety.max():.3f})")
    
    accuracy = thesis_results.groupby('pipeline')['correct'].mean()
    best_acc = accuracy.idxmax()
    print(f"Highest Accuracy: {best_acc} ({accuracy.max():.3f})")
    
    # Save final report
    report_lines = [
        "="*60,
        "THESIS EXPERIMENT FINAL REPORT",
        "="*60,
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"LLM: {LLM_MODEL} (temp={TEMPERATURE})",
        f"Embedding: {EMBEDDING_MODEL}",
        f"Chunking: FIXED {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap",
        f"Evaluation set: {len(eval_df)} questions",
        "",
        "AGGREGATE RESULTS:",
        str(summary),
        "",
        "BEST CONFIGURATION:",
        f"  Accuracy: {best_acc} = {accuracy.max():.3f}",
        f"  Faithfulness: {best_faith} = {faithfulness.max():.3f}",
        f"  Hallucination: {best_hall} = {hallucination.min():.3f}",
        f"  Safety: {best_safe} = {safety.max():.3f}",
        "="*60,
    ]
    
    report_text = "\n".join(report_lines)
    with open('reports/thesis_final_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nSaved final report to reports/thesis_final_report.txt")


# ========== ENTRY POINT ==========
if __name__ == '__main__':
    print("=== MEDQUAD RAG THESIS EXPERIMENT ===")
    print(f"Eval size: {EVAL_SIZE} | Corpus limit: {CORPUS_LIMIT}")
    print("="*60)
    
    # Load data
    print("\n=== DATA LOADING ===")
    df = load_medquad_data('data/medquad_fixed.csv')
    df = preprocess_data(df)
    
    # Sample evaluation set
    eval_df = stratified_sample(df, n=min(EVAL_SIZE, len(df)), stratify_col='type')
    
    # Use all remaining data as corpus (no limit - ensures coverage)
    remaining = df[~df.index.isin(eval_df.index)].reset_index(drop=True)
    if CORPUS_LIMIT > 0:
        corpus_df = remaining.head(CORPUS_LIMIT)
    else:
        corpus_df = remaining
    print(f"Corpus size: {len(corpus_df)}")
    print(f"Evaluation set: {len(eval_df)} | Corpus: {len(corpus_df)}")
    
    # Convert to LangChain Documents
    raw_docs = [
        Document(page_content=row['answer'], metadata={
            'source': row['source'],
            'type': row['type'],
            'question': row['question'],
        })
        for _, row in corpus_df.iterrows()
    ]
    
    # Chunking
    print("\n=== FIXED CHUNKING ===")
    chunks = fixed_chunk_documents(raw_docs)
    
    # Build vector store
    print("\n=== BUILDING VECTOR STORE ===")
    vs, emb = build_vectorstore(chunks, persist_dir='./chroma_thesis')
    
    # Initialize LLM
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    print(f"LLM: {LLM_MODEL} (temp={TEMPERATURE})")
    
    # Run experiment
    print("\n" + "="*60)
    print("STARTING THESIS-ALIGNED EXPERIMENT")
    print("="*60)
    
    thesis_results = run_thesis_experiment(eval_df, chunks, vs, llm)
    
    # Save results
    thesis_results.to_csv('reports/thesis_experiment_results.csv', index=False)
    print("\nSaved thesis results to reports/thesis_experiment_results.csv")
    
    # Print summary
    print_summary(thesis_results, eval_df)
    
    print("\n" + "="*60)
    print("THESIS EXPERIMENT COMPLETE")
    print("="*60)
