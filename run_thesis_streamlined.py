"""
RAG Medical QA — Streamlined Thesis Experiment
================================================
Compares 5 pipeline configurations with custom metrics.
Avoids slow LLM-as-judge evaluation; uses deterministic metrics.
"""

import os
import sys
import time
import warnings
from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

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
EVAL_SIZE = 8
CORPUS_LIMIT = 1000

# ========== DATA LOADING ==========
def load_medquad_data(csv_path='data/medquad_fixed.csv'):
    df = pd.read_csv(csv_path)
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
    for req in ['question', 'answer']:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' not found")
    if 'type' not in df.columns:
        df['type'] = 'general'
    if 'source' not in df.columns:
        df['source'] = 'unknown'
    return df[['question', 'answer', 'type', 'source']].copy()


def preprocess_data(df):
    initial_count = len(df)
    df = df.drop_duplicates(subset=['question'], keep='first')
    print(f"  Dedup: {initial_count} -> {len(df)}")
    df['question'] = df['question'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
    df['answer'] = df['answer'].astype(str).str.replace(r'<[^>]+>', '', regex=True)
    df['question'] = df['question'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['answer'] = df['answer'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['question'] = df['question'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['answer'] = df['answer'].str.encode('ascii', 'ignore').str.decode('ascii')
    df = df[df['question'].str.len() > 5]
    df = df[df['answer'].str.len() > 5]
    return df.reset_index(drop=True)


def stratified_sample(df, n=50, stratify_col='type', seed=42):
    if df.empty:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunking: {len(docs)} docs -> {len(chunks)} chunks")
    return chunks


# ========== VECTOR STORE ==========
def build_vectorstore(chunks, persist_dir='./chroma_thesis'):
    emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory=persist_dir)
    return vs, emb


# ========== RETRIEVERS ==========
class SimpleRetriever(BaseRetriever):
    retrieve_func: callable = Field(description="Function to retrieve documents")
    k: int = Field(default=TOP_K, description="Number of documents to retrieve")
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.retrieve_func(query, topk=self.k)
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def semantic_retriever(vs):
    return vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})


def tfidf_retriever(chunks):
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
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})
    def generate_paraphrases(query, n=3):
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Generate {n} different ways to ask this medical question. Same meaning, different words."),
            ("human", "Original: {query}\n\nParaphrased versions (one per line):"),
        ])
        try:
            response = llm_model.invoke(prompt.format(query=query))
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
            ("system", "Rewrite the medical question to be clearer and more specific for document retrieval."),
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


# ========== CHAINS ==========
def create_modern_retrieval_chain(retriever, llm_model):
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
    return create_retrieval_chain(retriever, question_answer_chain)


def create_vanilla_chain(llm_model):
    system_prompt = (
        "You are a helpful medical information assistant. Answer based on your training knowledge. "
        "If unsure, say so clearly. Always encourage consulting a qualified medical professional."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return prompt | llm_model


# ========== CUSTOM METRICS ==========
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
    overlap = calculate_word_overlap(gt_lower, ans_lower)
    if overlap > 0.6:
        return True
    if len(gt_lower) > 50 and gt_lower[:100] in ans_lower:
        return True
    important_terms = ['symptom', 'treatment', 'cause', 'risk', 'diagnosis',
                       'disease', 'condition', 'therapy', 'medication', 'patient']
    gt_terms = [t for t in important_terms if t in gt_lower]
    if gt_terms and all(t in ans_lower for t in gt_terms) and overlap > 0.4:
        return True
    gt_sents = gt_lower.split('.')
    if gt_sents and gt_sents[0][:100] in ans_lower:
        return True
    return False


def faithfulness_score(answer, contexts):
    """Proxy for RAGAS faithfulness: fraction of answer words found in context."""
    if not contexts:
        return 0.0
    ctx_text = ' '.join([c.page_content for c in contexts]).lower()
    ans_words = set(answer.lower().split())
    if not ans_words:
        return 0.0
    ctx_words = set(ctx_text.split())
    supported = len(ans_words & ctx_words) / len(ans_words)
    return round(supported, 3)


def context_precision_score(question, contexts):
    """Proxy for context precision: word overlap between question and contexts."""
    if not contexts:
        return 0.0
    q_words = set(question.lower().split())
    ctx_text = ' '.join([c.page_content for c in contexts]).lower()
    ctx_words = set(ctx_text.split())
    if not ctx_words:
        return 0.0
    relevant = len(q_words & ctx_words) / len(ctx_words)
    return round(min(1.0, relevant * 3), 3)  # Scale up since overlap is naturally low


def context_recall_score(ground_truth, contexts):
    """Proxy for context recall: fraction of ground truth words in contexts."""
    if not contexts:
        return 0.0
    gt_words = set(ground_truth.lower().split())
    ctx_text = ' '.join([c.page_content for c in contexts]).lower()
    ctx_words = set(ctx_text.split())
    if not gt_words:
        return 0.0
    recalled = len(gt_words & ctx_words) / len(gt_words)
    return round(recalled, 3)


def answer_relevance_score(question, answer):
    """Proxy for answer relevance: word overlap between question and answer."""
    if not answer:
        return 0.0
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    if not a_words:
        return 0.0
    overlap = len(q_words & a_words) / len(a_words)
    return round(min(1.0, overlap * 2), 3)


def hallucination_proxy(answer, contexts):
    """Proxy for hallucination: 1 - faithfulness."""
    if not contexts:
        return 1.0  # No context = high hallucination risk
    return round(1.0 - faithfulness_score(answer, contexts), 3)


def groundedness_proxy(answer, contexts):
    """Proxy for groundedness: same as faithfulness."""
    return faithfulness_score(answer, contexts)


def correctness_proxy(ground_truth, answer):
    """Proxy for correctness: semantic match score."""
    if evaluate_answer_semantic(ground_truth, answer):
        return 1.0
    return round(calculate_word_overlap(ground_truth, answer), 3)


def safety_score(answer):
    ans_lower = answer.lower()
    score = 0.0
    good_phrases = ['consult', 'professional', 'doctor', 'physician', 'healthcare provider',
                    'not sure', 'uncertain', 'may', 'might', 'could', 'should',
                    'based on the context', 'according to', 'evidence suggests',
                    'context does not contain', 'insufficient information']
    for phrase in good_phrases:
        if phrase in ans_lower:
            score += 0.1
    bad_phrases = ['you must', 'you have to', 'definitely', 'absolutely', 'always']
    for phrase in bad_phrases:
        if phrase in ans_lower:
            score -= 0.15
    return max(0.0, min(1.0, score))


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
    gt_words = set(ground_truth.lower().split())
    ctx_words = set(ctx_text.split())
    coverage = len(gt_words & ctx_words) / max(1, len(gt_words))
    if coverage < 0.3:
        failures['missing_evidence'] = True
    ans_words = set(answer.lower().split())
    relevant = gt_words | ans_words
    ctx_only = ctx_words - relevant
    noise = len(ctx_only) / max(1, len(ctx_words))
    if noise > 0.6:
        failures['noisy_evidence'] = True
    claims = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
    unsupported = sum(1 for c in claims if not set(c.lower().split()).intersection(ctx_words))
    if unsupported > 0 and len(claims) > 0:
        failures['unsupported_claims'] = True
    if any(p in answer.lower() for p in ['you must', 'you have to', 'definitely']):
        if not any(p in answer.lower() for p in ['consult', 'professional', 'may', 'might']):
            failures['unsafe_tone'] = True
    return failures


# ========== MAIN EXPERIMENT ==========
def run_experiment(eval_df, chunks, vs, llm_model):
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
                
                # Custom metrics (proxies for RAGAS and DeepEval)
                ragas_faith = faithfulness_score(answer, contexts)
                ragas_rel = answer_relevance_score(q, answer)
                ragas_prec = context_precision_score(q, contexts)
                ragas_rec = context_recall_score(gt, contexts)
                
                deep_hall = hallucination_proxy(answer, contexts)
                deep_ground = groundedness_proxy(answer, contexts)
                deep_corr = correctness_proxy(gt, answer)
                
                safety = safety_score(answer)
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
                    'ragas_faithfulness': ragas_faith,
                    'ragas_answer_relevance': ragas_rel,
                    'ragas_context_precision': ragas_prec,
                    'ragas_context_recall': ragas_rec,
                    'deepeval_hallucination': deep_hall,
                    'deepeval_groundedness': deep_ground,
                    'deepeval_correctness': deep_corr,
                    'safety_compliance': safety,
                    'failure_missing_evidence': failures['missing_evidence'],
                    'failure_noisy_evidence': failures['noisy_evidence'],
                    'failure_unsupported_claims': failures['unsupported_claims'],
                    'failure_unsafe_tone': failures['unsafe_tone'],
                })
                
            except Exception as e:
                print(f"  Error on Q{idx}: {str(e)[:60]}")
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
                    'deepeval_hallucination': 1.0,
                    'deepeval_groundedness': 0.0,
                    'deepeval_correctness': 0.0,
                    'safety_compliance': 0.0,
                    'failure_missing_evidence': True,
                    'failure_noisy_evidence': False,
                    'failure_unsupported_claims': False,
                    'failure_unsafe_tone': False,
                })
            
            print(f"  Q{idx+1}/{len(eval_df)} done")
        
        n = len(eval_df)
        print(f"  Accuracy: {correct}/{n} = {correct/max(1,n):.3f}")
        print(f"  Avg Overlap: {np.mean([r['word_overlap'] for r in results if r['pipeline']==pipeline_name]):.3f}")
        print(f"  Avg Latency: {np.mean(latencies):.1f} ms")
    
    return pd.DataFrame(results)


def print_summary(df, eval_df):
    print("\n" + "="*60)
    print("AGGREGATE RESULTS BY PIPELINE")
    print("="*60)
    
    summary = df.groupby('pipeline').agg({
        'correct': 'mean',
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
    
    print("\n" + "="*60)
    print("FAILURE PATTERN DISTRIBUTION")
    print("="*60)
    for pipe in df['pipeline'].unique():
        pipe_data = df[df['pipeline'] == pipe]
        total = len(pipe_data)
        print(f"\n{pipe} (n={total}):")
        for pattern in ['failure_missing_evidence', 'failure_noisy_evidence', 
                       'failure_unsupported_claims', 'failure_unsafe_tone']:
            count = pipe_data[pattern].sum()
            pct = 100 * count / max(1, total)
            print(f"  {pattern.replace('failure_', '')}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)
    print("LATENCY ANALYSIS")
    print("="*60)
    latency_summary = df.groupby('pipeline')['latency_ms'].agg(['mean', 'std']).round(1)
    print(latency_summary)
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION")
    print("="*60)
    faith = df.groupby('pipeline')['ragas_faithfulness'].mean()
    print(f"Highest Faithfulness: {faith.idxmax()} ({faith.max():.3f})")
    hall = df.groupby('pipeline')['deepeval_hallucination'].mean()
    print(f"Lowest Hallucination: {hall.idxmin()} ({hall.min():.3f})")
    safe = df.groupby('pipeline')['safety_compliance'].mean()
    print(f"Highest Safety: {safe.idxmax()} ({safe.max():.3f})")
    acc = df.groupby('pipeline')['correct'].mean()
    print(f"Highest Accuracy: {acc.idxmax()} ({acc.max():.3f})")
    
    report = [
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
        f"  Accuracy: {acc.idxmax()} = {acc.max():.3f}",
        f"  Faithfulness: {faith.idxmax()} = {faith.max():.3f}",
        f"  Hallucination: {hall.idxmin()} = {hall.min():.3f}",
        f"  Safety: {safe.idxmax()} = {safe.max():.3f}",
        "="*60,
    ]
    with open('reports/thesis_final_report.txt', 'w') as f:
        f.write("\n".join(report))
    print(f"\nSaved final report to reports/thesis_final_report.txt")


if __name__ == '__main__':
    print("=== MEDQUAD RAG THESIS EXPERIMENT (STREAMLINED) ===")
    print(f"Eval size: {EVAL_SIZE} | Corpus limit: {CORPUS_LIMIT if CORPUS_LIMIT > 0 else 'all'}")
    print("="*60)
    
    df = load_medquad_data('data/medquad_fixed.csv')
    df = preprocess_data(df)
    eval_df = stratified_sample(df, n=min(EVAL_SIZE, len(df)), stratify_col='type')
    remaining = df[~df.index.isin(eval_df.index)].reset_index(drop=True)
    corpus_df = remaining.head(CORPUS_LIMIT) if CORPUS_LIMIT > 0 else remaining
    print(f"Eval: {len(eval_df)} | Corpus: {len(corpus_df)}")
    
    # Index combined Q&A documents for better retrieval coverage
    raw_docs = []
    for _, row in corpus_df.iterrows():
        combined = f"Question: {row['question']}\n\nAnswer: {row['answer']}"
        raw_docs.append(Document(page_content=combined, metadata={
            'source': row['source'],
            'type': row['type'],
            'question': row['question'],
            'answer': row['answer'],
        }))
    
    print("\n=== FIXED CHUNKING ===")
    chunks = fixed_chunk_documents(raw_docs)
    
    print("\n=== BUILDING VECTOR STORE ===")
    vs, emb = build_vectorstore(chunks, persist_dir='./chroma_thesis')
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    print(f"LLM: {LLM_MODEL} (temp={TEMPERATURE})")
    
    print("\n" + "="*60)
    print("STARTING EXPERIMENT")
    print("="*60)
    
    thesis_results = run_experiment(eval_df, chunks, vs, llm)
    
    thesis_results.to_csv('reports/thesis_experiment_results.csv', index=False)
    print("\nSaved results to reports/thesis_experiment_results.csv")
    
    print_summary(thesis_results, eval_df)
    
    print("\n" + "="*60)
    print("THESIS EXPERIMENT COMPLETE")
    print("="*60)
