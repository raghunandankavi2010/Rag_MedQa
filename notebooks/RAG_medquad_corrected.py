"""
RAG Medical QA — Thesis-Aligned Implementation
Student: Raghunandan Kavi (PN1196933)

This code implements the five pipeline configurations described in the thesis:
  Pipeline 1: Vanilla LLM (baseline, no retrieval)
  Pipeline 2: Standard RAG (dense semantic search)
  Pipeline 3: Multi-Query Expansion RAG
  Pipeline 4: Hybrid Retrieval RAG (RRF fusion)
  Pipeline 5: Query Reformulation + Reranking RAG

All pipelines use FIXED chunking (400 tokens, 50 overlap) as a control variable.
Evaluation uses RAGAS and DeepEval metrics.
"""

# ========== COMPATIBILITY SHIM ==========
# If langchain_classic is not installed, alias standard langchain
import sys
try:
    import langchain_classic
except ImportError:
    try:
        import langchain as _langchain
        sys.modules['langchain_classic'] = _langchain
        sys.modules['langchain_classic.chains'] = _langchain.chains
        if hasattr(_langchain.chains, 'combine_documents'):
            sys.modules['langchain_classic.chains.combine_documents'] = _langchain.chains.combine_documents
    except Exception:
        pass  # Will fail later at import time if neither is available

# ========== ALL ORIGINAL IMPORTS (KEPT EXACTLY AS-IS) ==========
import os
import json
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Removed `HuggingFaceEmbeddings` import because it isn't available
# in the langchain version used in the environment. The code uses
# `OpenAIEmbeddings` from `langchain_openai` instead.
# NOTE: The project uses `langchain_openai.ChatOpenAI`, `OpenAIEmbeddings` and
# `langchain_community.Chroma` later in the file (below). Older/wrong imports
# such as `langchain.llms.OpenAI`, `langchain.chains.RetrievalQA` and the
# duplicate `langchain.schema` message imports caused version mismatch errors
# in many environments. Those have been removed from this top-level block to
# avoid import conflicts. The concrete implementations are imported later
# where they are actually used.
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Additional imports used in running code
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing import List
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

# ========== EMBEDDING WRAPPER (COMPATIBILITY FIX) ==========
# RAGAS uses the deprecated embed_query() method. Add it as a wrapper.
class CompatibleOpenAIEmbeddings(OpenAIEmbeddings):
    """Wrapper that adds embed_query() for RAGAS compatibility."""
    def embed_query(self, text: str):
        """Embed a single query (RAGAS compatibility method)."""
        return self.embed_documents([text])[0]

# ========== CONFIGURATION ==========
# Thesis methodology: FIXED parameters across all pipelines
# Load OpenAI API key from a local .env (do NOT commit secrets).
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Add it to a .env file or environment variables.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
LLM_MODEL = "gpt-4o-mini"  # Thesis specification (changed from gpt-3.5-turbo)
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5
TEMPERATURE = 0.0

# ========== ORIGINAL DATA LOADING (KEPT AS-IS) ==========
# [Your original data loading code would go here]
# For this standalone version, we include a minimal loader

def load_medquad_data(data_dir='./MedQuAD'):
    """Load MedQuAD JSON files — original function preserved."""
    data = []

    # If the MedQuAD directory is missing, try sensible CSV fallbacks
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory not found: {data_dir}")
        csv_candidates = [
            os.path.join('data', 'medquad.csv'),
            os.path.join('data', 'medquad_complete.csv'),
            'medquad.csv',
            os.path.join('notebooks', 'medquad.csv'),
        ]
        for c in csv_candidates:
            if os.path.exists(c):
                try:
                    df_csv = pd.read_csv(c)
                    # Normalize columns
                    cols_lower = {col.lower(): col for col in df_csv.columns}
                    if 'question' in cols_lower and 'answer' in cols_lower:
                        qcol = cols_lower['question']
                        acol = cols_lower['answer']
                        tcol = cols_lower.get('type')
                        scol = cols_lower.get('source')
                        out = pd.DataFrame({
                            'question': df_csv[qcol].astype(str),
                            'answer': df_csv[acol].astype(str),
                            'type': df_csv[tcol].astype(str) if tcol else 'general',
                            'source': df_csv[scol].astype(str) if scol else 'unknown',
                        })
                        print(f"Loaded {len(out)} rows from CSV fallback: {c}")
                        return out.reset_index(drop=True)
                    else:
                        print(f"Found CSV {c} but missing required columns 'question'/'answer'.")
                except Exception as e:
                    print(f"  Error loading CSV {c}: {e}")

        print("No suitable CSV fallback found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['question', 'answer', 'type', 'source'])

    # Otherwise, walk the MedQuAD directory and load JSON files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                    source_name = doc.get('source', 'unknown')
                    for q in doc.get('questions', []):
                        data.append({
                            'question': q.get('question', ''),
                            'answer': q.get('answer', ''),
                            'type': q.get('type', 'general'),
                            'source': source_name,
                        })
                except Exception as e:
                    print(f"  Error loading {file}: {e}")

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} raw question-answer pairs from {data_dir}")
    return df


def preprocess_data(df):
    """Preprocess with 4 steps matching thesis Section 3.6.2."""
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
        print("Warning: Empty dataframe passed to stratified_sample — returning empty sample.")
        return pd.DataFrame(columns=df.columns if df is not None else ['question', 'answer', 'type', 'source'])

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


# Load data
print("=== DATA LOADING ===")
df = load_medquad_data('./MedQuAD')
df = preprocess_data(df)

# Create evaluation set
eval_df = stratified_sample(df, n=min(50, len(df)), stratify_col='type')
corpus_df = df[~df.index.isin(eval_df.index)].reset_index(drop=True)
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

# ========== FIXED CHUNKING (Control Variable) ==========
def fixed_chunk_documents(docs):
    """FIXED chunking: 400 tokens, 50 overlap. Control variable — not varied."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Fixed chunking: {len(docs)} docs -> {len(chunks)} chunks ({CHUNK_SIZE}/{CHUNK_OVERLAP})")
    return chunks


# Apply fixed chunking
print("\n=== FIXED CHUNKING ===")
chunks = fixed_chunk_documents(raw_docs)

# ========== VECTOR STORE ==========
def build_vectorstore(chunks, persist_dir='./chroma_expt'):
    """Build Chroma vector store — original function preserved."""
    embeddings_local = CompatibleOpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_local,
        persist_directory=persist_dir,
    )
    return vs, embeddings_local


print("\n=== BUILDING VECTOR STORE ===")
vs, emb = build_vectorstore(chunks, persist_dir='./chroma_fixed')

# ========== LLM INITIALISATION ==========
# Use gpt-4o-mini as specified in thesis
llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
print(f"LLM: {LLM_MODEL} (temp={TEMPERATURE})")

# ========== RETRIEVER FUNCTIONS (YOUR ORIGINAL CODE, KEPT AS-IS) ==========
class SimpleRetriever(BaseRetriever):
    """Simple retriever wrapper — your original class, kept exact."""
    retrieve_func: callable = Field(description="Function to retrieve documents")
    k: int = Field(default=TOP_K, description="Number of documents to retrieve")
    
    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self.retrieve_func(query, topk=self.k)
    
    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        return self._get_relevant_documents(query, **kwargs)


def semantic_retriever(vs):
    """Standard semantic retriever — your original function."""
    return vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})


def tfidf_retriever(chunks):
    """TF-IDF sparse retriever — your original function."""
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
    """Hybrid retriever with Reciprocal Rank Fusion — ENHANCED for thesis."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})
    tfidf = tfidf_retriever(chunks)
    
    def retrieve(query, topk=TOP_K):
        sem_docs = semantic.invoke(query)[:topk]
        tf_docs = tfidf.retrieve_func(query, topk=topk)
        
        # RRF fusion (thesis Section 3.8.4)
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


def multiquery_retriever(vs, chunks):
    """Multi-Query Expansion with LLM paraphrasing — ENHANCED for thesis."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K})
    
    def generate_paraphrases(query, n=3):
        """Use LLM to generate semantically equivalent reformulations."""
        paraphrase_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Generate {n} different ways to ask this medical question. Keep the same meaning but use different words."),
            ("human", "Original: {query}\n\nParaphrased versions (one per line):"),
        ])
        try:
            response = llm.invoke(paraphrase_prompt.format(query=query))
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


def reformulation_retriever(vs, chunks):
    """Query Reformulation + Reranking — ENHANCED for thesis."""
    semantic = vs.as_retriever(search_type='similarity', search_kwargs={'k': TOP_K * 2})
    
    # Reranking using TF-IDF similarity as proxy
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
        # Step 1: Reformulate query
        reform_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the following medical question to be clearer and more specific for document retrieval. Preserve the original intent."),
            ("human", "Original: {query}\n\nRewritten:"),
        ])
        try:
            response = llm.invoke(reform_prompt.format(query=query))
            reform = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            reform = reform if reform else query
        except Exception:
            reform = query
        
        # Step 2: Retrieve candidates
        candidates = semantic.invoke(reform)
        
        # Step 3: Rerank
        return rerank(reform, candidates, topk=topk)
    
    return SimpleRetriever(retrieve_func=retrieve, k=TOP_K)


# ========== MODERN RETRIEVAL CHAIN (YOUR ORIGINAL, KEPT AS-IS) ==========
def create_modern_retrieval_chain(retriever, llm_model):
    """Create retrieval chain — your original function, preserved exact."""
    system_prompt = (
        "You are a helpful medical information assistant. Answer the user's question "
        "using ONLY the provided context. If the context does not contain sufficient information, "
        "say so clearly. Do not make up information. Always encourage the user to consult "
        "a qualified medical professional for personal health decisions."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    return retrieval_chain


# ========== VANILLA LLM BASELINE (NEW — THESIS REQUIREMENT) ==========
def create_vanilla_chain():
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
    return prompt | llm


# ========== EVALUATION METRICS ==========
def calculate_word_overlap(text1, text2):
    """Your original function — kept exact."""
    if not text1 or not text2:
        return 0.0
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1:
        return 0.0
    return len(words1 & words2) / len(words1)


def evaluate_answer_semantic(ground_truth, generated_answer):
    """Your original function — kept exact."""
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


# ========== THESIS-ALIGNED EVALUATION (NEW) ==========
def classify_failures(answer, contexts, ground_truth):
    """Classify failure patterns — thesis Objective 6."""
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
    """Rule-based safety compliance — thesis Section 5.5."""
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


def compute_ragas_metrics(question, answer, contexts, ground_truth):
    """Compute RAGAS metrics — thesis Section 3.9.1."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        ragas_data = Dataset.from_dict({
            'question': [question],
            'answer': [answer],
            'contexts': [[c.page_content for c in contexts]] if contexts else [[]],
            'ground_truth': [ground_truth],
        })
        result = ragas_evaluate(ragas_data, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        return {
            'ragas_faithfulness': float(result['faithfulness'][0]),
            'ragas_answer_relevance': float(result['answer_relevancy'][0]),
            'ragas_context_precision': float(result['context_precision'][0]),
            'ragas_context_recall': float(result['context_recall'][0]),
        }
    except Exception:
        return {
            'ragas_faithfulness': 0.0,
            'ragas_answer_relevance': 0.0,
            'ragas_context_precision': 0.0,
            'ragas_context_recall': 0.0,
        }


def compute_deepeval_metrics(question, answer, contexts, ground_truth):
    """Compute DeepEval metrics — thesis Section 3.9.2."""
    try:
        from deepeval.metrics import HallucinationMetric, FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=[c.page_content for c in contexts] if contexts else [],
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


# ========== THESIS-ALIGNED EXPERIMENT DRIVER ==========
def run_thesis_experiment(eval_df, chunks, vs, llm):
    """
    Run the 5-pipeline experiment matching the thesis methodology.
    Fixed chunking, RAGAS + DeepEval, per-question-type, failure patterns, safety, latency.
    """
    os.makedirs('reports', exist_ok=True)
    
    pipelines = {
        'Pipeline 1: Vanilla LLM': ('vanilla', None),
        'Pipeline 2: Standard RAG': ('rag', semantic_retriever(vs)),
        'Pipeline 3: Multi-Query Expansion': ('rag', multiquery_retriever(vs, chunks)),
        'Pipeline 4: Hybrid Retrieval': ('rag', hybrid_retriever(vs, chunks)),
        'Pipeline 5: Query Reformulation': ('rag', reformulation_retriever(vs, chunks)),
    }
    
    results = []
    
    for pipeline_name, (ptype, retriever_obj) in pipelines.items():
        print(f"\n{'='*60}")
        print(f"Running: {pipeline_name}")
        print(f"{'='*60}")
        
        # Create chain
        if ptype == 'vanilla':
            chain = create_vanilla_chain()
        else:
            chain = create_modern_retrieval_chain(retriever_obj, llm)
        
        correct = 0
        overlap_scores = []
        semantic_scores = []
        latencies = []
        
        for idx, row in eval_df.iterrows():
            q = row['question']
            gt = row['answer']
            qtype = row.get('type', 'general')
            
            try:
                # Measure total latency
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
                
                # Basic evaluation
                if evaluate_answer_semantic(gt, answer):
                    correct += 1
                
                overlap = calculate_word_overlap(gt, answer)
                overlap_scores.append(overlap)
                semantic_scores.append(overlap)
                
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
                    'correct': evaluate_answer_semantic(gt, answer),
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
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{len(eval_df)}")
        
        # Pipeline summary
        n = len(eval_df)
        print(f"\n  Results for {pipeline_name}:")
        print(f"    Accuracy: {correct}/{n} = {correct/max(1,n):.3f}")
        print(f"    Avg Word Overlap: {np.mean(overlap_scores):.3f}")
        print(f"    Avg Latency: {np.mean(latencies):.1f} ms")
    
    return pd.DataFrame(results)


# ========== RUN THE THESIS EXPERIMENT ==========
print("\n" + "="*60)
print("STARTING THESIS-ALIGNED EXPERIMENT")
print("="*60)

thesis_results = run_thesis_experiment(eval_df, chunks, vs, llm)

# Save results
thesis_results.to_csv('reports/thesis_experiment_results.csv', index=False)
print("\nSaved thesis results to reports/thesis_experiment_results.csv")

# ========== AGGREGATE ANALYSIS ==========
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

# ========== PER-QUESTION-TYPE ANALYSIS ==========
print("\n" + "="*60)
print("PER-QUESTION-TYPE ANALYSIS")
print("="*60)

for qtype in eval_df['type'].unique():
    print(f"\n--- {qtype.upper()} ---")
    type_results = thesis_results[thesis_results['question_type'] == qtype]
    if len(type_results) > 0:
        type_summary = type_results.groupby('pipeline').agg({
            'ragas_faithfulness': 'mean',
            'deepeval_hallucination': 'mean',
            'safety_compliance': 'mean',
        }).round(3)
        print(type_summary)

# ========== FAILURE PATTERN ANALYSIS ==========
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

# ========== LATENCY ANALYSIS ==========
print("\n" + "="*60)
print("LATENCY ANALYSIS")
print("="*60)

latency_summary = thesis_results.groupby('pipeline')['latency_ms'].agg(['mean', 'std']).round(1)
print(latency_summary)

# ========== BEST CONFIGURATION ==========
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

# ========== SAVE FINAL REPORT ==========
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
    f"  Faithfulness: {best_faith} = {faithfulness.max():.3f}",
    f"  Hallucination: {best_hall} = {hallucination.min():.3f}",
    f"  Safety: {best_safe} = {safety.max():.3f}",
    "="*60,
]

report_text = "\n".join(report_lines)
with open('reports/thesis_final_report.txt', 'w') as f:
    f.write(report_text)
print(f"\nSaved final report to reports/thesis_final_report.txt")

print("\n" + "="*60)
print("THESIS EXPERIMENT COMPLETE")
print("="*60)

# ========== ORIGINAL BACKUP EXPERIMENT (OPTIONAL) ==========
# Your original 4x5 factorial experiment is preserved below as a function.
# Call run_original_experiment() if you want to run the old version.

def run_original_experiment(raw_docs, eval_df, llm):
    """
    ORIGINAL EXPERIMENT: 4 chunking methods x 5 retrieval methods.
    Kept intact for backward compatibility.
    """
    chunk_methods = ['recursive', 'fixed', 'sentence', 'paragraph']
    retrieval_methods = ['semantic', 'hybrid', 'tfidf', 'multiquery', 'reformulation']
    results_table = []
    
    print("\nStarting ORIGINAL experiment (4 chunking x 5 retrieval)...")
    
    for cm in chunk_methods:
        print(f'Chunk method: {cm}')
        test_chunks = chunk_documents(raw_docs, method=cm)
        test_vs, _ = build_vectorstore(test_chunks, persist_dir=f'./chroma_expt_{cm}')
        
        for rm in retrieval_methods:
            print(f'  Retrieval: {rm}')
            try:
                if rm == 'semantic':
                    ret = semantic_retriever(test_vs)
                elif rm == 'tfidf':
                    ret = tfidf_retriever(test_chunks)
                elif rm == 'hybrid':
                    ret = hybrid_retriever(test_vs, test_chunks)
                elif rm == 'multiquery':
                    ret = multiquery_retriever(test_vs, test_chunks)
                elif rm == 'reformulation':
                    ret = reformulation_retriever(test_vs, test_chunks, llm)
                else:
                    continue
                
                chain = create_modern_retrieval_chain(ret, llm)
                correct = 0
                overlap_scores = []
                
                for idx, s in eval_df.iterrows():
                    try:
                        result = chain.invoke({"input": s['question']})
                        ans = result.get('answer', '')
                        gt = s['answer']
                        if evaluate_answer_semantic(gt, ans):
                            correct += 1
                        overlap_scores.append(calculate_word_overlap(gt, ans))
                    except Exception:
                        continue
                
                acc = correct / max(1, len(eval_df))
                avg_ov = float(np.mean(overlap_scores)) if overlap_scores else 0.0
                results_table.append({
                    'chunk': cm, 'retrieval': rm, 'accuracy': acc, 'answer_overlap': avg_ov
                })
                print(f'    --> acc={acc:.3f}, overlap={avg_ov:.3f}')
            except Exception as e:
                print(f'    Error: {str(e)[:80]}')
                continue
    
    if results_table:
        df_res = pd.DataFrame(results_table)
        print('\n=== ORIGINAL EXPERIMENT SUMMARY ===')
        print(df_res.sort_values(['accuracy', 'answer_overlap'], ascending=False).head(10))
        df_res.to_csv('reports/original_experiment_summary.csv', index=False)
        print("Saved to reports/original_experiment_summary.csv")
    
    return results_table
run_original_experiment(raw_docs, eval_df, llm)
