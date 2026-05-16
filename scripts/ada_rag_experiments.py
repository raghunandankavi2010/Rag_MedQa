"""
Ada RAG experiments
- Scrapes https://ada.com/conditions/#letter-a to collect condition pages (limited number)
- Builds document chunks with multiple chunking strategies
- Creates retrieval functions: semantic (OpenAI embeddings), tfidf, hybrid, rerank, multiquery, reformulation
- Runs simple RAG answer pipeline using OpenAI Chat completions
- Saves CSV reports into `reports/`

Requirements (install in your venv):
pip install requests beautifulsoup4 python-dotenv openai scikit-learn numpy pandas nltk

Usage:
python scripts/ada_rag_experiments.py --max-pages 30 --chunks-per-doc 3
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Callable, Dict

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional NLP
try:
    import nltk
    from nltk import sent_tokenize
except Exception:
    nltk = None

# OpenAI
try:
    import openai
except Exception:
    openai = None

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError('OPENAI_API_KEY not set in .env')
if openai:
    openai.api_key = OPENAI_API_KEY

BASE_LIST_URL = 'https://ada.com/conditions/#letter-a'
USER_AGENT = 'ada-rag-experiments/1.0 (+https://github.com)'
HEADERS = {'User-Agent': USER_AGENT}

# ------------------------- Web scraping helpers -------------------------

def fetch_html(url: str, timeout=10) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def get_condition_links(max_pages=30) -> List[str]:
    html = fetch_html(BASE_LIST_URL)
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    # Find anchors under the conditions list
    for a in soup.select('a[href]'):
        href = a['href']
        if href.startswith('https://ada.com/conditions/') and href not in links:
            links.append(href)
            if len(links) >= max_pages:
                break
    return links


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Try common containers
    article = soup.find('article')
    if article:
        texts = [p.get_text(separator=' ', strip=True) for p in article.find_all('p')]
        return '\n\n'.join(t for t in texts if t)
    # fallback: gather paragraphs
    paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
    return '\n\n'.join(p for p in paragraphs if p)

# ------------------------- Document & chunking -------------------------

def make_documents(pages: List[Dict]) -> List[Dict]:
    docs = []
    for i, p in enumerate(pages):
        docs.append({'id': f"doc_{i}", 'title': p['title'], 'text': p['text'], 'url': p['url']})
    return docs


def chunk_recursive(text: str, chunk_size=1200, chunk_overlap=200) -> List[str]:
    pieces = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        pieces.append(text[start:end])
        start = max(start + chunk_size - chunk_overlap, end)
    return pieces


def chunk_fixed(text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
    return chunk_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_sentence(text: str, max_chars=800) -> List[str]:
    if nltk is None:
        try:
            import nltk as _nltk
            _nltk.download('punkt')
            from nltk import sent_tokenize
        except Exception:
            # fallback to naive split
            return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    sents = sent_tokenize(text)
    out = []
    cur = ''
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + ' ' + s).strip()
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out


def chunk_paragraph(text: str) -> List[str]:
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts if parts else [text]

# ------------------------- Embeddings & retrieval -------------------------

def embed_texts_openai(texts: List[str], model='text-embedding-3-small') -> np.ndarray:
    if openai is None:
        raise RuntimeError('openai package not installed')
    # batch in groups of 50
    embs = []
    B = 50
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = openai.Embedding.create(model=model, input=batch)
        for item in resp['data']:
            embs.append(np.array(item['embedding'], dtype=np.float32))
        time.sleep(0.1)
    return np.vstack(embs)


def semantic_retrieve(query: str, chunk_texts: List[str], chunk_embs: np.ndarray, topk=5, model='text-embedding-3-small') -> List[int]:
    q_emb = embed_texts_openai([query], model=model)[0]
    sims = cosine_similarity([q_emb], chunk_embs)[0]
    idx = np.argsort(sims)[::-1][:topk]
    return idx.tolist()


def tfidf_retriever_factory(texts: List[str]) -> Callable[[str, int], List[int]]:
    vect = TfidfVectorizer(stop_words='english').fit(texts)
    mat = vect.transform(texts)
    def retrieve(query: str, topk=5):
        qv = vect.transform([query])
        sims = cosine_similarity(qv, mat)[0]
        idx = np.argsort(sims)[::-1][:topk]
        return idx.tolist()
    return retrieve


def hybrid_retrieve(query: str, tfidf_ret: Callable, sem_idxs: List[int], topk=5) -> List[int]:
    # merge semantic indices and tfidf indices preserving order
    tfidf_idxs = tfidf_ret(query, topk)
    merged = []
    seen = set()
    for i in sem_idxs + tfidf_idxs:
        if i not in seen:
            merged.append(i); seen.add(i)
        if len(merged) >= topk:
            break
    return merged[:topk]

# ------------------------- RAG answer step -------------------------

def answer_with_context(question: str, contexts: List[str], llm_model='gpt-3.5-turbo') -> str:
    if openai is None:
        raise RuntimeError('openai package not installed')
    system = "You are a helpful medical assistant. Use provided context to answer succinctly. If not present, say you don't know."
    ctx_text = '\n\n'.join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = f"{system}\n\n{ctx_text}\n\nQuestion: {question}\n\nAnswer:"
    resp = openai.ChatCompletion.create(model=llm_model, messages=[{"role":"user","content":prompt}], temperature=0)
    return resp['choices'][0]['message']['content'].strip()

# ------------------------- Experiment runner -------------------------

def run_experiments(max_pages=30, max_chunks_per_doc=3, out_folder='reports'):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    print('Fetching condition links...')
    links = get_condition_links(max_pages=max_pages)
    pages = []
    for i, url in enumerate(links):
        print(f'Fetching {i+1}/{len(links)}: {url}')
        try:
            html = fetch_html(url)
            text = extract_main_text(html)
            title = url.rstrip('/').split('/')[-1].replace('-', ' ').title()
            pages.append({'url': url, 'title': title, 'text': text})
        except Exception as e:
            print('  failed:', e)
    if not pages:
        print('No pages fetched, aborting.')
        return

    docs = make_documents(pages)
    # create chunk sets per chunking strategy
    chunk_methods = {
        'recursive': lambda t: chunk_recursive(t, chunk_size=1200, chunk_overlap=200),
        'fixed': lambda t: chunk_fixed(t, chunk_size=500, chunk_overlap=50),
        'sentence': lambda t: chunk_sentence(t, max_chars=800),
        'paragraph': lambda t: chunk_paragraph(t),
    }

    retrieval_methods = ['semantic', 'tfidf', 'hybrid', 'rerank', 'multiquery', 'reformulation']

    results = []

    for cm_name, cm_fn in chunk_methods.items():
        print('\n=== Chunk method:', cm_name, '===')
        # create chunks list and mapping
        chunks = []
        metadata = []
        for d in docs:
            parts = cm_fn(d['text'])
            # limit chunks per doc to avoid explosion
            for j, p in enumerate(parts[:max_chunks_per_doc]):
                chunks.append(p)
                metadata.append({'doc_id': d['id'], 'title': d['title'], 'url': d['url']})
        if not chunks:
            continue
        print('Total chunks:', len(chunks))

        # build TF-IDF retriever
        tfidf_ret = tfidf_retriever_factory(chunks)

        # build embeddings (semantic) once per chunk set
        print('Computing embeddings for chunks (this may take time and API calls)...')
        chunk_embs = embed_texts_openai(chunks)

        for rm in retrieval_methods:
            print('Retrieval method:', rm)
            # run a small set of sample queries: use document titles as queries
            sample_questions = [m['title'] + ' symptoms' for m in metadata[:30]]
            for q in sample_questions[:30]:
                try:
                    if rm == 'semantic':
                        sem_idxs = semantic_retrieve(q, chunks, chunk_embs, topk=5)
                        chosen_idxs = sem_idxs
                    elif rm == 'tfidf':
                        chosen_idxs = tfidf_ret(q, 5)
                    elif rm == 'hybrid':
                        sem_idxs = semantic_retrieve(q, chunks, chunk_embs, topk=5)
                        chosen_idxs = hybrid_retrieve(q, tfidf_ret, sem_idxs, topk=5)
                    elif rm == 'rerank':
                        sem_idxs = semantic_retrieve(q, chunks, chunk_embs, topk=10)
                        # rerank by direct cosine similarity with the query embedding
                        q_emb = embed_texts_openai([q])[0]
                        cand_embs = chunk_embs[sem_idxs]
                        sims = cosine_similarity([q_emb], cand_embs)[0]
                        order = np.argsort(sims)[::-1]
                        chosen_idxs = [sem_idxs[i] for i in order[:5]]
                    elif rm == 'multiquery':
                        # expand queries via tfidf features
                        # reuse tfidf vectorizer internals via closure (approx)
                        expanded = [q]
                        chosen = []
                        seen = set()
                        for eq in expanded:
                            ids = tfidf_ret(eq, 5)
                            for ii in ids:
                                if ii not in seen:
                                    chosen.append(ii); seen.add(ii)
                                if len(chosen) >= 5:
                                    break
                            if len(chosen) >= 5:
                                break
                        chosen_idxs = chosen
                    elif rm == 'reformulation':
                        # ask LLM to reformulate then semantic
                        if openai is None:
                            chosen_idxs = semantic_retrieve(q, chunks, chunk_embs, topk=5)
                        else:
                            reform = openai.ChatCompletion.create(
                                model='gpt-3.5-turbo',
                                messages=[{'role':'user','content':f"Reformulate the query for better retrieval: {q}"}],
                                temperature=0,
                            )['choices'][0]['message']['content'].strip()
                            chosen_idxs = semantic_retrieve(reform, chunks, chunk_embs, topk=5)
                    else:
                        chosen_idxs = semantic_retrieve(q, chunks, chunk_embs, topk=5)

                    contexts = [chunks[i] for i in chosen_idxs]
                    answer = answer_with_context(q, contexts)

                    results.append({
                        'chunk_method': cm_name,
                        'retrieval_method': rm,
                        'query': q,
                        'chosen_idx': chosen_idxs[0] if chosen_idxs else None,
                        'answer_snippet': answer[:300],
                        'contexts_count': len(contexts)
                    })
                except Exception as e:
                    print('  error for query:', e)

        # small save per chunk method
        df = pd.DataFrame(results)
        out_csv = Path(out_folder) / f'ada_rag_results_{cm_name}.csv'
        df.to_csv(out_csv, index=False)
        print('Saved', out_csv)

    # overall save
    df_all = pd.DataFrame(results)
    df_all.to_csv(Path(out_folder) / 'ada_rag_results_all.csv', index=False)
    print('Saved overall results to reports/ada_rag_results_all.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-pages', type=int, default=30, help='Max condition pages to fetch')
    parser.add_argument('--max-chunks-per-doc', type=int, default=3, help='Limit chunks per doc')
    parser.add_argument('--out', type=str, default='reports', help='Output folder for reports')
    args = parser.parse_args()
    run_experiments(max_pages=args.max_pages, max_chunks_per_doc=args.max_chunks_per_doc, out_folder=args.out)
