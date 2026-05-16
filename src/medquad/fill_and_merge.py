"""
Fill missing answers using Wayback and broader extraction, then merge into a CSV.
Usage:
  python -m src.medquad.fill_and_merge --medquad-path data/MedQuAD --workers 4

By default only processes the three subsets (ADAM, Drugs, Herbs). No LLM used.
"""
import os
import time
import argparse
from pathlib import Path
import requests
from lxml import html, etree
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def extract_text_from_html(content: bytes) -> str:
    try:
        tree = html.fromstring(content)
    except Exception:
        return ''
    # Try common containers
    paths = [
        '//div[@id="mainContent"]//p/text()',
        '//div[@id="main-content"]//p/text()',
        '//div[contains(@class,"content")]//p/text()',
        '//article//p/text()',
        '//main//p/text()',
        '//body//p/text()'
    ]
    parts = []
    for p in paths:
        res = tree.xpath(p)
        if res:
            parts.extend([r.strip() for r in res if r and r.strip()])
    if parts:
        return ' '.join(parts)
    # fallback: all text
    text = tree.xpath('string(.)') or ''
    return ' '.join([t.strip() for t in text.splitlines() if t.strip()])


def wayback_snapshot(url: str) -> str:
    """Return snapshot HTML from Wayback if available, else empty string."""
    try:
        cdx = 'https://web.archive.org/cdx/search/cdx'
        params = {'url': url, 'output': 'json', 'limit': '1', 'filter': 'statuscode:200', 'from': '1996', 'to': '2026'}
        r = requests.get(cdx, params=params, timeout=20, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        if len(data) >= 2:
            # second row contains timestamp at index 1
            ts = data[1][1]
            snap = f'https://web.archive.org/web/{ts}id_/{url}'
            sr = requests.get(snap, timeout=30, headers=HEADERS)
            if sr.status_code == 200:
                return sr.content
        return ''
    except Exception:
        return ''


def fetch_live(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers=HEADERS)
        r.raise_for_status()
        return r.content
    except Exception:
        return ''


def parse_qapairs_from_file(xml_path: str):
    rows = []
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        focus = root.findtext('Focus', default='')
        syns = [s.text for s in root.findall('.//Synonym') if s.text]
        synonyms_str = ', '.join(syns)
        group = root.findtext('.//SemanticGroup', default='')
        doc_url = root.get('url') or (root.findtext('Url') or '')
        for qa in root.findall('.//QAPair'):
            pid = qa.get('pid')
            q_elem = qa.find('Question')
            qid = q_elem.get('qid') if q_elem is not None and q_elem.get('qid') else ''
            question = q_elem.text if q_elem is not None and q_elem.text else ''
            answer = qa.findtext('Answer', default='')
            qtype = q_elem.get('qtype') if q_elem is not None and q_elem.get('qtype') else ''
            rows.append({
                'source_file': os.path.basename(xml_path),
                'pid': pid,
                'qid': qid,
                'focus': focus,
                'group': group,
                'synonyms': synonyms_str,
                'qtype': qtype,
                'question': question,
                'answer': answer or '',
                'url': doc_url
            })
    except Exception:
        pass
    return rows


def attempt_fill(row):
    if row['answer'] and row['answer'].strip():
        return row, None
    url = row.get('url')
    if not url:
        return row, 'no_url'
    # try wayback
    content = wayback_snapshot(url)
    if content:
        text = extract_text_from_html(content)
        if text:
            row['answer'] = text
            return row, 'wayback'
    # try live fetch
    content = fetch_live(url)
    if content:
        text = extract_text_from_html(content)
        if text:
            row['answer'] = text
            return row, 'live'
    return row, 'not_found'


def process_subset(base_path, subset_name, max_workers=4):
    subset_path = os.path.join(base_path, subset_name)
    filled_path = os.path.join(base_path, f'filled_{subset_name}')
    xml_dir = filled_path if os.path.exists(filled_path) and any(Path(filled_path).glob('*.xml')) else subset_path
    xml_files = list(Path(xml_dir).glob('*.xml'))
    print(f'Processing {subset_name}: {len(xml_files)} files from {xml_dir}')
    all_rows = []
    for xf in xml_files:
        all_rows.extend(parse_qapairs_from_file(str(xf)))
    print(f'Parsed {len(all_rows)} QAPairs; attempting to fill missing answers...')
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(attempt_fill, row): row for row in all_rows}
        for fut in as_completed(futures):
            row, status = fut.result()
            row['_filled_by'] = status
            results.append(row)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--medquad-path', required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    base = args.medquad_path
    subsets = ['10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA']
    all_results = []
    for s in subsets:
        res = process_subset(base, s, max_workers=args.workers)
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    out = Path('reports') / 'medquad_complete.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Saved merged CSV to', out)
    if '_filled_by' in df.columns:
        filled = df[df['_filled_by'].isin(['wayback','live'])]
        print('Filled answers count:', len(filled))
    else:
        print('Filled answers count: 0')
