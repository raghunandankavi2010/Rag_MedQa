"""
Scrape missing answers for MedQuAD subsets 10,11,12 and save filled XML + parsed CSV.
Place this file in `src/medquad/` and run from project root.

Example:
python -m src.medquad.scrape_missing_answers --medquad-path data/MedQuAD --workers 4

This script reuses `src.medquad.io.parse_folder` to produce CSVs from the filled XMLs.
"""
import os
import time
import argparse
from pathlib import Path
from typing import Optional, Dict
from lxml import html, etree
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from src.medquad.io import parse_folder


DEF_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}


def scrape_adam_answer(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=30, headers=DEF_HEADERS)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        content = tree.xpath('//div[@id="mainContent"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="content"]//p/text()')
        if not content:
            content = tree.xpath('//article//p/text()')
        if content:
            return ' '.join(c.strip() for c in content if c and c.strip())
        return None
    except Exception as e:
        print(f"Error scraping ADAM URL {url}: {e}")
        return None


def scrape_drugs_answer(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=30, headers=DEF_HEADERS)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        content = tree.xpath('//div[@id="main-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="drug-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="section"]//p/text()')
        if content:
            return ' '.join(c.strip() for c in content if c and c.strip())
        return None
    except Exception as e:
        print(f"Error scraping Drugs URL {url}: {e}")
        return None


def scrape_herbs_answer(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=30, headers=DEF_HEADERS)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        content = tree.xpath('//div[@id="main-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="herb-content"]//p/text()')
        if not content:
            content = tree.xpath('//section[@class="content"]//p/text()')
        if content:
            return ' '.join(c.strip() for c in content if c and c.strip())
        return None
    except Exception as e:
        print(f"Error scraping Herbs URL {url}: {e}")
        return None


def process_xml_file(xml_path: str, scraper_func, output_dir: str) -> Dict:
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        questions = root.xpath('//QAPair')
        scraped_count = 0
        for q in questions:
            answer_elem = q.find('Answer')
            if answer_elem is None or not (answer_elem.text and answer_elem.text.strip()):
                question_elem = q.find('Question')
                if question_elem is not None and question_elem.text:
                    # Try to get URL from several places
                    url_elem = q.find('Url')
                    if url_elem is None:
                        url_elem = root.find('Url')
                    if url_elem is not None and url_elem.text:
                        url = url_elem.text.strip()
                        scraped_answer = scraper_func(url)
                        if scraped_answer:
                            if answer_elem is None:
                                answer_elem = etree.SubElement(q, 'Answer')
                            answer_elem.text = scraped_answer
                            scraped_count += 1
                            time.sleep(0.2)
        output_path = os.path.join(output_dir, os.path.basename(xml_path))
        os.makedirs(output_dir, exist_ok=True)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        return {'file': xml_path, 'scraped': scraped_count, 'status': 'success'}
    except Exception as e:
        return {'file': xml_path, 'error': str(e), 'status': 'failed'}


def scrape_missing_answers(medquad_path: str, max_workers: int = 4):
    subsets = {
        '10_MPlus_ADAM_QA': scrape_adam_answer,
        '11_MPlusDrugs_QA': scrape_drugs_answer,
        '12_MPlusHerbsSupplements_QA': scrape_herbs_answer
    }
    all_results = []
    for subset_name, scraper_func in subsets.items():
        subset_path = os.path.join(medquad_path, subset_name)
        if not os.path.exists(subset_path):
            print(f"Warning: {subset_name} not found at {subset_path}")
            continue
        output_dir = os.path.join(medquad_path, f'filled_{subset_name}')
        if os.path.exists(output_dir) and len(list(Path(output_dir).glob('*.xml')))>0:
            print(f"✅ {subset_name} already scraped. Skipping.")
            # still parse existing filled dir into CSV
            try:
                df = parse_folder(output_dir)
                csv_out = Path('reports') / f'filled_{subset_name}.csv'
                csv_out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_out, index=False)
                print('Saved parsed CSV to', csv_out)
            except Exception as e:
                print('Could not parse existing filled dir:', e)
            continue
        xml_files = list(Path(subset_path).glob('*.xml'))
        print(f"Found {len(xml_files)} files in {subset_name}")
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_xml_file, str(x), scraper_func, output_dir): x for x in xml_files}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                if res.get('status')=='success':
                    print(f"✓ {Path(res['file']).name}: Scraped {res.get('scraped',0)} answers")
                else:
                    print(f"✗ {Path(res['file']).name}: {res.get('error')}")
        all_results.extend(results)
        # parse the filled folder to CSV
        try:
            df = parse_folder(output_dir)
            csv_out = Path('reports') / f'filled_{subset_name}.csv'
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_out, index=False)
            print('Saved parsed CSV to', csv_out)
        except Exception as e:
            print('Could not parse filled XMLs for', subset_name, e)
    total_scraped = sum(r.get('scraped',0) for r in all_results if r.get('status')=='success')
    print('\n' + '='*40)
    print('Scraping Complete!')
    print('Total files processed:', len(all_results))
    print('Total answers scraped:', total_scraped)
    print('='*40)
    return all_results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--medquad-path', required=True, help='Path to the MedQuAD root folder (contains dataset subdirs)')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    scrape_missing_answers(args.medquad_path, max_workers=args.workers)
