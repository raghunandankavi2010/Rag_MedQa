"""Scraper for missing answers in MedQuAD database."""
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict
from lxml import html, etree
from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape_adam_answer(url: str) -> Optional[str]:
    """Scrape answer from ADAM (A.D.A.M. Medical Encyclopedia) pages."""
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        tree = html.fromstring(response.content)
        
        content = tree.xpath('//div[@id="mainContent"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="content"]//p/text()')
        if not content:
            content = tree.xpath('//article//p/text()')
        
        if content:
            return ' '.join(content).strip()
        return None
    except Exception as e:
        print(f"Error scraping ADAM URL {url}: {e}")
        return None

def scrape_drugs_answer(url: str) -> Optional[str]:
    """Scrape answer from MedlinePlus Drug Information pages."""
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        tree = html.fromstring(response.content)
        
        content = tree.xpath('//div[@id="main-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="drug-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="section"]//p/text()')
        
        if content:
            return ' '.join(content).strip()
        return None
    except Exception as e:
        print(f"Error scraping Drugs URL {url}: {e}")
        return None

def scrape_herbs_answer(url: str) -> Optional[str]:
    """Scrape answer from MedlinePlus Herbs & Supplements pages."""
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        tree = html.fromstring(response.content)
        
        content = tree.xpath('//div[@id="main-content"]//p/text()')
        if not content:
            content = tree.xpath('//div[@class="herb-content"]//p/text()')
        if not content:
            content = tree.xpath('//section[@class="content"]//p/text()')
        
        if content:
            return ' '.join(content).strip()
        return None
    except Exception as e:
        print(f"Error scraping Herbs URL {url}: {e}")
        return None

def process_xml_file(xml_path: str, scraper_func, output_dir: str) -> Dict:
    """Process a single XML file and scrape missing answers."""
    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()
        
        questions = root.xpath('//QAPair')
        scraped_count = 0
        
        for q in questions:
            answer_elem = q.find('Answer')
            if answer_elem is None or not answer_elem.text or answer_elem.text.strip() == '':
                question_elem = q.find('Question')
                if question_elem is not None and question_elem.text:
                    # Try to get URL from various possible locations
                    # Try URL in several places: QAPair Url element, Document 'url' attribute, Document Url element
                    url = None
                    url_elem = q.find('Url')
                    if url_elem is not None and url_elem.text and url_elem.text.strip():
                        url = url_elem.text.strip()
                    else:
                        # document-level attribute (common in MedQuAD files)
                        url_attr = root.get('url') or root.get('URL') or root.get('source')
                        if url_attr and str(url_attr).strip():
                            url = str(url_attr).strip()
                        else:
                            root_url_elem = root.find('Url')
                            if root_url_elem is not None and root_url_elem.text and root_url_elem.text.strip():
                                url = root_url_elem.text.strip()

                    if url:
                        scraped_answer = scraper_func(url)
                        if scraped_answer:
                            if answer_elem is None:
                                answer_elem = etree.SubElement(q, 'Answer')
                            answer_elem.text = scraped_answer
                            scraped_count += 1
                            time.sleep(0.5)
        
        output_path = os.path.join(output_dir, os.path.basename(xml_path))
        os.makedirs(output_dir, exist_ok=True)
        tree.write(output_path, encoding='UTF-8', xml_declaration=True)
        
        return {'file': xml_path, 'scraped': scraped_count, 'status': 'success'}
    except Exception as e:
        return {'file': xml_path, 'error': str(e), 'status': 'failed'}

def scrape_missing_answers(medquad_path: str, max_workers: int = 4):
    """Main function to scrape all missing answers."""
    subsets = {
        '10_MPlus_ADAM_QA': scrape_adam_answer,
        '11_MPlusDrugs_QA': scrape_drugs_answer,
        '12_MPlusHerbsSupplements_QA': scrape_herbs_answer
    }
    
    results = []
    
    for subset_name, scraper_func in subsets.items():
        subset_path = os.path.join(medquad_path, subset_name)
        if not os.path.exists(subset_path):
            print(f"Warning: {subset_name} not found at {subset_path}")
            continue
        
        output_dir = os.path.join(medquad_path, f'filled_{subset_name}')
        
        # Check if already processed
        if os.path.exists(output_dir) and len(list(Path(output_dir).glob('*.xml'))) > 0:
            print(f"✅ {subset_name} already scraped. Skipping.")
            continue
        
        xml_files = list(Path(subset_path).glob('*.xml'))
        print(f"Found {len(xml_files)} files in {subset_name}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_xml_file, str(xml_file), scraper_func, output_dir): xml_file
                for xml_file in xml_files
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['status'] == 'success':
                    print(f"✓ {os.path.basename(result['file'])}: Scraped {result['scraped']} answers")
                else:
                    print(f"✗ {os.path.basename(result['file'])}: {result['error']}")
    
    total_scraped = sum(r.get('scraped', 0) for r in results if r['status'] == 'success')
    print(f"\n{'='*50}")
    print(f"Scraping Complete!")
    print(f"Total files processed: {len(results)}")
    print(f"Total answers scraped: {total_scraped}")
    print(f"{'='*50}")
    
    return results