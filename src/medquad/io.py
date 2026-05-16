import os
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

# Try to import scraper, but don't fail if it's not there
try:
    from .scraper import scrape_missing_answers
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    print("WARNING: Scraper module not found. Will only parse existing answers.")

MEDQUAD_GIT = "https://github.com/abachaa/MedQuAD.git"


def clone_repo(url: str, dest: str) -> None:
    """Clone the repository to `dest` if it doesn't already exist or is empty."""
    if os.path.exists(dest) and any(os.scandir(dest)):
        print(f"OK: Repository already exists at {dest}")
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Cloning repository...")
    subprocess.run(["git", "clone", url, dest], check=True)
    print(f"Clone complete!")


def parse_medquad_xml(file_path: str) -> list:
    results = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        focus = root.findtext('Focus', default='')
        group = root.findtext('.//SemanticGroup', default='Unknown')

        syns = [s.text for s in root.findall('.//Synonym') if s.text]
        synonyms_str = ", ".join(syns)

        for qa in root.findall('.//QAPair'):
            q_elem = qa.find('Question')
            question = q_elem.text if q_elem is not None else ''
            answer = qa.findtext('Answer', default='')
            qtype = q_elem.get('qtype', 'general') if q_elem is not None else 'general'

            if answer and len(answer.strip()) > 0:
                results.append({
                    "focus": focus,
                    "group": group,
                    "synonyms": synonyms_str,
                    "qtype": qtype,
                    "question": question,
                    "answer": answer,
                    "source_file": os.path.basename(file_path)
                })
    except Exception as e:
        pass
    return results


def parse_folder(base_path: str, scrape_missing: bool = True, max_workers: int = 4) -> pd.DataFrame:
    """Walk `base_path` and parse all XML files into a DataFrame."""
    
    # First, scrape missing answers if requested and scraper is available
    if scrape_missing and SCRAPER_AVAILABLE:
        print("\nChecking for missing answers in MedQuAD dataset...")
        print("This will fetch answers from A.D.A.M., Drugs, and Herbs/Supplements databases.")
        print("This may take 30-60 minutes on first run.\n")
        
        # Check if we already have filled directories
        filled_dirs_exist = any(
            os.path.exists(os.path.join(base_path, f'filled_{subset}'))
            for subset in ['10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA']
        )
        
        if filled_dirs_exist:
            print("Filled directories already exist. Skipping scraping.")
            print("   (To re-scrape, delete the 'filled_*' directories)\n")
        else:
            print("Starting scraper...")
            scrape_missing_answers(base_path, max_workers)
            print("\nScraping complete! Filled directories created.\n")
    elif scrape_missing and not SCRAPER_AVAILABLE:
        print("\nWARNING: Scraper not available. Install required packages:")
        print("   pip install requests lxml")
        print("   And ensure src/medquad/scraper.py exists\n")
    
    # Now parse all files (prioritize filled directories)
    all_rows = []
    
    print("Parsing XML files...")
    
    for root_dir, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.xml'):
                full_path = os.path.join(root_dir, file)
                
                # Prefer filled version if it exists
                if 'filled_' not in full_path:
                    for subset in ['10_MPlus_ADAM_QA', '11_MPlusDrugs_QA', '12_MPlusHerbsSupplements_QA']:
                        if subset in full_path:
                            filled_path = full_path.replace(subset, f'filled_{subset}')
                            if os.path.exists(filled_path):
                                full_path = filled_path
                                break
                
                rows = parse_medquad_xml(full_path)
                all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    print(f"\nTotal records parsed: {len(df)}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)