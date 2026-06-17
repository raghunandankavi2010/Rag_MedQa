#!/usr/bin/env python3
"""Run the MedQuAD scraper to fetch missing answers."""
import os
import sys

# Add the project root to Python path (this is the fix)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Now this import will work
from src.medquad.scrapper import scrape_missing_answers

def main():
    medquad_path = "data/MedQuAD"
    workers = 4
    
    if not os.path.exists(medquad_path):
        print(f"❌ MedQuAD not found at {medquad_path}")
        print("First run: python scripts/prepare_medquad.py --clone")
        sys.exit(1)
    
    print(f"🕸️  Starting scraper on {medquad_path}")
    print(f"⚙️  Using {workers} workers")
    scrape_missing_answers(medquad_path, workers)
    print("✅ Scraping complete!")

if __name__ == '__main__':
    main()