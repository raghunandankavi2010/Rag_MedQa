#!/usr/bin/env python3
"""Thin CLI wrapper that uses `src.medquad.io` for work.

This keeps parsing logic inside the `src` package so it can be
imported and reused (for EDA, tests, etc.).
"""
import os
import sys
import argparse

# Ensure project root is on sys.path so `src` can be imported when running the script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.medquad.io import clone_repo, parse_folder


def main():
    parser = argparse.ArgumentParser(description="Clone and parse MedQuAD into CSV")
    parser.add_argument("--clone", action="store_true", help="Clone MedQuAD from GitHub into --dest")
    parser.add_argument("--dest", default="data/MedQuAD", help="Destination folder for the MedQuAD repo or existing dataset")
    parser.add_argument("--output", default="data/medquad.csv", help="CSV output path (optional)")
    args = parser.parse_args()

    dest = args.dest
    if args.clone:
        clone_repo("https://github.com/abachaa/MedQuAD.git", dest)

    if not os.path.exists(dest):
        raise SystemExit(f"Destination '{dest}' does not exist. Run with --clone or provide a valid --dest")

    df = parse_folder(dest)
    print(f"Processing Complete. Total rows extracted: {len(df)}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Wrote CSV to: {args.output}")
    else:
        print(df.head())


if __name__ == '__main__':
    main()
