#!/usr/bin/env python3
"""Exploratory Data Analysis for the MedQuAD corpus.

Loads the parsed MedQuAD CSV, prints summary statistics, and writes plots
plus a machine-readable summary CSV to the reports/ directory.

Usage:
    python scripts/eda.py
    python scripts/eda.py --input data/medquad.csv --reports reports
"""
import os
import sys
import argparse

# Ensure project root is importable when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(
            f"Input CSV '{path}' not found. Run:\n"
            f"  python scripts/prepare_medquad.py --clone\n"
            f"to generate it."
        )
    df = pd.read_csv(path)
    # Drop rows without question/answer and add length features (in words)
    df = df.dropna(subset=["question", "answer"])
    df["answer_len"] = df["answer"].astype(str).str.split().str.len()
    df["question_len"] = df["question"].astype(str).str.split().str.len()
    return df


def summarize(df: pd.DataFrame, reports_dir: str) -> pd.DataFrame:
    """Print key stats and return a tidy summary DataFrame."""
    print("=" * 70)
    print("MedQuAD EDA SUMMARY")
    print("=" * 70)
    print(f"Total QA pairs            : {len(df):,}")
    print(f"Unique focus topics       : {df['focus'].nunique():,}" if "focus" in df else "")
    print(f"Unique source files       : {df['source_file'].nunique():,}" if "source_file" in df else "")
    print()
    print("Answer length (words)     :")
    print(df["answer_len"].describe().round(1).to_string())
    print()
    print("Question length (words)   :")
    print(df["question_len"].describe().round(1).to_string())

    if "qtype" in df.columns:
        print("\nTop question types (qtype):")
        print(df["qtype"].value_counts().head(15).to_string())

    if "group" in df.columns:
        print("\nSemantic groups:")
        print(df["group"].value_counts().head(15).to_string())

    # Build a tidy summary table for downstream use
    rows = [
        ("total_qa_pairs", len(df)),
        ("unique_focus", df["focus"].nunique() if "focus" in df else None),
        ("unique_source_files", df["source_file"].nunique() if "source_file" in df else None),
        ("answer_len_mean", round(df["answer_len"].mean(), 2)),
        ("answer_len_median", df["answer_len"].median()),
        ("answer_len_max", df["answer_len"].max()),
        ("question_len_mean", round(df["question_len"].mean(), 2)),
        ("question_len_median", df["question_len"].median()),
        ("question_len_max", df["question_len"].max()),
    ]
    summary = pd.DataFrame(rows, columns=["metric", "value"])
    summary_path = os.path.join(reports_dir, "eda_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV  -> {summary_path}")

    # Also save the qtype distribution if present
    if "qtype" in df.columns:
        qpath = os.path.join(reports_dir, "eda_qtype_distribution.csv")
        df["qtype"].value_counts().rename_axis("qtype").reset_index(name="count").to_csv(qpath, index=False)
        print(f"Saved qtype counts -> {qpath}")

    return summary


def make_plots(df: pd.DataFrame, reports_dir: str) -> None:
    # Answer length histogram
    plt.figure(figsize=(8, 5))
    df["answer_len"].clip(upper=df["answer_len"].quantile(0.99)).hist(bins=50, color="#4C72B0")
    plt.title("Answer Length Distribution (words, 99th-pctile clipped)")
    plt.xlabel("Words per answer")
    plt.ylabel("Count")
    plt.tight_layout()
    p1 = os.path.join(reports_dir, "answer_length_hist.png")
    plt.savefig(p1, dpi=120)
    plt.close()
    print(f"Saved plot         -> {p1}")

    # Question length histogram
    plt.figure(figsize=(8, 5))
    df["question_len"].clip(upper=df["question_len"].quantile(0.99)).hist(bins=40, color="#55A868")
    plt.title("Question Length Distribution (words, 99th-pctile clipped)")
    plt.xlabel("Words per question")
    plt.ylabel("Count")
    plt.tight_layout()
    p2 = os.path.join(reports_dir, "question_length_hist.png")
    plt.savefig(p2, dpi=120)
    plt.close()
    print(f"Saved plot         -> {p2}")

    # qtype bar chart (top 15)
    if "qtype" in df.columns:
        plt.figure(figsize=(9, 5))
        df["qtype"].value_counts().head(15).plot(kind="barh", color="#C44E52")
        plt.title("Top Question Types")
        plt.xlabel("Count")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        p3 = os.path.join(reports_dir, "qtype_distribution.png")
        plt.savefig(p3, dpi=120)
        plt.close()
        print(f"Saved plot         -> {p3}")


def main():
    parser = argparse.ArgumentParser(description="EDA for MedQuAD corpus")
    parser.add_argument("--input", default="data/medquad.csv", help="Parsed MedQuAD CSV")
    parser.add_argument("--reports", default="reports", help="Output directory for plots/summaries")
    args = parser.parse_args()

    os.makedirs(args.reports, exist_ok=True)
    df = load(args.input)
    summarize(df, args.reports)
    make_plots(df, args.reports)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
