#!/usr/bin/env python3
"""Compute all statistics needed to populate the thesis from the real run.

Reads reports/pipeline_comparison_raw.csv (per-question, per-pipeline) and
writes reports/thesis_stats.json plus a human-readable console summary:
  - per-pipeline mean, SD, and 95% CI for each metric
  - full one-way ANOVA table for faithfulness (SS/df/MS/F/p)
  - per-question-type faithfulness by pipeline
  - failure-pattern percentages by pipeline
  - Pearson r between faithfulness and hallucination (pipeline means)
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RAW = "reports/pipeline_comparison_raw.csv"
OUT = "reports/thesis_stats.json"

LABELS = {
    "P1_Vanilla_LLM": "Vanilla LLM",
    "P2_Standard_RAG": "Standard RAG",
    "P3_Multi_Query_Expansion": "Multi-Query Expansion",
    "P4_Hybrid_Retrieval": "Hybrid Retrieval",
    "P5_Query_Reformulation": "Query Reformulation",
}
ORDER = list(LABELS)
METRICS = [
    "ragas_faithfulness", "ragas_answer_relevance", "ragas_context_precision",
    "ragas_context_recall", "deepeval_hallucination", "deepeval_relevance",
    "rouge1_f", "rougeL_f", "safety_compliance", "word_overlap", "latency_ms",
    "num_docs",
]
FAILURES = ["failure_missing_evidence", "failure_noisy_evidence",
            "failure_unsupported_claims", "failure_unsafe_tone"]


def ci95(series):
    s = series.dropna()
    n = len(s)
    if n < 2:
        return (float(s.mean()) if n else 0.0, 0.0, 0.0)
    m, sd = s.mean(), s.std(ddof=1)
    half = 1.96 * sd / np.sqrt(n)
    return float(m), float(m - half), float(m + half)


def main():
    df = pd.read_csv(RAW)
    out = {"per_pipeline": {}, "n_per_pipeline": int(df.groupby("pipeline").size().iloc[0])}

    for pid in ORDER:
        g = df[df["pipeline"] == pid]
        rec = {"label": LABELS[pid], "n": int(len(g))}
        for m in METRICS:
            if m in g.columns:
                mean, lo, hi = ci95(g[m])
                rec[m] = {"mean": round(mean, 4),
                          "sd": round(float(g[m].std(ddof=1)), 4),
                          "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        for fcol in FAILURES:
            if fcol in g.columns:
                rec[fcol] = round(float(g[fcol].mean()) * 100, 1)  # percentage
        out["per_pipeline"][pid] = rec

    # ── Full ANOVA table for faithfulness ───────────────────────────────────
    groups = [df[df["pipeline"] == p]["ragas_faithfulness"].dropna().values for p in ORDER]
    allv = np.concatenate(groups)
    grand = allv.mean()
    k = len(groups)
    N = len(allv)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, N - k
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    F, p = stats.f_oneway(*groups)
    out["anova_faithfulness"] = {
        "ss_between": round(float(ss_between), 4), "df_between": df_b,
        "ms_between": round(float(ms_b), 4),
        "ss_within": round(float(ss_within), 4), "df_within": df_w,
        "ms_within": round(float(ms_w), 4),
        "ss_total": round(float(ss_between + ss_within), 4), "df_total": N - 1,
        "F": round(float(F), 4), "p": float(p),
    }

    # ── Per-question-type faithfulness ──────────────────────────────────────
    qt = df.pivot_table(index="qtype", columns="pipeline",
                        values="ragas_faithfulness", aggfunc="mean").round(4)
    qt = qt.reindex(columns=ORDER)
    out["qtype_counts"] = df[df.pipeline == ORDER[0]]["qtype"].value_counts().to_dict()
    out["qtype_faithfulness"] = {idx: {p: (None if pd.isna(qt.loc[idx, p]) else round(float(qt.loc[idx, p]), 4)) for p in ORDER} for idx in qt.index}

    # ── Pearson r: faithfulness vs hallucination (pipeline means) ───────────
    fm = [df[df.pipeline == p]["ragas_faithfulness"].mean() for p in ORDER]
    hm = [df[df.pipeline == p]["deepeval_hallucination"].mean() for p in ORDER]
    r, rp = stats.pearsonr(fm, hm)
    out["pearson_faith_halluc"] = {"r": round(float(r), 4), "p": round(float(rp), 4)}

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"n per pipeline = {out['n_per_pipeline']}")
    print("\nFaithfulness  mean [95% CI]  (SD):")
    for p in ORDER:
        r_ = out["per_pipeline"][p]["ragas_faithfulness"]
        print(f"  {LABELS[p]:22s} {r_['mean']:.3f}  [{r_['ci_lo']:.3f}, {r_['ci_hi']:.3f}]  (SD {r_['sd']:.3f})")
    a = out["anova_faithfulness"]
    print(f"\nANOVA: F({a['df_between']},{a['df_within']})={a['F']:.3f}, p={a['p']:.4f}")
    print(f"  SSb={a['ss_between']}, SSw={a['ss_within']}, MSb={a['ms_between']}, MSw={a['ms_within']}")
    print(f"\nPearson r (faithfulness vs hallucination, pipeline means) = {out['pearson_faith_halluc']['r']:.3f} (p={out['pearson_faith_halluc']['p']:.3f})")
    print(f"\nqtype counts: {out['qtype_counts']}")
    print("\nFailure patterns (% of answered):")
    for p in ORDER:
        rec = out["per_pipeline"][p]
        print(f"  {LABELS[p]:22s} miss={rec['failure_missing_evidence']:5.1f} noisy={rec['failure_noisy_evidence']:5.1f} unsup={rec['failure_unsupported_claims']:5.1f} tone={rec['failure_unsafe_tone']:5.1f}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
