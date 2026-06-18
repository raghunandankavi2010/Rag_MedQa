#!/usr/bin/env python3
"""Regenerate the Chapter 5 figures from the real comparison results.

Outputs PNGs to reports/figures/ matching the thesis figure list (5.1-5.9),
using a clean, consistent style. Latency uses the median (robust to API-retry
outliers). No API calls.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "reports/figures"
os.makedirs(OUTDIR, exist_ok=True)
RAW = "reports/pipeline_comparison_raw.csv"
STATS = json.load(open("reports/thesis_stats.json"))

ORDER = ["P1_Vanilla_LLM", "P2_Standard_RAG", "P3_Multi_Query_Expansion",
         "P4_Hybrid_Retrieval", "P5_Query_Reformulation"]
SHORT = ["Vanilla\nLLM", "Standard\nRAG", "Multi-Query", "Hybrid", "Reformulation"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
df = pd.read_csv(RAW)
PP = STATS["per_pipeline"]


def means(metric):
    return [PP[p][metric]["mean"] for p in ORDER]


def cis(metric):
    los = [PP[p][metric]["mean"] - PP[p][metric]["ci_lo"] for p in ORDER]
    return los


def barfig(values, title, ylabel, fname, errs=None, lower_better=False, ymax=None, pct=False):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(SHORT, values, color=COLORS, yerr=errs, capsize=4,
                  edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax if ymax else max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(values) * 0.02,
                f"{v:.3f}" if not pct else f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=130)
    plt.close()
    print("wrote", fname)


# Fig 5.1 Faithfulness (with 95% CI)
barfig(means("ragas_faithfulness"), "RAGAS Faithfulness by Pipeline (mean, 95% CI)",
       "Faithfulness", "fig5_1_faithfulness.png", errs=cis("ragas_faithfulness"), ymax=0.9)

# Fig 5.2 Context precision & recall (grouped)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(ORDER)); w = 0.38
ax.bar(x - w/2, means("ragas_context_precision"), w, label="Context Precision", color="#4C72B0", edgecolor="black", linewidth=0.4)
ax.bar(x + w/2, means("ragas_context_recall"), w, label="Context Recall", color="#DD8452", edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
ax.set_title("RAGAS Context Precision and Recall by Pipeline", fontsize=12, fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig5_2_precision_recall.png"), dpi=130); plt.close()
print("wrote fig5_2")

# Fig 5.3 Hallucination rate
barfig(means("deepeval_hallucination"), "DeepEval Hallucination Rate by Pipeline (lower is better)",
       "Hallucination rate", "fig5_3_hallucination.png", ymax=0.45)

# Fig 5.4 Answer relevance (RAGAS)
barfig(means("ragas_answer_relevance"), "RAGAS Answer Relevance by Pipeline",
       "Answer relevance", "fig5_4_answer_relevance.png", ymax=0.6)

# Fig 5.5 Faithfulness by question type (top types by count)
qcounts = STATS["qtype_counts"]
qtypes = [q for q, _ in sorted(qcounts.items(), key=lambda kv: -kv[1])][:5]
qf = STATS["qtype_faithfulness"]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(qtypes)); w = 0.16
for i, p in enumerate(ORDER):
    vals = [qf.get(q, {}).get(p) or 0 for q in qtypes]
    ax.bar(x + (i - 2) * w, vals, w, label=SHORT[i].replace("\n", " "), color=COLORS[i], edgecolor="black", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels([q.capitalize() for q in qtypes]); ax.set_ylabel("Faithfulness"); ax.set_ylim(0, 1.0)
ax.set_title("Faithfulness by Question Type and Pipeline", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12)); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig5_5_faithfulness_by_qtype.png"), dpi=130); plt.close()
print("wrote fig5_5")

# Fig 5.6 Safety-orientation score
barfig(means("safety_compliance"), "Safety-Orientation Score by Pipeline",
       "Safety-orientation score", "fig5_6_safety.png", ymax=0.4)

# Fig 5.7 Failure patterns (grouped: missing, noisy, unsupported)
fails = ["failure_missing_evidence", "failure_noisy_evidence", "failure_unsupported_claims"]
flab = ["Missing evidence", "Noisy evidence", "Unsupported claims"]
fcol = ["#4C72B0", "#DD8452", "#C44E52"]
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(ORDER)); w = 0.26
for i, (fk, fl) in enumerate(zip(fails, flab)):
    vals = [PP[p][fk] for p in ORDER]
    ax.bar(x + (i - 1) * w, vals, w, label=fl, color=fcol[i], edgecolor="black", linewidth=0.3)
ax.set_xticks(x); ax.set_xticklabels(SHORT); ax.set_ylabel("% of answered questions"); ax.set_ylim(0, 109)
ax.set_title("Primary Failure Pattern Distribution by Pipeline", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig5_7_failures.png"), dpi=130); plt.close()
print("wrote fig5_7")

# Fig 5.8 Median latency
med = [df[df.pipeline == p]["latency_ms"].median() / 1000.0 for p in ORDER]
barfig(med, "Median End-to-End Latency by Pipeline", "Latency (seconds)", "fig5_8_latency.png", ymax=max(med) * 1.3)

# Fig 5.9 Faithfulness vs hallucination scatter
fm = means("ragas_faithfulness"); hm = means("deepeval_hallucination")
r = STATS["pearson_faith_halluc"]["r"]
fig, ax = plt.subplots(figsize=(7.5, 5))
for i in range(len(ORDER)):
    ax.scatter(fm[i], hm[i], s=130, color=COLORS[i], edgecolor="black", zorder=3)
    ax.annotate(SHORT[i].replace("\n", " "), (fm[i], hm[i]), textcoords="offset points", xytext=(8, 5), fontsize=9)
z = np.polyfit(fm, hm, 1); xs = np.linspace(min(fm) - 0.02, max(fm) + 0.02, 50)
ax.plot(xs, np.polyval(z, xs), "--", color="grey", label=f"trend (r = {r:.2f})")
ax.set_xlabel("RAGAS faithfulness"); ax.set_ylabel("DeepEval hallucination rate")
ax.set_title("Faithfulness versus Hallucination Rate (pipeline means)", fontsize=12, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "fig5_9_faith_vs_halluc.png"), dpi=130); plt.close()
print("wrote fig5_9")

print("\nAll figures written to", OUTDIR)
