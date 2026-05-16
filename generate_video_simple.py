"""
Generate Thesis Defence Video using matplotlib + imageio
Simpler and more reliable than MoviePy 2.x
"""

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import imageio
import imageio_ffmpeg

os.makedirs('reports/video/frames', exist_ok=True)

W, H = 1280, 720
DPI = 100
FIG_W, FIG_H = W / DPI, H / DPI

# Color scheme
PRIMARY = '#003366'
SECONDARY = '#006699'
ACCENT = '#FF6600'
TEXT_COLOR = '#333333'
BG_COLOR = '#FAFAFA'

def save_frame(fig, name):
    path = f'reports/video/frames/{name:04d}.png'
    fig.savefig(path, dpi=DPI, facecolor=BG_COLOR, edgecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return path

def create_title_slide(text, subtitle, author, duration_secs, fps):
    frames = []
    for _ in range(int(duration_secs * fps)):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        fig.patch.set_facecolor(PRIMARY)
        ax.set_facecolor(PRIMARY)
        
        # Title
        ax.text(W/2, H*0.55, text, fontsize=32, color='white', weight='bold',
                ha='center', va='center', wrap=True)
        # Subtitle
        ax.text(W/2, H*0.38, subtitle, fontsize=18, color='#C8DCE8',
                ha='center', va='center', wrap=True)
        # Author
        ax.text(W/2, H*0.22, author, fontsize=16, color='white',
                ha='center', va='center')
        # Institution
        ax.text(W/2, H*0.15, "Liverpool John Moores University | MSc AI and ML", fontsize=12, color='#B4C8DC',
                ha='center', va='center')
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        frames.append(img)
        plt.close(fig)
    return frames

def create_section_slide(number, title, duration_secs, fps):
    frames = []
    for _ in range(int(duration_secs * fps)):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        fig.patch.set_facecolor(SECONDARY)
        ax.set_facecolor(SECONDARY)
        
        ax.text(W/2, H*0.55, str(number), fontsize=100, color='white', weight='bold',
                ha='center', va='center')
        ax.text(W/2, H*0.35, title.upper(), fontsize=36, color='white', weight='bold',
                ha='center', va='center')
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        frames.append(img)
        plt.close(fig)
    return frames

def create_bullet_slide(title, bullets, duration_secs, fps):
    frames = []
    for _ in range(int(duration_secs * fps)):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        
        # Header bar
        rect = patches.Rectangle((0, H-80), W, 80, linewidth=0, edgecolor='none', facecolor=PRIMARY)
        ax.add_patch(rect)
        ax.text(30, H-40, title, fontsize=22, color='white', weight='bold',
                ha='left', va='center')
        
        # Bullets
        y_start = H - 150
        for i, bullet in enumerate(bullets):
            ax.text(50, y_start - i*60, f"• {bullet}", fontsize=15, color=TEXT_COLOR,
                    ha='left', va='top', wrap=True)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        frames.append(img)
        plt.close(fig)
    return frames

def create_figure_slide(title, image_path, caption, duration_secs, fps):
    if not os.path.exists(image_path):
        return create_bullet_slide(title, [f"[Figure: {caption}]"], duration_secs, fps)
    
    frames = []
    for _ in range(int(duration_secs * fps)):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        
        # Header bar
        rect = patches.Rectangle((0, H-80), W, 80, linewidth=0, edgecolor='none', facecolor=PRIMARY)
        ax.add_patch(rect)
        ax.text(30, H-40, title, fontsize=20, color='white', weight='bold',
                ha='left', va='center')
        
        # Image
        try:
            from matplotlib.image import imread
            img_data = imread(image_path)
            img_h = 450
            img_w = int(img_h * img_data.shape[1] / img_data.shape[0])
            if img_w > W - 100:
                img_w = W - 100
                img_h = int(img_w * img_data.shape[0] / img_data.shape[1])
            ax.imshow(img_data, extent=[(W-img_w)/2, (W+img_w)/2, H-100-img_h, H-100])
        except Exception:
            pass
        
        # Caption
        ax.text(W/2, 50, caption, fontsize=13, color=SECONDARY, style='italic',
                ha='center', va='center')
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        frames.append(img)
        plt.close(fig)
    return frames

def create_two_col_slide(title, left_title, left_items, right_title, right_items, duration_secs, fps):
    frames = []
    for _ in range(int(duration_secs * fps)):
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.axis('off')
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        
        rect = patches.Rectangle((0, H-80), W, 80, linewidth=0, edgecolor='none', facecolor=PRIMARY)
        ax.add_patch(rect)
        ax.text(30, H-40, title, fontsize=22, color='white', weight='bold', ha='left', va='center')
        
        # Left column
        ax.text(50, H-130, left_title, fontsize=18, color=SECONDARY, weight='bold', ha='left', va='top')
        for i, item in enumerate(left_items):
            ax.text(50, H-180 - i*50, f"• {item}", fontsize=14, color=TEXT_COLOR, ha='left', va='top', wrap=True)
        
        # Divider
        ax.plot([W/2, W/2], [H-150, 100], color=SECONDARY, linewidth=2)
        
        # Right column
        ax.text(W/2 + 30, H-130, right_title, fontsize=18, color=SECONDARY, weight='bold', ha='left', va='top')
        for i, item in enumerate(right_items):
            ax.text(W/2 + 30, H-180 - i*50, f"• {item}", fontsize=14, color=TEXT_COLOR, ha='left', va='top', wrap=True)
        
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        frames.append(img)
        plt.close(fig)
    return frames

# ==================== BUILD ALL FRAMES ====================
FPS = 12
all_frames = []

# 1. Title
all_frames.extend(create_title_slide(
    "A Comparative Study of RAG Pipelines\nfor Medical Question Answering",
    "Reducing Hallucination in Clinical Decision Support Systems",
    "Raghunandan Kavi | PN1196933",
    4, FPS))

# 2. Outline
all_frames.extend(create_bullet_slide("Presentation Outline", [
    "Background & Motivation — Why medical AI needs grounded answers",
    "Problem Statement — Hallucination risks in clinical settings",
    "Research Questions & Hypotheses",
    "Methodology — Five pipeline configurations",
    "Results — RAGAS and DeepEval metrics",
    "Key Findings & Clinical Implications",
    "Conclusion & Future Work"
], 5, FPS))

# 3. Section 1
all_frames.extend(create_section_slide(1, "Background & Motivation", 3, FPS))

# 4. Problem
all_frames.extend(create_bullet_slide("The Hallucination Problem in Medical AI", [
    "LLMs produce confident but factually incorrect medical advice",
    "Documented case: chatbot said chemo drug was safe during pregnancy",
    "Hallucination rates: 15-30% on complex medical queries (Singhal et al., 2023)",
    "Oncology meta-analysis: 23% across 6,523 responses (Yoon et al., 2026)",
    "Current systems lack evidence grounding — no source verification"
], 5, FPS))

# 5. What is RAG
all_frames.extend(create_bullet_slide("What is Retrieval-Augmented Generation?", [
    "RAG grounds LLM outputs in external knowledge sources",
    "Three components: Retriever → Encoder → Generator (Lewis et al., 2020)",
    "Dense retrieval: vector embeddings for semantic similarity",
    "Sparse retrieval: keyword matching for exact terminology",
    "Medical docs need BOTH: semantic coverage + terminological precision"
], 5, FPS))

# 6. Section 2
all_frames.extend(create_section_slide(2, "Research Questions", 3, FPS))

# 7. RQs
all_frames.extend(create_bullet_slide("Research Questions", [
    "RQ1: How do RAG pipelines differ from vanilla LLM in reducing hallucinations?",
    "RQ2: How does retrieval depth affect faithfulness and hallucination risk?",
    "RQ3: How do query expansion, hybrid retrieval, and reformulation compare?",
    "RQ4: Where do failures occur and how do patterns vary by strategy?"
], 5, FPS))

# 8. Hypotheses
all_frames.extend(create_bullet_slide("Hypotheses", [
    "H1: All four RAG pipelines achieve significantly better metrics than baseline",
    "H2: Hybrid retrieval wins due to combining semantic + lexical search",
    "H3: Multi-query expansion trades precision for recall"
], 4, FPS))

# 9. Section 3
all_frames.extend(create_section_slide(3, "Methodology", 3, FPS))

# 10. Dataset
all_frames.extend(create_bullet_slide("Dataset: MedQuAD", [
    "47,457 Q&A pairs from 12 NIH sources (Ben Abacha & Demner-Fushman, 2019)",
    "Covers symptoms, treatments, diagnosis, medication, prognosis",
    "Consumer health information — no PII",
    "Preprocessing: deduplication, HTML stripping, normalization",
    "Stratified sampling across question types"
], 5, FPS))

# 11. Controlled Design
all_frames.extend(create_two_col_slide(
    "Controlled Experimental Design",
    "Fixed Across All Pipelines",
    ["LLM: GPT-4o-mini (temp=0.0)", "Embedding: text-embedding-3-small", "Chunking: 400 tokens, 50 overlap", "Top-k: 5 documents", "Safety-oriented prompt"],
    "Five Pipeline Configurations",
    ["1. Vanilla LLM (no retrieval)", "2. Standard RAG (dense semantic)", "3. Multi-Query Expansion", "4. Hybrid Retrieval (RRF)", "5. Query Reformulation + Reranking"],
    6, FPS))

# 12. Evaluation
all_frames.extend(create_two_col_slide(
    "Evaluation Framework",
    "RAGAS (Grounding Quality)",
    ["Faithfulness — claims supported", "Answer Relevance — on-topic", "Context Precision — signal/noise", "Context Recall — completeness"],
    "DeepEval (Safety & Reliability)",
    ["Hallucination Rate", "Groundedness", "Correctness", "Safety Compliance"],
    5, FPS))

# 13. Section 4
all_frames.extend(create_section_slide(4, "Results & Analysis", 3, FPS))

# 14-21. Figures
fig_data = [
    ("RAGAS Faithfulness: Hybrid Retrieval Wins", "reports/figures/figure_5_1_faithfulness.png", "Hybrid: 0.89 | Standard: 0.78 | Vanilla: 0.42"),
    ("Hallucination Rate: 81% Reduction", "reports/figures/figure_5_3_hallucination.png", "Hybrid: 0.11 | Standard: 0.22 | Vanilla: 0.58"),
    ("Context Precision & Recall", "reports/figures/figure_5_2_precision_recall.png", "Hybrid achieves best balance: 0.86 precision, 0.83 recall"),
    ("Safety Compliance", "reports/figures/figure_5_6_safety.png", "Clinical threshold: 0.80 | Hybrid: 0.92 | Vanilla: 0.45"),
    ("Faithfulness by Question Type", "reports/figures/figure_5_5_per_type.png", "Symptom questions easiest; treatment most challenging"),
    ("Failure Pattern Distribution", "reports/figures/figure_5_7_failures.png", "Hybrid retrieval minimizes all failure categories"),
    ("Latency vs Accuracy", "reports/figures/figure_5_8_latency.png", "Hybrid: 1,680ms — best accuracy-to-latency ratio"),
    ("Faithfulness vs Hallucination: r = -0.97", "reports/figures/figure_5_9_correlation.png", "Faithfulness is a reliable proxy for hallucination risk"),
]

for title, path, caption in fig_data:
    all_frames.extend(create_figure_slide(title, path, caption, 6, FPS))

# 22. Section 5
all_frames.extend(create_section_slide(5, "Key Findings", 3, FPS))

# 23. Findings
all_frames.extend(create_bullet_slide("Key Findings", [
    "Retrieval grounding DRAMATICALLY reduces hallucination — 62-81% improvement",
    "Hybrid retrieval is standout: highest faithfulness (0.89), lowest hallucination (0.11)",
    "Multi-query expansion improves recall but introduces noise",
    "Query reformulation improves precision but adds latency (3,200ms)",
    "All RAG pipelines exceed clinical safety threshold of 0.80",
    "Symptom questions easiest; treatment questions remain challenging"
], 6, FPS))

# 24. Implications
all_frames.extend(create_bullet_slide("Implications for Clinical Practice", [
    "Vanilla LLM is UNSAFE for clinical deployment (hallucination 0.58)",
    "Standard RAG provides good balance: 62% reduction, 1,240ms latency",
    "Hybrid retrieval recommended for high-stakes applications",
    "Retrieval grounding naturally promotes cautious language",
    "Goal: augment clinicians, not replace them"
], 5, FPS))

# 25. Contributions
all_frames.extend(create_bullet_slide("Contributions to Knowledge", [
    "First controlled benchmark isolating retrieval strategy effects",
    "Empirical evidence hybrid retrieval outperforms single-method",
    "Per-question-type analysis for nuanced guidance",
    "Structured failure pattern classification",
    "Safety compliance evaluation addressing critical gap"
], 5, FPS))

# 26. Section 6
all_frames.extend(create_section_slide(6, "Conclusion", 3, FPS))

# 27. Limitations
all_frames.extend(create_bullet_slide("Limitations & Future Work", [
    "Limited to consumer health questions — not clinician-facing content",
    "Automatic metrics need clinician validation",
    "English-only — multilingual medical RAG unexplored",
    "Future: knowledge graphs, time-aware retrieval, clinician evaluation",
    "Future: multilingual datasets, granular safety taxonomy"
], 5, FPS))

# 28. Thank you
all_frames.extend(create_title_slide("Thank You", "Questions & Discussion", "", 4, FPS))

# ==================== RENDER VIDEO ====================
print(f"Total frames: {len(all_frames)}")
print(f"Estimated duration: {len(all_frames)/FPS:.1f} seconds ({len(all_frames)/FPS/60:.1f} minutes)")

output_path = r'C:\Users\HP\Downloads\LJMU_Thesis_Defence_Video.mp4'
imageio.mimsave(output_path, all_frames, fps=FPS, quality=8, codec='libx264', pixelformat='yuv420p')
print(f"Video saved to: {output_path}")
