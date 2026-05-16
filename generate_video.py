"""
Generate Thesis Defence Video
=============================
Creates an MP4 video walkthrough of the thesis using moviepy.
Combines slides, figures, and narration text.
"""

import os
from moviepy import (
    ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips,
    ColorClip
)

os.makedirs('reports/video', exist_ok=True)

# Video settings
W, H = 1280, 720
DURATION_PER_SLIDE = 6  # seconds per content slide
TITLE_DURATION = 5
FIGURE_DURATION = 8

# Color scheme
BG_COLOR = (250, 250, 250)
PRIMARY = (0, 51, 102)
SECONDARY = (0, 102, 153)
ACCENT = (255, 102, 0)
TEXT_COLOR = (51, 51, 51)

def make_bg(duration):
    return ColorClip(size=(W, H), color=BG_COLOR).with_duration(duration)

def make_header_bar(duration):
    bar = ColorClip(size=(W, 80), color=PRIMARY).with_duration(duration)
    return bar.with_position((0, 0))

def make_title_clip(text, duration):
    bg = make_bg(duration)
    header = make_header_bar(duration)
    
    txt = TextClip(text, fontsize=48, color='white', font='Arial-Bold',
                   method='caption', size=(W-100, 200), align='center')
    txt = txt.with_duration(duration).with_position(('center', 200))
    
    return CompositeVideoClip([bg, header, txt])

def make_bullet_slide(title, bullets, duration):
    bg = make_bg(duration)
    header = make_header_bar(duration)
    
    title_txt = TextClip(title, fontsize=32, color='white', font='Arial-Bold',
                         method='caption', size=(W-80, 60), align='left')
    title_txt = title_txt.with_duration(duration).with_position((40, 15))
    
    bullet_text = "\n".join([f"• {b}" for b in bullets])
    body = TextClip(bullet_text, fontsize=22, color='black', font='Arial',
                    method='caption', size=(W-120, 500), align='left')
    body = body.with_duration(duration).with_position((60, 110))
    
    return CompositeVideoClip([bg, header, title_txt, body])

def make_figure_slide(title, image_path, caption, duration):
    if not os.path.exists(image_path):
        return make_bullet_slide(title, [f"[Figure: {caption}]"], duration)
    
    bg = make_bg(duration)
    header = make_header_bar(duration)
    
    title_txt = TextClip(title, fontsize=28, color='white', font='Arial-Bold',
                         method='caption', size=(W-80, 60), align='left')
    title_txt = title_txt.with_duration(duration).with_position((40, 15))
    
    img = ImageClip(image_path).with_duration(duration)
    img_h = 480
    img_w = int(img_h * img.size[0] / img.size[1])
    if img_w > W - 100:
        img_w = W - 100
        img_h = int(img_w * img.size[1] / img.size[0])
    img = img.resize((img_w, img_h))
    img = img.with_position(('center', 100))
    
    cap = TextClip(caption, fontsize=16, color=SECONDARY, font='Arial-Italic',
                   method='caption', size=(W-100, 60), align='center')
    cap = cap.with_duration(duration).with_position(('center', 620))
    
    return CompositeVideoClip([bg, header, title_txt, img, cap])

def make_two_column_slide(title, left_title, left_items, right_title, right_items, duration):
    bg = make_bg(duration)
    header = make_header_bar(duration)
    
    title_txt = TextClip(title, fontsize=30, color='white', font='Arial-Bold',
                         method='caption', size=(W-80, 60), align='left')
    title_txt = title_txt.with_duration(duration).with_position((40, 15))
    
    lt = TextClip(left_title, fontsize=24, color=SECONDARY, font='Arial-Bold',
                  method='caption', size=(550, 40), align='left')
    lt = lt.with_duration(duration).with_position((50, 110))
    
    left_text = "\n".join([f"• {i}" for i in left_items])
    lb = TextClip(left_text, fontsize=18, color='black', font='Arial',
                  method='caption', size=(550, 500), align='left')
    lb = lb.with_duration(duration).with_position((50, 160))
    
    rt = TextClip(right_title, fontsize=24, color=SECONDARY, font='Arial-Bold',
                  method='caption', size=(550, 40), align='left')
    rt = rt.with_duration(duration).with_position((680, 110))
    
    right_text = "\n".join([f"• {i}" for i in right_items])
    rb = TextClip(right_text, fontsize=18, color='black', font='Arial',
                  method='caption', size=(550, 500), align='left')
    rb = rb.with_duration(duration).with_position((680, 160))
    
    divider = ColorClip(size=(3, 500), color=SECONDARY).with_duration(duration).with_position((635, 130))
    
    return CompositeVideoClip([bg, header, title_txt, lt, lb, rt, rb, divider])

def make_section_slide(number, title, duration):
    bg = ColorClip(size=(W, H), color=SECONDARY).with_duration(duration)
    
    num = TextClip(str(number), fontsize=120, color='white', font='Arial-Bold')
    num = num.with_duration(duration).with_position(('center', 200))
    
    txt = TextClip(title.upper(), fontsize=50, color='white', font='Arial-Bold',
                   method='caption', size=(W-100, 150), align='center')
    txt = txt.with_duration(duration).with_position(('center', 380))
    
    return CompositeVideoClip([bg, num, txt])

# ==================== BUILD VIDEO ====================
clips = []

# 1. Title
clips.append(make_title_clip(
    "A Comparative Study of RAG Pipelines\nfor Medical Question Answering",
    TITLE_DURATION
))

# 2. Author info
bg = make_bg(4)
author = TextClip("Raghunandan Kavi  |  PN1196933\n\nLiverpool John Moores University\nMSc Artificial Intelligence and Machine Learning",
                  fontsize=28, color=PRIMARY, font='Arial', method='caption', size=(W-100, 300), align='center')
author = author.with_duration(4).with_position(('center', 280))
clips.append(CompositeVideoClip([bg, author]))

# 3. Section 1
clips.append(make_section_slide(1, "Background & Motivation", 4))

# 4. The Problem
clips.append(make_bullet_slide(
    "The Hallucination Problem in Medical AI",
    [
        "LLMs produce confident but factually incorrect medical advice",
        "Documented case: chatbot said chemo drug was safe during pregnancy",
        "Hallucination rates: 15-30% on complex medical queries",
        "Oncology meta-analysis: 23% across 6,523 responses",
        "Current systems lack evidence grounding — no source verification"
    ], DURATION_PER_SLIDE
))

# 5. What is RAG
clips.append(make_bullet_slide(
    "What is Retrieval-Augmented Generation?",
    [
        "RAG grounds LLM outputs in external knowledge sources",
        "Three components: Retriever → Encoder → Generator",
        "Dense retrieval: vector embeddings for semantic similarity",
        "Sparse retrieval: keyword matching for exact terminology",
        "Medical docs need BOTH: semantic coverage + terminological precision"
    ], DURATION_PER_SLIDE
))

# 6. Section 2
clips.append(make_section_slide(2, "Research Questions", 4))

# 7. RQs
clips.append(make_bullet_slide(
    "Research Questions",
    [
        "RQ1: How do RAG pipelines differ from vanilla LLM in reducing hallucinations?",
        "RQ2: How does retrieval depth affect faithfulness and hallucination risk?",
        "RQ3: How do query expansion, hybrid retrieval, and reformulation compare?",
        "RQ4: Where do failures occur and how do patterns vary by strategy?"
    ], DURATION_PER_SLIDE
))

# 8. Hypotheses
clips.append(make_bullet_slide(
    "Hypotheses",
    [
        "H1: All four RAG pipelines achieve significantly better metrics than baseline",
        "H2: Hybrid retrieval wins due to combining semantic + lexical search",
        "H3: Multi-query expansion trades precision for recall"
    ], DURATION_PER_SLIDE
))

# 9. Section 3
clips.append(make_section_slide(3, "Methodology", 4))

# 10. Dataset
clips.append(make_bullet_slide(
    "Dataset: MedQuAD",
    [
        "47,457 Q&A pairs from 12 NIH sources (Ben Abacha & Demner-Fushman, 2019)",
        "Covers symptoms, treatments, diagnosis, medication, prognosis",
        "Consumer health information — no PII",
        "Preprocessing: deduplication, HTML stripping, normalization",
        "Stratified sampling across question types"
    ], DURATION_PER_SLIDE
))

# 11. Controlled Design
clips.append(make_two_column_slide(
    "Controlled Experimental Design",
    "Fixed Across All Pipelines",
    ["LLM: GPT-4o-mini (temp=0.0)", "Embedding: text-embedding-3-small", "Chunking: 400 tokens, 50 overlap", "Top-k: 5 documents", "Safety-oriented prompt"],
    "Five Pipeline Configurations",
    ["1. Vanilla LLM (no retrieval)", "2. Standard RAG (dense semantic)", "3. Multi-Query Expansion", "4. Hybrid Retrieval (RRF fusion)", "5. Query Reformulation + Reranking"],
    DURATION_PER_SLIDE
))

# 12. Evaluation
clips.append(make_two_column_slide(
    "Evaluation Framework",
    "RAGAS (Grounding Quality)",
    ["Faithfulness — claims supported", "Answer Relevance — on-topic", "Context Precision — signal/noise", "Context Recall — completeness"],
    "DeepEval (Safety & Reliability)",
    ["Hallucination Rate", "Groundedness", "Correctness", "Safety Compliance"],
    DURATION_PER_SLIDE
))

# 13. Section 4
clips.append(make_section_slide(4, "Results", 4))

# 14. Faithfulness
clips.append(make_figure_slide(
    "RAGAS Faithfulness: Hybrid Retrieval Wins",
    "reports/figures/figure_5_1_faithfulness.png",
    "Hybrid: 0.89  |  Standard RAG: 0.78  |  Vanilla: 0.42",
    FIGURE_DURATION
))

# 15. Hallucination
clips.append(make_figure_slide(
    "Hallucination Rate: 81% Reduction Achieved",
    "reports/figures/figure_5_3_hallucination.png",
    "Hybrid: 0.11  |  Standard RAG: 0.22  |  Vanilla: 0.58",
    FIGURE_DURATION
))

# 16. Context Quality
clips.append(make_figure_slide(
    "Context Precision & Recall",
    "reports/figures/figure_5_2_precision_recall.png",
    "Hybrid achieves best balance: 0.86 precision, 0.83 recall",
    FIGURE_DURATION
))

# 17. Safety
clips.append(make_figure_slide(
    "Safety Compliance: All RAG Exceeds Clinical Threshold",
    "reports/figures/figure_5_6_safety.png",
    "Clinical threshold: 0.80  |  Hybrid: 0.92  |  Vanilla: 0.45",
    FIGURE_DURATION
))

# 18. Per-Type
clips.append(make_figure_slide(
    "Faithfulness by Question Type",
    "reports/figures/figure_5_5_per_type.png",
    "Symptom questions easiest; treatment questions most challenging",
    FIGURE_DURATION
))

# 19. Failures
clips.append(make_figure_slide(
    "Failure Pattern Distribution",
    "reports/figures/figure_5_7_failures.png",
    "Hybrid retrieval minimizes all failure categories",
    FIGURE_DURATION
))

# 20. Latency
clips.append(make_figure_slide(
    "Latency vs Accuracy Trade-off",
    "reports/figures/figure_5_8_latency.png",
    "Hybrid: 1,680ms — best accuracy-to-latency ratio",
    FIGURE_DURATION
))

# 21. Correlation
clips.append(make_figure_slide(
    "Faithfulness vs Hallucination: r = -0.97",
    "reports/figures/figure_5_9_correlation.png",
    "Faithfulness is a reliable proxy for hallucination risk",
    FIGURE_DURATION
))

# 22. Section 5
clips.append(make_section_slide(5, "Key Findings", 4))

# 23. Findings
clips.append(make_bullet_slide(
    "Key Findings",
    [
        "Retrieval grounding DRAMATICALLY reduces hallucination — 62-81% improvement",
        "Hybrid retrieval is standout: highest faithfulness (0.89), lowest hallucination (0.11)",
        "Multi-query expansion improves recall but introduces noise",
        "Query reformulation improves precision but adds latency (3,200ms)",
        "All RAG pipelines exceed clinical safety threshold of 0.80",
        "Symptom questions easiest; treatment questions remain challenging"
    ], DURATION_PER_SLIDE + 2
))

# 24. Clinical Implications
clips.append(make_bullet_slide(
    "Implications for Clinical Practice",
    [
        "Vanilla LLM is UNSAFE for clinical deployment (hallucination 0.58)",
        "Standard RAG provides good balance: 62% reduction, 1,240ms latency",
        "Hybrid retrieval recommended for high-stakes applications",
        "Retrieval grounding naturally promotes cautious language",
        "Goal: augment clinicians, not replace them"
    ], DURATION_PER_SLIDE
))

# 25. Contributions
clips.append(make_bullet_slide(
    "Contributions to Knowledge",
    [
        "First controlled benchmark isolating retrieval strategy effects",
        "Empirical evidence hybrid retrieval outperforms single-method",
        "Per-question-type analysis for nuanced guidance",
        "Structured failure pattern classification",
        "Safety compliance evaluation addressing critical gap"
    ], DURATION_PER_SLIDE
))

# 26. Section 6
clips.append(make_section_slide(6, "Conclusion", 4))

# 27. Limitations
clips.append(make_bullet_slide(
    "Limitations & Future Work",
    [
        "Limited to consumer health questions — not clinician-facing content",
        "Automatic metrics need clinician validation",
        "English-only — multilingual medical RAG unexplored",
        "Future: knowledge graphs, time-aware retrieval, clinician evaluation",
        "Future: multilingual datasets, granular safety taxonomy"
    ], DURATION_PER_SLIDE
))

# 28. Thank you
bg = ColorClip(size=(W, H), color=PRIMARY).with_duration(5)
thanks = TextClip("Thank You\n\nQuestions & Discussion", fontsize=60, color='white',
                  font='Arial-Bold', method='caption', size=(W-100, 300), align='center')
thanks = thanks.with_duration(5).with_position(('center', 250))
clips.append(CompositeVideoClip([bg, thanks]))

# ==================== RENDER ====================
print(f"Rendering video with {len(clips)} clips...")
final_clip = concatenate_videoclips(clips, method="compose")
output_path = r'C:\Users\HP\Downloads\LJMU_Thesis_Defence_Video.mp4'
final_clip.write_videofile(output_path, fps=24, codec='libx264', audio=False,
                           threads=4, preset='fast')
print(f"Video saved to: {output_path}")
print(f"Total duration: {final_clip.duration:.1f} seconds ({final_clip.duration/60:.1f} minutes)")
