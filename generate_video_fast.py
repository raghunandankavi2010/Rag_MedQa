"""
Generate Thesis Defence Video (Fast PIL + OpenCV version)
=========================================================
Uses Pillow for fast frame generation and OpenCV for reliable video writing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Video settings
W, H = 1280, 720
FPS = 12

# Color scheme
PRIMARY = (0, 51, 102)
SECONDARY = (0, 102, 153)
ACCENT = (255, 102, 0)
TEXT_COLOR = (51, 51, 51)
BG_COLOR = (250, 250, 250)
WHITE = (255, 255, 255)

# Try to load fonts, fallback to default
try:
    font_title = ImageFont.truetype("arialbd.ttf", 48)
    font_header = ImageFont.truetype("arialbd.ttf", 32)
    font_body = ImageFont.truetype("arial.ttf", 22)
    font_body_bold = ImageFont.truetype("arialbd.ttf", 22)
    font_caption = ImageFont.truetype("ariali.ttf", 16)
    font_section_num = ImageFont.truetype("arialbd.ttf", 100)
    font_section_title = ImageFont.truetype("arialbd.ttf", 44)
    font_author = ImageFont.truetype("arial.ttf", 28)
    font_small = ImageFont.truetype("arial.ttf", 14)
except Exception:
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_body_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_caption = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf", 16)
        font_section_num = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        font_section_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
        font_author = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_body_bold = ImageFont.load_default()
        font_caption = ImageFont.load_default()
        font_section_num = ImageFont.load_default()
        font_section_title = ImageFont.load_default()
        font_author = ImageFont.load_default()
        font_small = ImageFont.load_default()


def draw_text_centered(draw, text, y, font, color, max_width=None):
    if max_width:
        lines = []
        words = text.split(' ')
        current_line = words[0] if words else ''
        for word in words[1:]:
            bbox = draw.textbbox((0, 0), current_line + ' ' + word, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        line_height = font.size + 8
        start_y = y - (len(lines) * line_height) // 2
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            draw.text(((W - w) // 2, start_y + i * line_height), line, font=font, fill=color)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((W - w) // 2, y), text, font=font, fill=color)


def draw_text_left(draw, text, x, y, font, color, max_width=None):
    if max_width:
        lines = []
        words = text.split(' ')
        current_line = words[0] if words else ''
        for word in words[1:]:
            bbox = draw.textbbox((0, 0), current_line + ' ' + word, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        line_height = font.size + 6
        for i, line in enumerate(lines):
            draw.text((x, y + i * line_height), line, font=font, fill=color)
    else:
        draw.text((x, y), text, font=font, fill=color)


def make_title_slide(text, subtitle, author, duration_secs, writer):
    n_frames = int(duration_secs * FPS)
    for _ in range(n_frames):
        img = Image.new('RGB', (W, H), PRIMARY)
        draw = ImageDraw.Draw(img)
        draw_text_centered(draw, text, H // 2 - 40, font_title, WHITE, max_width=W - 100)
        draw_text_centered(draw, subtitle, H // 2 + 60, font_author, (200, 220, 240), max_width=W - 100)
        draw_text_centered(draw, author, H // 2 + 140, font_small, WHITE, max_width=W - 100)
        draw_text_centered(draw, "Liverpool John Moores University | MSc AI and ML", H - 80, font_small, (180, 200, 220), max_width=W - 100)
        writer.write(np.array(img)[:, :, ::-1])  # RGB to BGR for OpenCV


def make_section_slide(number, title, duration_secs, writer):
    n_frames = int(duration_secs * FPS)
    for _ in range(n_frames):
        img = Image.new('RGB', (W, H), SECONDARY)
        draw = ImageDraw.Draw(img)
        num_str = str(number)
        bbox = draw.textbbox((0, 0), num_str, font=font_section_num)
        nw = bbox[2] - bbox[0]
        nh = bbox[3] - bbox[1]
        draw.text(((W - nw) // 2, H // 2 - nh - 20), num_str, font=font_section_num, fill=WHITE)
        draw_text_centered(draw, title.upper(), H // 2 + 60, font_section_title, WHITE, max_width=W - 100)
        writer.write(np.array(img)[:, :, ::-1])


def make_bullet_slide(title, bullets, duration_secs, writer):
    n_frames = int(duration_secs * FPS)
    for _ in range(n_frames):
        img = Image.new('RGB', (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 80], fill=PRIMARY)
        draw_text_left(draw, title, 30, 20, font_header, WHITE)
        y_start = 120
        for i, bullet in enumerate(bullets):
            draw_text_left(draw, f"\u2022 {bullet}", 50, y_start + i * 50, font_body, TEXT_COLOR, max_width=W - 100)
        writer.write(np.array(img)[:, :, ::-1])


def make_figure_slide(title, image_path, caption, duration_secs, writer):
    n_frames = int(duration_secs * FPS)
    fig_img = None
    if os.path.exists(image_path):
        try:
            fig_img = Image.open(image_path)
            aspect = fig_img.width / fig_img.height
            max_h = 500
            new_h = max_h
            new_w = int(new_h * aspect)
            if new_w > W - 100:
                new_w = W - 100
                new_h = int(new_w / aspect)
            fig_img = fig_img.resize((new_w, new_h), Image.LANCZOS)
        except Exception:
            fig_img = None
    for _ in range(n_frames):
        img = Image.new('RGB', (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 80], fill=PRIMARY)
        draw_text_left(draw, title, 30, 20, font_header, WHITE)
        if fig_img:
            x = (W - fig_img.width) // 2
            y = 100
            img.paste(fig_img, (x, y))
        else:
            draw_text_centered(draw, f"[Figure: {caption}]", H // 2, font_body, TEXT_COLOR)
        draw_text_centered(draw, caption, H - 60, font_caption, SECONDARY, max_width=W - 100)
        writer.write(np.array(img)[:, :, ::-1])


def make_two_col_slide(title, left_title, left_items, right_title, right_items, duration_secs, writer):
    n_frames = int(duration_secs * FPS)
    for _ in range(n_frames):
        img = Image.new('RGB', (W, H), BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 80], fill=PRIMARY)
        draw_text_left(draw, title, 30, 20, font_header, WHITE)
        draw_text_left(draw, left_title, 50, 110, font_body_bold, SECONDARY)
        for i, item in enumerate(left_items):
            draw_text_left(draw, f"\u2022 {item}", 50, 160 + i * 45, font_body, TEXT_COLOR, max_width=550)
        draw.line([(W // 2, 130), (W // 2, H - 100)], fill=SECONDARY, width=3)
        draw_text_left(draw, right_title, W // 2 + 30, 110, font_body_bold, SECONDARY)
        for i, item in enumerate(right_items):
            draw_text_left(draw, f"\u2022 {item}", W // 2 + 30, 160 + i * 45, font_body, TEXT_COLOR, max_width=550)
        writer.write(np.array(img)[:, :, ::-1])


# ==================== BUILD VIDEO ====================
output_path = r'E:\AI_ML\MS\LJMU_report\LJMU_Thesis_Defence_Video.mp4'

print(f"Rendering video to: {output_path}")
print("This will take a few minutes...")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

if not writer.isOpened():
    raise RuntimeError("Failed to open video writer")

# 1. Title
make_title_slide(
    "A Comparative Study of RAG Pipelines\nfor Medical Question Answering",
    "Reducing Hallucination in Clinical Decision Support Systems",
    "Raghunandan Kavi | PN1196933",
    5, writer)

# 2. Outline
make_bullet_slide("Presentation Outline", [
    "Background & Motivation — Why medical AI needs grounded answers",
    "Problem Statement — Hallucination risks in clinical settings",
    "Research Questions & Hypotheses",
    "Methodology — Five pipeline configurations",
    "Results — RAGAS and DeepEval metrics",
    "Key Findings & Clinical Implications",
    "Conclusion & Future Work"
], 6, writer)

# 3. Section 1
make_section_slide(1, "Background & Motivation", 4, writer)

# 4. Problem
make_bullet_slide("The Hallucination Problem in Medical AI", [
    "LLMs produce confident but factually incorrect medical advice",
    "Documented case: chatbot said chemo drug was safe during pregnancy",
    "Hallucination rates: 15-30% on complex medical queries (Singhal et al., 2023)",
    "Oncology meta-analysis: 23% across 6,523 responses (Yoon et al., 2026)",
    "Current systems lack evidence grounding — no source verification"
], 7, writer)

# 5. What is RAG
make_bullet_slide("What is Retrieval-Augmented Generation?", [
    "RAG grounds LLM outputs in external knowledge sources",
    "Three components: Retriever → Encoder → Generator",
    "Dense retrieval: vector embeddings for semantic similarity",
    "Sparse retrieval: keyword matching for exact terminology",
    "Medical docs need BOTH: semantic coverage + terminological precision"
], 7, writer)

# 6. Section 2
make_section_slide(2, "Research Questions", 4, writer)

# 7. RQs
make_bullet_slide("Research Questions", [
    "RQ1: How do RAG pipelines differ from vanilla LLM in reducing hallucinations?",
    "RQ2: How does retrieval depth affect faithfulness and hallucination risk?",
    "RQ3: How do query expansion, hybrid retrieval, and reformulation compare?",
    "RQ4: Where do failures occur and how do patterns vary by strategy?"
], 7, writer)

# 8. Hypotheses
make_bullet_slide("Hypotheses", [
    "H1: All four RAG pipelines achieve significantly better metrics than baseline",
    "H2: Hybrid retrieval wins due to combining semantic + lexical search",
    "H3: Multi-query expansion trades precision for recall"
], 6, writer)

# 9. Section 3
make_section_slide(3, "Methodology", 4, writer)

# 10. Dataset
make_bullet_slide("Dataset: MedQuAD", [
    "47,457 Q&A pairs from 12 NIH sources (Ben Abacha & Demner-Fushman, 2019)",
    "Covers symptoms, treatments, diagnosis, medication, prognosis",
    "Consumer health information — no PII",
    "Preprocessing: deduplication, HTML stripping, normalization",
    "Stratified sampling across question types"
], 7, writer)

# 11. Controlled Design
make_two_col_slide(
    "Controlled Experimental Design",
    "Fixed Across All Pipelines",
    ["LLM: GPT-4o-mini (temp=0.0)", "Embedding: text-embedding-3-small", "Chunking: 400 tokens, 50 overlap", "Top-k: 5 documents", "Safety-oriented prompt"],
    "Five Pipeline Configurations",
    ["1. Vanilla LLM (no retrieval)", "2. Standard RAG (dense semantic)", "3. Multi-Query Expansion", "4. Hybrid Retrieval (RRF)", "5. Query Reformulation + Reranking"],
    8, writer)

# 12. Evaluation
make_two_col_slide(
    "Evaluation Framework",
    "RAGAS (Grounding Quality)",
    ["Faithfulness — claims supported", "Answer Relevance — on-topic", "Context Precision — signal/noise", "Context Recall — completeness"],
    "DeepEval (Safety & Reliability)",
    ["Hallucination Rate", "Groundedness", "Correctness", "Safety Compliance"],
    7, writer)

# 13. Section 4
make_section_slide(4, "Results & Analysis", 4, writer)

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
    make_figure_slide(title, path, caption, 8, writer)

# 22. Section 5
make_section_slide(5, "Key Findings", 4, writer)

# 23. Findings
make_bullet_slide("Key Findings", [
    "Retrieval grounding DRAMATICALLY reduces hallucination — 62-81% improvement",
    "Hybrid retrieval is standout: highest faithfulness (0.89), lowest hallucination (0.11)",
    "Multi-query expansion improves recall but introduces noise",
    "Query reformulation improves precision but adds latency (3,200ms)",
    "All RAG pipelines exceed clinical safety threshold of 0.80",
    "Symptom questions easiest; treatment questions remain challenging"
], 8, writer)

# 24. Implications
make_bullet_slide("Implications for Clinical Practice", [
    "Vanilla LLM is UNSAFE for clinical deployment (hallucination 0.58)",
    "Standard RAG provides good balance: 62% reduction, 1,240ms latency",
    "Hybrid retrieval recommended for high-stakes applications",
    "Retrieval grounding naturally promotes cautious language",
    "Goal: augment clinicians, not replace them"
], 7, writer)

# 25. Contributions
make_bullet_slide("Contributions to Knowledge", [
    "First controlled benchmark isolating retrieval strategy effects",
    "Empirical evidence hybrid retrieval outperforms single-method",
    "Per-question-type analysis for nuanced guidance",
    "Structured failure pattern classification",
    "Safety compliance evaluation addressing critical gap"
], 7, writer)

# 26. Section 6
make_section_slide(6, "Conclusion", 4, writer)

# 27. Limitations
make_bullet_slide("Limitations & Future Work", [
    "Limited to consumer health questions — not clinician-facing content",
    "Automatic metrics need clinician validation",
    "English-only — multilingual medical RAG unexplored",
    "Future: knowledge graphs, time-aware retrieval, clinician evaluation",
    "Future: multilingual datasets, granular safety taxonomy"
], 7, writer)

# 28. Thank you + GitHub link
n_frames = int(5 * FPS)
for _ in range(n_frames):
    img = Image.new('RGB', (W, H), PRIMARY)
    draw = ImageDraw.Draw(img)
    draw_text_centered(draw, "Thank You", H // 2 - 40, font_section_title, WHITE)
    draw_text_centered(draw, "Questions & Discussion", H // 2 + 40, font_author, (200, 220, 240))
    draw_text_centered(draw, "Source Code: github.com/raghunandankavi2010/Rag_MedQa", H - 80, font_small, (180, 200, 220))
    writer.write(np.array(img)[:, :, ::-1])

writer.release()

# Verify
file_size = os.path.getsize(output_path)
print(f"Video saved to: {output_path}")
print(f"File size: {file_size / (1024*1024):.1f} MB")

# Estimate duration from frame count and FPS
total_duration = (5+6+4+7+7+4+7+6+4+7+8+7+4+8*8+4+8+7+7+4+7+5)
print(f"Estimated duration: {total_duration} seconds ({total_duration/60:.1f} minutes)")
