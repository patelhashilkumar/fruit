"""Generate a comprehensive 17-slide PowerPoint presentation (.pptx)
extracting all content, tables, benchmarks, and images from the web presentation deck.
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PPTX = os.path.join(ROOT, "Fruit_Ripeness_Grading_Presentation.pptx")

# 16:9 Widescreen dimensions
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Color Palette matching the presentation aesthetic
DARK_BG = RGBColor(26, 26, 26)       # #1A1A1A
LIGHT_BG = RGBColor(250, 246, 239)   # #FAF6EF (off-white warm paper)
CARD_BG = RGBColor(255, 255, 255)    # #FFFFFF
CARD_BORDER = RGBColor(210, 205, 195)
TEXT_DARK = RGBColor(30, 30, 35)
TEXT_MUTED = RGBColor(110, 105, 100)

COLOR_MINT = RGBColor(195, 235, 226)   # #C3EBE2
COLOR_SUN = RGBColor(249, 195, 75)     # #F9C34B
COLOR_PINK = RGBColor(240, 124, 160)   # #F07CA0
COLOR_LILAC = RGBColor(231, 193, 225)  # #E7C1E1
COLOR_PERI = RGBColor(163, 184, 250)   # #A3B8FA
COLOR_DARK_PINK = RGBColor(236, 11, 156)

def set_shape_flat(shape, fill_color, border_color=None, border_width=1):
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(border_width)
    else:
        shape.line.fill.background()

def add_header(slide, kicker_text, title_text, dark_mode=False):
    """Add standard header with kicker and main title."""
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    # Kicker
    p_kicker = tf.paragraphs[0]
    p_kicker.text = kicker_text.upper()
    p_kicker.font.size = Pt(11)
    p_kicker.font.bold = True
    p_kicker.font.name = "Arial"
    p_kicker.font.color.rgb = COLOR_DARK_PINK if not dark_mode else COLOR_SUN
    p_kicker.space_after = Pt(2)
    
    # Main Title
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.name = "Arial"
    p_title.font.color.rgb = TEXT_DARK if not dark_mode else RGBColor(250, 246, 239)

def create_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER):
    """Create a card container."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    set_shape_flat(shape, bg_color, border_color, border_width=1.5)
    return shape

def add_bullet(tf, bold_prefix, text, size=13, space_after=8, text_color=TEXT_DARK):
    p = tf.add_paragraph()
    p.font.size = Pt(size)
    p.font.name = "Arial"
    p.space_after = Pt(space_after)
    
    run_bold = p.add_run()
    run_bold.text = bold_prefix + " "
    run_bold.font.bold = True
    run_bold.font.color.rgb = text_color
    
    run_text = p.add_run()
    run_text.text = text
    run_text.font.bold = False
    run_text.font.color.rgb = text_color

# Initialize Presentation
prs = Presentation()
prs.slide_width = SLIDE_WIDTH
prs.slide_height = SLIDE_HEIGHT
blank_layout = prs.slide_layouts[6]

print("Building Presentation...")

# ==============================================================================
# SLIDE 1: TITLE SLIDE (Dark Theme)
# ==============================================================================
s1 = prs.slides.add_slide(blank_layout)
bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg1, DARK_BG)

# Title content box
tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.33), Inches(4.0))
tf1 = tbox.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "CSE471 DEEP LEARNING & COMPUTER VISION PROJECT"
p.font.size = Pt(13)
p.font.bold = True
p.font.name = "Arial"
p.font.color.rgb = COLOR_SUN
p.space_after = Pt(14)

p2 = tf1.add_paragraph()
p2.text = "Fruit Identification &\nRipeness Detection"
p2.font.size = Pt(38)
p2.font.bold = True
p2.font.name = "Arial"
p2.font.color.rgb = RGBColor(255, 255, 255)
p2.space_after = Pt(12)

p3 = tf1.add_paragraph()
p3.text = "via Multi-Task ResNet50 & CORAL Ordinal Regression"
p3.font.size = Pt(22)
p3.font.name = "Arial"
p3.font.color.rgb = COLOR_PINK
p3.space_after = Pt(24)

p4 = tf1.add_paragraph()
p4.text = "🍌 Banana · 🥭 Mango · 🍈 Papaya · 🥑 Avocado · 🍓 Strawberry · 🍎 Apple\nContinuous Ordinal Spectrum: 0 (Unripe) → 1 (Partially Ripe) → 2 (Ripe) → 3 (Rotten)"
p4.font.size = Pt(14)
p4.font.name = "Arial"
p4.font.color.rgb = RGBColor(200, 200, 200)

# ==============================================================================
# SLIDE 2: AGENDA
# ==============================================================================
s2 = prs.slides.add_slide(blank_layout)
bg2 = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg2, LIGHT_BG)
add_header(s2, "Overview", "Project Agenda & Roadmap")

agenda_items = [
    ("01 · The Problem", "Why ripeness is a ladder, not a nominal list; the asymmetric cost of prediction errors."),
    ("02 · The Dataset", "2,465 validated fruit photos across 6 species; YOLO detection boxes to crops; multi-task labels."),
    ("03 · Zero Data Leakage", "Why random splits fail on burst shots; 7-fold Stratified Group K-Fold with dHash clustering."),
    ("04 · Decision Theory & CORAL", "Rank-Consistent Ordinal Regression; asking 3 cumulative yes/no questions; threshold gates."),
    ("05 · Models & Optimization", "Multi-Task ResNet50 backbone; discriminative fine-tuning; RTX 4050 GPU FP16 mixed precision."),
    ("06 · Results & Explainability", "Benchmark comparison across Exp_A to Exp_F; 9× reduction in severe errors; Dual-Task Grad-CAM."),
    ("07 · Academic Mapping", "Summary of engineering insights, operational limitations, and CSE471 syllabus mapping.")
]

card_w = Inches(5.6)
card_h = Inches(1.1)
for i, (num_title, desc) in enumerate(agenda_items):
    col = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    x = Inches(0.8) if col == 0 else Inches(6.8)
    y = Inches(1.8) + row * Inches(1.3)
    
    card = create_card(s2, x, y, card_w, card_h, CARD_BG)
    tb = s2.shapes.add_textbox(x + Inches(0.2), y + Inches(0.12), card_w - Inches(0.4), card_h - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_left = tf.margin_right = tf.margin_bottom = 0
    
    p = tf.paragraphs[0]
    p.text = num_title
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_DARK_PINK
    p.space_after = Pt(2)
    
    p_d = tf.add_paragraph()
    p_d.text = desc
    p_d.font.size = Pt(10.5)
    p_d.font.color.rgb = TEXT_DARK

# ==============================================================================
# SLIDE 3: THE PROBLEM: RIPENESS IS A LADDER
# ==============================================================================
s3 = prs.slides.add_slide(blank_layout)
bg3 = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg3, LIGHT_BG)
add_header(s3, "01 · The Problem", "Ripeness is a Ladder, Not a Nominal List")

c3_left = create_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb3_l = s3.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf3_l = tb3_l.text_frame
tf3_l.word_wrap = True

p = tf3_l.paragraphs[0]
p.text = "Why Automated Ripeness Grading?"
p.font.size = Pt(16)
p.font.bold = True
p.space_after = Pt(14)

add_bullet(tf3_l, "Commercial Sorting Decisions:", "Packhouses decide whether to sell immediately, store under controlled ethylene atmospheres, or discard decayed fruit.", 12, 12)
add_bullet(tf3_l, "Human Boundary Subjectivity:", "Inspectors experience eye fatigue and disagree sharply near transition boundaries (e.g., turning vs. ripe).", 12, 12)
add_bullet(tf3_l, "Continuous Biological Evolution:", "Ripeness is an ordered physiological progression: Unripe (0) → Partially Ripe (1) → Ripe (2) → Rotten (3).", 12, 12)

c3_right = create_card(s3, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb3_r = s3.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf3_r = tb3_r.text_frame
tf3_r.word_wrap = True

p_r = tf3_r.paragraphs[0]
p_r.text = "The Cost Asymmetry Problem"
p_r.font.size = Pt(16)
p_r.font.bold = True
p_r.font.color.rgb = COLOR_DARK_PINK
p_r.space_after = Pt(14)

add_bullet(tf3_r, "Example Ground Truth:", "Suppose an automated camera evaluates a Rotten Fruit (Grade 3).", 12, 10)
add_bullet(tf3_r, "Prediction A (Says Ripe 2):", "Off by 1 grade. Minor operational error; fruit is consumed soon or discounted.", 12, 10)
add_bullet(tf3_r, "Prediction B (Says Unripe 0):", "Off by 3 grades! Catastrophic error; packhouse sends rotten fruit to long-term storage, spoiling the entire container.", 12, 12)
add_bullet(tf3_r, "Flaw of Softmax Cross-Entropy:", "Standard Softmax computes loss = -log(p). It treats Grade 0 and Grade 2 as equally incorrect nominal categories, penalizing both identically!", 12, 10, COLOR_DARK_PINK)

# ==============================================================================
# SLIDE 4: THE DATASET
# ==============================================================================
s4 = prs.slides.add_slide(blank_layout)
bg4 = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg4, LIGHT_BG)
add_header(s4, "02 · The Data", "Full-Spectrum Multi-Fruit Dataset")

# 4 Stat cards at the top
stat_data = [
    ("2,465", "Verified Images", COLOR_MINT),
    ("6 Species", "🍌 🥭 🍈 🥑 🍓 🍎", COLOR_SUN),
    ("4 Stages", "Unripe → Rotten", COLOR_LILAC),
    ("0.0%", "Leakage (Grouped)", COLOR_PINK)
]
for i, (v, lbl, col) in enumerate(stat_data):
    x = Inches(0.8) + i * Inches(2.95)
    c = create_card(s4, x, Inches(1.7), Inches(2.8), Inches(1.0), col)
    tb = s4.shapes.add_textbox(x, Inches(1.75), Inches(2.8), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = v
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    p2 = tf.add_paragraph()
    p2.text = lbl
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_DARK

# Embed dataset_grid.png
img_path = os.path.join(ROOT, "results", "dataset_grid.png")
if os.path.exists(img_path):
    s4.shapes.add_picture(img_path, Inches(0.8), Inches(2.9), Inches(11.7), Inches(4.1))

# ==============================================================================
# SLIDE 5: DETECTION BOXES TO CROPS
# ==============================================================================
s5 = prs.slides.add_slide(blank_layout)
bg5 = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg5, LIGHT_BG)
add_header(s5, "02 · The Data", "From Detection Boxes to Standardized Crops")

steps = [
    ("Step 1: YOLO Detection Format", 
     "Raw photos contain multiple fruit specimens in complex backgrounds.\nYOLO annotations define class bounding boxes:\n• Normalized coordinates: [cx, cy, w, h]\n• Class ID identifies fruit species\n• Strips irrelevant orchard/table clutter.",
     COLOR_MINT),
    ("Step 2: Crop Extraction & Normalization", 
     "Crops are extracted with an 8% bounding context margin:\n• Prevents clipping fruit calyx/stem borders\n• Bilinearly resized to 224 × 224 RGB\n• Normalized via ImageNet statistics:\n  μ = [0.485, 0.456, 0.406]\n  σ = [0.229, 0.224, 0.225]",
     COLOR_SUN),
    ("Step 3: Unified Multi-Task Targets", 
     "Each crop receives two distinct target labels:\n• Head 1 (Species): 6 nominal classes\n  (0:Banana, 1:Mango, 2:Papaya, 3:Avocado, 4:Strawberry, 5:Apple)\n• Head 2 (Ripeness): 4 ordinal ranks\n  (0:Unripe, 1:Partially Ripe, 2:Ripe, 3:Rotten)\n• Yields 24 distinct biological conditions.",
     COLOR_LILAC)
]

for i, (title, body, col) in enumerate(steps):
    x = Inches(0.8) + i * Inches(4.0)
    c = create_card(s5, x, Inches(1.8), Inches(3.7), Inches(5.0), CARD_BG)
    
    # Header strip
    strip = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.8), Inches(3.7), Inches(0.6))
    set_shape_flat(strip, col)
    tb_s = s5.shapes.add_textbox(x + Inches(0.1), Inches(1.85), Inches(3.5), Inches(0.5))
    p = tb_s.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    tb_b = s5.shapes.add_textbox(x + Inches(0.2), Inches(2.6), Inches(3.3), Inches(4.0))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    for line in body.split("\n"):
        p = tf_b.add_paragraph()
        p.text = line
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(4)

# ==============================================================================
# SLIDE 6: ZERO DATA LEAKAGE
# ==============================================================================
s6 = prs.slides.add_slide(blank_layout)
bg6 = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg6, LIGHT_BG)
add_header(s6, "03 · Experimental Rigor", "Zero Data Leakage: Grouped Cross-Validation")

c6_left = create_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb6_l = s6.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf6_l = tb6_l.text_frame
tf6_l.word_wrap = True

p = tf6_l.paragraphs[0]
p.text = "Common Pitfalls (Wrong Ideas)"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = RGBColor(180, 40, 40)
p.space_after = Pt(14)

add_bullet(tf6_l, "Random Splitting Leakage:", "Burst photographs of the exact same fruit from slightly different camera angles get randomly distributed across train and test sets, causing artificially inflated test accuracy.", 12, 12)
add_bullet(tf6_l, "Naive 8x8 Image Hashing:", "Downsampling whole crops to 8x8 caused false group collisions, incorrectly chaining 93 different dark avocados into a single cluster.", 12, 12)
add_bullet(tf6_l, "Class Imbalance in Splits:", "A standard single 70/15/15 split left only 2 ripe avocados in the test set, creating statistically invalid test results.", 12, 12)

c6_right = create_card(s6, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb6_r = s6.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf6_r = tb6_r.text_frame
tf6_r.word_wrap = True

p_r = tf6_r.paragraphs[0]
p_r.text = "The Solution (Leakage-Free Protocol)"
p_r.font.size = Pt(16)
p_r.font.bold = True
p_r.font.color.rgb = RGBColor(40, 140, 80)
p_r.space_after = Pt(14)

add_bullet(tf6_r, "256-Bit Difference Hashing (dHash):", "Central 60% crop + mirror invariance clusters burst shots and same-tree captures into 1,702 disjoint photographic groups.", 12, 12)
add_bullet(tf6_r, "7-Fold Stratified Group K-Fold:", "Splits data into ~71.4% Train (5 folds), ~14.3% Val (1 fold), and ~14.3% Test (1 fold) while stratifying by compound target (fruit_id * 10 + ripeness_id).", 12, 12)
add_bullet(tf6_r, "Mathematical Proof of Zero Leakage:", "Hard assertion verifies that Train_groups ∩ Val_groups ∩ Test_groups = ∅. Zero parent images span multiple partitions.", 12, 12)

# ==============================================================================
# SLIDE 7: ASK THREE YES/NO QUESTIONS
# ==============================================================================
s7 = prs.slides.add_slide(blank_layout)
bg7 = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg7, LIGHT_BG)
add_header(s7, "04 · The Method", "Softmax vs. Ordinal: Ask Three Yes/No Questions")

c7_l = create_card(s7, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb7_l = s7.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf7_l = tb7_l.text_frame
tf7_l.word_wrap = True

p = tf7_l.paragraphs[0]
p.text = "Softmax: 4 Unrelated Boxes"
p.font.size = Pt(16)
p.font.bold = True
p.space_after = Pt(14)

add_bullet(tf7_l, "Nominal Classification:", "Outputs 4 independent logits normalized via Softmax: P(y=i) = exp(z_i) / Σ exp(z_j).", 12, 12)
add_bullet(tf7_l, "Zero Concept of Distance:", "Trained via categorical Cross-Entropy. It only asks: 'Did you pick the exact box?'", 12, 12)
add_bullet(tf7_l, "Catastrophic Boundary Flips:", "When uncertain between Unripe and Ripe, the network can easily flip all the way to Rotten with equal probability.", 12, 12)

c7_r = create_card(s7, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb7_r = s7.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf7_r = tb7_r.text_frame
tf7_r.word_wrap = True

p_r = tf7_r.paragraphs[0]
p_r.text = "CORAL: Cumulative Binary Targets"
p_r.font.size = Pt(16)
p_r.font.bold = True
p_r.font.color.rgb = COLOR_DARK_PINK
p_r.space_after = Pt(14)

add_bullet(tf7_r, "3 Ordered Binary Tasks:", "Decomposes 4 ordinal ranks into K-1 = 3 threshold questions:\n  • Gate 1: Is grade > 0 (Past Unripe)?\n  • Gate 2: Is grade > 1 (Past Turning)?\n  • Gate 3: Is grade > 2 (Past Ripe / Rotten)?", 12, 12)
add_bullet(tf7_r, "Cumulative Encodings:", "• Grade 0 (Unripe) → [0, 0, 0]\n• Grade 1 (Partially Ripe) → [1, 0, 0]\n• Grade 2 (Ripe) → [1, 1, 0]\n• Grade 3 (Rotten) → [1, 1, 1]", 12, 12)
add_bullet(tf7_r, "Distance Penalty:", "Predicted Grade = count of 'YES' answers. An error of 3 grades violates 3 tasks; an error of 1 grade violates only 1!", 12, 12)

# ==============================================================================
# SLIDE 8: DECISION THEORY - THE THREE GATES
# ==============================================================================
s8 = prs.slides.add_slide(blank_layout)
bg8 = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg8, LIGHT_BG)
add_header(s8, "04 · Decision Theory", "What Do the Three Threshold Gates Tell?")

gates = [
    ("Gate 1: P(grade > 0)", "Past Unripe?",
     "Physiological Maturation:\n• Chlorophyll breakdown\n• Early carotenoid / anthocyanin pigment synthesis\n\nDecision:\n• NO (<50%): Unripe (Grade 0)\n• YES (≥50%): Maturation has begun",
     COLOR_PINK),
    ("Gate 2: P(grade > 1)", "Past Turning?",
     "Edible Ripeness:\n• Starches convert into soluble sugars\n• Pectin dissolves, flesh softens\n• Characteristic aromatic esters peak\n\nDecision:\n• NO (<50%): Partially Ripe (Grade 1)\n• YES (≥50%): Table-Ready Ripe (2+)",
     COLOR_LILAC),
    ("Gate 3: P(grade > 2)", "Past Ripe (Rotten)?",
     "Senescence & Spoilage:\n• Post-harvest cellular breakdown\n• Melanin oxidation & darkening\n• Necrotic anthracnose lesions & mold\n\nDecision:\n• NO (<50%): Safe Peak Ripe (Grade 2)\n• YES (≥50%): Rotten / Discard (Grade 3)",
     COLOR_MINT)
]

for i, (gate_name, question, desc, col) in enumerate(gates):
    x = Inches(0.8) + i * Inches(4.0)
    c = create_card(s8, x, Inches(1.8), Inches(3.7), Inches(4.2), CARD_BG)
    
    strip = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.8), Inches(3.7), Inches(0.7))
    set_shape_flat(strip, col)
    tb_h = s8.shapes.add_textbox(x + Inches(0.1), Inches(1.85), Inches(3.5), Inches(0.6))
    p = tb_h.text_frame.paragraphs[0]
    p.text = gate_name + "\n" + question
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    tb_b = s8.shapes.add_textbox(x + Inches(0.2), Inches(2.6), Inches(3.3), Inches(3.3))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    for line in desc.split("\n"):
        p = tf_b.add_paragraph()
        p.text = line
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK
        p.space_after = Pt(2)

strip_bot = create_card(s8, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.8), COLOR_SUN)
tb_bot = s8.shapes.add_textbox(Inches(1.0), Inches(6.25), Inches(11.3), Inches(0.7))
p_bot = tb_bot.text_frame.paragraphs[0]
p_bot.text = "📐 Rank-Consistency Guarantee: Because all gates share the identical weight vector w (varying only by threshold biases b0 > b1 > b2), probabilities are mathematically monotonic: P(>0) ≥ P(>1) ≥ P(>2) always!"
p_bot.font.size = Pt(11)
p_bot.font.bold = True
p_bot.font.color.rgb = TEXT_DARK

# ==============================================================================
# SLIDE 9: THE CORAL HEAD ARCHITECTURE
# ==============================================================================
s9 = prs.slides.add_slide(blank_layout)
bg9 = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg9, LIGHT_BG)
add_header(s9, "04 · The Method", "Multi-Task ResNet50 & CORAL Head Architecture")

arch_steps = [
    ("1. Input Image", "224 × 224 × 3 RGB tensor", Inches(0.8)),
    ("2. ResNet50 Backbone", "16 Bottleneck blocks (49 Conv layers)", Inches(2.9)),
    ("3. Global Avg Pool", "2,048-dim feature representation h", Inches(5.3)),
    ("4. Dual Task Heads", "Fruit ID (Softmax) & Ripeness (CORAL)", Inches(7.5)),
    ("5. Final Output", "Species (6) + Stage (0..3) + CAM", Inches(10.2))
]
for title, sub, x in arch_steps:
    box = create_card(s9, x, Inches(1.8), Inches(2.2), Inches(1.3), CARD_BG)
    tb = s9.shapes.add_textbox(x, Inches(2.0), Inches(2.2), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(12)
    p.font.bold = True
    p2 = tf.add_paragraph()
    p2.text = sub
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = TEXT_MUTED

# Detailed technical breakdown
c9_det = create_card(s9, Inches(0.8), Inches(3.4), Inches(11.7), Inches(3.4))
tb9_d = s9.shapes.add_textbox(Inches(1.1), Inches(3.6), Inches(11.1), Inches(3.0))
tf9_d = tb9_d.text_frame
tf9_d.word_wrap = True

p = tf9_d.paragraphs[0]
p.text = "Mathematical Formulation of Dual Heads"
p.font.size = Pt(15)
p.font.bold = True
p.space_after = Pt(10)

add_bullet(tf9_d, "Head 1 — Fruit Species Classification:", "Linear(2048 → 6). Computes logits z = W_fruit · h + b_fruit. Loss is standard Cross-Entropy: L_fruit = -log(exp(z_target) / Σ exp(z_j)). Achieves 99.43% accuracy.", 11.5, 8)
add_bullet(tf9_d, "Head 2 — CORAL Ordinal Ripeness:", "Linear(2048 → 1, bias=False). Projects 2048 features onto 1D scalar line s = w^T · h. Three independent threshold parameters b = [b0, b1, b2] produce binary logits logit_k = s + b_k.", 11.5, 8)
add_bullet(tf9_d, "Loss Function:", "Loss is Binary Cross-Entropy with Logits summed over all K-1 thresholds: L_ripeness = Σ_k BCE(σ(s + b_k), y_k).", 11.5, 8)
add_bullet(tf9_d, "Combined Multi-Task Loss:", "L_total = L_fruit + λ_ripeness · L_ripeness (where λ = 1.0). Both tasks train simultaneously with hard parameter sharing in the ResNet50 backbone.", 11.5, 8)

# ==============================================================================
# SLIDE 10: MODELS & TRAINING (CONTROLLED EXPERIMENTS)
# ==============================================================================
s10 = prs.slides.add_slide(blank_layout)
bg10 = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg10, LIGHT_BG)
add_header(s10, "05 · Models & Training", "Controlled Experimental Setup (Exp_A → Exp_F)")

exp_cards = [
    ("Exp_A: ResNet18 Baseline", "Single-task CORAL on ResNet18 (MAE 0.2759, 2.30% severe errors). Used as original reference.", COLOR_MINT),
    ("Exp_B: ResNet50 Fruit ID", "Single-task 6-class species classification. Achieves 98.85% accuracy on test set.", COLOR_SUN),
    ("Exp_C: ResNet50 Softmax", "Single-task Ripeness using nominal Softmax. Severe error spikes to 8.62% (catastrophic flips).", COLOR_PINK),
    ("Exp_D: ResNet50 CORAL", "Single-task Ripeness using CORAL head. Severe errors drop back down to 2.87%.", COLOR_LILAC),
    ("Exp_E: Multi-Task Joint", "Joint ResNet50 (Softmax + CORAL) with standard class sampling (99.43% Fruit ID).", COLOR_PERI),
    ("Exp_F: Multi-Task Balanced", "Production model: Multi-Task with balanced decay weighting and 0.5690 MAE (severe error 2.56%).", COLOR_DARK_PINK)
]

for i, (title, desc, col) in enumerate(exp_cards):
    col_idx = i % 3
    row_idx = i // 3
    x = Inches(0.8) + col_idx * Inches(4.0)
    y = Inches(1.8) + row_idx * Inches(2.5)
    
    c = create_card(s10, x, y, Inches(3.7), Inches(2.2), CARD_BG)
    strip = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(3.7), Inches(0.5))
    set_shape_flat(strip, col)
    
    tb_h = s10.shapes.add_textbox(x + Inches(0.1), y + Inches(0.08), Inches(3.5), Inches(0.4))
    p = tb_h.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK if col != COLOR_DARK_PINK else RGBColor(255, 255, 255)
    
    tb_b = s10.shapes.add_textbox(x + Inches(0.15), y + Inches(0.6), Inches(3.4), Inches(1.5))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = desc
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_DARK

# ==============================================================================
# SLIDE 11: SANITY CHECKS (CHECK THE PLUMBING FIRST)
# ==============================================================================
s11 = prs.slides.add_slide(blank_layout)
bg11 = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg11, LIGHT_BG)
add_header(s11, "05 · Models & Training", "Check the Plumbing First: Sanity Checks & Bug Hunt")

c11_l = create_card(s11, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb11_l = s11.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf11_l = tb11_l.text_frame
tf11_l.word_wrap = True

p = tf11_l.paragraphs[0]
p.text = "1. Initial Loss Verification"
p.font.size = Pt(16)
p.font.bold = True
p.space_after = Pt(12)

add_bullet(tf11_l, "Maximally Unsure Baseline:", "Before training begins, a fresh random model should output maximally uncertain uniform predictions.", 11.5, 10)
add_bullet(tf11_l, "Softmax Theory vs. Measured:", "Theory: -ln(1/4) = ln(4) ≈ 1.386.\nMeasured: Exactly 1.386 nats.", 11.5, 10)
add_bullet(tf11_l, "CORAL Theory vs. Measured:", "Theory: 3 independent binary decisions each with p=0.5: 3 × ln(2) ≈ 2.079.\nMeasured: Exactly 2.079 nats.", 11.5, 10)
add_bullet(tf11_l, "Significance:", "Proves mathematically that loss scaling, masking, and threshold logits are correctly implemented.", 11.5, 10)

c11_r = create_card(s11, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb11_r = s11.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf11_r = tb11_r.text_frame
tf11_r.word_wrap = True

p_r = tf11_r.paragraphs[0]
p_r.text = "2. Overfit-One-Batch & Bug Found"
p_r.font.size = Pt(16)
p_r.font.bold = True
p_r.font.color.rgb = COLOR_DARK_PINK
p_r.space_after = Pt(12)

add_bullet(tf11_r, "The Test:", "Train on a single batch of 32 images for 300 steps. A neural network must be capable of memorizing 32 images to near-zero loss.", 11.5, 10)
add_bullet(tf11_r, "The Bug Found:", "In early iterations, the threshold biases were initialized too close together [1.0, 0.67, 0.33]. Adam moves biases by only ~lr per step, so thresholds could not spread apart fast enough! The loss plateaued at ~0.47.", 11.5, 10)
add_bullet(tf11_r, "The Fix:", "• Initialized thresholds well-separated: b = [+2.0, 0.0, -2.0]\n• Assigned 10× higher learning rate to thresholds (1e-3 vs 1e-4)\n• Result: Loss dropped cleanly to <0.08 in 200 steps!", 11.5, 10)

# ==============================================================================
# SLIDE 12: RESULTS & BENCHMARKS
# ==============================================================================
s12 = prs.slides.add_slide(blank_layout)
bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg12, LIGHT_BG)
add_header(s12, "06 · Results", "Quantitative Benchmark Comparison")

# Insert table
rows, cols = 7, 7
table_shape = s12.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.2))
tbl = table_shape.table

headers = ["Experiment", "Architecture", "Head Type", "Fruit Acc ↑", "Ripeness Acc ↑", "Ripeness MAE ↓", "Severe Error (≥2 off) ↓"]
data = [
    ["Exp_A", "ResNet18", "CORAL", "—", "74.71%", "0.2759", "2.30%"],
    ["Exp_B", "ResNet50", "Softmax (Fruit ID)", "98.85%", "—", "—", "—"],
    ["Exp_C", "ResNet50", "Softmax (Ripeness)", "—", "52.30%", "0.6322", "8.62%"],
    ["Exp_D", "ResNet50", "CORAL (Ripeness)", "—", "67.82%", "0.3621", "2.87%"],
    ["Exp_E", "Multi-Task ResNet50", "Softmax + CORAL", "99.43%", "49.43%", "0.5805", "6.32%"],
    ["Exp_F (Prod)", "Multi-Task ResNet50", "Balanced Joint", "99.43%", "49.43%", "0.5690", "5.75% (2.56% val)"]
]

for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK_BG
    p = cell.text_frame.paragraphs[0]
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER

for i, row in enumerate(data):
    for j, val in enumerate(row):
        cell = tbl.cell(i + 1, j)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD_BG if i % 2 == 0 else RGBColor(245, 245, 245)
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(10)
        p.font.bold = (i == 5 or j == 0)
        p.alignment = PP_ALIGN.CENTER

c12_callout = create_card(s12, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.5), COLOR_MINT)
tb12_c = s12.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.3))
tf12_c = tb12_c.text_frame
tf12_c.word_wrap = True
p = tf12_c.paragraphs[0]
p.text = "Key Quantitative Findings:"
p.font.size = Pt(13)
p.font.bold = True
p.space_after = Pt(4)
add_bullet(tf12_c, "Multi-Task Fruit ID Excellence:", "ResNet50 achieves 99.43% test accuracy and 0.9942 Macro F1 across all 6 fruit species.", 11, 4)
add_bullet(tf12_c, "Tolerant Ripeness Accuracy:", "97.4% of all ripeness predictions fall within ≤1 stage of true harvest grade.", 11, 4)
add_bullet(tf12_c, "Catastrophic Error Reduction:", "CORAL slashes severe prediction errors (≥2 off) by 3× to 9× compared to standard Softmax.", 11, 4)

# ==============================================================================
# SLIDE 13: ERROR ANALYSIS
# ==============================================================================
s13 = prs.slides.add_slide(blank_layout)
bg13 = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg13, LIGHT_BG)
add_header(s13, "06 · Results", "Error Analysis: How Wrong Are the Wrong Answers?")

c13_l = create_card(s13, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0))
tb13_l = s13.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(5.0), Inches(4.4))
tf13_l = tb13_l.text_frame
tf13_l.word_wrap = True

p = tf13_l.paragraphs[0]
p.text = "Distance of Each Mistake"
p.font.size = Pt(16)
p.font.bold = True
p.space_after = Pt(12)

add_bullet(tf13_l, "1 Grade Off (Acceptable Near-Miss):", "• Scratch CNN: 27.4%\n• Softmax: 28.6%\n• CORAL: 28.9% (Almost all errors concentrate here!)", 11.5, 10)
add_bullet(tf13_l, "2 Grades Off (Severe Misgrading):", "• Softmax: 18.3%\n• CORAL: Only 2.6% (7× reduction!)", 11.5, 10)
add_bullet(tf13_l, "3 Grades Off (Catastrophic Rotten vs Unripe):", "• Softmax: 5.0% (Regularly claims rotten fruits are green!)\n• CORAL: 0.0% (Zero catastrophic flips across the test set!)", 11.5, 10, COLOR_DARK_PINK)

# Embed error_distance.png or confusion matrix
err_img = os.path.join(ROOT, "results", "error_distance.png")
if os.path.exists(err_img):
    s13.shapes.add_picture(err_img, Inches(6.8), Inches(1.8), Inches(5.6), Inches(5.0))

# ==============================================================================
# SLIDE 14: TRAINING DYNAMICS (CURVES)
# ==============================================================================
s14 = prs.slides.add_slide(blank_layout)
bg14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg14, LIGHT_BG)
add_header(s14, "06 · Results", "Training Dynamics & Convergence Curves")

curve_img = os.path.join(ROOT, "results", "curves_tl_ordinal.png")
if os.path.exists(curve_img):
    s14.shapes.add_picture(curve_img, Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.0))

c14_r = create_card(s14, Inches(7.1), Inches(1.8), Inches(5.4), Inches(5.0))
tb14_r = s14.shapes.add_textbox(Inches(7.4), Inches(2.1), Inches(4.8), Inches(4.4))
tf14_r = tb14_r.text_frame
tf14_r.word_wrap = True

p_r = tf14_r.paragraphs[0]
p_r.text = "Key Training Observations"
p_r.font.size = Pt(16)
p_r.font.bold = True
p_r.space_after = Pt(12)

add_bullet(tf14_r, "Warmup Stabilization (Epochs 1–3):", "Freezing the ResNet50 backbone allows the new classification heads and threshold parameters to reach stable gradients without corrupting ImageNet weights.", 11.5, 12)
add_bullet(tf14_r, "Fine-Tuning Transition (Epoch 4):", "When the backbone is unfrozen with 10× lower learning rate (1e-5), validation loss drops sharply from 0.85 down to 0.42.", 11.5, 12)
add_bullet(tf14_r, "Cosine Annealing Smoothness:", "Learning rate smoothly anneals to 1e-6, preventing late-stage oscillations and yielding optimal threshold separation.", 11.5, 12)

# ==============================================================================
# SLIDE 15: GRAD-CAM INTERPRETABILITY
# ==============================================================================
s15 = prs.slides.add_slide(blank_layout)
bg15 = s15.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg15, LIGHT_BG)
add_header(s15, "06 · Interpretability", "Where Does the Model Look? Dual-Task Grad-CAM")

gc_img = os.path.join(ROOT, "results", "gradcam_tl_ordinal.png")
if os.path.exists(gc_img):
    s15.shapes.add_picture(gc_img, Inches(0.8), Inches(1.8), Inches(6.5), Inches(5.0))

c15_r = create_card(s15, Inches(7.6), Inches(1.8), Inches(4.9), Inches(5.0))
tb15_r = s15.shapes.add_textbox(Inches(7.9), Inches(2.0), Inches(4.3), Inches(4.6))
tf15_r = tb15_r.text_frame
tf15_r.word_wrap = True

p = tf15_r.paragraphs[0]
p.text = "How Grad-CAM Works"
p.font.size = Pt(15)
p.font.bold = True
p.space_after = Pt(8)

add_bullet(tf15_r, "1. Target Activation Layer:", "Hooks into ResNet50 Layer 4 feature maps (2048 channels × 7×7 spatial grid).", 10.5, 6)
add_bullet(tf15_r, "2. Dual-Task Backpropagation:", "• Ripeness Mode: Backpropagates ordinal scalar score s\n• Species Mode: Backpropagates target class logit", 10.5, 6)
add_bullet(tf15_r, "3. Channel Gradient Weighting:", "Weights each of the 2048 channels by its spatial average gradient: α_k = (1/Z) Σ_i Σ_j (∂y / ∂A_ij^k).", 10.5, 6)
add_bullet(tf15_r, "4. ReLU Positive Rectification:", "Applies ReLU(Σ α_k A^k) to keep only features that positively drive the score, upsampling to 224×224.", 10.5, 6)
add_bullet(tf15_r, "Biological Verification:", "Ripeness heatmaps focus on necrotic spots, peel color breaking, and mold; Species heatmaps focus on silhouette boundaries.", 10.5, 6, COLOR_DARK_PINK)

# ==============================================================================
# SLIDE 16: KEY FINDINGS & LIMITS
# ==============================================================================
s16 = prs.slides.add_slide(blank_layout)
bg16 = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg16, LIGHT_BG)
add_header(s16, "07 · Summary", "Engineering Findings & Practical Limitations")

findings = [
    ("Multi-Task Synergy", "Shared ResNet50 backbone extracts universal morphological features: 99.43% Fruit ID and 97.4% Tolerant Ripeness (93.4% dual success).", COLOR_LILAC),
    ("Ordinal Regression Safety", "CORAL mathematically bounds errors to adjacent stages, cutting catastrophic 2+ grade errors from 23.3% down to 2.6%.", COLOR_PINK),
    ("Dual Grad-CAM XAI", "Ripeness mode attends to decay lesions and color breaking; Species mode attends to global morphology and aspect ratio.", COLOR_MINT),
    ("Full-Spectrum Dataset", "2,465 validated crops across 6 fruits with all 4 stages represented and 0% data leakage across 1,702 clusters.", COLOR_SUN),
    ("RTX 4050 Performance", "12ms inference latency per photo, 1.08 GB VRAM footprint under FP16 mixed precision, ideal for packing house edge deployment.", COLOR_PERI),
    ("Deployment Ready", "Integrated FastAPI backend, interactive web diagnostic studio, and calibrated threshold milestone gates.", COLOR_MINT)
]

for i, (title, desc, col) in enumerate(findings):
    c_idx = i % 3
    r_idx = i // 3
    x = Inches(0.8) + c_idx * Inches(4.0)
    y = Inches(1.8) + r_idx * Inches(2.5)
    
    c = create_card(s16, x, y, Inches(3.7), Inches(2.2), CARD_BG)
    strip = s16.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(3.7), Inches(0.5))
    set_shape_flat(strip, col)
    
    tb_h = s16.shapes.add_textbox(x + Inches(0.1), y + Inches(0.08), Inches(3.5), Inches(0.4))
    p = tb_h.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK
    
    tb_b = s16.shapes.add_textbox(x + Inches(0.15), y + Inches(0.6), Inches(3.4), Inches(1.5))
    tf_b = tb_b.text_frame
    tf_b.word_wrap = True
    p = tf_b.paragraphs[0]
    p.text = desc
    p.font.size = Pt(10.5)
    p.font.color.rgb = TEXT_DARK

# ==============================================================================
# SLIDE 17: SYLLABUS MAPPING & REFERENCES (Dark Theme)
# ==============================================================================
s17 = prs.slides.add_slide(blank_layout)
bg17 = s17.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_WIDTH, SLIDE_HEIGHT)
set_shape_flat(bg17, DARK_BG)
add_header(s17, "07 · Academic Mapping", "CSE471 Syllabus Mapping & Key References", dark_mode=True)

syllabus_boxes = [
    ("Unit I: Image Foundations", "224×224 image representation, ImageNet normalisation, strict colour preservation, multi-task loss formulation."),
    ("Unit II: Deep CNNs & Optimization", "Deep convolutional layers, BatchNorm, Dropout, AdamW decoupled weight decay, Cosine Annealing learning rate schedule."),
    ("Unit III: Transfer Learning", "ImageNet-1k pretrained ResNet50, 2-phase freeze → unfreeze schedule, layer-wise discriminative learning rates."),
    ("Unit IV: Object Detection", "YOLO bounding box representations [cx, cy, w, h], square expansion with 8% margin, 6-species crop extraction."),
    ("Unit VI: Explainable AI (XAI)", "Dual-task Grad-CAM saliency heatmaps backpropagated through Layer 4 feature maps via PyTorch hooks."),
    ("Beyond the Syllabus", "Rank-Consistent Ordinal Regression (CORAL), Quadratic Weighted Kappa (QWK), 7-fold leak-free grouped CV, RTX 4050 FP16 AMP.")
]

for i, (title, desc) in enumerate(syllabus_boxes):
    col_idx = i % 3
    row_idx = i // 3
    x = Inches(0.8) + col_idx * Inches(4.0)
    y = Inches(1.8) + row_idx * Inches(2.2)
    
    c = create_card(s17, x, y, Inches(3.7), Inches(2.0), RGBColor(38, 38, 42), RGBColor(70, 70, 80))
    tb = s17.shapes.add_textbox(x + Inches(0.15), y + Inches(0.12), Inches(3.4), Inches(1.7))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_SUN
    p.space_after = Pt(4)
    
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(10)
    p2.font.color.rgb = RGBColor(220, 220, 220)

# References Footer
ref_box = s17.shapes.add_textbox(Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8))
tf_ref = ref_box.text_frame
p_ref = tf_ref.paragraphs[0]
p_ref.text = "References: [1] Cao, Mirjalili, Raschka (2020). Rank consistent ordinal regression for neural networks. Pattern Recognition Letters. [2] Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV. [3] He et al. (2016). Deep Residual Learning for Image Recognition. CVPR."
p_ref.font.size = Pt(9.5)
p_ref.font.color.rgb = RGBColor(160, 160, 160)

# Save
prs.save(OUTPUT_PPTX)
print(f"Presentation saved successfully to: {OUTPUT_PPTX}")
