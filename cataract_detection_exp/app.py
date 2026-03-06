import os
import gradio as gr
import torch
from PIL import Image
import gradio as gr
print(gr.__version__)
import random
import numpy as np
from PIL import Image
from utils.model_loader import load_model
from utils.preprocessing import preprocess
from utils.inference import run_ensemble
from sanity_check_models import run_sanity_check

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Model paths
# ----------------------------
MODEL_A_PATH  = "models/modelA.pth"
MODEL_B1_PATH = "models/modelB1.pth"
MODEL_B2_PATH = "models/modelB2.pth"

# ----------------------------
# Calibration thresholds (FINAL)
# ----------------------------
ZONE_1_MAX = 0.08
ZONE_2A_MAX = 0.10
ZONE_2B_MAX = 0.12
CATARACT_OPACITY_THRESHOLD = 0.70

# ----------------------------
# Sanity check
# ----------------------------
run_sanity_check()

# ----------------------------
# Force HF Xet materialization
# ----------------------------
for p in [MODEL_A_PATH, MODEL_B1_PATH, MODEL_B2_PATH]:
    if not os.path.exists(p):
        raise RuntimeError(f"Model file missing: {p}")
    with open(p, "rb") as f:
        f.read(1)

# ----------------------------
#demo folders
# ----------------------------
DEMO_FOLDERS = {
    "Natural — No Cataract": "natural_no_cataract",
    "Natural — Immature Cataract": "natural_immature",
    "Natural — Mature Cataract": "natural_mature",
    "Intraocular Lens (IOL)": "iol"
}
# ----------------------------
# Load models ONCE
# ----------------------------
modelA  = load_model(MODEL_A_PATH, device)
modelB1 = load_model(MODEL_B1_PATH, device)
modelB2 = load_model(MODEL_B2_PATH, device)
def render_cataract_card(p_nc, p_cat):

    nc_pct = round(p_nc * 100)
    cat_pct = round(p_cat * 100)

    # Color + zone based strictly on YOUR thresholds
    if p_nc >= ZONE_2B_MAX:
        zone_label = "Clear Non-Cataract Zone"
        zone_color = "#10b981"
    elif ZONE_2A_MAX <= p_nc < ZONE_2B_MAX:
        zone_label = "Overlap / Monitoring Zone"
        zone_color = "#f59e0b"
    elif ZONE_1_MAX < p_nc < ZONE_2A_MAX:
        zone_label = "Early Signal Zone"
        zone_color = "#f97316"
    else:
        zone_label = "Strong Cataract Signal"
        zone_color = "#ef4444"

    return f"""
    <div style="
        border-radius:14px;
        padding:22px;
        background:#ffffff;
        box-shadow:0 10px 30px rgba(0,0,0,0.08);
        margin-top:16px;
        border:1px solid #e5e7eb;
    ">
        <div style="
            font-size:18px;
            font-weight:600;
            margin-bottom:18px;
            color:#111827;
        ">
            Cataract Screening Confidence
        </div>
        <div style="
            display:inline-block;
            padding:6px 14px;
            border-radius:20px;
            background:{zone_color};
            color:white;
            font-weight:600;
            margin-bottom:20px;
        ">
            {zone_label}
        </div>
        <div style="margin-bottom:18px;">
            <div style="display:flex; justify-content:space-between; font-weight:500; color:#374151;">
                <span>Non-Cataract Probability (Decision Anchor)</span>
                <span>{nc_pct}%</span>
            </div>
            <div style="height:12px; background:#f3f4f6; border-radius:8px; margin-top:6px;">
                <div style="width:{nc_pct}%; height:100%; background:{zone_color}; border-radius:8px;"></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; font-weight:500; color:#6b7280;">
                <span>Raw Cataract Pattern Confidence</span>
                <span>{cat_pct}%</span>
            </div>
            <div style="height:10px; background:#f3f4f6; border-radius:8px; margin-top:6px;">
                <div style="width:{cat_pct}%; height:100%; background:#d1d5db; border-radius:8px;"></div>
            </div>
        </div>
        <div style="
            font-size:13px;
            color:#6b7280;
            margin-top:18px;
            line-height:1.5;
        ">
            Final classification is determined using calibrated non-cataract 
            probability thresholds validated during model development.
            Raw cataract confidence alone does not determine the final result.
        </div>
        <details style="margin-top:8px;">
            <summary style="
                cursor:pointer;
                font-weight:500;
                color:#2563eb;
                font-size:13px;
            ">
                How the screening decision is made
            </summary>
            <div style="
                margin-top:6px;
                font-size:12.5px;
                color:#4b5563;
                line-height:1.45;
            ">
                <strong>Decision Logic</strong><br>
                ≤ 8% → Cataract Present<br>
                8–10% → Likely Early Cataract<br>
                10–12% → Monitoring Zone<br>
                ≥ 12% → No Cataract Detected<br><br>
                Classification is based on calibrated non-cataract probability thresholds.
                The system prioritizes reducing false positives.
                A high raw cataract percentage alone does not determine the result.
            </div>
        </details>
    </div>
    """
def render_lens_card(lens_type, p_iol, p_natural):
    natural_pct = round(p_natural * 100)
    iol_pct = round(p_iol * 100)

    # Highlight color based on FINAL decision
    if lens_type == "Natural Lens":
        natural_color = "#10b981"   # emerald
        iol_color = "#e5e7eb"       # light grey
        badge_color = "#10b981"
    else:
        natural_color = "#e5e7eb"
        iol_color = "#2563eb"       # strong blue
        badge_color = "#2563eb"

    return f"""
    <div style="
        border-radius:14px;
        padding:22px;
        background:#ffffff;
        box-shadow:0 10px 30px rgba(0,0,0,0.08);
        margin-top:16px;
        border:1px solid #e5e7eb;
    ">

        <div style="
            font-size:18px;
            font-weight:600;
            margin-bottom:18px;
            color:#111827;
        ">
            Lens Analysis
        </div>

        <div style="
            display:inline-block;
            padding:6px 14px;
            border-radius:20px;
            background:{badge_color};
            color:white;
            font-weight:600;
            margin-bottom:20px;
        ">
            Final Decision: {lens_type}
        </div>

        <div style="margin-bottom:18px;">
            <div style="
                display:flex;
                justify-content:space-between;
                font-weight:500;
                color:#374151;
            ">
                <span>Natural Lens</span>
                <span>{natural_pct}%</span>
            </div>
            <div style="
                height:12px;
                background:#f3f4f6;
                border-radius:8px;
                margin-top:6px;
            ">
                <div style="
                    width:{natural_pct}%;
                    height:100%;
                    background:{natural_color};
                    border-radius:8px;
                "></div>
            </div>
        </div>

        <div style="margin-bottom:18px;">
            <div style="
                display:flex;
                justify-content:space-between;
                font-weight:500;
                color:#374151;
            ">
                <span>Intraocular Lens (IOL)</span>
                <span>{iol_pct}%</span>
            </div>
            <div style="
                height:12px;
                background:#f3f4f6;
                border-radius:8px;
                margin-top:6px;
            ">
                <div style="
                    width:{iol_pct}%;
                    height:100%;
                    background:{iol_color};
                    border-radius:8px;
                "></div>
            </div>
        </div>

        <div style="
            font-size:13px;
            color:#6b7280;
            margin-top:18px;
            line-height:1.5;
        ">
            Final classification is based on validated probability thresholds 
            established during model calibration. Percentage values reflect 
            internal pattern confidence. The highest percentage alone does 
            not determine the outcome.
        </div>

    </div>
    """

# ----------------------------
# Prediction function
# ----------------------------
def predict(image):

    if image is None:
       return ("—","—","—","—","—","","")

    img = Image.fromarray(image).convert("RGB")
    img_224, img_256 = preprocess(img)

    result = run_ensemble(
        img_224,
        img_256,
        modelA,
        modelB1,
        modelB2,
        device
    )
    

    probs = result["severity_probs"]  # [No Cataract, Immature, Mature]
    lens_probs = result["lens_probs"]
    print("Raw lens_probs:", lens_probs)
    p_iol = float(lens_probs[0])
    p_natural = float(lens_probs[1])

    print("IOL prob:", p_iol, "| Natural prob:", p_natural)
    p_nc = float(probs[0])
    p_cat = 1.0 - p_nc
    
    # ----------------------------
    # Classification + Action
    # ----------------------------
    if p_nc <= ZONE_1_MAX:
        assessment = "Cataract Present"
        note = (
            "Clear imaging evidence of cataract is detected. "
            "Lens opacity patterns are consistent with clinically significant cataract."
        )
        suggested_action = "Ophthalmologic evaluation recommended."

    elif ZONE_1_MAX < p_nc < ZONE_2A_MAX:
        assessment = "Likely Early (Immature) Cataract"
        note = (
            "Imaging patterns suggest early or immature cataract formation. "
            "Structural lens changes are present but not advanced."
        )
        suggested_action = "Routine monitoring or clinical correlation advised."

    elif ZONE_2A_MAX <= p_nc < ZONE_2B_MAX:
        assessment = "Likely Non-Cataract (Early Overlap)"
        note = (
            "No definitive cataract detected. "
            "Subtle lens patterns may overlap with very early cataract features "
            "or normal physiological variation."
        )
        suggested_action = "Monitoring recommended if symptoms develop."

    else:
        assessment = "No Cataract Detected"
        note = (
            "Lens appearance is consistent with a healthy lens. "
            "No imaging evidence of cataract is observed."
        )
        suggested_action = "No immediate follow-up required."

    
    # ----------------------------
    # Lens Type Interpretation (Independent of Severity)
    # ----------------------------

    if p_iol >= 0.75:
        lens_type = "Intraocular Lens Detected"
        lens_note = "High-confidence detection of post-surgical intraocular lens."

    elif 0.60 <= p_iol < 0.75:
        lens_type = "Possible Intraocular Lens"
        lens_note = "Moderate confidence of post-surgical lens features."

    else:
        lens_type = "Natural Lens"
        lens_note = "Lens appearance consistent with native crystalline lens."

    cataract_card = render_cataract_card(p_nc, p_cat)

    lens_card = render_lens_card(lens_type, p_iol, p_natural)
    return assessment, note, suggested_action, lens_type, lens_note, cataract_card, lens_card

# ----------------------------
# demo loader
# ---------------------------- 
def load_demo_image(category):

    base_path = "demo_validation_set"
    folder_name = DEMO_FOLDERS[category]

    category_path = os.path.join(base_path, folder_name)

    images = [
        f for f in os.listdir(category_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not images:
        raise ValueError(f"No images found in {category_path}")

    chosen_file = random.choice(images)

    img_path = os.path.join(category_path, chosen_file)

    img = Image.open(img_path).convert("RGB")

    return np.array(img)
# ----------------------------
# demo pipeline autorun
# ----------------------------
def demo_pipeline(category):

    img_array = load_demo_image(category)

    return (
        gr.update(visible=True, value=img_array),
        *predict(img_array)
    )

# ----------------------------
# UI
# ----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🩺 Cataract Screening — Research Demo
    **Calibrated deep-learning ensemble optimized for cataract screening.**
    ⚠️ Research & educational use only  
    Not a diagnostic or clinical medical device
    """)
    with gr.Row():
        with gr.Column(scale=1):

            mode_selector = gr.Radio(
                choices=["Upload Eye Image", "Test on Independent Samples"],
                value="Upload Eye Image",
                label="Image Source"
            )

            image_input = gr.Image(
                label="Upload Eye Image",
                type="numpy",
                height=260
            )
            predict_btn = gr.Button("Run Screening")
            
            demo_dropdown = gr.Dropdown(
                choices=list(DEMO_FOLDERS.keys()),
                label="Select Demo Category",
                visible=False
            )

            demo_note = gr.Markdown(
                "These images were not used during model training. "
                "They are part of an independent demo validation set.",
                visible=False
            )

            demo_button = gr.Button(
                "Load Sample Image",
                visible=False
            )


        with gr.Column(scale=1):
            assessment_out = gr.Textbox(label="Result", interactive=False)
            note_out = gr.Textbox(label="Explanation", interactive=False)
            action_out = gr.Textbox(label="Suggested Action", interactive=False)
            lens_out = gr.Textbox(label="Lens Type", interactive=False)
            lens_note_out = gr.Textbox(label="Lens Interpretation", interactive=False)
            cataract_plot = gr.HTML()
            lens_plot = gr.HTML()


            # ----------------------------
            # Result interpretation guide
            # ----------------------------
            with gr.Accordion("How to Interpret Results", open=False):

                gr.Markdown("""
<div style="margin-top:10px; font-size:14px; color:#374151;">

This system categorizes eye images into four outcomes based on detected lens patterns.

<div style="margin-top:18px; padding:16px; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
<strong>Cataract Present</strong><br>
Clear imaging evidence of clinically significant lens opacity.<br>
<span style="color:#6b7280;">Recommended:</span> Clinical evaluation by an eye specialist.
</div>

<div style="margin-top:14px; padding:16px; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
<strong>Likely Early (Immature) Cataract</strong><br>
Early structural lens changes detected. Not advanced.<br>
<span style="color:#6b7280;">Recommended:</span> Routine monitoring or clinical correlation.
</div>

<div style="margin-top:14px; padding:16px; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
<strong>Likely Non-Cataract (Early Overlap)</strong><br>
No definitive cataract detected. Patterns may overlap with early cataract or normal variation.<br>
<span style="color:#6b7280;">Recommended:</span> Monitor if symptoms develop.
</div>

<div style="margin-top:14px; padding:16px; border-radius:12px; background:#f9fafb; border:1px solid #e5e7eb;">
<strong>No Cataract Detected</strong><br>
Lens appearance consistent with healthy structure.<br>
<span style="color:#6b7280;">Recommended:</span> No follow-up required at this time.
</div>

<div style="margin-top:18px; font-size:12px; color:#9ca3af;">
This tool is designed for screening and educational purposes and does not replace a clinical eye examination.
</div>

</div>
                """)    
    

    # ----------------------------
    def toggle_mode(mode):
        if mode == "Upload Eye Image":
            return (
                gr.update(visible=True, value=None),   # image_input
                gr.update(visible=False),              # demo_dropdown
                gr.update(visible=False),              # demo_note
                gr.update(visible=False),              # demo_button
                gr.update(visible=True),               # predict_btn
                "",                                    # assessment_out
                "",                                    # note_out
                "",                                    # action_out
                "",                                    # lens_out
                "",                                    # lens_note_out
                "",                                    # cataract_plot
                "",                                    # lens_plot
            )
        else:
            return (
                gr.update(visible=False, value=None),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            )
    
    mode_selector.change(
        fn=toggle_mode,
        inputs=mode_selector,
        outputs=[
             image_input,
             demo_dropdown,
             demo_note,
             demo_button,
             predict_btn,
             assessment_out,
             note_out,
             action_out,
             lens_out,
             lens_note_out,
             cataract_plot,
             lens_plot,
        ]
    )

    
    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            assessment_out,
            note_out,
            action_out,
            lens_out,
            lens_note_out,
            cataract_plot,
            lens_plot,
        ]
    )
    demo_button.click(
        fn=demo_pipeline,
        inputs=demo_dropdown,
        outputs=[
            image_input,
            assessment_out,
            note_out,
            action_out,
            lens_out,
            lens_note_out,
            cataract_plot,
            lens_plot,
        ]
    )


demo.launch()