import os
import io
import random
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.model_loader import load_model
from utils.preprocessing import preprocess
from utils.inference import run_ensemble
from sanity_check_models import run_sanity_check
from fastapi.responses import FileResponse

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────
MODEL_A_PATH  = "models/modelA.pth"
MODEL_B1_PATH = "models/modelB1.pth"
MODEL_B2_PATH = "models/modelB2.pth"

# ─────────────────────────────────────────
# Calibration thresholds
# ─────────────────────────────────────────
ZONE_1_MAX  = 0.08
ZONE_2A_MAX = 0.10
ZONE_2B_MAX = 0.12

# ─────────────────────────────────────────
# Demo folder mapping
# ─────────────────────────────────────────
DEMO_BASE = "demo_validation_set"
DEMO_FOLDERS = {
    "Natural — No Cataract":       "natural_no_cataract",
    "Natural — Immature Cataract": "natural_immature",
    "Natural — Mature Cataract":   "natural_mature",
    "Intraocular Lens (IOL)":      "iol",
}

# ─────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────
run_sanity_check()

for p in [MODEL_A_PATH, MODEL_B1_PATH, MODEL_B2_PATH]:
    if not os.path.exists(p):
        raise RuntimeError(f"Model file missing: {p}")
    with open(p, "rb") as f:
        f.read(1)

# ─────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────
modelA  = load_model(MODEL_A_PATH,  device)
modelB1 = load_model(MODEL_B1_PATH, device)
modelB2 = load_model(MODEL_B2_PATH, device)

# ─────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────
app = FastAPI(
    title="CataractNet API",
    description="Calibrated deep-learning ensemble for cataract screening.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────
class PredictResponse(BaseModel):
    assessment:        str
    note:              str
    suggested_action:  str
    lens_type:         str
    lens_note:         str   # ← always populated
    confidence_dist:   dict  # {"Cataract": float, "No Cataract": float}
    lens_probs:        dict  # {"IOL": float, "Natural": float}
    zone:              str
    zone_color:        str
    disclaimer:        str
    sample_image_b64:  str = ""

# ─────────────────────────────────────────
# Shared classification logic
# ─────────────────────────────────────────
def _classify(p_nc: float, p_iol: float, p_natural: float) -> dict:
    """
    Pure classification from probabilities.
    Returns every field the PredictResponse schema requires.
    """
    p_cat = 1.0 - p_nc

    # ── Cataract zone
    if p_nc <= ZONE_1_MAX:
        assessment       = "Cataract Present"
        note             = (
            "Clear imaging evidence of cataract is detected. "
            "Lens opacity patterns are consistent with clinically significant cataract."
        )
        suggested_action = "Ophthalmologic evaluation recommended."
        zone             = "Strong Cataract Signal"
        zone_color       = "#ef4444"

    elif ZONE_1_MAX < p_nc < ZONE_2A_MAX:
        assessment       = "Likely Early (Immature) Cataract"
        note             = (
            "Imaging patterns suggest early or immature cataract formation. "
            "Structural lens changes are present but not advanced."
        )
        suggested_action = "Routine monitoring or clinical correlation advised."
        zone             = "Early Signal Zone"
        zone_color       = "#f97316"

    elif ZONE_2A_MAX <= p_nc < ZONE_2B_MAX:
        assessment       = "Likely Non-Cataract (Early Overlap)"
        note             = (
            "No definitive cataract detected. "
            "Subtle lens patterns may overlap with very early cataract features "
            "or normal physiological variation."
        )
        suggested_action = "Monitoring recommended if symptoms develop."
        zone             = "Overlap / Monitoring Zone"
        zone_color       = "#f59e0b"

    else:
        assessment       = "No Cataract Detected"
        note             = (
            "Lens appearance is consistent with a healthy lens. "
            "No imaging evidence of cataract is observed."
        )
        suggested_action = "No immediate follow-up required."
        zone             = "Clear Non-Cataract Zone"
        zone_color       = "#10b981"

    # ── Lens type
    if p_iol >= 0.75:
        lens_type = "Intraocular Lens (IOL) Detected"
        lens_note = "High-confidence detection of post-surgical intraocular lens."
    elif 0.60 <= p_iol < 0.75:
        lens_type = "Possible Intraocular Lens"
        lens_note = "Moderate confidence of post-surgical lens features."
    else:
        lens_type = "Natural Lens"
        lens_note = "Lens appearance consistent with native crystalline lens."

    return dict(
        assessment       = assessment,
        note             = note,
        suggested_action = suggested_action,
        lens_type        = lens_type,
        lens_note        = lens_note,
        confidence_dist  = {
            "Cataract":    round(p_cat,     4),
            "No Cataract": round(p_nc,      4),
        },
        lens_probs = {
            "IOL":     round(p_iol,     4),
            "Natural": round(p_natural, 4),
        },
        zone       = zone,
        zone_color = zone_color,
        disclaimer = (
            "Research and educational use only. "
            "Not a diagnostic or clinical medical device. "
            "Always consult a qualified ophthalmologist."
        ),
    )

# ─────────────────────────────────────────
# Shared inference logic
# ─────────────────────────────────────────
def _run_inference(img: Image.Image) -> PredictResponse:
    img_224, img_256 = preprocess(img)
    result = run_ensemble(img_224, img_256, modelA, modelB1, modelB2, device)

    probs      = result["severity_probs"]  # [No Cataract, Immature, Mature]
    lens_probs = result["lens_probs"]      # [IOL, Natural]

    p_nc      = float(probs[0])
    p_iol     = float(lens_probs[0])
    p_natural = float(lens_probs[1])

    fields = _classify(p_nc, p_iol, p_natural)
    return PredictResponse(**fields, sample_image_b64="")

# ─────────────────────────────────────────
# /predict  — upload mode
# ─────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    return _run_inference(img)

# ─────────────────────────────────────────
# /demo  — independent samples mode
# ─────────────────────────────────────────
@app.get("/demo", response_model=PredictResponse)
def demo(category: str = Query(..., description="One of the four demo category names")):
    import base64

    if category not in DEMO_FOLDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown category '{category}'. Valid: {list(DEMO_FOLDERS.keys())}"
        )

    folder_path = os.path.join(DEMO_BASE, DEMO_FOLDERS[category])
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=500, detail=f"Demo folder not found: {folder_path}")

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        raise HTTPException(status_code=500, detail=f"No images found in {folder_path}")

    img_path = os.path.join(folder_path, random.choice(images))
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open demo image: {exc}")

    response = _run_inference(img)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    response.sample_image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return response

# ─────────────────────────────────────────
# /demo/categories
# ─────────────────────────────────────────
@app.get("/demo/categories")
def demo_categories():
    return {"categories": list(DEMO_FOLDERS.keys())}

# ─────────────────────────────────────────
# /test  — mock response for frontend dev
#          Uses _classify() so it is always
#          in sync with real inference output.
# ─────────────────────────────────────────
@app.get("/test", response_model=PredictResponse)
def test_response():
    """
    Returns a fully-formed PredictResponse built through the real
    _classify() path so every field (including lens_note) is guaranteed
    to be present and consistent.

    Open: http://127.0.0.1:8000/test
    """
    # Simulate: strong cataract signal, natural lens
    fields = _classify(p_nc=0.07, p_iol=0.36, p_natural=0.64)
    return PredictResponse(**fields, sample_image_b64="")

# ─────────────────────────────────────────
# /health
# ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# serve  model.html
@app.get("/")
async def read_index():
    # Adjust path if main.py is in a subdirectory
    return FileResponse('../model.html')
@app.get("/index_detailed.html")
async def read_technical_report():
    # Make sure this path matches how you point to model.html
    return FileResponse('../index_detailed.html')
# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
