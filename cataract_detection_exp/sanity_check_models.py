# sanity_check_models.py
import torch
from utils.model_loader import load_model

def run_sanity_check():
    device = torch.device("cpu")

    print("Loading Model A...")
    load_model("models/modelA.pth", device)
    print("✓ Model A loaded")

    print("Loading Model B1...")
    load_model("models/modelB1.pth", device)
    print("✓ Model B1 loaded")

    print("Loading Model B2...")
    load_model("models/modelB2.pth", device)
    print("✓ Model B2 loaded")

    print("🎉 ALL MODELS LOADED SUCCESSFULLY")

