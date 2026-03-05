import torch
import torch.nn.functional as F

LABELS = ["No Cataract", "Immature", "Mature"]
AUX_LABELS = ["IOL", "Native Lens (Non-IOL)"]


# Temperatures (fixed from calibration)
T_A  = 1.600
T_B1 = 1.265
T_B2 = 1.924


def run_ensemble(img_224, img_256, modelA, modelB1, modelB2, device):
    img_224 = img_224.to(device)
    img_256 = img_256.to(device)

    with torch.no_grad():
        logits_A, aux_A   = modelA(img_224)
        logits_B1, aux_B1 = modelB1(img_256)
        logits_B2, aux_B2 = modelB2(img_256)

        probs_A  = F.softmax(logits_A / T_A, dim=1)
        probs_B1 = F.softmax(logits_B1 / T_B1, dim=1)
        probs_B2 = F.softmax(logits_B2 / T_B2, dim=1)

        probs = (probs_A + probs_B1 + probs_B2) / 3.0
        severity_idx = torch.argmax(probs, dim=1).item()

        # Aux head (average)
        aux_probs = (
            F.softmax(aux_A, dim=1) +
            F.softmax(aux_B1, dim=1) +
            F.softmax(aux_B2, dim=1)
        ) / 3.0
        aux_idx = torch.argmax(aux_probs, dim=1).item()

    aux_probs_np = aux_probs.squeeze().cpu().numpy()

    return {
        "severity": LABELS[severity_idx],
        "severity_probs": probs.squeeze().cpu().numpy(),
        "lens_type": AUX_LABELS[aux_idx],
        "lens_probs": aux_probs_np
    }
