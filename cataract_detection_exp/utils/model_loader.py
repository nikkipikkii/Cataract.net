import torch
import timm
import torch.nn as nn

class CataractNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features

        self.severity_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )

        self.aux_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.severity_head(feats), self.aux_head(feats)


def load_model(path, device):
    model = CataractNet().to(device)
    ckpt = torch.load(path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"{path} is not a valid checkpoint. "
            "Expected key: 'model_state_dict'"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

