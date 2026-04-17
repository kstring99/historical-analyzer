"""EfficientNet-B0 classifier for aerial photo structure identification."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


TARGET_CLASSES = [
    "ust_farm",
    "ast",
    "lagoon_pit",
    "industrial_structure",
    "vegetation_stress",
    "surface_staining",
    "benign",
]

NUM_CLASSES = len(TARGET_CLASSES)


def get_device() -> torch.device:
    """Get best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with custom classification head.

    Uses ImageNet pretrained weights. Freezes backbone by default for
    fine-tuning on small datasets (which is our case).
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        model = efficientnet_b0(weights=None)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, NUM_CLASSES),
    )

    return model


def load_model(checkpoint_path: str, device: torch.device | None = None) -> nn.Module:
    """Load a trained model from checkpoint."""
    if device is None:
        device = get_device()
    model = build_model(pretrained=False, freeze_backbone=False)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> dict:
    """Run inference on a single image tensor.

    Returns {"class", "confidence", "probabilities"}.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    top_idx = int(probs.argmax())
    return {
        "class": TARGET_CLASSES[top_idx],
        "confidence": round(float(probs[top_idx]), 3),
        "probabilities": {
            TARGET_CLASSES[i]: round(float(probs[i]), 3)
            for i in range(NUM_CLASSES)
        },
    }
