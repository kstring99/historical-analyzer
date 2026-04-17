"""Dataset class for training the structure classifier."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from cv_analyzer.classifier.model import TARGET_CLASSES


# Standard ImageNet normalization (EfficientNet was trained with these)
TRANSFORM_TRAIN = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_VAL = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class StructureDataset(Dataset):
    """Dataset of labeled image crops for structure classification.

    Expected directory layout:
        dataset_dir/
        ├── labels.json     # {"images": [{"file": "001.jpg", "class": "ast"}, ...]}
        └── crops/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
    """

    def __init__(self, dataset_dir: str | Path, split: str = "train"):
        self.root = Path(dataset_dir)
        self.transform = TRANSFORM_TRAIN if split == "train" else TRANSFORM_VAL

        labels_path = self.root / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"No labels.json found in {self.root}")

        with open(labels_path) as f:
            data = json.load(f)

        self.samples = []
        class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}

        for entry in data["images"]:
            img_path = self.root / "crops" / entry["file"]
            cls = entry["class"]
            if cls not in class_to_idx:
                continue
            if not img_path.exists():
                continue
            self.samples.append((str(img_path), class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Return a blank image if file is corrupt
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, label


def prepare_tensor(crop: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV crop to a normalized tensor for inference."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return TRANSFORM_VAL(rgb)
