"""Training pipeline for the structure classifier. Supports Apple Silicon MPS."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from cv_analyzer.classifier.model import (
    build_model,
    get_device,
    TARGET_CLASSES,
    NUM_CLASSES,
)
from cv_analyzer.classifier.dataset import StructureDataset


def train(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    unfreeze_after: int = 10,
):
    """Train the classifier from a labeled dataset.

    Two-phase training:
    1. Epochs 1-unfreeze_after: frozen backbone, train head only (fast convergence)
    2. Epochs unfreeze_after+1 to end: unfreeze backbone, lower LR (fine-tune)
    """
    device = get_device()
    print(f"Device: {device}")
    print(f"Classes: {TARGET_CLASSES}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    full_dataset = StructureDataset(dataset_dir, split="train")
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    if n_total < 10:
        print(f"Warning: only {n_total} samples. Need more labeled data for useful training.")

    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Use val transforms for validation split
    val_ds.dataset.transform = StructureDataset(dataset_dir, split="val").transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {n_train}, Val: {n_val}")

    # Build model (frozen backbone initially)
    model = build_model(pretrained=True, freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Phase 2: unfreeze backbone
        if epoch == unfreeze_after + 1:
            print(f"\n--- Unfreezing backbone at epoch {epoch} ---")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - unfreeze_after
            )

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        })

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train loss={avg_train_loss:.4f} acc={train_acc:.3f} | "
            f"Val loss={avg_val_loss:.4f} acc={val_acc:.3f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "classes": TARGET_CLASSES,
                },
                output_path / "best_model.pt",
            )
            print(f"  → Saved best model (val_acc={val_acc:.3f})")

    # Save final model + training history
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "classes": TARGET_CLASSES,
        },
        output_path / "final_model.pt",
    )

    with open(output_path / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc}, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Models saved to: {output_path}")
