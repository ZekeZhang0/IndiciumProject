"""
CheXpert+ Multi-Label Chest X-Ray Classification
Training Pipeline — DenseNet121 with Transfer Learning

Assumes organized_data/ structure from the data organization scripts:
    organized_data/
        train/
            Atelectasis/
            Cardiomegaly/
            ...
            No_Finding/
        val/
            ...
        test/
            ...

Each image belongs to exactly one folder (primary label), but during training
we reconstruct the full multi-label vector from the JSON labels file.

Usage:
    python train_chexpert.py
    python train_chexpert.py --epochs 20 --batch_size 32
"""

import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PATHO_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax",
    "Support Devices", "No Finding"
]
NUM_CLASSES = len(PATHO_LABELS)

# ImageNet mean/std (standard for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class CheXpertDataset(Dataset):
    """
    Multi-label CheXpert dataset.

    Reads images from organized_data/{split}/<label>/*.png
    and looks up their full multi-label vector from report_fixed.json.

    If a JSON path is not supplied, falls back to single-label mode
    (only the folder label is positive).
    """

    def __init__(self, data_dir: str, split: str, transform=None, json_path: str = None):
        self.data_dir  = Path(data_dir) / split
        self.transform = transform
        self.label_lookup = {}

        # Build label lookup from JSON
        if json_path and os.path.exists(json_path):
            with open(json_path) as f:
                json_data = json.load(f)
            for item in json_data:
                key = item["path_to_image"]
                vec = []
                for l in PATHO_LABELS[:-1]:          # first 13 pathologies
                    val = item.get(l, 0.0) or 0.0
                    vec.append(1.0 if val == 1.0 else 0.0)
                # "No Finding" = 1 only when all others are 0
                vec.append(1.0 if sum(vec) == 0 else 0.0)
                self.label_lookup[key] = vec

        # Collect all image paths
        self.samples = []   # list of (Path, label_vector)
        for label_dir in sorted(self.data_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            folder_label = label_dir.name.replace("_", " ")
            for img_path in label_dir.glob("*.png"):
                label_vec = self._resolve_labels(img_path, folder_label)
                self.samples.append((img_path, label_vec))

        print(f"[{split}] {len(self.samples)} images across {NUM_CLASSES} classes")

    def _resolve_labels(self, img_path: Path, folder_label: str):
        """Convert an image path back to the JSON key and look up labels."""
        # img_path example: organized_data/train/Atelectasis/patient12_study1_view1.png
        # JSON key example: train/patient12/study1/view1.jpg
        parts = img_path.stem.split("_")
        if len(parts) >= 3:
            patient, study = parts[0], parts[1]
            view = "_".join(parts[2:]) + ".jpg"
            split_name = img_path.parent.parent.name    # 'train' / 'val' / 'test'
            json_key = f"{split_name}/{patient}/{study}/{view}"
            if json_key in self.label_lookup:
                return self.label_lookup[json_key]

        # Fallback: single-label from folder name
        vec = [0.0] * NUM_CLASSES
        label_name = folder_label if folder_label != "No_Finding" else "No Finding"
        if label_name in PATHO_LABELS:
            vec[PATHO_LABELS.index(label_name)] = 1.0
        else:
            vec[-1] = 1.0   # No Finding
        return vec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_vec = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label_vec, dtype=torch.float32)
        return image, label


# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Build DenseNet121 pretrained on ImageNet with a custom
    multi-label classification head.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss   = total_loss / len(loader)

    # Compute per-class AUC (skip classes with no positive samples)
    aucs = []
    for i in range(NUM_CLASSES):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            aucs.append(auc)
        else:
            aucs.append(float("nan"))

    macro_auc = np.nanmean(aucs)
    return avg_loss, macro_auc, aucs, all_probs, all_labels


def print_per_class_auc(aucs):
    print("\n  Per-Class AUC:")
    for name, auc in zip(PATHO_LABELS, aucs):
        bar = "█" * int(auc * 20) if not np.isnan(auc) else "N/A"
        val = f"{auc:.4f}" if not np.isnan(auc) else " N/A "
        print(f"    {name:<30} {val}  {bar}")
    print()


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────
def train(args):
    # ── Device ─────────────────────────────
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  CheXpert+ Training Pipeline")
    print(f"  Model:   DenseNet121")
    print(f"  Device:  {device}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch_size}")
    print(f"  ImgSize: {args.image_size}x{args.image_size}")
    print(f"{'='*60}\n")

    # ── Transforms & Datasets ───────────────
    train_tf, val_tf = get_transforms(args.image_size)

    train_ds = CheXpertDataset(args.data_dir, "train", train_tf, args.json_path)
    val_ds   = CheXpertDataset(args.data_dir, "val",   val_tf,   args.json_path)
    test_ds  = CheXpertDataset(args.data_dir, "test",  val_tf,   args.json_path)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ───────────────────────────────
    model = build_model(NUM_CLASSES, pretrained=True).to(device)

    # ── Loss, Optimizer, Scheduler ──────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Checkpoint Dir ──────────────────────
    ckpt_dir = Path(args.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_densenet121.pth"

    # ── Training Loop ───────────────────────
    best_auc = 0.0
    history  = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'─'*50}")
        print(f"  Epoch {epoch}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'─'*50}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_auc, val_aucs, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"\n  → Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_auc":    val_auc,
                "labels":     PATHO_LABELS,
            }, best_model_path)
            print(f"  ✓ New best model saved (AUC={val_auc:.4f})")

        print_per_class_auc(val_aucs)

    # ── Final Test Evaluation ────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL TEST EVALUATION (loading best model)")
    print(f"{'='*60}\n")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_loss, test_auc, test_aucs, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"  Test Loss: {test_loss:.4f} | Test Macro AUC: {test_auc:.4f}\n")
    print_per_class_auc(test_aucs)

    # Threshold predictions at 0.5 for F1
    preds = (test_probs >= 0.5).astype(int)
    print("\n  Classification Report (threshold=0.5):")
    print(classification_report(test_labels, preds,
                                target_names=PATHO_LABELS, zero_division=0))

    # Save training history
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history saved to {history_path}")
    print(f"  Best model saved to {best_model_path}")
    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_single(image_path: str, model_path: str, image_size: int = 224, threshold: float = 0.5):
    """
    Run inference on a single image using a saved DenseNet121 checkpoint.
    Returns a dict of {label: probability}.
    """
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    _, val_tf = get_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = val_tf(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

    results = {label: float(prob) for label, prob in zip(PATHO_LABELS, probs)}

    print(f"\nPredictions for: {image_path}")
    print(f"{'─'*40}")
    for label, prob in sorted(results.items(), key=lambda x: -x[1]):
        flag = "✓" if prob >= threshold else " "
        print(f"  {flag} {label:<30} {prob:.4f}")

    return results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CheXpert+ DenseNet121 Training Pipeline")
    parser.add_argument("--data_dir",    type=str,   default="./organized_data/",
                        help="Root dir of organized train/val/test folders")
    parser.add_argument("--json_path",   type=str,   default="report_fixed.json",
                        help="Path to report_fixed.json for multi-label lookup")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--output_dir",  type=str,   default="./checkpoints/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
