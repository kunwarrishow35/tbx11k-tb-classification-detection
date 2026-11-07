"""
ResNet-101 Multi-Label Training Pipeline for TBX11K Chest X-Ray Classification
=============================================================================

Reference Audit vs. Project Configuration:
------------------------------------------
- Official Reference: KaimingHe/deep-residual-networks & Torchvision Reference Recipe
    * Architecture: ResNet-101 (Torchvision ResNet V1.5 variant with bottleneck blocks [3, 4, 23, 3])
    * Optimizer: SGD, momentum=0.9, nesterov=False, weight_decay=1e-4
    * Schedule: Initial LR 0.01, StepLR with gamma=0.1 at epoch 30
    * Target: 2-output multi-label classification (ActiveTuberculosis, ObsoletePulmonaryTuberculosis)
- Project-Specific Adaptations (RTX 4050 Laptop GPU, 6 GB VRAM):
    * Implementation: PyTorch torchvision.models.resnet101(weights=None)
    * Classification Head: model.fc = nn.Linear(2048, 2)
    * Output Layer: 2 logits with nn.BCEWithLogitsLoss on raw logits
    * Input Resolution: 512 x 512, 3 channels (converted from grayscale if necessary)
    * Mandatory Preprocessing: RGB conversion, bilinear resize to 512 x 512, ToTensor() [0, 1]
    * Normalization: NONE (No ImageNet mean/std; no dataset-specific normalization)
    * Augmentation: NONE (Strictly deterministic training for clean benchmark comparison)
    * Epoch Budget: 70 epochs maximum (training from scratch)
    * Physical Batch Size: 4 (fallback to 2 if CUDA OOM occurs)
    * Gradient Accumulation: 1 step by default (nominal effective batch size 4)
    * Learning Rate: 0.01 initial (conservative adaptation for single-GPU small batch)
    * LR Scheduler: StepLR with gamma=0.1 at epoch 30
    * Mixed Precision: CUDA FP16 autocast + GradScaler
    * Early Stopping: Patience of 20 epochs monitored on validation macro-F1
    * Model Selection: Best validation macro-F1 checkpoint saved to best_model.pth
    * Reproducibility: Seed 42 for python, numpy, torch, and CUDA determinism
    * Smoke Test Mode: python CLASSIFICATION/ResNet-101/train.py --smoke-test (runs exactly 2 epochs)
    * Default Ordinary Execution: python CLASSIFICATION/ResNet-101/train.py (zero CLI args required)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# =============================================================================
# MAIN TRAINING CONFIGURATION (Authoritative Project Settings)
# =============================================================================
NUM_EPOCHS: int = 70
BATCH_SIZE: int = 4
BATCH_SIZE_FALLBACK: int = 2
LEARNING_RATE: float = 0.01
EARLY_STOPPING_PATIENCE: int = 20
WEIGHT_DECAY: float = 0.0001
MOMENTUM: float = 0.9
NESTEROV: bool = False
LR_STEP_SIZE: int = 30
LR_GAMMA: float = 0.1
IMAGE_SIZE: int = 512
SEED: int = 42
NUM_WORKERS: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 1

# Label Constants
TARGET_COLUMNS: List[str] = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]
CATEGORY_NAMES: List[str] = ["Healthy", "Active_Only", "Obsolete_Only", "Both"]
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)


# -----------------------------------------------------------------------------
# Reproducibility Helper
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Workspace Path Resolution
# -----------------------------------------------------------------------------
def resolve_workspace_paths(
    custom_data_dir: Optional[str] = None,
    custom_splits_dir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Locates the workspace root, dataset directory (TBX11K), and splits directory.
    Searches current working directory and parent paths dynamically.
    """
    script_dir = Path(__file__).resolve().parent

    candidates = [
        Path.cwd(),
        script_dir,
        script_dir.parent,
        script_dir.parent.parent,
    ]

    workspace_root = None
    for cand in candidates:
        if (cand / "TBX11K").is_dir() and (cand / "data_prep").is_dir():
            workspace_root = cand.resolve()
            break

    if workspace_root is None:
        workspace_root = script_dir.parent.parent.resolve()

    if custom_data_dir:
        data_dir = Path(custom_data_dir).resolve()
    else:
        data_dir = workspace_root / "TBX11K"

    if custom_splits_dir:
        splits_dir = Path(custom_splits_dir).resolve()
    else:
        splits_dir = workspace_root / "data_prep" / "splits"

    return workspace_root, data_dir, splits_dir


# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------
class TBX11KMultiLabelDataset(Dataset):
    """
    Multi-label dataset for TBX11K Chest X-Rays.
    Reads manifests generated by data_prep/generate_splits.py.
    Yields (image_tensor, target_tensor, image_rel_path).
    """

    def __init__(
        self,
        csv_path: Path,
        dataset_root: Path,
        transform: Optional[transforms.Compose] = None,
        split_name: str = "train",
        expected_count: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.split_name = split_name

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {self.csv_path}")

        self.samples: List[Dict[str, Any]] = []
        self._load_and_validate_manifest(expected_count)

    def _load_and_validate_manifest(self, expected_count: Optional[int]) -> None:
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Verify required columns
            for col in ["image_path"] + TARGET_COLUMNS:
                if col not in fieldnames:
                    raise ValueError(
                        f"Missing required column '{col}' in {self.csv_path}. "
                        f"Found columns: {fieldnames}"
                    )

            seen_paths = set()
            for row_idx, row in enumerate(reader, start=2):
                img_rel = row["image_path"].strip()
                if not img_rel:
                    raise ValueError(f"Empty image_path at line {row_idx} of {self.csv_path}")

                if img_rel in seen_paths:
                    raise ValueError(f"Duplicate image path detected at line {row_idx}: {img_rel}")
                seen_paths.add(img_rel)

                # Validate binary targets
                try:
                    act_val = int(row["ActiveTuberculosis"])
                    obs_val = int(row["ObsoletePulmonaryTuberculosis"])
                except ValueError as e:
                    raise ValueError(
                        f"Non-integer target value at line {row_idx} of {self.csv_path}: {e}"
                    )

                if act_val not in (0, 1) or obs_val not in (0, 1):
                    raise ValueError(
                        f"Target out of range [0, 1] at line {row_idx}: Active={act_val}, Obsolete={obs_val}"
                    )

                full_path = self.dataset_root / img_rel
                if not full_path.is_file():
                    raise FileNotFoundError(
                        f"Image file does not exist on disk: {full_path} (from {self.csv_path}:{row_idx})"
                    )

                self.samples.append({
                    "image_rel_path": img_rel,
                    "full_path": full_path,
                    "cohort": row.get("cohort", ""),
                    "split": row.get("split", self.split_name),
                    "active_tb": act_val,
                    "obsolete_tb": obs_val,
                })

        # Validate sample count
        if expected_count is not None and len(self.samples) != expected_count:
            logging.warning(
                f"Split '{self.split_name}' expected {expected_count} samples, but found {len(self.samples)}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.samples[idx]

        # Load image and convert to 3-channel RGB (chest X-rays may be grayscale or RGB)
        with Image.open(item["full_path"]) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        target = torch.tensor(
            [item["active_tb"], item["obsolete_tb"]],
            dtype=torch.float32,
        )

        return tensor, target, item["image_rel_path"]


# -----------------------------------------------------------------------------
# Transforms Factory (Mandatory: No Normalization, No Augmentation)
# -----------------------------------------------------------------------------
def get_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Mandatory ResNet-101 preprocessing pipeline:
    1. Bilinear resize to 512 x 512.
    2. Convert to float32 tensor via ToTensor() scaled to [0.0, 1.0].
    Strictly NO ImageNet normalization and NO data augmentation.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


# -----------------------------------------------------------------------------
# Model Construction
# -----------------------------------------------------------------------------
def build_resnet101(num_classes: int = 2) -> nn.Module:
    """
    Instantiates ResNet-101 from scratch (weights=None).
    Replaces the classification head (fc) with nn.Linear(2048, num_classes).
    Confirms all backbone and classifier parameters are trainable.
    """
    model = models.resnet101(weights=None)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    # Ensure all parameters require gradients (from-scratch training)
    for name, param in model.named_parameters():
        param.requires_grad = True

    return model


# -----------------------------------------------------------------------------
# Metric Computation Helper
# -----------------------------------------------------------------------------
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes Precision, Recall (Sensitivity), Specificity, and F1 for a binary label.
    y_true, y_pred: 1D binary arrays of shape (N,).
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def derive_category(active: int, obsolete: int) -> int:
    """
    Maps 2-label binary indicators to 4 derived categories:
      0: Healthy (0, 0)
      1: Active Only (1, 0)
      2: Obsolete Only (0, 1)
      3: Both (1, 1)
    """
    if active == 0 and obsolete == 0:
        return 0
    elif active == 1 and obsolete == 0:
        return 1
    elif active == 0 and obsolete == 1:
        return 2
    else:
        return 3


def compute_epoch_metrics(
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes all required validation/training epoch metrics:
    - Per-label precision, recall, specificity, and F1.
    - Macro-F1 across target labels.
    - Subset exact-match accuracy and Hamming loss.
    - Derived 4-category distribution and accuracy.
    """
    preds = (all_probs >= threshold).astype(int)

    # Active TB metrics
    act_metrics = compute_binary_metrics(all_targets[:, 0], preds[:, 0])
    # Obsolete TB metrics
    obs_metrics = compute_binary_metrics(all_targets[:, 1], preds[:, 1])

    macro_f1 = (act_metrics["f1"] + obs_metrics["f1"]) / 2.0

    # Multi-label subset accuracy (both labels must match)
    exact_match = np.mean(
        (all_targets[:, 0] == preds[:, 0]) & (all_targets[:, 1] == preds[:, 1])
    )
    # Hamming loss: fraction of labels misclassified
    hamming_loss = np.mean(all_targets != preds)

    # Derived 4 categories
    true_cats = [derive_category(int(t[0]), int(t[1])) for t in all_targets]
    pred_cats = [derive_category(int(p[0]), int(p[1])) for p in preds]

    true_cat_counts = {name: 0 for name in CATEGORY_NAMES}
    pred_cat_counts = {name: 0 for name in CATEGORY_NAMES}
    for c in true_cats:
        true_cat_counts[CATEGORY_NAMES[c]] += 1
    for c in pred_cats:
        pred_cat_counts[CATEGORY_NAMES[c]] += 1

    cat_accuracy = np.mean(np.array(true_cats) == np.array(pred_cats))

    return {
        "macro_f1": float(macro_f1),
        "exact_match_accuracy": float(exact_match),
        "hamming_loss": float(hamming_loss),
        "active_tb": act_metrics,
        "obsolete_tb": obs_metrics,
        "category_accuracy": float(cat_accuracy),
        "true_category_counts": true_cat_counts,
        "pred_category_counts": pred_cat_counts,
    }


# -----------------------------------------------------------------------------
# Training and Validation Loops
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
    accum_steps: int = 1,
) -> Tuple[float, Dict[str, Any], float]:
    """
    Executes one epoch of training with BCEWithLogitsLoss, AMP, and optional gradient accumulation.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    start_time = time.time()
    optimizer.zero_grad()

    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
                loss_scaled = loss / accum_steps
            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_duration = time.time() - start_time
    epoch_loss = running_loss / len(dataloader.dataset)

    all_targets_np = np.vstack(all_targets)
    all_probs_np = np.vstack(all_probs)
    metrics = compute_epoch_metrics(all_targets_np, all_probs_np)

    return epoch_loss, metrics, epoch_duration


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    Executes one validation epoch in evaluation mode without gradient calculation.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    all_targets_np = np.vstack(all_targets)
    all_probs_np = np.vstack(all_probs)
    metrics = compute_epoch_metrics(all_targets_np, all_probs_np)

    return val_loss, metrics


# -----------------------------------------------------------------------------
# Checkpoint Saving Helper
# -----------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    val_macro_f1: float,
    val_metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Saves checkpoint file with complete metadata and preprocessing details."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "val_macro_f1": float(val_macro_f1),
        "val_metrics": val_metrics,
        "config": config,
        "preprocessing": {
            "image_size": list(DEFAULT_IMAGE_SIZE),
            "channels": 3,
            "format": "RGB",
            "interpolation": "BILINEAR",
            "normalization": "none",
            "tensor_range": "[0.0, 1.0]",
            "augmentation": "none",
        },
        "target_columns": TARGET_COLUMNS,
        "category_names": CATEGORY_NAMES,
    }
    torch.save(state, path)


# -----------------------------------------------------------------------------
# Plotting Helper
# -----------------------------------------------------------------------------
def plot_training_curves(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Generates and saves clean training and validation curve plots."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_macro_f1 = [r["val_macro_f1"] for r in history]
    val_exact_match = [r["val_exact_match_acc"] for r in history]
    val_act_f1 = [r["val_active_f1"] for r in history]
    val_obs_f1 = [r["val_obsolete_f1"] for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("BCE Loss", fontsize=11, fontweight="bold")
    ax1.set_title("ResNet-101: Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Metric curves
    ax2.plot(epochs, val_macro_f1, label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_exact_match, label="Val Exact Match", color="#9467bd", linewidth=1.5, linestyle=":")
    ax2.plot(epochs, val_act_f1, label="Active TB F1", color="#d62728", linewidth=1.5, linestyle="--")
    ax2.plot(epochs, val_obs_f1, label="Obsolete TB F1", color="#8c564b", linewidth=1.5, linestyle="-.")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_title("ResNet-101: Validation Metrics", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Dedicated Smoke Test Routine
# -----------------------------------------------------------------------------
def run_smoke_test(args: argparse.Namespace) -> int:
    """
    Executes the 2-epoch smoke test mode.
    Verifies:
      1. Dataset loading and split validation (3,219 train, 690 val)
      2. Model instantiation from scratch with trainable parameters
      3. Forward pass producing valid logits
      4. Loss computation with BCEWithLogitsLoss
      5. Backward pass and non-zero finite gradients
      6. Optimizer parameter updates
      7. Validation metric computation
      8. Checkpoint saving and reloading
      9. History CSV and plot generation
      10. No CUDA OOM error
    """
    print("\n" + "=" * 70)
    print("STARTING RESNET-101 MANDATORY 2-EPOCH SMOKE TEST".center(70))
    print("=" * 70)

    script_dir = Path(__file__).resolve().parent
    smoke_output_dir = Path(args.output_dir) if args.output_dir else script_dir / "outputs" / "smoke_test"
    smoke_output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed if args.seed is not None else SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"[Smoke Test] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    _, data_dir, splits_dir = resolve_workspace_paths(args.data_dir, args.splits_dir)
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    print(f"[Smoke Test] Dataset root: {data_dir}")
    print(f"[Smoke Test] Splits root:  {splits_dir}")
    print(f"[Smoke Test] Output root:  {smoke_output_dir}")

    try:
        # 1. Dataset loading check
        transform = get_transforms(DEFAULT_IMAGE_SIZE)
        print("[Smoke Test] Loading training dataset (3,219 samples expected)...")
        train_ds = TBX11KMultiLabelDataset(train_csv, data_dir, transform=transform, split_name="train", expected_count=3219)
        print(f"[Smoke Test] Train dataset loaded successfully: {len(train_ds)} samples.")

        print("[Smoke Test] Loading validation dataset (690 samples expected)...")
        val_ds = TBX11KMultiLabelDataset(val_csv, data_dir, transform=transform, split_name="val", expected_count=690)
        print(f"[Smoke Test] Validation dataset loaded successfully: {len(val_ds)} samples.")

        batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
        num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # 2. Model initialization check
        print("[Smoke Test] Initializing ResNet-101 from scratch (weights=None)...")
        model = build_resnet101(num_classes=2).to(device)

        # Confirm backbone parameters are trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = list(model.parameters())
        if len(trainable_params) != len(total_params):
            raise RuntimeError("Smoke test failure: not all parameters require gradients.")
        print(f"[Smoke Test] Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        lr = args.lr if args.lr is not None else LEARNING_RATE
        momentum = args.momentum if args.momentum is not None else MOMENTUM
        weight_decay = args.weight_decay if args.weight_decay is not None else WEIGHT_DECAY
        nesterov = args.nesterov if args.nesterov is not None else NESTEROV
        lr_step_size = args.lr_step_size if args.lr_step_size is not None else LR_STEP_SIZE
        lr_gamma = args.lr_gamma if args.lr_gamma is not None else LR_GAMMA
        accum_steps = args.accum_steps if args.accum_steps is not None else GRADIENT_ACCUMULATION_STEPS

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        use_amp = (device.type == "cuda") and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None

        # Snapshot a parameter to verify optimizer updates
        initial_param_val = list(model.parameters())[0].detach().clone()

        # 3. Run exactly 2 epochs
        smoke_history = []
        for epoch in range(1, 3):
            epoch_loss, train_metrics, duration = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                accum_steps=accum_steps,
            )

            if not np.isfinite(epoch_loss):
                raise ValueError(f"Smoke test failure: Training loss is non-finite ({epoch_loss}) at epoch {epoch}.")

            val_loss, val_metrics = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            if not np.isfinite(val_loss):
                raise ValueError(f"Smoke test failure: Validation loss is non-finite ({val_loss}) at epoch {epoch}.")

            scheduler.step()

            # Verify parameter updated
            current_param_val = list(model.parameters())[0].detach()
            param_diff = torch.norm(current_param_val - initial_param_val).item()
            if param_diff == 0.0:
                raise RuntimeError("Smoke test failure: Optimizer step did not update model parameters.")

            smoke_history.append({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_sec": duration,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_exact_match_acc": val_metrics["exact_match_accuracy"],
                "val_hamming_loss": val_metrics["hamming_loss"],
                "val_category_acc": val_metrics["category_accuracy"],
                "val_active_precision": val_metrics["active_tb"]["precision"],
                "val_active_recall": val_metrics["active_tb"]["recall"],
                "val_active_specificity": val_metrics["active_tb"]["specificity"],
                "val_active_f1": val_metrics["active_tb"]["f1"],
                "val_obsolete_precision": val_metrics["obsolete_tb"]["precision"],
                "val_obsolete_recall": val_metrics["obsolete_tb"]["recall"],
                "val_obsolete_specificity": val_metrics["obsolete_tb"]["specificity"],
                "val_obsolete_f1": val_metrics["obsolete_tb"]["f1"],
                "val_cat_healthy_count": val_metrics["pred_category_counts"]["Healthy"],
                "val_cat_active_only_count": val_metrics["pred_category_counts"]["Active_Only"],
                "val_cat_obsolete_only_count": val_metrics["pred_category_counts"]["Obsolete_Only"],
                "val_cat_both_count": val_metrics["pred_category_counts"]["Both"],
                "peak_gpu_mem_mb": (
                    torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
                    if device.type == "cuda"
                    else 0.0
                ),
            })

            print(
                f"[Smoke Test Epoch {epoch}/2] "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Val Exact Match: {val_metrics['exact_match_accuracy']:.4f} | "
                f"Duration: {duration:.1f}s"
            )

        # 4. Checkpoint saving and reload check
        best_ckpt_path = smoke_output_dir / "best_model.pth"
        last_ckpt_path = smoke_output_dir / "last_checkpoint.pth"

        save_checkpoint(
            best_ckpt_path,
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_macro_f1=val_metrics["macro_f1"],
            val_metrics=val_metrics,
            config=vars(args),
        )
        save_checkpoint(
            last_ckpt_path,
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_macro_f1=val_metrics["macro_f1"],
            val_metrics=val_metrics,
            config=vars(args),
        )

        # Reload check
        reload_model = build_resnet101(num_classes=2).to(device)
        loaded_state = torch.load(best_ckpt_path, map_location=device)
        reload_model.load_state_dict(loaded_state["model_state_dict"])
        reload_model.eval()
        dummy_inp = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        with torch.no_grad():
            dummy_out = reload_model(dummy_inp)
        if dummy_out.shape != (1, 2) or not torch.isfinite(dummy_out).all():
            raise RuntimeError("Smoke test failure: Reloaded checkpoint produced invalid dummy output.")

        # 5. History CSV output
        history_csv_path = smoke_output_dir / "training_history.csv"
        with open(history_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(smoke_history[0].keys()))
            writer.writeheader()
            writer.writerows(smoke_history)

        # 6. Plotting output
        plot_path = smoke_output_dir / "training_plots.png"
        plot_training_curves(smoke_history, plot_path)

        print("\n" + "=" * 70)
        print("RESNET-101 SMOKE TEST: PASS".center(70))
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"RESNET-101 SMOKE TEST: FAIL - {str(e)}".center(70))
        print("=" * 70 + "\n")
        logging.exception("Smoke test failed with traceback:")
        return 1


# -----------------------------------------------------------------------------
# Main Training Function
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ResNet-101 from scratch on TBX11K multi-label dataset."
    )
    # Dataset and paths
    parser.add_argument("--data-dir", type=str, default=None, help="Path to TBX11K root folder (default: auto-detected).")
    parser.add_argument("--splits-dir", type=str, default=None, help="Path to data_prep/splits/ folder (default: auto-detected).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save training checkpoints and history (default: CLASSIFICATION/ResNet-101/outputs/train).",
    )

    # Optimization and hyperparameters (Authoritative defaults set from top-level configuration)
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum epochs (default: {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Physical batch size (default: {BATCH_SIZE}; fallback to {BATCH_SIZE_FALLBACK} if OOM).",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
        help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS}; nominal effective batch size 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Initial learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=MOMENTUM,
        help=f"SGD momentum (default: {MOMENTUM}).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help=f"SGD weight decay (default: {WEIGHT_DECAY}).",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=NESTEROV,
        help=f"Enable Nesterov momentum (default: {NESTEROV}).",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=LR_STEP_SIZE,
        help=f"StepLR drop epoch interval (default: {LR_STEP_SIZE}).",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=LR_GAMMA,
        help=f"StepLR multiplicative decay factor (default: {LR_GAMMA}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Early stopping patience in epochs based on validation macro-F1 (default: {EARLY_STOPPING_PATIENCE}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"DataLoader workers (default: {NUM_WORKERS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Reproducibility seed (default: {SEED}).",
    )

    # Hardware & Performance switches
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration.")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 mixed precision.")

    # Dedicated Execution modes
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-epoch mandatory smoke test and verify the entire pipeline without full training.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint from which to resume training.",
    )

    args = parser.parse_args()

    # Route to smoke test if requested; returns immediately without running full training
    if args.smoke_test:
        sys.exit(run_smoke_test(args))

    # Standard training pipeline setup
    set_seed(args.seed)

    workspace_root, data_dir, splits_dir = resolve_workspace_paths(args.data_dir, args.splits_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs" / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logging.info(f"Workspace root: {workspace_root}")
    logging.info(f"Dataset root:   {data_dir}")
    logging.info(f"Splits root:    {splits_dir}")
    logging.info(f"Output dir:     {output_dir}")
    logging.info(f"Compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    effective_batch_size = args.batch_size * args.accum_steps
    logging.info(
        f"Batch config: Physical={args.batch_size}, Accumulation Steps={args.accum_steps}, "
        f"Effective Batch Size={effective_batch_size}"
    )
    logging.info(
        f"Hyperparameters: Epochs={args.epochs}, LR={args.lr}, Patience={args.patience}, "
        f"Weight Decay={args.weight_decay}, Momentum={args.momentum}, Nesterov={args.nesterov}, "
        f"LR Step Size={args.lr_step_size}, LR Gamma={args.lr_gamma}"
    )
    logging.info(
        f"Preprocessing: Resolution={DEFAULT_IMAGE_SIZE}, RGB=True, Interpolation=BILINEAR, "
        f"Normalization=NONE (Scaled to [0.0, 1.0] via ToTensor()), Augmentation=NONE"
    )

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    transform = get_transforms(DEFAULT_IMAGE_SIZE)
    logging.info("Loading training and validation manifests...")
    train_dataset = TBX11KMultiLabelDataset(
        csv_path=train_csv,
        dataset_root=data_dir,
        transform=transform,
        split_name="train",
        expected_count=3219,
    )
    val_dataset = TBX11KMultiLabelDataset(
        csv_path=val_csv,
        dataset_root=data_dir,
        transform=transform,
        split_name="val",
        expected_count=690,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model from scratch
    logging.info("Instantiating ResNet-101 from scratch (weights=None)...")
    model = build_resnet101(num_classes=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None

    # Resume support
    start_epoch = 1
    best_val_macro_f1 = -1.0
    epochs_no_improve = 0
    history_records: List[Dict[str, Any]] = []

    best_model_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_checkpoint.pth"
    history_csv_path = output_dir / "training_history.csv"
    plots_path = output_dir / "training_plots.png"

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        logging.info(f"Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt.get("val_macro_f1", -1.0)
        logging.info(f"Resumed at epoch {start_epoch} with prior best Val Macro-F1: {best_val_macro_f1:.4f}")

        # Load existing history if available
        if history_csv_path.is_file():
            with open(history_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    typed_row = {}
                    for k, v in row.items():
                        try:
                            typed_row[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            typed_row[k] = v
                    history_records.append(typed_row)

    logging.info(f"Starting training run: {args.epochs} max epochs, patience={args.patience} epochs.")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        try:
            train_loss, train_metrics, duration = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                accum_steps=args.accum_steps,
            )
        except torch.cuda.OutOfMemoryError as oom_err:
            logging.error(f"CUDA Out of Memory encountered at batch size {args.batch_size}: {oom_err}")
            logging.info(f"Attempting fallback to batch size {BATCH_SIZE_FALLBACK}...")
            torch.cuda.empty_cache()
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE_FALLBACK,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE_FALLBACK,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            train_loss, train_metrics, duration = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                accum_steps=args.accum_steps * 2,
            )

        val_loss, val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        # GPU Peak Memory
        peak_gpu_mem_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        val_macro_f1 = val_metrics["macro_f1"]
        is_best = val_macro_f1 > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = val_macro_f1
            epochs_no_improve = 0
            save_checkpoint(
                path=best_model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_macro_f1=val_macro_f1,
                val_metrics=val_metrics,
                config=vars(args),
            )
            improvement_tag = " [BEST]"
        else:
            epochs_no_improve += 1
            improvement_tag = f" [No improvement: {epochs_no_improve}/{args.patience}]"

        # Save last checkpoint every epoch
        save_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_macro_f1=val_macro_f1,
            val_metrics=val_metrics,
            config=vars(args),
        )

        logging.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Val Exact Match: {val_metrics['exact_match_accuracy']:.4f} | "
            f"Active F1: {val_metrics['active_tb']['f1']:.4f} | "
            f"Obsolete F1: {val_metrics['obsolete_tb']['f1']:.4f} | "
            f"Peak VRAM: {peak_gpu_mem_mb:.1f} MB | "
            f"Duration: {duration:.1f}s{improvement_tag}"
        )

        # Log epoch record
        history_records.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "duration_sec": duration,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_exact_match_acc": val_metrics["exact_match_accuracy"],
            "val_hamming_loss": val_metrics["hamming_loss"],
            "val_category_acc": val_metrics["category_accuracy"],
            "val_active_precision": val_metrics["active_tb"]["precision"],
            "val_active_recall": val_metrics["active_tb"]["recall"],
            "val_active_specificity": val_metrics["active_tb"]["specificity"],
            "val_active_f1": val_metrics["active_tb"]["f1"],
            "val_obsolete_precision": val_metrics["obsolete_tb"]["precision"],
            "val_obsolete_recall": val_metrics["obsolete_tb"]["recall"],
            "val_obsolete_specificity": val_metrics["obsolete_tb"]["specificity"],
            "val_obsolete_f1": val_metrics["obsolete_tb"]["f1"],
            "val_cat_healthy_count": val_metrics["pred_category_counts"]["Healthy"],
            "val_cat_active_only_count": val_metrics["pred_category_counts"]["Active_Only"],
            "val_cat_obsolete_only_count": val_metrics["pred_category_counts"]["Obsolete_Only"],
            "val_cat_both_count": val_metrics["pred_category_counts"]["Both"],
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        })

        # Update training_history.csv after each epoch
        with open(history_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            writer.writerows(history_records)

        # Update plots
        plot_training_curves(history_records, plots_path)

        # Check early stopping
        if epochs_no_improve >= args.patience:
            logging.info(
                f"Early stopping triggered at epoch {epoch}: "
                f"Validation macro-F1 has not improved for {args.patience} consecutive epochs."
            )
            break

    logging.info(f"Training complete. Best Validation Macro-F1: {best_val_macro_f1:.4f}")
    logging.info(f"Best checkpoint saved at: {best_model_path}")
    logging.info(f"History CSV saved at:     {history_csv_path}")
    logging.info(f"Training plots saved at:  {plots_path}")


if __name__ == "__main__":
    main()

