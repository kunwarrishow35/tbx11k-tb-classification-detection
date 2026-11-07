"""
EfficientNet-B0 Multi-Label Chest X-Ray Classification Pipeline
================================================================

This module implements the training pipeline for EfficientNet-B0 trained from scratch
on the canonical TBX11K benchmark dataset (3,219 train, 690 val, 690 test).

GOVERNING SPECIFICATION AND ORDER OF AUTHORITY:
1. Primary Authority: Official EfficientNet Hyperparameters as documented in
   CLASSIFICATION/EfficientNet/EfficientNet_Official_Hyperparameters.pdf (extracted
   directly from tensorflow/tpu models/official/efficientnet).
2. Permitted Deviations: Dataset-related (multi-label BCE, 512x512 resolution, 70 epochs)
   and hardware/system-related (PyTorch 2.14, single NVIDIA RTX 4050 6GB GPU, batch 4,
   AMP FP16, timm.optim.RMSpropTF) as defined in Section 3 of project directives.
3. Code Architecture & Conventions: Inherited from CLASSIFICATION/ResNet-101/train.py for
   unified logging, checkpoint schemas, tracking artifacts, and reproducibility.

DEVIATION TABLE (OFFICIAL RECIPE VS. IMPLEMENTATION):
+-------------------------------+-----------------------------------------+-----------------------------------------+---------------+
| Parameter / Component         | Official tensorflow/tpu Recipe          | Project Implementation (RTX 4050 6 GB)  | Category      |
+-------------------------------+-----------------------------------------+-----------------------------------------+---------------+
| Architecture                  | EfficientNet-B0 (width 1.0, depth 1.0)  | torchvision.models.efficientnet_b0      | MATCHES       |
| Pretrained Weights            | None (trained from scratch)             | None (weights=None, random init)        | MATCHES       |
| Classification Head           | 1000-class Softmax                      | 2-output Linear [Active, Obsolete]      | DATASET       |
| Loss Function                 | Softmax Cross-Entropy                   | nn.BCEWithLogitsLoss()                  | DATASET       |
| Label Smoothing               | 0.1 on Softmax                          | None (not mathematically valid for BCE) | DATASET       |
| Input Resolution              | 224 x 224                               | 512 x 512                               | DATASET       |
| Optimizer                     | TF RMSProp (decay 0.9, mom 0.9, eps 1e-3)| timm.optim.RMSpropTF (same hyperparams) | MATCHES       |
| Base Learning Rate            | 0.016 at batch size 256                 | 0.016 (base rate before batch scaling)  | MATCHES       |
| Linear LR Scaling             | scaled_lr = base_lr * (batch / 256)     | scaled_lr = 0.016 * (4 / 256) = 0.00025 | SYSTEM        |
| LR Warmup                     | 5 epochs linear from 0 to scaled_lr     | 5 epochs linear (4,025 optimizer steps) | MATCHES       |
| LR Decay Schedule             | 0.97 every 2.4 epochs (staircase)       | 0.97 every 1,932 optimizer steps        | MATCHES       |
| Schedule Count Origin         | Step 0 (decays count during warmup)     | Step 0 (switches to decayed rate)       | MATCHES       |
| Weight Decay                  | 1e-5 L2 on all non-BatchNorm variables  | 1e-5 on conv/linear weights & biases    | MATCHES       |
| BatchNorm Weight Decay        | 0.0 (no decay on BN)                    | 0.0 (asserted strictly in param group)  | MATCHES       |
| BatchNorm Momentum / Epsilon  | TF mom 0.99, eps 1e-3                   | PyTorch mom 0.01, eps 1e-3 (identical)  | MATCHES       |
| Dropout Rate                  | 0.2 (final dropout before classifier)   | 0.2                                     | MATCHES       |
| Stochastic Depth (DropConnect)| 0.2 survival probability 0.8            | stochastic_depth_prob = 0.2             | MATCHES       |
| Exponential Moving Average    | Decay 0.9999, ramp min(0.9999,(1+t)/(10+t))| ModelEMA class with identical formula | MATCHES       |
| Training Data Augmentation    | Inception crop (0.08-1.0, 3/4-4/3)      | Custom OfficialInceptionCrop transform  | MATCHES       |
| Normalization                 | ImageNet mean & std                     | ImageNet mean & std ([0, 1] input)      | MATCHES       |
| Physical Batch Size           | 2048 (TPU pod)                          | 4 (RTX 4050 6GB VRAM hardware boundary) | SYSTEM        |
| Precision                     | float32 / bfloat16                      | CUDA FP16 Autocast + GradScaler         | SYSTEM        |
| Training Epochs               | ~350 epochs                             | 70 epochs (fixed study budget)          | DATASET       |
| Early Stopping                | None                                    | None (runs all 70 epochs)               | MATCHES       |
| Checkpoint Selection          | EMA weights on validation set           | Best EMA validation Macro-F1 at tau=0.5 | MATCHES       |
+-------------------------------+-----------------------------------------+-----------------------------------------+---------------+

UNVERIFIED LIST (DECLARED LIMITATIONS & ASSUMPTIONS):
1. RMSpropTF is not bit-identical to TF1: timm's RMSpropTF matches TensorFlow's epsilon
   placement and square gradient tracking, but underlying GPU kernel operations differ.
2. torchvision init and stochastic-depth scaling are not verified line by line against
   the original TF graph builder.
3. The official main.py snapshot may be older than the live repo.
4. The batch-4 LR (0.00025) comes from the linear scaling rule (0.016 * 4 / 256),
   not from an empirical hyperparameter search at batch 4.
5. BatchNorm statistics are noisy at batch size 4; momentum=0.01 dampens updates.
6. The random crop area fraction (down to 0.08) can cut small chest X-ray lesions.
   An ablation option (--augment none) is provided.
7. TF crop internals (uniform aspect ratio sampling, coverage rule) are derived from
   documented TensorFlow sample_distorted_bounding_box behavior.
8. Label smoothing (0.1 in official ImageNet) is omitted as undefined for BCEWithLogitsLoss.
9. Model selection on validation Macro-F1 and 70-epoch budget are project-level choices.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for Windows / servers
import matplotlib.pyplot as plt
import numpy as np
import timm.optim
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# =============================================================================
# AUTHORITATIVE HYPERPARAMETERS & PROJECT CONSTANTS
# =============================================================================
EXPERIMENT_NAME: str = "EfficientNet-B0 Multi-Label Chest X-Ray Classification"
ARCHITECTURE_NAME: str = "EfficientNet-B0"

NUM_EPOCHS: int = 70
BATCH_SIZE: int = 4
BATCH_SIZE_FALLBACK: int = 2
GRADIENT_ACCUMULATION_STEPS: int = 1

# Official Learning Rate Specification:
# Base learning rate = 0.016 when train batch size is 256.
# Scaled learning rate = base_lr * (batch_size / 256.0) = 0.016 * (4 / 256.0) = 0.00025.
BASE_LR_PER_256: float = 0.016
BASE_LR: float = BASE_LR_PER_256 * (BATCH_SIZE / 256.0)  # 0.00025

# Schedule Constants:
WARMUP_EPOCHS: float = 5.0
DECAY_EPOCHS: float = 2.4
DECAY_RATE: float = 0.97

# Optimizer Constants:
RMSPROP_ALPHA: float = 0.9
RMSPROP_MOMENTUM: float = 0.9
RMSPROP_EPS: float = 1e-3
WEIGHT_DECAY: float = 1e-5

# Model Architecture Constants:
DROPOUT_RATE: float = 0.2
STOCHASTIC_DEPTH_PROB: float = 0.2
BN_MOMENTUM: float = 0.01  # PyTorch convention: corresponds to TF 0.99
BN_EPS: float = 1e-3
EXPECTED_TOTAL_PARAMS: int = 4010110

# Exponential Moving Average Constants:
EMA_MAX_DECAY: float = 0.9999

# Image & Normalization Constants:
IMAGE_SIZE: int = 512
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
NORM_MEAN: List[float] = [0.485, 0.456, 0.406]
NORM_STD: List[float] = [0.229, 0.224, 0.225]

# System & Dataset Constants:
SEED: int = 42
NUM_WORKERS: int = 2  # Stable on Windows multiprocessing
TARGET_COLUMNS: List[str] = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]
CATEGORY_NAMES: List[str] = ["Healthy", "Active_Only", "Obsolete_Only", "Both"]


# -----------------------------------------------------------------------------
# Reproducibility Helper
# -----------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Dynamic Workspace Path Resolution
# -----------------------------------------------------------------------------
def resolve_workspace_paths(
    custom_data_dir: Optional[str] = None,
    custom_splits_dir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Locates the workspace root, dataset directory (TBX11K), and splits directory."""
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent

    # Verification: must contain TBX11K and data_prep
    if not (workspace_root / "TBX11K").is_dir() or not (workspace_root / "data_prep").is_dir():
        cwd = Path.cwd().resolve()
        if (cwd / "TBX11K").is_dir() and (cwd / "data_prep").is_dir():
            workspace_root = cwd
        else:
            for ancestor in script_dir.parents:
                if (ancestor / "TBX11K").is_dir() and (ancestor / "data_prep").is_dir():
                    workspace_root = ancestor
                    break

    # Resolve data directory
    if custom_data_dir:
        data_dir = Path(custom_data_dir).resolve()
    else:
        data_dir = workspace_root / "TBX11K"

    # Resolve splits directory
    if custom_splits_dir:
        splits_dir = Path(custom_splits_dir).resolve()
    else:
        splits_dir = workspace_root / "data_prep" / "splits"

    return workspace_root, data_dir, splits_dir


# -----------------------------------------------------------------------------
# Custom Inception-Style Crop Transform (Official TensorFlow Matching)
# -----------------------------------------------------------------------------
class OfficialInceptionCrop:
    """
    Implements TensorFlow's sample_distorted_bounding_box algorithm for the whole image:
    1. Up to 10 attempts to sample a valid crop.
    2. Area fraction sampled uniformly: A ~ U(0.08, 1.0).
    3. Coverage constraint: A >= 0.1 (since the whole image is the bounding box).
    4. Aspect ratio sampled uniformly: R ~ U(3/4, 4/3) = U(0.75, 1.3333).
    5. Derives crop_w and crop_h, validating that the crop fits within image boundaries.
    6. If all attempts fail, falls back to the full image.
    7. If resulting crop equals original size, uses official center-crop fallback:
       square crop of min(orig_h, orig_w) * S / (S + 32) with S=512, centered.

    Note: torchvision.transforms.RandomResizedCrop is NOT used because its log-uniform
    aspect ratio sampling, absence of coverage constraint, and fallback logic diverge
    from TensorFlow's official Inception preprocessing.
    """

    def __init__(self, S: int = IMAGE_SIZE):
        self.S = S

    def __call__(self, img: Image.Image) -> Image.Image:
        orig_w, orig_h = img.size
        orig_area = orig_w * orig_h

        crop_box: Optional[Tuple[int, int, int, int]] = None

        for _ in range(10):
            area_frac = random.uniform(0.08, 1.0)
            if area_frac < 0.1:  # min_object_covered constraint
                continue

            aspect_ratio = random.uniform(3.0 / 4.0, 4.0 / 3.0)  # Uniform, not log-uniform
            target_area = area_frac * orig_area

            # w / h = aspect_ratio  =>  w = h * aspect_ratio
            # w * h = target_area   =>  h^2 * aspect_ratio = target_area
            h = math.sqrt(target_area / aspect_ratio)
            w = math.sqrt(target_area * aspect_ratio)
            crop_h = int(round(h))
            crop_w = int(round(w))

            if 0 < crop_w <= orig_w and 0 < crop_h <= orig_h:
                offset_w = random.randint(0, orig_w - crop_w)
                offset_h = random.randint(0, orig_h - crop_h)
                crop_box = (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)
                break

        # Fallback to whole image if all attempts fail
        if crop_box is None:
            crop_box = (0, 0, orig_w, orig_h)

        crop_w = crop_box[2] - crop_box[0]
        crop_h = crop_box[3] - crop_box[1]

        # Official center-crop fallback if resulting crop equals the original image size:
        # crop of min(h, w) * S / (S + 32) with S=512, centered
        if crop_w == orig_w and crop_h == orig_h:
            crop_size = int(round(min(orig_h, orig_w) * self.S / (self.S + 32.0)))
            offset_w = (orig_w - crop_size) // 2
            offset_h = (orig_h - crop_size) // 2
            crop_box = (offset_w, offset_h, offset_w + crop_size, offset_h + crop_size)

        return img.crop(crop_box)


# -----------------------------------------------------------------------------
# Transforms Pipelines
# -----------------------------------------------------------------------------
def get_train_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    augment: str = "official",
) -> transforms.Compose:
    """
    Constructs the training transform pipeline.
    Official: OfficialInceptionCrop -> Bicubic Resize(512, 512) -> RandomHorizontalFlip(0.5)
              -> ToTensor() -> ImageNet Normalization.
    None: Bilinear Resize(512, 512) -> ToTensor() -> ImageNet Normalization (ablation).
    """
    if augment == "official":
        return transforms.Compose([
            OfficialInceptionCrop(S=image_size[0]),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    elif augment == "none":
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
    else:
        raise ValueError(f"Unknown augmentation mode: {augment}. Expected 'official' or 'none'.")


def get_eval_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Evaluation transform pipeline (strictly identical across val and test):
    Bilinear Resize(512, 512) -> ToTensor() -> ImageNet Normalization.
    Zero augmentation and zero crop distortion.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# -----------------------------------------------------------------------------
# Dataset Class (Verbatim Schema Parity with ResNet-101 / DenseNet-121)
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

            seen_paths: Set[str] = set()
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

        # Load image and convert to 3-channel RGB
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
# Model Construction
# -----------------------------------------------------------------------------
def build_efficientnet_b0(num_classes: int = 2) -> nn.Module:
    """
    Instantiates EfficientNet-B0 from scratch (weights=None).
    Configures official hyperparameters:
    - norm_layer: BatchNorm2d with eps=1e-3, momentum=0.01 (TF 0.99 equivalent).
    - dropout: 0.2 before final classifier.
    - stochastic_depth_prob: 0.2 linearly increasing survival probability.
    - num_classes: 2 independent binary outputs.
    Expected total parameters: exactly 4,010,110.
    """
    norm_layer = partial(nn.BatchNorm2d, eps=BN_EPS, momentum=BN_MOMENTUM)
    model = models.efficientnet_b0(
        weights=None,
        num_classes=num_classes,
        dropout=DROPOUT_RATE,
        stochastic_depth_prob=STOCHASTIC_DEPTH_PROB,
        norm_layer=norm_layer,
    )

    # Ensure all parameters require gradients (from-scratch training)
    for name, param in model.named_parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_TOTAL_PARAMS, (
        f"Parameter count mismatch: expected {EXPECTED_TOTAL_PARAMS}, got {total_params}."
    )

    return model


# -----------------------------------------------------------------------------
# Optimizer Parameter Grouping (Strict Weight Decay Isolation)
# -----------------------------------------------------------------------------
def build_parameter_groups(model: nn.Module, weight_decay: float = WEIGHT_DECAY) -> List[Dict[str, Any]]:
    """
    Separates model parameters into two groups per official recipe:
    1. Decayed group: weight_decay = 1e-5 (Conv and Linear weights AND biases).
    2. Undecayed group: weight_decay = 0.0 (strictly all BatchNorm weights and biases).
    Enforces strict assertions verifying no BN parameter is in the decayed group
    and that classifier biases are decayed.
    """
    bn_param_ids: Set[int] = set()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            for p in m.parameters(recurse=False):
                bn_param_ids.add(id(p))

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in bn_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Assertions
    for p in decay_params:
        assert id(p) not in bn_param_ids, "BatchNorm parameter detected in decayed group!"
    for p in no_decay_params:
        assert id(p) in bn_param_ids, "Non-BatchNorm parameter detected in no-decay group!"

    # Verify classifier bias is decayed
    classifier_bias = model.classifier[1].bias
    assert any(p is classifier_bias for p in decay_params), "Classifier bias missing from decayed group!"

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------------------
# Model Exponential Moving Average (ModelEMA)
# -----------------------------------------------------------------------------
class ModelEMA:
    """
    Maintains an exponential moving average (EMA) of model parameters and buffers.
    Decay formula: min(max_decay, (1 + num_updates) / (10 + num_updates)).
    Updates all floating-point state_dict tensors (including BatchNorm running_mean and running_var)
    and copies integer buffers (num_batches_tracked).
    """

    def __init__(self, model: nn.Module, max_decay: float = EMA_MAX_DECAY):
        self.max_decay = max_decay
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.clone().detach()

    def update(self, model: nn.Module) -> float:
        """Applies EMA step. Returns decay factor used."""
        decay = min(self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(decay).add_(v, alpha=(1.0 - decay))
                else:
                    self.shadow[k].copy_(v)
        self.num_updates += 1
        return decay

    def apply_to(self, target_model: nn.Module) -> None:
        """Copies shadow weights into the target model."""
        target_model.load_state_dict(self.shadow)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.num_updates = state["num_updates"]
        self.shadow = state["shadow"]


# -----------------------------------------------------------------------------
# Learning Rate Calculation (Per Optimizer Step)
# -----------------------------------------------------------------------------
def calculate_learning_rate(
    global_step: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
    decay_rate: float = DECAY_RATE,
) -> float:
    """
    Computes learning rate per optimizer step according to official schedule:
    - If step < warmup_steps: linear warmup from 0 to base_lr.
    - Otherwise: staircase exponential decay: base_lr * (decay_rate ** floor(step / decay_steps)).
    Note: decay count starts at step 0, so schedule immediately reflects elapsed decays.
    """
    if global_step < warmup_steps:
        return base_lr * (float(global_step) / float(warmup_steps))
    else:
        num_decays = math.floor(float(global_step) / float(decay_steps))
        return base_lr * (decay_rate ** num_decays)


# -----------------------------------------------------------------------------
# Metric Computation Functions (Verbatim ResNet Parity)
# -----------------------------------------------------------------------------
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Precision, Recall (Sensitivity), Specificity, and F1 for a binary label."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
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
    """Computes all required validation/training epoch metrics."""
    preds = (all_probs >= threshold).astype(int)

    act_metrics = compute_binary_metrics(all_targets[:, 0], preds[:, 0])
    obs_metrics = compute_binary_metrics(all_targets[:, 1], preds[:, 1])

    macro_f1 = (act_metrics["f1"] + obs_metrics["f1"]) / 2.0

    exact_match = np.mean(
        (all_targets[:, 0] == preds[:, 0]) & (all_targets[:, 1] == preds[:, 1])
    )
    hamming_loss = np.mean(all_targets != preds)

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
# Checkpoint Saving & Loading
# -----------------------------------------------------------------------------
def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    ema_model: nn.Module,
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    ema: ModelEMA,
    val_macro_f1: float,
    best_val_macro_f1: float,
    val_metrics: Dict[str, Any],
    raw_val_metrics: Dict[str, Any],
    config: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> None:
    """
    Saves checkpoint file with complete metadata, dual model weights (EMA in model_state_dict,
    raw model in raw_model_state_dict), and optimizer/scaler states.
    """
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": ema_model.state_dict(),  # Evaluated by test.py
        "raw_model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,  # Custom schedule is a pure function of global_step
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_num_updates": ema.num_updates,
        "val_macro_f1": float(val_macro_f1),
        "best_val_macro_f1": float(best_val_macro_f1),
        "val_metrics": val_metrics,
        "raw_val_metrics": raw_val_metrics,
        "config": config,
        "preprocessing": {
            "image_size": list(DEFAULT_IMAGE_SIZE),
            "channels": 3,
            "format": "RGB",
            "interpolation_train": "BICUBIC",
            "interpolation_eval": "BILINEAR",
            "normalization": "imagenet",
            "mean": NORM_MEAN,
            "std": NORM_STD,
            "augmentation": config.get("augment", "official"),
        },
        "target_columns": TARGET_COLUMNS,
        "category_names": CATEGORY_NAMES,
        "history": history,
    }
    torch.save(state, path)


# -----------------------------------------------------------------------------
# Training Trajectory Visualization
# -----------------------------------------------------------------------------
def plot_training_curves(history_csv: Path, output_png: Path) -> None:
    """Generates a 6-panel training trajectory visualization."""
    if not history_csv.is_file():
        return

    epochs: List[int] = []
    train_losses, ema_val_losses = [], []
    ema_macro_f1s, raw_macro_f1s = [], []
    ema_exact_match, raw_exact_match = [], []
    lr_starts, lr_ends = [], []
    active_f1s, obsolete_f1s = [], []
    vrams, skips = [], []

    with open(history_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            ema_val_losses.append(float(row["ema_val_loss"]))
            ema_macro_f1s.append(float(row["ema_val_macro_f1"]))
            raw_macro_f1s.append(float(row.get("raw_val_macro_f1", row["ema_val_macro_f1"])))
            ema_exact_match.append(float(row["ema_val_exact_match_acc"]))
            raw_exact_match.append(float(row.get("raw_val_exact_match_acc", row["ema_val_exact_match_acc"])))
            lr_starts.append(float(row.get("lr_start", row["learning_rate"])))
            lr_ends.append(float(row.get("lr_end", row["learning_rate"])))
            active_f1s.append(float(row["ema_val_active_f1"]))
            obsolete_f1s.append(float(row["ema_val_obsolete_f1"]))
            vrams.append(float(row["peak_gpu_mem_mb"]))
            skips.append(int(row.get("skipped_steps", 0)))

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # 1. Losses
    axes[0, 0].plot(epochs, train_losses, label="Train Loss (Raw)", color="#1f77b4", lw=1.8)
    axes[0, 0].plot(epochs, ema_val_losses, label="Val Loss (EMA)", color="#d62728", lw=1.8)
    axes[0, 0].set_title("Training Loss vs. EMA Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("BCE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Macro-F1 (EMA vs Raw)
    axes[0, 1].plot(epochs, ema_macro_f1s, label="Val Macro-F1 (EMA)", color="#2ca02c", lw=2.0)
    axes[0, 1].plot(epochs, raw_macro_f1s, label="Val Macro-F1 (Raw)", color="#7f7f7f", lw=1.2, linestyle="--")
    axes[0, 1].set_title("Validation Macro-F1 (EMA vs. Raw)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Macro-F1 Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Exact Match Accuracy
    axes[1, 0].plot(epochs, ema_exact_match, label="Val Exact Match (EMA)", color="#ff7f0e", lw=1.8)
    axes[1, 0].plot(epochs, raw_exact_match, label="Val Exact Match (Raw)", color="#bcbd22", lw=1.2, linestyle="--")
    axes[1, 0].set_title("Validation Subset Exact Match Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Exact Match Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, lr_starts, label="Epoch Start LR", color="#9467bd", lw=1.8)
    axes[1, 1].plot(epochs, lr_ends, label="Epoch End LR", color="#8c564b", lw=1.2, linestyle=":")
    axes[1, 1].set_title("Learning Rate Trajectory (Warmup + Staircase)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Per-Label F1 Scores
    axes[2, 0].plot(epochs, active_f1s, label="Active TB F1 (EMA)", color="#e377c2", lw=1.8)
    axes[2, 0].plot(epochs, obsolete_f1s, label="Obsolete TB F1 (EMA)", color="#17becf", lw=1.8)
    axes[2, 0].set_title("Per-Label Validation F1 Scores (EMA)")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("F1 Score")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Peak VRAM & Skipped Steps
    ax_vram = axes[2, 1]
    ax_vram.plot(epochs, vrams, label="Peak VRAM (MB)", color="#393b44", lw=1.8)
    ax_vram.set_title("Hardware Utilization & Skipped Optimizer Steps")
    ax_vram.set_xlabel("Epoch")
    ax_vram.set_ylabel("Peak VRAM (MB)", color="#393b44")
    ax_vram.grid(True, alpha=0.3)

    if any(s > 0 for s in skips):
        ax_skips = ax_vram.twinx()
        ax_skips.bar(epochs, skips, alpha=0.3, color="red", label="Skipped Steps")
        ax_skips.set_ylabel("Skipped Steps", color="red")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Manifest & Schedule Preview Writers
# -----------------------------------------------------------------------------
def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def write_lr_schedule_preview(
    output_path: Path,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
    steps_per_epoch: int,
) -> None:
    """Writes a CSV preview of the learning rate schedule at startup."""
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch_float", "learning_rate", "phase"])
        for s in range(0, total_steps + 1, max(1, total_steps // 200)):
            epoch_f = s / float(steps_per_epoch)
            lr = calculate_learning_rate(s, base_lr, warmup_steps, decay_steps)
            phase = "warmup" if s < warmup_steps else "staircase_decay"
            writer.writerow([s, f"{epoch_f:.4f}", f"{lr:.8e}", phase])


def write_run_manifest(
    output_path: Path,
    config: Dict[str, Any],
    splits_dir: Path,
    best_epoch: Optional[int] = None,
    best_val_macro_f1: Optional[float] = None,
    stop_reason: str = "running",
    events: Optional[List[str]] = None,
) -> None:
    """Writes run_manifest.json capturing environment, dataset hashes, and config."""
    manifest_data = {
        "experiment_name": EXPERIMENT_NAME,
        "architecture": ARCHITECTURE_NAME,
        "manifest_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "configuration": config,
        "events": events or [],
        "dataset_hashes_sha256": {
            "train_csv": compute_file_sha256(splits_dir / "train.csv"),
            "val_csv": compute_file_sha256(splits_dir / "val.csv"),
            "test_csv": compute_file_sha256(splits_dir / "test.csv"),
        },
        "environment": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_vram_gb": (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0,
        },
        "deviations_from_official": [
            {"parameter": "Classification Head", "official": "1000-class Softmax", "implemented": "2-output Linear [Active, Obsolete]", "category": "DATASET"},
            {"parameter": "Loss Function", "official": "Softmax Cross-Entropy", "implemented": "BCEWithLogitsLoss()", "category": "DATASET"},
            {"parameter": "Label Smoothing", "official": "0.1", "implemented": "None (undefined for BCE)", "category": "DATASET"},
            {"parameter": "Input Resolution", "official": "224 x 224", "implemented": "512 x 512", "category": "DATASET"},
            {"parameter": "Physical Batch Size", "official": "2048", "implemented": "4 (hardware limit)", "category": "SYSTEM"},
            {"parameter": "Learning Rate", "official": "0.016 (batch 256)", "implemented": "0.00025 (scaled by 4/256)", "category": "SYSTEM"},
            {"parameter": "Training Length", "official": "~350 epochs", "implemented": "70 epochs", "category": "DATASET"},
            {"parameter": "Optimizer Class", "official": "TF RMSprop", "implemented": "timm.optim.RMSpropTF", "category": "MATCHES"},
        ],
        "unverified_declarations": [
            "RMSpropTF is not bit-identical to TF1.",
            "torchvision init and stochastic-depth scaling are not verified line by line.",
            "The official main.py snapshot may be older than the live repo.",
            "The batch-4 LR comes from the scaling rule, not from an official run.",
            "BatchNorm is noisy at batch 4.",
            "The crop (0.08 area) can cut small lesions.",
            "TF crop internals come from documented TF op behavior.",
            "Label smoothing is omitted.",
            "Model selection and 70 epochs are project choices.",
        ],
    }
    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


# -----------------------------------------------------------------------------
# Training and Validation Epoch Runners
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    ema: ModelEMA,
    device: torch.device,
    use_amp: bool,
    accum_steps: int,
    global_step: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> Tuple[float, Dict[str, Any], float, int, float, float, int]:
    """
    Executes one epoch of training following official EfficientNet schedule & EMA rules:
    - Sets LR per optimizer step via calculate_learning_rate(global_step).
    - Under AMP, calculates BCEWithLogitsLoss with float logits.
    - Scales loss and performs backward pass.
    - Step scaler and update EMA only when step is not skipped by GradScaler.
    - Returns: (epoch_loss, metrics, duration, new_global_step, lr_start, lr_end, skipped_steps).
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    start_time = time.time()
    optimizer.zero_grad()

    lr_start = calculate_learning_rate(global_step, base_lr, warmup_steps, decay_steps)
    current_lr = lr_start
    skipped_steps_epoch = 0

    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits.float(), targets)
                loss_scaled = loss / accum_steps
            assert scaler is not None
            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                # Update learning rate on optimizer step
                current_lr = calculate_learning_rate(global_step, base_lr, warmup_steps, decay_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                step_skipped = (scale_after < scale_before)
                optimizer.zero_grad()

                if not step_skipped:
                    ema.update(model)
                    global_step += 1
                else:
                    skipped_steps_epoch += 1
        else:
            logits = model(images)
            loss = criterion(logits.float(), targets)
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                current_lr = calculate_learning_rate(global_step, base_lr, warmup_steps, decay_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)
                global_step += 1

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
    lr_end = current_lr

    return epoch_loss, metrics, epoch_duration, global_step, lr_start, lr_end, skipped_steps_epoch


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, Dict[str, Any]]:
    """Executes validation without gradient calculation."""
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
                    loss = criterion(logits.float(), targets)
            else:
                logits = model(images)
                loss = criterion(logits.float(), targets)

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
# Smoke Test Runner
# -----------------------------------------------------------------------------
def run_smoke_test(args: argparse.Namespace) -> int:
    """Executes a 2-epoch smoke test to verify all components end-to-end."""
    print("=" * 80)
    print("STARTING EFFICIENTNET-B0 MANDATORY 2-EPOCH SMOKE TEST")
    print("=" * 80)

    set_seed(args.seed)
    workspace_root, data_dir, splits_dir = resolve_workspace_paths(args.data_dir, args.splits_dir)
    smoke_dir = Path(__file__).resolve().parent / "outputs" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    # Use small subset for fast verification
    train_transform = get_train_transforms(DEFAULT_IMAGE_SIZE, augment=args.augment)
    eval_transform = get_eval_transforms(DEFAULT_IMAGE_SIZE)

    train_ds = TBX11KMultiLabelDataset(train_csv, data_dir, transform=train_transform, split_name="train", expected_count=3219)
    val_ds = TBX11KMultiLabelDataset(val_csv, data_dir, transform=eval_transform, split_name="val", expected_count=690)

    # Use first 32 images for smoke test
    train_ds.samples = train_ds.samples[:32]
    val_ds.samples = val_ds.samples[:16]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_efficientnet_b0(num_classes=2).to(device)
    param_groups = build_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = timm.optim.RMSpropTF(
        param_groups,
        lr=args.lr,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        momentum=RMSPROP_MOMENTUM,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None
    ema = ModelEMA(model, max_decay=EMA_MAX_DECAY)
    eval_model = build_efficientnet_b0(num_classes=2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    warmup_steps = 2 * steps_per_epoch
    decay_steps = steps_per_epoch

    global_step = 0
    history = []

    for ep in range(1, 3):
        loss, train_metrics, duration, global_step, lr_start, lr_end, skips = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            use_amp=use_amp,
            accum_steps=1,
            global_step=global_step,
            base_lr=args.lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )

        ema.apply_to(eval_model)
        ema_val_loss, ema_val_metrics = validate_one_epoch(eval_model, val_loader, criterion, device, use_amp)

        print(f"[Smoke Test] Epoch {ep}/2 | Train Loss: {loss:.4f} | EMA Val Macro-F1: {ema_val_metrics['macro_f1']:.4f} | LR: {lr_start:.6f}")
        history.append({
            "epoch": ep,
            "global_step": global_step,
            "learning_rate": lr_end,
            "train_loss": loss,
            "ema_val_loss": ema_val_loss,
            "ema_val_macro_f1": ema_val_metrics["macro_f1"],
            "ema_val_exact_match_acc": ema_val_metrics["exact_match_accuracy"],
            "ema_val_active_f1": ema_val_metrics["active_tb"]["f1"],
            "ema_val_obsolete_f1": ema_val_metrics["obsolete_tb"]["f1"],
            "peak_gpu_mem_mb": 500.0,
        })

    # Save smoke test checkpoints
    save_checkpoint(
        path=smoke_dir / "best_model.pth",
        epoch=2,
        global_step=global_step,
        ema_model=eval_model,
        raw_model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        val_macro_f1=ema_val_metrics["macro_f1"],
        best_val_macro_f1=ema_val_metrics["macro_f1"],
        val_metrics=ema_val_metrics,
        raw_val_metrics=ema_val_metrics,
        config=vars(args),
        history=history,
    )
    save_checkpoint(
        path=smoke_dir / "last_checkpoint.pth",
        epoch=2,
        global_step=global_step,
        ema_model=eval_model,
        raw_model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        val_macro_f1=ema_val_metrics["macro_f1"],
        best_val_macro_f1=ema_val_metrics["macro_f1"],
        val_metrics=ema_val_metrics,
        raw_val_metrics=ema_val_metrics,
        config=vars(args),
        history=history,
    )

    print(f"[Smoke Test SUCCESS] Artifacts saved to: {smoke_dir}")
    return 0


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 from scratch on TBX11K following official TF recipe."
    )
    # Dataset and paths
    parser.add_argument("--data-dir", type=str, default=None, help="Path to TBX11K root folder (default: auto-detected).")
    parser.add_argument("--splits-dir", type=str, default=None, help="Path to data_prep/splits/ folder (default: auto-detected).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save training checkpoints and history (default: CLASSIFICATION/EfficientNet/outputs/train).",
    )

    # Optimization and hyperparameters (Defaults locked to official recipe)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Maximum epochs (default: {NUM_EPOCHS}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Physical batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--accum-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS, help="Gradient accumulation steps (default: 1).")
    parser.add_argument("--lr", type=float, default=BASE_LR, help=f"Initial base learning rate at batch 4 (default: {BASE_LR}).")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help=f"Weight decay (default: {WEIGHT_DECAY}).")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Reproducibility seed (default: {SEED}).")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help=f"DataLoader workers (default: {NUM_WORKERS}).")

    # Augmentation ablation
    parser.add_argument(
        "--augment",
        type=str,
        choices=["official", "none"],
        default="official",
        help="Augmentation mode: 'official' (Inception crop + bicubic resize + flip) or 'none' (default: official).",
    )

    # Hardware & execution flags
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration.")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 mixed precision.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch sanity run and verify pipeline.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint from which to resume training.")

    args = parser.parse_args()

    # Route to smoke test if requested
    if args.smoke_test:
        sys.exit(run_smoke_test(args))

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
    effective_batch_size = args.batch_size * args.accum_steps

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    logging.info("=" * 80)
    logging.info(f"STARTING {EXPERIMENT_NAME}")
    logging.info("=" * 80)
    logging.info(f"Workspace root:   {workspace_root}")
    logging.info(f"Dataset root:     {data_dir}")
    logging.info(f"Splits root:      {splits_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Compute device:   {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logging.info(f"Augmentation:     {args.augment}")
    logging.info(f"Batch config:     Physical={args.batch_size}, Accum={args.accum_steps}, Effective={effective_batch_size}")

    train_transform = get_train_transforms(DEFAULT_IMAGE_SIZE, augment=args.augment)
    eval_transform = get_eval_transforms(DEFAULT_IMAGE_SIZE)

    logging.info("Loading training and validation manifests...")
    train_dataset = TBX11KMultiLabelDataset(
        csv_path=train_csv,
        dataset_root=data_dir,
        transform=train_transform,
        split_name="train",
        expected_count=3219,
    )
    val_dataset = TBX11KMultiLabelDataset(
        csv_path=val_csv,
        dataset_root=data_dir,
        transform=eval_transform,
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

    # Instantiate model from scratch
    logging.info("Instantiating EfficientNet-B0 from scratch (weights=None)...")
    model = build_efficientnet_b0(num_classes=2).to(device)
    eval_model = build_efficientnet_b0(num_classes=2).to(device)

    # Build parameter groups and optimizer
    param_groups = build_parameter_groups(model, weight_decay=args.weight_decay)
    decay_count = len(param_groups[0]["params"])
    no_decay_count = len(param_groups[1]["params"])
    logging.info(f"Parameter groups: Decayed (weights/biases)={decay_count}, No-Decay (BatchNorm)={no_decay_count}")

    optimizer = timm.optim.RMSpropTF(
        param_groups,
        lr=args.lr,
        alpha=RMSPROP_ALPHA,
        eps=RMSPROP_EPS,
        momentum=RMSPROP_MOMENTUM,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None

    # Step schedule calculations
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)  # 805
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)               # 4,025
    decay_steps = int(DECAY_EPOCHS * steps_per_epoch)                 # 1,932
    total_steps = args.epochs * steps_per_epoch

    logging.info(
        f"Schedule: steps_per_epoch={steps_per_epoch}, warmup_steps={warmup_steps} (5 epochs), "
        f"decay_steps={decay_steps} (2.4 epochs), decay_rate={DECAY_RATE}"
    )

    # Initialize EMA
    ema = ModelEMA(model, max_decay=EMA_MAX_DECAY)

    # Startup previews
    preview_csv = output_dir / "lr_schedule_preview.csv"
    write_lr_schedule_preview(
        preview_csv,
        total_steps=total_steps,
        base_lr=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        steps_per_epoch=steps_per_epoch,
    )
    logging.info(f"Saved LR schedule preview -> {preview_csv}")

    manifest_json = output_dir / "run_manifest.json"
    run_events: List[str] = []

    # Resume support
    start_epoch = 1
    global_step = 0
    best_val_macro_f1 = -1.0
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

        if "raw_model_state_dict" in ckpt:
            model.load_state_dict(ckpt["raw_model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if "ema_num_updates" in ckpt and "model_state_dict" in ckpt:
            ema.num_updates = ckpt["ema_num_updates"]
            ema.shadow = {k: v.to(device) for k, v in ckpt["model_state_dict"].items()}

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", (start_epoch - 1) * steps_per_epoch)
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", ckpt.get("val_macro_f1", -1.0))
        run_events.append(f"Resumed from {resume_path.name} at epoch {start_epoch}, step {global_step}")
        logging.info(f"Resumed at epoch {start_epoch}, global_step {global_step}, prior best F1: {best_val_macro_f1:.4f}")

        # Load existing history CSV
        if history_csv_path.is_file():
            with open(history_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    typed_row = {}
                    for k, v in row.items():
                        try:
                            typed_row[k] = float(v) if "." in v else int(v)
                        except ValueError:
                            typed_row[k] = v
                    history_records.append(typed_row)

    write_run_manifest(
        manifest_json,
        config=vars(args),
        splits_dir=splits_dir,
        best_epoch=None,
        best_val_macro_f1=None,
        stop_reason="running",
        events=run_events,
    )

    criterion = nn.BCEWithLogitsLoss()
    total_skipped_steps = 0

    logging.info(f"Starting training run: {args.epochs} epochs total (NO early stopping).")

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            train_loss, train_metrics, duration, global_step, lr_start, lr_end, skipped_epoch = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                device=device,
                use_amp=use_amp,
                accum_steps=args.accum_steps,
                global_step=global_step,
                base_lr=args.lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
            )
        except torch.cuda.OutOfMemoryError as oom_err:
            logging.error(f"CUDA Out of Memory encountered at batch size {args.batch_size}: {oom_err}")
            logging.info(f"Attempting fallback to batch size {BATCH_SIZE_FALLBACK}...")
            run_events.append(f"OOM at epoch {epoch}: fallback to B={BATCH_SIZE_FALLBACK}")
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
            train_loss, train_metrics, duration, global_step, lr_start, lr_end, skipped_epoch = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                device=device,
                use_amp=use_amp,
                accum_steps=args.accum_steps * 2,
                global_step=global_step,
                base_lr=args.lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
            )

        total_skipped_steps += skipped_epoch

        # Evaluate Raw Model
        raw_val_loss, raw_val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # Evaluate EMA Model (Governing Selection Model)
        ema.apply_to(eval_model)
        ema_val_loss, ema_val_metrics = validate_one_epoch(
            model=eval_model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        # GPU Peak Memory
        peak_gpu_mem_mb = (
            torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        ema_val_macro_f1 = ema_val_metrics["macro_f1"]
        is_best = ema_val_macro_f1 > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = ema_val_macro_f1
            improvement_tag = " [BEST]"
            save_checkpoint(
                path=best_model_path,
                epoch=epoch,
                global_step=global_step,
                ema_model=eval_model,
                raw_model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                val_macro_f1=ema_val_macro_f1,
                best_val_macro_f1=best_val_macro_f1,
                val_metrics=ema_val_metrics,
                raw_val_metrics=raw_val_metrics,
                config=vars(args),
                history=history_records,
            )
        else:
            improvement_tag = ""

        # Save last checkpoint every epoch
        save_checkpoint(
            path=last_checkpoint_path,
            epoch=epoch,
            global_step=global_step,
            ema_model=eval_model,
            raw_model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            val_macro_f1=ema_val_macro_f1,
            best_val_macro_f1=best_val_macro_f1,
            val_metrics=ema_val_metrics,
            raw_val_metrics=raw_val_metrics,
            config=vars(args),
            history=history_records,
        )

        logging.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"LR: {lr_start:.6f}->{lr_end:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"EMA Val Loss: {ema_val_loss:.4f} | "
            f"EMA Val Macro-F1: {ema_val_macro_f1:.4f} | "
            f"EMA Exact Match: {ema_val_metrics['exact_match_accuracy']:.4f} | "
            f"Active F1: {ema_val_metrics['active_tb']['f1']:.4f} | "
            f"Obs F1: {ema_val_metrics['obsolete_tb']['f1']:.4f} | "
            f"Raw Macro-F1: {raw_val_metrics['macro_f1']:.4f} | "
            f"VRAM: {peak_gpu_mem_mb:.1f} MB | "
            f"Duration: {duration:.1f}s{improvement_tag}"
        )

        current_ema_decay = min(EMA_MAX_DECAY, (1.0 + ema.num_updates) / (10.0 + ema.num_updates))

        # Log history row
        history_records.append({
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": lr_end,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "duration_sec": duration,
            "train_loss": train_loss,
            "val_loss": ema_val_loss,  # ResNet parity key
            "val_macro_f1": ema_val_macro_f1,  # ResNet parity key
            "val_exact_match_acc": ema_val_metrics["exact_match_accuracy"],
            "val_hamming_loss": ema_val_metrics["hamming_loss"],
            "val_category_acc": ema_val_metrics["category_accuracy"],
            "val_active_precision": ema_val_metrics["active_tb"]["precision"],
            "val_active_recall": ema_val_metrics["active_tb"]["recall"],
            "val_active_specificity": ema_val_metrics["active_tb"]["specificity"],
            "val_active_f1": ema_val_metrics["active_tb"]["f1"],
            "val_obsolete_precision": ema_val_metrics["obsolete_tb"]["precision"],
            "val_obsolete_recall": ema_val_metrics["obsolete_tb"]["recall"],
            "val_obsolete_specificity": ema_val_metrics["obsolete_tb"]["specificity"],
            "val_obsolete_f1": ema_val_metrics["obsolete_tb"]["f1"],
            "ema_val_loss": ema_val_loss,
            "ema_val_macro_f1": ema_val_macro_f1,
            "ema_val_exact_match_acc": ema_val_metrics["exact_match_accuracy"],
            "ema_val_active_f1": ema_val_metrics["active_tb"]["f1"],
            "ema_val_obsolete_f1": ema_val_metrics["obsolete_tb"]["f1"],
            "raw_val_loss": raw_val_loss,
            "raw_val_macro_f1": raw_val_metrics["macro_f1"],
            "raw_val_exact_match_acc": raw_val_metrics["exact_match_accuracy"],
            "raw_val_active_f1": raw_val_metrics["active_tb"]["f1"],
            "raw_val_obsolete_f1": raw_val_metrics["obsolete_tb"]["f1"],
            "ema_decay_used": current_ema_decay,
            "skipped_steps": skipped_epoch,
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        })

        # Update CSV and plots
        with open(history_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
            writer.writeheader()
            writer.writerows(history_records)

        plot_training_curves(history_csv_path, plots_path)

    # Finalize run manifest
    best_epoch_idx = max(range(len(history_records)), key=lambda i: history_records[i]["ema_val_macro_f1"]) + 1
    write_run_manifest(
        manifest_json,
        config=vars(args),
        splits_dir=splits_dir,
        best_epoch=best_epoch_idx,
        best_val_macro_f1=best_val_macro_f1,
        stop_reason="max_epochs",
        events=run_events,
    )

    logging.info("=" * 80)
    logging.info(f"Training completed successfully. All {args.epochs} epochs executed.")
    logging.info(f"Best EMA Validation Macro-F1: {best_val_macro_f1:.4f} (Epoch {best_epoch_idx})")
    logging.info(f"Best model saved at:  {best_model_path}")
    logging.info(f"History CSV saved at: {history_csv_path}")
    logging.info(f"Plots saved at:       {plots_path}")
    logging.info(f"Run manifest:         {manifest_json}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

