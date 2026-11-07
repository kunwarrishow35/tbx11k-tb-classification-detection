"""
DenseNet-121 Multi-Label Test Evaluation Pipeline for TBX11K Chest X-Rays
========================================================================

Evaluation Protocol & Metric Specifications:
--------------------------------------------
- Architecture: DenseNet-121 / DenseNet-BC-121 equivalent instantiated via PyTorch.
- Target Split: Canonical test set (data_prep/splits/test.csv, 690 images).
- Target Labels: ActiveTuberculosis, ObsoletePulmonaryTuberculosis (2 output logits).
- Input Resolution: 512 x 512, 3 channels (RGB conversion for grayscale images).
- Normalization: ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].

Validation-Set Threshold Analysis:
----------------------------------
- Evaluates data_prep/splits/val.csv (690 images) using identical transforms and model weights.
- Saves val_predictions.csv with image_path, true labels, and continuous probabilities.
- Sweeps decision thresholds from 0.01 to 0.95 in steps of 0.01 (rounded to 4 decimals).
- Saves threshold_sweep_val.csv and val_threshold_curves.png.
- Selects F1-optimal per-label thresholds and shared macro-F1 threshold with median tie-breaking.
- Saves val_tuned_thresholds.json strictly before test-set inference.
- Evaluates test.csv at validation-tuned thresholds, saving a 7-file bundle in <output_dir>_val_tuned.
- Saves threshold_comparison.csv comparing fixed 0.5, shared tuned, and per-label tuned settings.
- Supports --no-val-sweep flag to skip validation analysis (default is enabled).

Generated Artifacts:
--------------------
- test_metrics.json: Structured JSON with complete classification and efficiency metrics.
- test_metrics.csv: Tabular flattened metrics summary.
- per_image_predictions.csv: Detailed image-level predictions with probabilities and categories.
- confusion_matrix_active_tb.png: 2x2 confusion matrix visualization for Active TB.
- confusion_matrix_obsolete_tb.png: 2x2 confusion matrix visualization for Obsolete TB.
- confusion_matrix_4category.png: 4x4 derived category confusion matrix visualization.
- test_report.txt: Comprehensive human-readable performance report.
- val_predictions.csv: Validation split predictions (outputs/test/).
- threshold_sweep_val.csv: Complete 95-step validation sweep metrics (outputs/test/).
- val_threshold_curves.png: Validation precision, recall, and F1 curves (outputs/test/).
- val_tuned_thresholds.json: Optimal validation thresholds and metadata (outputs/test/).
- threshold_comparison.csv: Comparison across fixed 0.5 and tuned settings (outputs/test/).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
TARGET_COLUMNS = ["ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]
CATEGORY_NAMES = ["Healthy", "Active_Only", "Obsolete_Only", "Both"]
DEFAULT_IMAGE_SIZE = (512, 512)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Path Resolution
# -----------------------------------------------------------------------------
def resolve_workspace_paths(
    custom_data_dir: Optional[str] = None,
    custom_splits_dir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Dynamically resolves workspace root, dataset directory, and splits directory."""
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

    data_dir = Path(custom_data_dir).resolve() if custom_data_dir else workspace_root / "TBX11K"
    splits_dir = Path(custom_splits_dir).resolve() if custom_splits_dir else workspace_root / "data_prep" / "splits"

    return workspace_root, data_dir, splits_dir


# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------
class TBX11KTestDataset(Dataset):
    """
    Test dataset loading strictly from split manifest CSV.
    Yields (image_tensor, target_tensor, image_rel_path, cohort).
    """

    def __init__(
        self,
        csv_path: Path,
        dataset_root: Path,
        transform: Optional[transforms.Compose] = None,
        expected_count: Optional[int] = 690,
    ):
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Test CSV manifest not found: {self.csv_path}")

        self.samples: List[Dict[str, Any]] = []
        self._load_and_validate(expected_count)

    def _load_and_validate(self, expected_count: Optional[int]) -> None:
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            for col in ["image_path"] + TARGET_COLUMNS:
                if col not in fieldnames:
                    raise ValueError(f"Missing required column '{col}' in test CSV {self.csv_path}.")

            seen = set()
            for row_idx, row in enumerate(reader, start=2):
                img_rel = row["image_path"].strip()
                if not img_rel:
                    raise ValueError(f"Empty image_path at line {row_idx} of {self.csv_path}")

                if img_rel in seen:
                    raise ValueError(f"Duplicate image path at line {row_idx}: {img_rel}")
                seen.add(img_rel)

                act_val = int(row["ActiveTuberculosis"])
                obs_val = int(row["ObsoletePulmonaryTuberculosis"])
                if act_val not in (0, 1) or obs_val not in (0, 1):
                    raise ValueError(f"Invalid binary target at line {row_idx}: ({act_val}, {obs_val})")

                full_path = self.dataset_root / img_rel
                if not full_path.is_file():
                    raise FileNotFoundError(f"Test image missing from disk: {full_path}")

                self.samples.append({
                    "image_rel_path": img_rel,
                    "full_path": full_path,
                    "cohort": row.get("cohort", ""),
                    "split": row.get("split", "test"),
                    "active_tb": act_val,
                    "obsolete_tb": obs_val,
                })

        if expected_count is not None and len(self.samples) != expected_count:
            logging.warning(
                f"Split expected {expected_count} samples, found {len(self.samples)}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        item = self.samples[idx]
        with Image.open(item["full_path"]) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)

        target = torch.tensor([item["active_tb"], item["obsolete_tb"]], dtype=torch.float32)
        return tensor, target, item["image_rel_path"], item["cohort"]


# -----------------------------------------------------------------------------
# Transforms & Model Construction
# -----------------------------------------------------------------------------
def get_test_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def build_densenet121(num_classes: int = 2) -> nn.Module:
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)
    return model


# -----------------------------------------------------------------------------
# Category Mapping Helper
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Classification Metric Suite
# -----------------------------------------------------------------------------
def compute_test_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: Union[float, List[float], np.ndarray] = 0.5,
) -> Dict[str, Any]:
    """
    Computes complete classification metrics across the multi-label test set:
    - Subset exact match accuracy and Hamming metrics
    - Per-label precision, sensitivity/recall, specificity, F1, ROC-AUC, PR-AUC
    - Macro-F1 and Weighted-F1
    - 4-category confusion matrix and per-category metrics

    Supports either a scalar threshold (e.g. 0.5) or a per-label list/array of thresholds.
    When a scalar is provided, outputs the exact schema identical to DenseNet baseline.
    When a per-label threshold is provided, adds 'thresholds_per_label' and 'threshold_source'.
    """
    if isinstance(threshold, (int, float)):
        th_scalar = float(threshold)
        y_pred = (y_probs >= th_scalar).astype(int)
        is_per_label = False
        prob_threshold_field: Any = th_scalar
    elif isinstance(threshold, (list, tuple, np.ndarray)):
        th_arr = np.array(threshold, dtype=float)
        if th_arr.ndim == 0:
            th_scalar = float(th_arr)
            y_pred = (y_probs >= th_scalar).astype(int)
            is_per_label = False
            prob_threshold_field = th_scalar
        else:
            y_pred = (y_probs >= th_arr).astype(int)
            is_per_label = True
            prob_threshold_field = [float(x) for x in th_arr]
    else:
        th_scalar = float(threshold)
        y_pred = (y_probs >= th_scalar).astype(int)
        is_per_label = False
        prob_threshold_field = th_scalar

    n_samples = len(y_true)

    # 1. Exact Match Accuracy: all labels must match for a sample
    exact_matches = (y_true[:, 0] == y_pred[:, 0]) & (y_true[:, 1] == y_pred[:, 1])
    exact_match_acc = float(np.mean(exact_matches)) if n_samples > 0 else 0.0

    # 2. Hamming Loss & Hamming Accuracy
    hamming_loss = float(np.mean(y_true != y_pred)) if n_samples > 0 else 0.0
    hamming_acc = 1.0 - hamming_loss

    # 3. Per-label metrics
    per_label_metrics = {}
    label_f1s = []
    label_supports = []

    for idx, name in enumerate(TARGET_COLUMNS):
        t = y_true[:, idx]
        p = y_pred[:, idx]
        prob = y_probs[:, idx]

        tp = int(np.sum((t == 1) & (p == 1)))
        fp = int(np.sum((t == 0) & (p == 1)))
        fn = int(np.sum((t == 1) & (p == 0)))
        tn = int(np.sum((t == 0) & (p == 0)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # ROC-AUC and PR-AUC
        try:
            roc_auc = float(roc_auc_score(t, prob))
        except Exception:
            roc_auc = None

        try:
            pr_auc = float(average_precision_score(t, prob))
        except Exception:
            pr_auc = None

        support = int(np.sum(t == 1))
        label_f1s.append(f1)
        label_supports.append(support)

        per_label_metrics[name] = {
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": float(prec),
            "sensitivity_recall": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support_positives": support,
            "support_negatives": int(np.sum(t == 0)),
        }

    # Macro-F1 (unweighted arithmetic mean)
    macro_f1 = float(np.mean(label_f1s)) if label_f1s else 0.0

    # Weighted-F1 (weighted by positive support)
    total_support = sum(label_supports)
    if total_support > 0:
        weighted_f1 = float(sum(f * s for f, s in zip(label_f1s, label_supports)) / total_support)
    else:
        weighted_f1 = 0.0

    # 4. Derived 4-Category Metrics
    true_cats = np.array([derive_category(int(t[0]), int(t[1])) for t in y_true])
    pred_cats = np.array([derive_category(int(p[0]), int(p[1])) for p in y_pred])

    cat_cm = confusion_matrix(true_cats, pred_cats, labels=[0, 1, 2, 3]).tolist()

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        true_cats,
        pred_cats,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    per_category_metrics = {}
    for i, cat_name in enumerate(CATEGORY_NAMES):
        per_category_metrics[cat_name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1_score": float(f1s[i]),
            "support": int(supports[i]),
        }

    cat_accuracy = float(np.mean(true_cats == pred_cats)) if len(true_cats) > 0 else 0.0

    classification_summary = {
        "exact_match_accuracy": exact_match_acc,
        "hamming_loss": hamming_loss,
        "hamming_accuracy": hamming_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "category_accuracy": cat_accuracy,
        "total_test_samples": n_samples,
        "probability_threshold": prob_threshold_field,
    }

    if is_per_label:
        classification_summary["thresholds_per_label"] = {
            name: float(th_arr[idx]) for idx, name in enumerate(TARGET_COLUMNS)
        }
        classification_summary["threshold_source"] = "val.csv sweep"

    return {
        "classification_summary": classification_summary,
        "per_label_metrics": per_label_metrics,
        "per_category_metrics": per_category_metrics,
        "four_category_confusion_matrix": cat_cm,
        "derived_category_names": CATEGORY_NAMES,
    }


# -----------------------------------------------------------------------------
# Efficiency Benchmarking Suite
# -----------------------------------------------------------------------------
def benchmark_efficiency(
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    eval_batch_size: int = 4,
    warmup_iters: int = 10,
    timed_iters: int = 50,
) -> Dict[str, Any]:
    """Measures FLOPs, MACs, parameter counts, latency, and throughput."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops_count = None
    macs_count = None
    flops_error = None

    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        model_copy = build_densenet121(num_classes=2).to(device)
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()

        with torch.inference_mode():
            macs_count, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        flops_count = 2.0 * macs_count
        del model_copy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        flops_error = str(e)

    # Benchmark single-image latency
    single_tensor = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    with torch.inference_mode():
        for _ in range(warmup_iters):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_tensor)
            else:
                _ = model(single_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies_ms = []
        for _ in range(timed_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_tensor)
            else:
                _ = model(single_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))
    fps_single = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Benchmark batched throughput
    batched_tensor = torch.randn(eval_batch_size, 3, input_size[0], input_size[1], device=device)
    with torch.inference_mode():
        for _ in range(warmup_iters):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(batched_tensor)
            else:
                _ = model(batched_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_batch_start = time.perf_counter()
        for _ in range(timed_iters):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(batched_tensor)
            else:
                _ = model(batched_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_batch_end = time.perf_counter()

    total_batched_images = eval_batch_size * timed_iters
    batched_time_sec = t_batch_end - t_batch_start
    batched_throughput_fps = total_batched_images / batched_time_sec if batched_time_sec > 0 else 0.0

    return {
        "model_architecture": "DenseNet-121",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "computational_complexity": {
            "macs": macs_count,
            "flops": flops_count,
            "convention": "1 MAC = 2 FLOPs",
            "tool": "thop (torch-profile)",
            "error": flops_error,
        },
        "inference_benchmark": {
            "single_image": {
                "batch_size": 1,
                "input_resolution": f"{input_size[0]}x{input_size[1]}x3",
                "mean_latency_ms": round(mean_lat, 2),
                "std_latency_ms": round(std_lat, 2),
                "min_latency_ms": round(min_lat, 2),
                "max_latency_ms": round(max_lat, 2),
                "fps": round(fps_single, 2),
                "timed_iterations": timed_iters,
            },
            "batched": {
                "batch_size": eval_batch_size,
                "input_resolution": f"{input_size[0]}x{input_size[1]}x3",
                "throughput_images_per_sec": round(batched_throughput_fps, 2),
                "mean_batch_latency_ms": round((batched_time_sec / timed_iters) * 1000.0, 2),
                "timed_iterations": timed_iters,
            },
        },
        "hardware_environment": {
            "device": str(device),
            "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "precision_mode": "FP16 Autocast" if (use_amp and device.type == "cuda") else "FP32",
        },
    }


# -----------------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------------
def plot_binary_confusion_matrix(
    cm_dict: Dict[str, int],
    label_name: str,
    output_path: Path,
) -> None:
    """Plots and saves a clean 2x2 binary confusion matrix."""
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])
    total = np.sum(matrix)

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            pct = (val / total * 100.0) if total > 0 else 0.0
            color = "white" if val > (matrix.max() / 2.0) else "black"
            ax.text(
                j, i, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold",
            )

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"Confusion Matrix: {label_name}", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_4category_confusion_matrix(
    cm_matrix: List[List[int]],
    output_path: Path,
) -> None:
    """Plots and saves a 4x4 derived category confusion matrix."""
    matrix = np.array(cm_matrix)
    total = np.sum(matrix)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    display_labels = ["Healthy\n(0,0)", "Active Only\n(1,0)", "Obsolete Only\n(0,1)", "Both\n(1,1)"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(4):
        for j in range(4):
            val = matrix[i, j]
            pct = (val / total * 100.0) if total > 0 else 0.0
            color = "white" if val > (matrix.max() / 2.0) else "black"
            ax.text(
                j, i, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="center", color=color, fontsize=9, fontweight="bold",
            )

    ax.set_xlabel("Predicted Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Derived 4-Category Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_val_threshold_curves(
    sweep_records: List[Dict[str, Any]],
    best_th_act: float,
    best_th_obs: float,
    best_th_shared: float,
    output_path: Path,
) -> None:
    """Plots validation precision, recall, and F1 across thresholds with vertical reference lines."""
    thresholds = [r["threshold"] for r in sweep_records]

    act_prec = [r["active_precision"] for r in sweep_records]
    act_rec = [r["active_recall"] for r in sweep_records]
    act_f1 = [r["active_f1"] for r in sweep_records]

    obs_prec = [r["obsolete_precision"] for r in sweep_records]
    obs_rec = [r["obsolete_recall"] for r in sweep_records]
    obs_f1 = [r["obsolete_f1"] for r in sweep_records]

    shared_f1 = [r["shared_macro_f1"] for r in sweep_records]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Active Tuberculosis
    ax1 = axes[0]
    ax1.plot(thresholds, act_prec, label="Precision", color="tab:green", linewidth=1.8)
    ax1.plot(thresholds, act_rec, label="Recall (Sensitivity)", color="tab:blue", linewidth=1.8)
    ax1.plot(thresholds, act_f1, label="F1-Score", color="tab:red", linewidth=2.4)
    ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.8, label="Default τ = 0.50")
    ax1.axvline(best_th_act, color="purple", linestyle="-", linewidth=2.0, label=f"Tuned τ = {best_th_act:.2f}")
    ax1.set_title("Active Tuberculosis (val.csv, N=108+)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax1.set_ylabel("Metric Score", fontsize=11)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    # Panel 2: Obsolete Pulmonary Tuberculosis
    ax2 = axes[1]
    ax2.plot(thresholds, obs_prec, label="Precision", color="tab:green", linewidth=1.8)
    ax2.plot(thresholds, obs_rec, label="Recall (Sensitivity)", color="tab:blue", linewidth=1.8)
    ax2.plot(thresholds, obs_f1, label="F1-Score", color="tab:red", linewidth=2.4)
    ax2.axvline(0.50, color="gray", linestyle="--", alpha=0.8, label="Default τ = 0.50")
    ax2.axvline(best_th_obs, color="purple", linestyle="-", linewidth=2.0, label=f"Tuned τ = {best_th_obs:.2f}")
    ax2.set_title("Obsolete Pulmonary TB (val.csv, N=17+)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax2.set_ylabel("Metric Score", fontsize=11)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9)

    # Panel 3: Shared Threshold Macro-F1
    ax3 = axes[2]
    ax3.plot(thresholds, shared_f1, label="Macro-F1 (Shared τ)", color="darkblue", linewidth=2.2)
    ax3.axvline(0.50, color="gray", linestyle="--", alpha=0.8, label="Default τ = 0.50")
    ax3.axvline(best_th_shared, color="crimson", linestyle="-", linewidth=2.0, label=f"Tuned τ = {best_th_shared:.2f}")
    ax3.set_title("Shared Threshold Macro-F1 (val.csv)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax3.set_ylabel("Macro-F1 Score", fontsize=11)
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(-0.02, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info(f"Saved validation threshold curves -> {output_path}")


# -----------------------------------------------------------------------------
# Text Report Generator
# -----------------------------------------------------------------------------
def generate_text_report(
    test_metrics: Dict[str, Any],
    efficiency_metrics: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
) -> str:
    """Builds a comprehensive, formatted human-readable evaluation report."""
    summary = test_metrics["classification_summary"]
    per_label = test_metrics["per_label_metrics"]
    per_cat = test_metrics["per_category_metrics"]
    bench = efficiency_metrics.get("inference_benchmark", {})
    hw = efficiency_metrics.get("hardware_environment", {})
    flops = efficiency_metrics.get("computational_complexity", {})

    is_tuned = summary.get("threshold_source") == "val.csv sweep"
    if is_tuned:
        th_dict = summary.get("thresholds_per_label", {})
        th_lines = ["  Probability Threshold:     Validation-Tuned (val.csv F1 maximization)"]
        for l_name in TARGET_COLUMNS:
            th_lines.append(f"                             - {l_name}: tau = {th_dict.get(l_name, 'N/A'):.4f}")
        th_desc = "\n".join(th_lines)
    else:
        th_desc = f"  Probability Threshold:     {summary['probability_threshold']}"

    report_lines = [
        "=" * 78,
        "DENSENET-121 MULTI-LABEL TEST EVALUATION REPORT".center(78),
    ]
    if is_tuned:
        th_dict = summary.get("thresholds_per_label", {})
        report_lines.append("  Threshold Selection: Thresholds tuned exclusively on validation split (val.csv)")
        for l_name in TARGET_COLUMNS:
            report_lines.append(f"    * {l_name}: tau = {th_dict.get(l_name, 0.5):.4f}")
    report_lines.extend([
        "=" * 78,
        "",
        "1. EXPERIMENTAL CONFIGURATION & HARDWARE METADATA",
        "-" * 78,
        f"  Model Architecture:        DenseNet-121 (DenseNet-BC-121 equivalent)",
        f"  Pretrained Weights:        None (Trained from scratch)",
        f"  Input Resolution:          512 x 512, 3 channels (RGB)",
        f"  Normalization:             ImageNet Mean {NORM_MEAN}, Std {NORM_STD}",
        f"  Target Split:              Canonical test.csv ({summary['total_test_samples']} images)",
        th_desc,
        f"  Best Checkpoint Loaded:    Epoch {checkpoint_info.get('epoch', 'N/A')} "
        f"(Val Macro-F1: {checkpoint_info.get('val_macro_f1', 'N/A')})",
        f"  Evaluation Compute Device: {hw.get('device', 'N/A')} ({hw.get('gpu_model', 'N/A')})",
        f"  Inference Precision:       {hw.get('precision_mode', 'N/A')}",
        f"  PyTorch Version:           {hw.get('torch_version', 'N/A')}",
        "",
        "2. OVERALL CLASSIFICATION PERFORMANCE",
        "-" * 78,
        f"  Subset Exact-Match Acc:    {summary['exact_match_accuracy'] * 100:.2f}%  "
        f"(Definition: both binary labels match GT)",
        f"  Hamming Accuracy:          {summary['hamming_accuracy'] * 100:.2f}%  "
        f"(1.0 - Hamming Loss)",
        f"  Hamming Loss:              {summary['hamming_loss']:.4f}  "
        f"(Fraction of misclassified labels)",
        f"  Macro-F1 (Unweighted):     {summary['macro_f1']:.4f}  "
        f"(Arithmetic mean of Active & Obsolete F1)",
        f"  Weighted-F1 (By Support):  {summary['weighted_f1']:.4f}  "
        f"(Weighted by positive class prevalence)",
        f"  Derived Category Accuracy: {summary['category_accuracy'] * 100:.2f}%",
        f"  Total Test Images:         {summary['total_test_samples']:,}",
        "",
        "3. PER-LABEL CLASSIFICATION METRICS",
        "-" * 78,
    ])

    for label_name, metrics in per_label.items():
        cm = metrics["confusion_matrix"]
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
        pr_str = f"{metrics['pr_auc']:.4f}" if metrics["pr_auc"] is not None else "N/A"
        report_lines.extend([
            f"  [{label_name}]",
            f"    Positive Support:        {metrics['support_positives']:,} "
            f"({metrics['support_positives'] / summary['total_test_samples'] * 100:.2f}%)",
            f"    Precision:               {metrics['precision']:.4f}",
            f"    Sensitivity (Recall):    {metrics['sensitivity_recall']:.4f}",
            f"    Specificity:             {metrics['specificity']:.4f}",
            f"    F1-Score:                {metrics['f1_score']:.4f}",
            f"    ROC-AUC:                 {roc_str}",
            f"    PR-AUC (Avg Precision):  {pr_str}",
            f"    Confusion Matrix:        TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}",
            "",
        ])

    report_lines.extend([
        "4. DERIVED FOUR-CATEGORY CLASSIFICATION PERFORMANCE",
        "-" * 78,
        f"  {'Category':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
        f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}",
    ])

    for cat_name, metrics in per_cat.items():
        report_lines.append(
            f"  {cat_name:<16} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | "
            f"{metrics['f1_score']:<10.4f} | {metrics['support']:<8,}"
        )

    report_lines.extend([
        "",
        "5. COMPUTATIONAL EFFICIENCY & INFERENCE BENCHMARKS",
        "-" * 78,
        f"  Total Parameter Count:     {efficiency_metrics['total_parameters']:,}",
        f"  Trainable Parameters:      {efficiency_metrics['trainable_parameters']:,}",
    ])

    if flops.get("macs") is not None:
        report_lines.extend([
            f"  MACs (Multiply-Accumulate):{flops['macs'] / 1e9:.2f} GigaMACs ({flops['tool']})",
            f"  FLOPs:                     {flops['flops'] / 1e9:.2f} GigaFLOPs ({flops['convention']})",
        ])
    else:
        report_lines.append(f"  FLOPs Counting:            Not Available ({flops.get('error', 'N/A')})")

    bs1 = bench["single_image"]
    batch = bench["batched"]
    report_lines.extend([
        f"  Single-Image Latency (B=1):{bs1['mean_latency_ms']:.2f} ms (+/- {bs1['std_latency_ms']:.2f} ms)",
        f"  Single-Image Range:        Min: {bs1['min_latency_ms']:.2f} ms, Max: {bs1['max_latency_ms']:.2f} ms",
        f"  Single-Image Throughput:   {bs1['fps']:.2f} FPS (frames per second)",
        f"  Batched Throughput (B={batch['batch_size']}):   {batch['throughput_images_per_sec']:.2f} images/sec "
        f"({batch['mean_batch_latency_ms']:.2f} ms per batch)",
        "",
        "6. METHODOLOGY & CLINICAL INTERPRETATION NOTES",
        "-" * 78,
    ])
    if is_tuned:
        report_lines.extend([
            "  - Multi-label decision thresholds tuned exclusively on validation split (val.csv).",
            "    No test-set threshold tuning performed.",
        ])
    else:
        report_lines.extend([
            "  - Multi-label sigmoid threshold fixed at 0.5. No test-set threshold tuning performed.",
        ])
    report_lines.extend([
        "  - The 4 categories are strictly derived from independent binary predictions:",
        "      Healthy=(0,0), Active_Only=(1,0), Obsolete_Only=(0,1), Both=(1,1).",
        "  - Exact-match requires correctly predicting both disease states simultaneously.",
        "  - Latency timing employs CUDA synchronization to prevent asynchronous GPU queue bias.",
        "=" * 78,
    ])

    return "\n".join(report_lines)


# -----------------------------------------------------------------------------
# Save Standard 7-File Evaluation Bundle
# -----------------------------------------------------------------------------
def save_evaluation_bundle(
    bundle_dir: Path,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    all_img_paths: List[str],
    all_cohorts: List[str],
    test_metrics: Dict[str, Any],
    efficiency_metrics: Dict[str, Any],
    checkpoint: Dict[str, Any],
    ckpt_path: Path,
) -> None:
    """Saves the authoritative 7-file evaluation bundle into bundle_dir."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary = test_metrics["classification_summary"]

    if "thresholds_per_label" in summary:
        th_dict = summary["thresholds_per_label"]
        th_arr = np.array([th_dict[name] for name in TARGET_COLUMNS])
        y_pred = (y_probs >= th_arr).astype(int)
    else:
        th_val = summary["probability_threshold"]
        y_pred = (y_probs >= th_val).astype(int)

    # 1. Per-image predictions CSV
    image_records = []
    for idx in range(len(all_img_paths)):
        p_act, p_obs = float(y_probs[idx, 0]), float(y_probs[idx, 1])
        t_act, t_obs = int(y_true[idx, 0]), int(y_true[idx, 1])
        pred_act = int(y_pred[idx, 0])
        pred_obs = int(y_pred[idx, 1])

        true_cat = CATEGORY_NAMES[derive_category(t_act, t_obs)]
        pred_cat = CATEGORY_NAMES[derive_category(pred_act, pred_obs)]
        exact_match = int(t_act == pred_act and t_obs == pred_obs)

        image_records.append({
            "image_path": all_img_paths[idx],
            "cohort": all_cohorts[idx],
            "true_ActiveTuberculosis": t_act,
            "true_ObsoletePulmonaryTuberculosis": t_obs,
            "prob_ActiveTuberculosis": round(p_act, 6),
            "prob_ObsoletePulmonaryTuberculosis": round(p_obs, 6),
            "pred_ActiveTuberculosis": pred_act,
            "pred_ObsoletePulmonaryTuberculosis": pred_obs,
            "true_category": true_cat,
            "pred_category": pred_cat,
            "exact_match": exact_match,
        })

    per_image_csv_path = bundle_dir / "per_image_predictions.csv"
    with open(per_image_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_records[0].keys()))
        writer.writeheader()
        writer.writerows(image_records)
    logging.info(f"Saved per-image predictions to {per_image_csv_path}")

    # 2. JSON metrics
    metrics_json_path = bundle_dir / "test_metrics.json"
    full_output = {
        "checkpoint_evaluated": str(ckpt_path),
        "checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "val_macro_f1": checkpoint.get("val_macro_f1"),
        },
        "classification_metrics": test_metrics,
        "efficiency_metrics": efficiency_metrics,
    }
    with open(metrics_json_path, mode="w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    logging.info(f"Saved complete metrics JSON to {metrics_json_path}")

    # 3. CSV metrics
    metrics_csv_path = bundle_dir / "test_metrics.csv"
    summary_rows = []
    for k, v in test_metrics["classification_summary"].items():
        val_str = json.dumps(v) if isinstance(v, (dict, list)) else v
        summary_rows.append({"metric_group": "overall_classification", "metric_name": k, "value": val_str})
    for label, lmetrics in test_metrics["per_label_metrics"].items():
        for k, v in lmetrics.items():
            if k != "confusion_matrix":
                summary_rows.append({"metric_group": f"label_{label}", "metric_name": k, "value": v})
    for cat, cmetrics in test_metrics["per_category_metrics"].items():
        for k, v in cmetrics.items():
            summary_rows.append({"metric_group": f"category_{cat}", "metric_name": k, "value": v})

    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric_group", "metric_name", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)
    logging.info(f"Saved summary metrics CSV to {metrics_csv_path}")

    # 4. Confusion matrix plots
    active_cm_path = bundle_dir / "confusion_matrix_active_tb.png"
    obsolete_cm_path = bundle_dir / "confusion_matrix_obsolete_tb.png"
    cat_cm_path = bundle_dir / "confusion_matrix_4category.png"

    plot_binary_confusion_matrix(
        test_metrics["per_label_metrics"]["ActiveTuberculosis"]["confusion_matrix"],
        label_name="Active Tuberculosis",
        output_path=active_cm_path,
    )
    plot_binary_confusion_matrix(
        test_metrics["per_label_metrics"]["ObsoletePulmonaryTuberculosis"]["confusion_matrix"],
        label_name="Obsolete Pulmonary Tuberculosis",
        output_path=obsolete_cm_path,
    )
    plot_4category_confusion_matrix(
        test_metrics["four_category_confusion_matrix"],
        output_path=cat_cm_path,
    )
    logging.info(f"Saved confusion matrix figures to {bundle_dir}")

    # 5. Text report
    report_text = generate_text_report(
        test_metrics=test_metrics,
        efficiency_metrics=efficiency_metrics,
        checkpoint_info=checkpoint,
    )
    report_path = bundle_dir / "test_report.txt"
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Saved text evaluation report to {report_path}")


# -----------------------------------------------------------------------------
# Validation Threshold Sweep Execution
# -----------------------------------------------------------------------------
def run_validation_threshold_sweep(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    output_dir: Path,
    ckpt_path: Path,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes validation scoring, threshold sweep from 0.01 to 0.95 in 0.01 steps,
    generates validation curve plots, selects F1-maximizing thresholds with median tie-breaking,
    and writes val_tuned_thresholds.json strictly before test evaluation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Executing validation inference over val.csv (690 samples)...")

    all_val_targets: List[np.ndarray] = []
    all_val_probs: List[np.ndarray] = []
    all_val_paths: List[str] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device, non_blocking=True)
            targets = batch[1]
            rel_paths = batch[2]

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_val_probs.append(probs)
            all_val_targets.append(targets.numpy())
            all_val_paths.extend(rel_paths)

    y_val_true = np.vstack(all_val_targets)
    y_val_probs = np.vstack(all_val_probs)

    # 1. Save val_predictions.csv
    val_records = []
    for idx, rel_path in enumerate(all_val_paths):
        val_records.append({
            "image_path": rel_path,
            "true_active": int(y_val_true[idx, 0]),
            "true_obsolete": int(y_val_true[idx, 1]),
            "prob_active": round(float(y_val_probs[idx, 0]), 6),
            "prob_obsolete": round(float(y_val_probs[idx, 1]), 6),
        })

    val_pred_csv = output_dir / "val_predictions.csv"
    with open(val_pred_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_records[0].keys()))
        writer.writeheader()
        writer.writerows(val_records)
    logging.info(f"Saved validation predictions -> {val_pred_csv}")

    # 2. Threshold sweep from 0.01 to 0.95 in steps of 0.01
    thresholds = [round(i * 0.01, 4) for i in range(1, 96)]
    sweep_records: List[Dict[str, Any]] = []

    for tau in thresholds:
        # Active TB
        t_act = y_val_true[:, 0]
        p_act = (y_val_probs[:, 0] >= tau).astype(int)
        tp_act = int(np.sum((t_act == 1) & (p_act == 1)))
        fp_act = int(np.sum((t_act == 0) & (p_act == 1)))
        fn_act = int(np.sum((t_act == 1) & (p_act == 0)))
        tn_act = int(np.sum((t_act == 0) & (p_act == 0)))
        prec_act = tp_act / (tp_act + fp_act) if (tp_act + fp_act) > 0 else 0.0
        rec_act = tp_act / (tp_act + fn_act) if (tp_act + fn_act) > 0 else 0.0
        spec_act = tn_act / (tn_act + fp_act) if (tn_act + fp_act) > 0 else 0.0
        f1_act = (2.0 * prec_act * rec_act) / (prec_act + rec_act) if (prec_act + rec_act) > 0 else 0.0

        # Obsolete TB
        t_obs = y_val_true[:, 1]
        p_obs = (y_val_probs[:, 1] >= tau).astype(int)
        tp_obs = int(np.sum((t_obs == 1) & (p_obs == 1)))
        fp_obs = int(np.sum((t_obs == 0) & (p_obs == 1)))
        fn_obs = int(np.sum((t_obs == 1) & (p_obs == 0)))
        tn_obs = int(np.sum((t_obs == 0) & (p_obs == 0)))
        prec_obs = tp_obs / (tp_obs + fp_obs) if (tp_obs + fp_obs) > 0 else 0.0
        rec_obs = tp_obs / (tp_obs + fn_obs) if (tp_obs + fn_obs) > 0 else 0.0
        spec_obs = tn_obs / (tn_obs + fp_obs) if (tn_obs + fp_obs) > 0 else 0.0
        f1_obs = (2.0 * prec_obs * rec_obs) / (prec_obs + rec_obs) if (prec_obs + rec_obs) > 0 else 0.0

        shared_macro_f1 = (f1_act + f1_obs) / 2.0

        sweep_records.append({
            "threshold": tau,
            "active_tp": tp_act,
            "active_fp": fp_act,
            "active_fn": fn_act,
            "active_tn": tn_act,
            "active_precision": round(prec_act, 6),
            "active_recall": round(rec_act, 6),
            "active_specificity": round(spec_act, 6),
            "active_f1": round(f1_act, 6),
            "obsolete_tp": tp_obs,
            "obsolete_fp": fp_obs,
            "obsolete_fn": fn_obs,
            "obsolete_tn": tn_obs,
            "obsolete_precision": round(prec_obs, 6),
            "obsolete_recall": round(rec_obs, 6),
            "obsolete_specificity": round(spec_obs, 6),
            "obsolete_f1": round(f1_obs, 6),
            "shared_macro_f1": round(shared_macro_f1, 6),
        })

    # Save threshold_sweep_val.csv
    sweep_csv_path = output_dir / "threshold_sweep_val.csv"
    with open(sweep_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_records[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_records)
    logging.info(f"Saved validation threshold sweep -> {sweep_csv_path}")

    # 3. Select optimal thresholds with tie-breaking and fallback flags
    # Active TB
    max_f1_act = max(r["active_f1"] for r in sweep_records)
    if max_f1_act <= 0.0:
        best_th_act = 0.50
        fallback_act = True
        tie_act = [0.50]
        prec_best_act, rec_best_act, spec_best_act, f1_best_act = 0.0, 0.0, 1.0, 0.0
    else:
        fallback_act = False
        tie_act = [r["threshold"] for r in sweep_records if np.isclose(r["active_f1"], max_f1_act, atol=1e-12, rtol=1e-9)]
        best_th_act = round(float(np.median(tie_act)), 4)
        p_best_act = (y_val_probs[:, 0] >= best_th_act).astype(int)
        tp_b = int(np.sum((y_val_true[:, 0] == 1) & (p_best_act == 1)))
        fp_b = int(np.sum((y_val_true[:, 0] == 0) & (p_best_act == 1)))
        fn_b = int(np.sum((y_val_true[:, 0] == 1) & (p_best_act == 0)))
        tn_b = int(np.sum((y_val_true[:, 0] == 0) & (p_best_act == 0)))
        prec_best_act = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        rec_best_act = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        spec_best_act = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
        f1_best_act = (2.0 * prec_best_act * rec_best_act) / (prec_best_act + rec_best_act) if (prec_best_act + rec_best_act) > 0 else 0.0

    # Obsolete TB
    max_f1_obs = max(r["obsolete_f1"] for r in sweep_records)
    if max_f1_obs <= 0.0:
        best_th_obs = 0.50
        fallback_obs = True
        tie_obs = [0.50]
        prec_best_obs, rec_best_obs, spec_best_obs, f1_best_obs = 0.0, 0.0, 1.0, 0.0
    else:
        fallback_obs = False
        tie_obs = [r["threshold"] for r in sweep_records if np.isclose(r["obsolete_f1"], max_f1_obs, atol=1e-12, rtol=1e-9)]
        best_th_obs = round(float(np.median(tie_obs)), 4)
        p_best_obs = (y_val_probs[:, 1] >= best_th_obs).astype(int)
        tp_b = int(np.sum((y_val_true[:, 1] == 1) & (p_best_obs == 1)))
        fp_b = int(np.sum((y_val_true[:, 1] == 0) & (p_best_obs == 1)))
        fn_b = int(np.sum((y_val_true[:, 1] == 1) & (p_best_obs == 0)))
        tn_b = int(np.sum((y_val_true[:, 1] == 0) & (p_best_obs == 0)))
        prec_best_obs = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        rec_best_obs = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        spec_best_obs = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
        f1_best_obs = (2.0 * prec_best_obs * rec_best_obs) / (prec_best_obs + rec_best_obs) if (prec_best_obs + rec_best_obs) > 0 else 0.0

    # Shared Threshold
    max_macro_f1 = max(r["shared_macro_f1"] for r in sweep_records)
    if max_macro_f1 <= 0.0:
        best_th_shared = 0.50
        fallback_shared = True
        tie_shared = [0.50]
    else:
        fallback_shared = False
        tie_shared = [r["threshold"] for r in sweep_records if np.isclose(r["shared_macro_f1"], max_macro_f1, atol=1e-12, rtol=1e-9)]
        best_th_shared = round(float(np.median(tie_shared)), 4)

    # 4. Generate curves plot
    plot_val_threshold_curves(
        sweep_records=sweep_records,
        best_th_act=best_th_act,
        best_th_obs=best_th_obs,
        best_th_shared=best_th_shared,
        output_path=output_dir / "val_threshold_curves.png",
    )

    # 5. Save val_tuned_thresholds.json BEFORE test inference
    val_pos_act = int(np.sum(y_val_true[:, 0] == 1))
    val_pos_obs = int(np.sum(y_val_true[:, 1] == 1))

    tuned_metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "threshold_selection_protocol": "Validation F1-maximization (sweep 0.01 to 0.95 in 0.01 steps)",
        "validation_dataset": "data_prep/splits/val.csv (690 samples)",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "per_label": {
            "ActiveTuberculosis": {
                "threshold": best_th_act,
                "validation_f1": round(f1_best_act, 6),
                "validation_precision": round(prec_best_act, 6),
                "validation_recall": round(rec_best_act, 6),
                "validation_specificity": round(spec_best_act, 6),
                "validation_positives": val_pos_act,
                "tie_set": tie_act,
                "fallback_to_0.5": fallback_act,
            },
            "ObsoletePulmonaryTuberculosis": {
                "threshold": best_th_obs,
                "validation_f1": round(f1_best_obs, 6),
                "validation_precision": round(prec_best_obs, 6),
                "validation_recall": round(rec_best_obs, 6),
                "validation_specificity": round(spec_best_obs, 6),
                "validation_positives": val_pos_obs,
                "tie_set": tie_obs,
                "fallback_to_0.5": fallback_obs,
            },
        },
        "shared_threshold": {
            "threshold": best_th_shared,
            "validation_macro_f1": round(max_macro_f1, 6),
            "tie_set": tie_shared,
            "fallback_to_0.5": fallback_shared,
        },
        "validation_positives_per_label": {
            "ActiveTuberculosis": val_pos_act,
            "ObsoletePulmonaryTuberculosis": val_pos_obs,
        },
    }

    tuned_json_path = output_dir / "val_tuned_thresholds.json"
    with open(tuned_json_path, mode="w", encoding="utf-8") as f:
        json.dump(tuned_metadata, f, indent=2)
    logging.info(f"Saved validation tuned thresholds JSON -> {tuned_json_path}")

    return {
        "best_th_act": best_th_act,
        "best_th_obs": best_th_obs,
        "best_th_shared": best_th_shared,
        "metadata": tuned_metadata,
        "sweep_records": sweep_records,
        "y_val_true": y_val_true,
        "y_val_probs": y_val_probs,
    }


# -----------------------------------------------------------------------------
# Threshold Comparison CSV Writer
# -----------------------------------------------------------------------------
def generate_threshold_comparison(
    y_test_true: np.ndarray,
    y_test_probs: np.ndarray,
    best_th_act: float,
    best_th_obs: float,
    best_th_shared: float,
    output_path: Path,
) -> None:
    """
    Generates threshold_comparison.csv comparing fixed 0.5, shared tuned, and per-label tuned settings.
    Columns: setting, thresholds, macro_f1, active_f1, active_precision, active_recall,
             obsolete_f1, obsolete_precision, obsolete_recall, exact_match_accuracy,
             active_roc_auc, active_pr_auc, obsolete_roc_auc, obsolete_pr_auc.
    """
    try:
        act_roc = float(roc_auc_score(y_test_true[:, 0], y_test_probs[:, 0]))
    except Exception:
        act_roc = float("nan")
    try:
        act_pr = float(average_precision_score(y_test_true[:, 0], y_test_probs[:, 0]))
    except Exception:
        act_pr = float("nan")

    try:
        obs_roc = float(roc_auc_score(y_test_true[:, 1], y_test_probs[:, 1]))
    except Exception:
        obs_roc = float("nan")
    try:
        obs_pr = float(average_precision_score(y_test_true[:, 1], y_test_probs[:, 1]))
    except Exception:
        obs_pr = float("nan")

    settings = [
        ("fixed_0.5", 0.50, 0.50, "0.50 / 0.50"),
        ("shared_val_tuned", best_th_shared, best_th_shared, f"{best_th_shared:.4f} / {best_th_shared:.4f}"),
        ("per_label_val_tuned", best_th_act, best_th_obs, f"{best_th_act:.4f} / {best_th_obs:.4f}"),
    ]

    rows = []
    for setting_name, th_act, th_obs, th_str in settings:
        m = compute_test_metrics(y_test_true, y_test_probs, threshold=[th_act, th_obs])
        summ = m["classification_summary"]
        act_m = m["per_label_metrics"]["ActiveTuberculosis"]
        obs_m = m["per_label_metrics"]["ObsoletePulmonaryTuberculosis"]

        rows.append({
            "setting": setting_name,
            "thresholds": th_str,
            "macro_f1": round(summ["macro_f1"], 6),
            "active_f1": round(act_m["f1_score"], 6),
            "active_precision": round(act_m["precision"], 6),
            "active_recall": round(act_m["sensitivity_recall"], 6),
            "obsolete_f1": round(obs_m["f1_score"], 6),
            "obsolete_precision": round(obs_m["precision"], 6),
            "obsolete_recall": round(obs_m["sensitivity_recall"], 6),
            "exact_match_accuracy": round(summ["exact_match_accuracy"], 6),
            "active_roc_auc": round(act_roc, 6),
            "active_pr_auc": round(act_pr, 6),
            "obsolete_roc_auc": round(obs_roc, 6),
            "obsolete_pr_auc": round(obs_pr, 6),
        })

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Saved threshold comparison CSV -> {output_path}")


# -----------------------------------------------------------------------------
# Main Test Evaluation Execution
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DenseNet-121 best checkpoint on TBX11K test set."
    )
    # Checkpoint and data paths
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to TBX11K dataset directory.")
    parser.add_argument("--splits-dir", type=str, default=None, help="Path to data_prep/splits/ folder.")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to specific test.csv.")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to specific val.csv.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save test evaluation metrics, CSVs, and plots.",
    )

    # Evaluation settings
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size (default: 4).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold (default: 0.5).")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration.")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 autocast during inference.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency and throughput benchmarking.")
    parser.add_argument(
        "--no-val-sweep",
        action="store_true",
        help="Skip validation-set threshold sweep and tuned evaluation (default: sweep enabled).",
    )

    args = parser.parse_args()

    workspace_root, default_data_dir, default_splits_dir = resolve_workspace_paths(args.data_dir, args.splits_dir)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else default_data_dir
    splits_dir = Path(args.splits_dir).resolve() if args.splits_dir else default_splits_dir
    test_csv = Path(args.test_csv).resolve() if args.test_csv else splits_dir / "test.csv"
    val_csv = Path(args.val_csv).resolve() if args.val_csv else splits_dir / "val.csv"

    # Resolve output directory
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(__file__).resolve().parent / "outputs" / "test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tuned_output_dir = output_dir.parent / f"{output_dir.name}_val_tuned"

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        # Search candidate locations
        script_dir = Path(__file__).resolve().parent
        cand_ckpts = [
            script_dir / "outputs" / "train" / "best_model.pth",
            script_dir / "best_model.pth",
            workspace_root / "outputs" / "train" / "best_model.pth",
        ]
        ckpt_path = None
        for cand in cand_ckpts:
            if cand.is_file():
                ckpt_path = cand
                break
        if ckpt_path is None:
            ckpt_path = cand_ckpts[0]  # Default fallback for logging/error reporting

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path}. "
            f"Please run training first or specify --checkpoint <path>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logging.info(f"Device: {device} ({gpu_name})")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_densenet121(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info(f"Loaded DenseNet-121 model state from epoch {checkpoint.get('epoch', 'N/A')}.")

    transform = get_test_transforms(DEFAULT_IMAGE_SIZE)

    # -------------------------------------------------------------------------
    # STEP A: Validation Threshold Sweep (Strictly BEFORE test evaluation)
    # -------------------------------------------------------------------------
    val_sweep_results = None
    if not args.no_val_sweep:
        if not val_csv.is_file():
            raise FileNotFoundError(f"Validation CSV not found at: {val_csv}")

        val_dataset = TBX11KTestDataset(
            csv_path=val_csv,
            dataset_root=data_dir,
            transform=transform,
            expected_count=690,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )
        val_sweep_results = run_validation_threshold_sweep(
            model=model,
            val_loader=val_loader,
            device=device,
            use_amp=use_amp,
            output_dir=output_dir,
            ckpt_path=ckpt_path,
            checkpoint=checkpoint,
        )

    # -------------------------------------------------------------------------
    # STEP B: Test Set Inference Loop (Executed ONCE)
    # -------------------------------------------------------------------------
    logging.info(f"Loading test dataset from {test_csv}...")
    test_dataset = TBX11KTestDataset(
        csv_path=test_csv,
        dataset_root=data_dir,
        transform=transform,
        expected_count=690,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logging.info(f"Evaluating {len(test_dataset)} test samples...")
    all_targets: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_img_paths: List[str] = []
    all_cohorts: List[str] = []

    with torch.no_grad():
        for images, targets, rel_paths, cohorts in test_loader:
            images = images.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
            all_img_paths.extend(rel_paths)
            all_cohorts.extend(cohorts)

    all_targets_arr = np.vstack(all_targets)
    all_probs_arr = np.vstack(all_probs)

    # -------------------------------------------------------------------------
    # STEP C: Efficiency Benchmarking Suite (Executed ONCE)
    # -------------------------------------------------------------------------
    if not args.skip_benchmark:
        logging.info("Benchmarking latency, FPS, and computational efficiency...")
        efficiency_metrics = benchmark_efficiency(
            model=model,
            device=device,
            use_amp=use_amp,
            input_size=DEFAULT_IMAGE_SIZE,
            eval_batch_size=args.batch_size,
        )
    else:
        efficiency_metrics = {
            "model_architecture": "DenseNet-121",
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "computational_complexity": {"note": "Benchmark skipped via --skip-benchmark"},
            "inference_benchmark": {"note": "Benchmark skipped via --skip-benchmark"},
            "hardware_environment": {
                "device": str(device),
                "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            },
        }

    # -------------------------------------------------------------------------
    # STEP D: Baseline Evaluation at Fixed tau = 0.50 (Exact Schema Identity)
    # -------------------------------------------------------------------------
    logging.info("Computing baseline evaluation metrics suite at fixed tau = 0.50...")
    baseline_metrics = compute_test_metrics(
        y_true=all_targets_arr,
        y_probs=all_probs_arr,
        threshold=args.threshold,
    )
    save_evaluation_bundle(
        bundle_dir=output_dir,
        y_true=all_targets_arr,
        y_probs=all_probs_arr,
        all_img_paths=all_img_paths,
        all_cohorts=all_cohorts,
        test_metrics=baseline_metrics,
        efficiency_metrics=efficiency_metrics,
        checkpoint=checkpoint,
        ckpt_path=ckpt_path,
    )

    # -------------------------------------------------------------------------
    # STEP E: Tuned Evaluation and Comparison (When Sweep Enabled)
    # -------------------------------------------------------------------------
    if val_sweep_results is not None:
        best_th_act = val_sweep_results["best_th_act"]
        best_th_obs = val_sweep_results["best_th_obs"]
        best_th_shared = val_sweep_results["best_th_shared"]

        logging.info(
            f"Computing validation-tuned test metrics (Active: {best_th_act:.4f}, Obsolete: {best_th_obs:.4f})..."
        )
        tuned_test_metrics = compute_test_metrics(
            y_true=all_targets_arr,
            y_probs=all_probs_arr,
            threshold=[best_th_act, best_th_obs],
        )

        save_evaluation_bundle(
            bundle_dir=tuned_output_dir,
            y_true=all_targets_arr,
            y_probs=all_probs_arr,
            all_img_paths=all_img_paths,
            all_cohorts=all_cohorts,
            test_metrics=tuned_test_metrics,
            efficiency_metrics=efficiency_metrics,
            checkpoint=checkpoint,
            ckpt_path=ckpt_path,
        )

        # Write threshold_comparison.csv into output_dir
        comp_csv_path = output_dir / "threshold_comparison.csv"
        generate_threshold_comparison(
            y_test_true=all_targets_arr,
            y_test_probs=all_probs_arr,
            best_th_act=best_th_act,
            best_th_obs=best_th_obs,
            best_th_shared=best_th_shared,
            output_path=comp_csv_path,
        )

    # Print baseline report to stdout
    report_text = generate_text_report(
        test_metrics=baseline_metrics,
        efficiency_metrics=efficiency_metrics,
        checkpoint_info=checkpoint,
    )
    print("\n" + report_text)


if __name__ == "__main__":
    main()
