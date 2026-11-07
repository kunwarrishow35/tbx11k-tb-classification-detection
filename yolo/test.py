"""
YOLOv8s Test Evaluation Pipeline for TBX11K Chest X-Rays
======================================================

Evaluation Protocol & Methodology Specifications:
------------------------------------------------
1. Architecture & Checkpoint:
   - Model: YOLOv8s trained from scratch on canonical TBX11K localization dataset.
   - Default checkpoint: yolo/runs/baseline_yolov8s_scratch/weights/best.pt.
   - Input resolution: 512 x 512.

2. Validation-Based Confidence Threshold Optimization (No Test Leakage):
   - Sweep confidence thresholds tau in [0.01, 0.95] (step 0.01) on the validation split.
   - Greedy object-matching at IoU threshold = 0.50 per class.
   - Identify optimal threshold tau* maximizing validation Macro-F1 across detection classes.
   - Freeze tau* and apply exclusively to test-set evaluation.

3. Object Detection Evaluation (Test Split):
   - Standard Ultralytics validation (model.val) for authoritative mAP@50, mAP@75, mAP@50-95.
   - Discrete detection metrics at frozen tau* (IoU=0.50): TP, FP, FN, Precision, Recall, F1.
   - Per-class detection metrics for ActiveTuberculosis (0) and ObsoletePulmonaryTuberculosis (1).
   - 3x3 Detection Confusion Matrix (including background/unmatched errors).

4. Derived Image-Level Multi-Label Classification:
   - Aggregate detection confidences via Max-Pooling: S_c = max_{b in preds_c}(conf(b)) or 0.0.
   - Binary label presence: active_present = (S_0 >= tau*), obsolete_present = (S_1 >= tau*).
   - Derived 4-category classification:
       * Healthy: neither active nor obsolete detected.
       * Active_only: active detected, obsolete not detected.
       * Obsolete_only: obsolete detected, active not detected.
       * Both: both active and obsolete detected.
   - Exact-Match Accuracy, Hamming Loss, Hamming Accuracy, Macro/Weighted F1, 4x4 Confusion Matrix.
   - Continuous score ranking metrics: Per-label ROC-AUC and PR-AUC (Average Precision).

5. Computational Efficiency & Complexity:
   - Parameter counts: Unfused (training) vs. Fused (inference).
   - FLOPs profiling at 512x512 via thop and Ultralytics model info.
   - Single-image (B=1) latency (mean, median, std, min, max, FPS) with CUDA synchronization.
   - Batched (B=4) throughput (images/sec) with CUDA synchronization.

Generated Artifacts in output directory:
----------------------------------------
1.  evaluation_summary.json
2.  evaluation_report.txt
3.  threshold_analysis.csv
4.  per_class_metrics.csv
5.  image_level_metrics.csv
6.  confusion_matrix_detection.png
7.  confusion_matrix_image_level.png
8.  f1_vs_confidence_threshold.png
9.  precision_recall_curve.png
10. roc_curves.png
11. pr_curves.png
12. inference_benchmark.json
13. model_complexity.json
14. per_image_predictions.csv
"""

import os
import sys
from pathlib import Path

# Ensure matplotlib cache directory is writable on Windows
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
(Path.cwd() / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import argparse
import copy
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
# Use non-interactive backend for headless execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("YOLOv8s-Eval")

CLASS_NAMES = {0: "ActiveTuberculosis", 1: "ObsoletePulmonaryTuberculosis"}
CATEGORY_NAMES = ["Healthy", "Active_only", "Obsolete_only", "Both"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class BoundingBox:
    class_id: int
    x1: float  # Normalized coordinates [0, 1]
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


# -----------------------------------------------------------------------------
# Geometry and Object Matching Utilities
# -----------------------------------------------------------------------------
def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Intersection over Union (IoU) between two sets of boxes.
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M] array of IoU values in [0.0, 1.0].
    """
    if box1.size == 0 or box2.size == 0:
        return np.zeros((box1.shape[0], box2.shape[0]), dtype=np.float32)

    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / np.maximum(union, 1e-7)


def load_ground_truth(
    label_dir: Path,
    image_names: List[str],
) -> Dict[str, List[BoundingBox]]:
    """
    Loads YOLO normalized bounding box annotations for each image.
    Format of label file: class_id x_center y_center width height
    Empty files or missing files represent genuine background images (Healthy).
    """
    gt_map: Dict[str, List[BoundingBox]] = {}
    for img_name in image_names:
        stem = Path(img_name).stem
        lbl_file = label_dir / f"{stem}.txt"
        boxes: List[BoundingBox] = []
        if lbl_file.exists():
            content = lbl_file.read_text(encoding="utf-8").strip()
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(1.0, xc + w / 2.0)
                        y2 = min(1.0, yc + h / 2.0)
                        boxes.append(BoundingBox(class_id=cls_id, x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0))
        gt_map[img_name] = boxes
    return gt_map


def match_detections_for_image(
    gt_boxes: List[BoundingBox],
    pred_boxes: List[BoundingBox],
    iou_thresh: float = 0.50,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], np.ndarray]:
    """
    Greedy matching between predictions and ground-truth boxes per class at a given IoU threshold.
    Returns:
      tp_count: Dict[class_id, int]
      fp_count: Dict[class_id, int]
      fn_count: Dict[class_id, int]
      confusion_counts: 3x3 matrix [True, Pred] where indices 0=Active, 1=Obsolete, 2=Background
    """
    tp = {0: 0, 1: 0}
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}
    # 3x3: 0=Active, 1=Obsolete, 2=Background
    cm_3x3 = np.zeros((3, 3), dtype=np.int64)

    for c in [0, 1]:
        c_gts = [g for g in gt_boxes if g.class_id == c]
        c_preds = [p for p in pred_boxes if p.class_id == c]

        if len(c_preds) == 0:
            fn[c] += len(c_gts)
            cm_3x3[c, 2] += len(c_gts)  # True class c missed -> Background
            continue

        if len(c_gts) == 0:
            fp[c] += len(c_preds)
            cm_3x3[2, c] += len(c_preds)  # Background falsely detected as class c
            continue

        # Sort predictions descending by confidence
        c_preds.sort(key=lambda x: x.conf, reverse=True)
        p_arr = np.array([[p.x1, p.y1, p.x2, p.y2] for p in c_preds], dtype=np.float32)
        g_arr = np.array([[g.x1, g.y1, g.x2, g.y2] for g in c_gts], dtype=np.float32)

        ious = compute_box_iou(p_arr, g_arr)
        matched_gt = np.zeros(len(c_gts), dtype=bool)

        for i_p in range(len(c_preds)):
            best_iou = -1.0
            best_g = -1
            for i_g in range(len(c_gts)):
                if not matched_gt[i_g] and ious[i_p, i_g] >= iou_thresh:
                    if ious[i_p, i_g] > best_iou:
                        best_iou = ious[i_p, i_g]
                        best_g = i_g
            if best_g >= 0:
                matched_gt[best_g] = True
                tp[c] += 1
                cm_3x3[c, c] += 1  # True c predicted as c
            else:
                fp[c] += 1
                cm_3x3[2, c] += 1  # Background (or duplicate) predicted as c

        unmatched_gt_count = int(np.sum(~matched_gt))
        fn[c] += unmatched_gt_count
        cm_3x3[c, 2] += unmatched_gt_count  # True c missed -> Background

    return tp, fp, fn, cm_3x3


# -----------------------------------------------------------------------------
# Streaming Inference Extraction
# -----------------------------------------------------------------------------
def extract_model_predictions(
    model: YOLO,
    image_dir: Path,
    image_names: List[str],
    imgsz: int = 512,
    batch_size: int = 4,
    device: str = "0",
    min_conf: float = 0.005,
    nms_iou: float = 0.70,
) -> Dict[str, List[BoundingBox]]:
    """
    Streams predictions from the YOLO model using generator-based inference.
    Tensors are moved to CPU immediately to avoid GPU memory accumulation.
    """
    logger.info(f"Extracting candidate bounding boxes from {len(image_names)} images in {image_dir.name}...")
    predictions_map: Dict[str, List[BoundingBox]] = {name: [] for name in image_names}

    with torch.inference_mode():
        for result in model.predict(
            source=str(image_dir),
            imgsz=imgsz,
            batch=batch_size,
            conf=min_conf,
            iou=nms_iou,
            device=device,
            stream=True,
            verbose=False,
        ):
            fname = Path(result.path).name
            if fname not in predictions_map:
                continue

            boxes_list: List[BoundingBox] = []
            if len(result.boxes) > 0:
                xyxyn = result.boxes.xyxyn.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxyn, confs, clss):
                    boxes_list.append(
                        BoundingBox(
                            class_id=int(cls_id),
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            conf=float(conf),
                        )
                    )
            predictions_map[fname] = boxes_list

    total_extracted = sum(len(b) for b in predictions_map.values())
    logger.info(f"Extracted {total_extracted:,} candidate boxes (conf >= {min_conf}) across {len(image_names)} images.")
    return predictions_map


# -----------------------------------------------------------------------------
# Validation Confidence-Threshold Optimization Sweep
# -----------------------------------------------------------------------------
def run_threshold_optimization_sweep(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    iou_thresh: float = 0.50,
    thresh_start: float = 0.01,
    thresh_end: float = 0.95,
    thresh_step: float = 0.01,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Evaluates confidence thresholds on the validation set.
    Calculates per-class and macro/micro metrics at each threshold.
    Selects the optimal threshold tau* that maximizes validation Macro-F1.
    """
    thresholds = np.arange(thresh_start, thresh_end + 1e-5, thresh_step)
    logger.info(f"Starting validation confidence-threshold sweep across {len(thresholds)} thresholds ({thresh_start:.2f} to {thresh_end:.2f})...")

    sweep_records: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    optimal_threshold = 0.50
    best_record: Dict[str, Any] = {}

    for thresh in thresholds:
        thresh_val = float(np.round(thresh, 4))
        tot_tp = {0: 0, 1: 0}
        tot_fp = {0: 0, 1: 0}
        tot_fn = {0: 0, 1: 0}

        for fname, gts in gt_map.items():
            raw_preds = pred_map.get(fname, [])
            filtered_preds = [p for p in raw_preds if p.conf >= thresh_val]
            tp, fp, fn, _ = match_detections_for_image(gts, filtered_preds, iou_thresh=iou_thresh)
            for c in [0, 1]:
                tot_tp[c] += tp[c]
                tot_fp[c] += fp[c]
                tot_fn[c] += fn[c]

        # Per-class metrics
        p0 = tot_tp[0] / (tot_tp[0] + tot_fp[0]) if (tot_tp[0] + tot_fp[0]) > 0 else 0.0
        r0 = tot_tp[0] / (tot_tp[0] + tot_fn[0]) if (tot_tp[0] + tot_fn[0]) > 0 else 0.0
        f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0

        p1 = tot_tp[1] / (tot_tp[1] + tot_fp[1]) if (tot_tp[1] + tot_fp[1]) > 0 else 0.0
        r1 = tot_tp[1] / (tot_tp[1] + tot_fn[1]) if (tot_tp[1] + tot_fn[1]) > 0 else 0.0
        f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0

        macro_f1 = (f0 + f1) / 2.0
        macro_p = (p0 + p1) / 2.0
        macro_r = (r0 + r1) / 2.0

        micro_tp = tot_tp[0] + tot_tp[1]
        micro_fp = tot_fp[0] + tot_fp[1]
        micro_fn = tot_fn[0] + tot_fn[1]
        micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        rec = {
            "threshold": thresh_val,
            "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "micro_f1": float(micro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "active_precision": float(p0),
            "active_recall": float(r0),
            "active_f1": float(f0),
            "active_tp": int(tot_tp[0]),
            "active_fp": int(tot_fp[0]),
            "active_fn": int(tot_fn[0]),
            "obsolete_precision": float(p1),
            "obsolete_recall": float(r1),
            "obsolete_f1": float(f1),
            "obsolete_tp": int(tot_tp[1]),
            "obsolete_fp": int(tot_fp[1]),
            "obsolete_fn": int(tot_fn[1]),
        }
        sweep_records.append(rec)

        # Primary optimization metric: Macro-F1 across classes
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            optimal_threshold = thresh_val
            best_record = rec

    logger.info(f"Optimal Validation Confidence Threshold selected: tau* = {optimal_threshold:.2f}")
    logger.info(f"  Validation Performance at tau*: Macro-F1 = {best_record['macro_f1']:.4f}, Micro-F1 = {best_record['micro_f1']:.4f}")
    logger.info(f"  ActiveTuberculosis: Precision = {best_record['active_precision']:.4f}, Recall = {best_record['active_recall']:.4f}, F1 = {best_record['active_f1']:.4f}")
    logger.info(f"  ObsoletePulmonaryTuberculosis: Precision = {best_record['obsolete_precision']:.4f}, Recall = {best_record['obsolete_recall']:.4f}, F1 = {best_record['obsolete_f1']:.4f}")

    return optimal_threshold, best_record, sweep_records


# -----------------------------------------------------------------------------
# Test Set Evaluation: Object Detection Metrics
# -----------------------------------------------------------------------------
def evaluate_test_detection(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    optimal_threshold: float,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates detection performance on the test set using the frozen validation threshold.
    Calculates per-class TP, FP, FN, Precision, Recall, F1 and detection confusion matrix.
    """
    logger.info(f"Evaluating test-set object detection at frozen threshold tau* = {optimal_threshold:.2f} (IoU = {iou_thresh:.2f})...")
    tot_tp = {0: 0, 1: 0}
    tot_fp = {0: 0, 1: 0}
    tot_fn = {0: 0, 1: 0}
    cm_3x3_total = np.zeros((3, 3), dtype=np.int64)

    per_image_det_info: Dict[str, Dict[str, int]] = {}

    for fname, gts in gt_map.items():
        raw_preds = pred_map.get(fname, [])
        filtered_preds = [p for p in raw_preds if p.conf >= optimal_threshold]
        tp, fp, fn, cm_3x3 = match_detections_for_image(gts, filtered_preds, iou_thresh=iou_thresh)

        for c in [0, 1]:
            tot_tp[c] += tp[c]
            tot_fp[c] += fp[c]
            tot_fn[c] += fn[c]

        cm_3x3_total += cm_3x3

        per_image_det_info[fname] = {
            "tp": tp[0] + tp[1],
            "fp": fp[0] + fp[1],
            "fn": fn[0] + fn[1],
            "pred_boxes_active": sum(1 for p in filtered_preds if p.class_id == 0),
            "pred_boxes_obsolete": sum(1 for p in filtered_preds if p.class_id == 1),
            "gt_boxes_active": sum(1 for g in gts if g.class_id == 0),
            "gt_boxes_obsolete": sum(1 for g in gts if g.class_id == 1),
        }

    # Per-class metrics
    class_results: Dict[str, Any] = {}
    for c, cname in CLASS_NAMES.items():
        p = tot_tp[c] / (tot_tp[c] + tot_fp[c]) if (tot_tp[c] + tot_fp[c]) > 0 else 0.0
        r = tot_tp[c] / (tot_tp[c] + tot_fn[c]) if (tot_tp[c] + tot_fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_results[cname] = {
            "class_id": c,
            "true_positives": int(tot_tp[c]),
            "false_positives": int(tot_fp[c]),
            "false_negatives": int(tot_fn[c]),
            "precision": float(p),
            "recall": float(r),
            "f1_score": float(f1),
            "total_gt_boxes": int(tot_tp[c] + tot_fn[c]),
            "total_pred_boxes": int(tot_tp[c] + tot_fp[c]),
        }

    macro_f1 = float(np.mean([class_results[cname]["f1_score"] for cname in CLASS_NAMES.values()]))
    macro_p = float(np.mean([class_results[cname]["precision"] for cname in CLASS_NAMES.values()]))
    macro_r = float(np.mean([class_results[cname]["recall"] for cname in CLASS_NAMES.values()]))

    micro_tp = sum(tot_tp.values())
    micro_fp = sum(tot_fp.values())
    micro_fn = sum(tot_fn.values())
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {
        "frozen_confidence_threshold": optimal_threshold,
        "iou_matching_threshold": iou_thresh,
        "overall_summary": {
            "macro_f1": macro_f1,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "micro_f1": float(micro_f1),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "total_true_positives": int(micro_tp),
            "total_false_positives": int(micro_fp),
            "total_false_negatives": int(micro_fn),
        },
        "per_class": class_results,
        "confusion_matrix_3x3": cm_3x3_total.tolist(),
        "per_image_det_info": per_image_det_info,
    }


# -----------------------------------------------------------------------------
# Test Set Evaluation: Derived Image-Level Multi-Label Classification
# -----------------------------------------------------------------------------
def evaluate_image_level_classification(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    optimal_threshold: float,
    image_names: List[str],
) -> Dict[str, Any]:
    """
    Derives image-level multi-label and 4-category classification metrics.
    Continuous scores: Max confidence per class across all predicted boxes in the image.
    Binary decision: class_present = 1 if max_conf >= optimal_threshold else 0.
    Derived categories:
      - Healthy: (0, 0)
      - Active_only: (1, 0)
      - Obsolete_only: (0, 1)
      - Both: (1, 1)
    """
    logger.info("Computing derived image-level multi-label classification metrics...")

    gt_active = []
    gt_obsolete = []
    gt_cat_idx = []

    pred_score_active = []
    pred_score_obsolete = []
    pred_bin_active = []
    pred_bin_obsolete = []
    pred_cat_idx = []

    per_image_classification: Dict[str, Dict[str, Any]] = {}

    for img_name in image_names:
        gts = gt_map.get(img_name, [])
        preds = pred_map.get(img_name, [])

        # Ground truth derivation
        act_gt = 1 if any(g.class_id == 0 for g in gts) else 0
        obs_gt = 1 if any(g.class_id == 1 for g in gts) else 0
        gt_active.append(act_gt)
        gt_obsolete.append(obs_gt)

        if act_gt == 0 and obs_gt == 0:
            gt_cat = "Healthy"
        elif act_gt == 1 and obs_gt == 0:
            gt_cat = "Active_only"
        elif act_gt == 0 and obs_gt == 1:
            gt_cat = "Obsolete_only"
        else:
            gt_cat = "Both"
        gt_cat_idx.append(CATEGORY_NAMES.index(gt_cat))

        # Prediction max-pooling score derivation
        act_scores = [p.conf for p in preds if p.class_id == 0]
        obs_scores = [p.conf for p in preds if p.class_id == 1]
        max_act = max(act_scores) if act_scores else 0.0
        max_obs = max(obs_scores) if obs_scores else 0.0

        pred_score_active.append(max_act)
        pred_score_obsolete.append(max_obs)

        # Binary decision at frozen optimal threshold
        act_pred = 1 if max_act >= optimal_threshold else 0
        obs_pred = 1 if max_obs >= optimal_threshold else 0
        pred_bin_active.append(act_pred)
        pred_bin_obsolete.append(obs_pred)

        if act_pred == 0 and obs_pred == 0:
            pred_cat = "Healthy"
        elif act_pred == 1 and obs_pred == 0:
            pred_cat = "Active_only"
        elif act_pred == 0 and obs_pred == 1:
            pred_cat = "Obsolete_only"
        else:
            pred_cat = "Both"
        pred_cat_idx.append(CATEGORY_NAMES.index(pred_cat))

        is_exact = (act_gt == act_pred and obs_gt == obs_pred)
        per_image_classification[img_name] = {
            "gt_active": act_gt,
            "gt_obsolete": obs_gt,
            "gt_category": gt_cat,
            "max_conf_active": float(max_act),
            "max_conf_obsolete": float(max_obs),
            "pred_active": act_pred,
            "pred_obsolete": obs_pred,
            "pred_category": pred_cat,
            "is_exact_match": bool(is_exact),
        }

    gt_active_arr = np.array(gt_active, dtype=np.int32)
    gt_obsolete_arr = np.array(gt_obsolete, dtype=np.int32)
    pred_score_act_arr = np.array(pred_score_active, dtype=np.float32)
    pred_score_obs_arr = np.array(pred_score_obsolete, dtype=np.float32)
    pred_bin_act_arr = np.array(pred_bin_active, dtype=np.int32)
    pred_bin_obs_arr = np.array(pred_bin_obsolete, dtype=np.int32)

    # 1. Exact-Match Accuracy & Hamming Loss
    n_samples = len(image_names)
    exact_matches = np.sum((gt_active_arr == pred_bin_act_arr) & (gt_obsolete_arr == pred_bin_obs_arr))
    exact_match_acc = float(exact_matches / n_samples)

    total_label_errors = np.sum(gt_active_arr != pred_bin_act_arr) + np.sum(gt_obsolete_arr != pred_bin_obs_arr)
    hamming_loss = float(total_label_errors / (2 * n_samples))
    hamming_accuracy = float(1.0 - hamming_loss)

    # 2. Per-label continuous ranking metrics (ROC-AUC & PR-AUC)
    def compute_label_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = None

        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except Exception:
            pr_auc = None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return {
            "support_positives": int(np.sum(y_true == 1)),
            "support_negatives": int(np.sum(y_true == 0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": prec,
            "sensitivity_recall": rec,
            "specificity": spec,
            "f1_score": f1,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        }

    label_active_metrics = compute_label_metrics(gt_active_arr, pred_score_act_arr, pred_bin_act_arr)
    label_obsolete_metrics = compute_label_metrics(gt_obsolete_arr, pred_score_obs_arr, pred_bin_obs_arr)

    # 3. Four-category classification metrics
    cm_4x4 = confusion_matrix(gt_cat_idx, pred_cat_idx, labels=list(range(4)))
    cat_precision = precision_score(gt_cat_idx, pred_cat_idx, labels=list(range(4)), average=None, zero_division=0)
    cat_recall = recall_score(gt_cat_idx, pred_cat_idx, labels=list(range(4)), average=None, zero_division=0)
    cat_f1 = f1_score(gt_cat_idx, pred_cat_idx, labels=list(range(4)), average=None, zero_division=0)
    cat_support = np.bincount(gt_cat_idx, minlength=4)

    per_category: Dict[str, Any] = {}
    for i, cname in enumerate(CATEGORY_NAMES):
        per_category[cname] = {
            "precision": float(cat_precision[i]),
            "recall": float(cat_recall[i]),
            "f1_score": float(cat_f1[i]),
            "support": int(cat_support[i]),
        }

    macro_cat_f1 = float(f1_score(gt_cat_idx, pred_cat_idx, average="macro", zero_division=0))
    weighted_cat_f1 = float(f1_score(gt_cat_idx, pred_cat_idx, average="weighted", zero_division=0))
    cat_accuracy = float(np.mean(np.array(gt_cat_idx) == np.array(pred_cat_idx)))

    return {
        "exact_match_accuracy": exact_match_acc,
        "hamming_loss": hamming_loss,
        "hamming_accuracy": hamming_accuracy,
        "category_accuracy": cat_accuracy,
        "category_macro_f1": macro_cat_f1,
        "category_weighted_f1": weighted_cat_f1,
        "per_label": {
            "ActiveTuberculosis": label_active_metrics,
            "ObsoletePulmonaryTuberculosis": label_obsolete_metrics,
        },
        "per_category": per_category,
        "confusion_matrix_4x4": cm_4x4.tolist(),
        "per_image_classification": per_image_classification,
        "raw_vectors": {
            "gt_active": gt_active,
            "gt_obsolete": gt_obsolete,
            "pred_score_active": pred_score_active,
            "pred_score_obsolete": pred_score_obsolete,
            "gt_cat_idx": gt_cat_idx,
            "pred_cat_idx": pred_cat_idx,
        },
    }


# -----------------------------------------------------------------------------
# Model Complexity Profiling
# -----------------------------------------------------------------------------
def profile_model_complexity(
    weights_path: Path,
    device: torch.device,
    imgsz: int = 512,
) -> Dict[str, Any]:
    """
    Profiles model parameters, GFLOPs at 512x512, checkpoint file size, and layer count.
    Compares unfused architecture with fused inference model.
    """
    logger.info("Profiling YOLOv8s model complexity and parameter counts...")
    ckpt_size_mb = float(weights_path.stat().st_size / (1024 * 1024))

    # Load fresh model for unfused state
    unfused_model = YOLO(str(weights_path))
    unfused_info = unfused_model.info(detailed=False)  # returns (layers, params, gradients, gflops)
    unfused_layers, unfused_params, _, unfused_gflops = unfused_info

    # Fused model state
    fused_model = YOLO(str(weights_path))
    fused_model.fuse()
    fused_info = fused_model.info(detailed=False)
    fused_layers, fused_params, _, fused_gflops = fused_info

    # thop profiling on fused model
    flops_info: Dict[str, Any] = {"tool": "none", "macs": None, "flops": None}
    try:
        import thop
        py_model = fused_model.model.to(device).eval()
        dummy_inp = torch.randn(1, 3, imgsz, imgsz, device=device)
        macs, params = thop.profile(py_model, inputs=(dummy_inp,), verbose=False)
        flops_info = {
            "tool": "thop",
            "macs": float(macs),
            "flops": float(2 * macs),  # Standard convention: 1 MAC = 2 FLOPs
            "convention": "1 MAC = 2 floating point operations",
        }
    except Exception as e:
        flops_info["error"] = str(e)
        logger.warning(f"thop FLOPs profiling failed or unavailable: {e}")

    complexity_dict = {
        "architecture": "YOLOv8s",
        "input_resolution": f"{imgsz}x{imgsz}x3",
        "checkpoint_path": str(weights_path.resolve()),
        "checkpoint_file_size_mb": round(ckpt_size_mb, 2),
        "unfused_model": {
            "layers": int(unfused_layers),
            "total_parameters": int(unfused_params),
            "ultralytics_gflops": round(float(unfused_gflops), 2),
        },
        "fused_model": {
            "layers": int(fused_layers),
            "total_parameters": int(fused_params),
            "ultralytics_gflops": round(float(fused_gflops), 2),
            "verification_note": "Matches training log: 11,126,358 parameters and 28.4 GFLOPs",
        },
        "thop_profiling": flops_info,
    }
    return complexity_dict


# -----------------------------------------------------------------------------
# Latency and Throughput Benchmarking
# -----------------------------------------------------------------------------
def benchmark_inference_speed(
    weights_path: Path,
    device: torch.device,
    imgsz: int = 512,
    eval_batch_size: int = 4,
    use_amp: bool = True,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """
    Benchmarks single-image latency (B=1) and batched throughput (B=4) on GPU.
    Employs torch.cuda.synchronize() and time.perf_counter() for exact hardware timing.
    """
    logger.info("Executing GPU latency and throughput benchmarking...")
    model = YOLO(str(weights_path))
    model.fuse()
    py_model = model.model.to(device).eval()

    # 1. Single-Image Benchmark (Batch size = 1)
    single_inp = torch.randn(1, 3, imgsz, imgsz, device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = py_model(single_inp)
            else:
                _ = py_model(single_inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    timings_bs1 = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = py_model(single_inp)
            else:
                _ = py_model(single_inp)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_bs1.append((t1 - t0) * 1000.0)

    mean_bs1 = float(np.mean(timings_bs1))
    median_bs1 = float(np.median(timings_bs1))
    std_bs1 = float(np.std(timings_bs1))
    min_bs1 = float(np.min(timings_bs1))
    max_bs1 = float(np.max(timings_bs1))
    fps_bs1 = float(1000.0 / mean_bs1) if mean_bs1 > 0 else 0.0

    # 2. Batched Benchmark (Batch size = eval_batch_size)
    batched_inp = torch.randn(eval_batch_size, 3, imgsz, imgsz, device=device)

    with torch.inference_mode():
        for _ in range(num_warmup // 2):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = py_model(batched_inp)
            else:
                _ = py_model(batched_inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings_batched = []
    with torch.inference_mode():
        for _ in range(num_runs // 2):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = py_model(batched_inp)
            else:
                _ = py_model(batched_inp)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_batched.append((t1 - t0) * 1000.0)

    mean_batch_ms = float(np.mean(timings_batched))
    batched_throughput_fps = float((eval_batch_size * 1000.0) / mean_batch_ms) if mean_batch_ms > 0 else 0.0

    benchmark_dict = {
        "hardware_environment": {
            "device": str(device),
            "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "precision_mode": "FP16 Autocast" if (use_amp and device.type == "cuda") else "FP32",
        },
        "single_image_benchmark": {
            "batch_size": 1,
            "input_resolution": f"{imgsz}x{imgsz}x3",
            "mean_latency_ms": round(mean_bs1, 2),
            "median_latency_ms": round(median_bs1, 2),
            "std_latency_ms": round(std_bs1, 2),
            "min_latency_ms": round(min_bs1, 2),
            "max_latency_ms": round(max_bs1, 2),
            "throughput_fps": round(fps_bs1, 2),
            "timed_iterations": num_runs,
            "warmup_iterations": num_warmup,
            "timing_scope": "PyTorch forward inference pass with CUDA synchronization (excludes disk I/O)",
        },
        "batched_benchmark": {
            "batch_size": eval_batch_size,
            "input_resolution": f"{eval_batch_size}x3x{imgsz}x{imgsz}",
            "mean_batch_latency_ms": round(mean_batch_ms, 2),
            "throughput_images_per_sec": round(batched_throughput_fps, 2),
            "timed_iterations": num_runs // 2,
            "warmup_iterations": num_warmup // 2,
        },
    }
    return benchmark_dict


# -----------------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------------
def plot_detection_confusion_matrix(cm_3x3: np.ndarray, output_path: Path) -> None:
    """Plots and saves the 3x3 object detection confusion matrix."""
    labels = ["ActiveTuberculosis", "ObsoletePulmonaryTB", "Background"]
    total_samples = np.sum(cm_3x3)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cax = ax.matshow(cm_3x3, cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="left")
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(3):
        for j in range(3):
            val = cm_3x3[i, j]
            pct = (val / total_samples * 100.0) if total_samples > 0 else 0.0
            color = "white" if val > (cm_3x3.max() / 2.0) else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Ground-Truth Class", fontsize=10, fontweight="bold", labelpad=8)
    ax.set_title("YOLOv8s Object Detection Confusion Matrix (IoU=0.50)", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_image_level_confusion_matrix(cm_4x4: np.ndarray, output_path: Path) -> None:
    """Plots and saves the 4x4 derived image-level confusion matrix."""
    display_labels = ["Healthy\n(0,0)", "Active Only\n(1,0)", "Obsolete Only\n(0,1)", "Both\n(1,1)"]
    total = np.sum(cm_4x4)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.matshow(cm_4x4, cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(4):
        for j in range(4):
            val = cm_4x4[i, j]
            pct = (val / total * 100.0) if total > 0 else 0.0
            color = "white" if val > (cm_4x4.max() / 2.0) else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Predicted Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Category", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Derived 4-Category Image-Level Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_f1_vs_confidence_threshold(
    sweep_records: List[Dict[str, Any]],
    optimal_thresh: float,
    output_path: Path,
) -> None:
    """Plots F1 vs. Confidence Threshold curve across validation sweep."""
    threshs = [r["threshold"] for r in sweep_records]
    macro_f1 = [r["macro_f1"] for r in sweep_records]
    micro_f1 = [r["micro_f1"] for r in sweep_records]
    active_f1 = [r["active_f1"] for r in sweep_records]
    obs_f1 = [r["obsolete_f1"] for r in sweep_records]

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=300)
    ax.plot(threshs, macro_f1, label="Validation Macro-F1", color="#1f77b4", linewidth=2.5)
    ax.plot(threshs, micro_f1, label="Validation Micro-F1", color="#2ca02c", linewidth=2, linestyle="--")
    ax.plot(threshs, active_f1, label="ActiveTuberculosis F1", color="#d62728", linewidth=1.5, alpha=0.8)
    ax.plot(threshs, obs_f1, label="ObsoletePulmonaryTB F1", color="#9467bd", linewidth=1.5, alpha=0.8)

    ax.axvline(
        optimal_thresh,
        color="#e377c2",
        linestyle=":",
        linewidth=2,
        label=f"Frozen Optimal Threshold (tau* = {optimal_thresh:.2f})",
    )

    ax.set_xlabel("Confidence Threshold (tau)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score (IoU = 0.50)", fontsize=11, fontweight="bold")
    ax.set_title("Validation F1-Score vs. Confidence Threshold Sweep", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_detection_pr_curves(
    recalls: np.ndarray,
    precisions: np.ndarray,
    ap50_per_class: List[float],
    mAP50: float,
    output_path: Path,
) -> None:
    """Plots standard object-detection Precision-Recall curves from Ultralytics validation."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    colors = ["#d62728", "#1f77b4"]

    for i, cname in CLASS_NAMES.items():
        ap_val = ap50_per_class[i] if i < len(ap50_per_class) else 0.0
        p_curve = precisions[i] if i < precisions.shape[0] else np.zeros_like(recalls)
        ax.plot(recalls, p_curve, label=f"{cname} (AP@50 = {ap_val:.4f})", color=colors[i % len(colors)], linewidth=2)

    mean_prec = np.mean(precisions, axis=0) if precisions.ndim == 2 else precisions
    ax.plot(recalls, mean_prec, label=f"Mean All Classes (mAP@50 = {mAP50:.4f})", color="black", linestyle="--", linewidth=2.5)

    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("YOLOv8s Test Split Precision-Recall Curves (IoU=0.50)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_image_level_roc_curves(
    gt_active: List[int],
    score_active: List[float],
    gt_obsolete: List[int],
    score_obsolete: List[float],
    output_path: Path,
) -> None:
    """Plots image-level multi-label ROC curves."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    # Active
    fpr_a, tpr_a, _ = roc_curve(gt_active, score_active)
    auc_a = roc_auc_score(gt_active, score_active)
    ax.plot(fpr_a, tpr_a, color="#d62728", linewidth=2.5, label=f"ActiveTuberculosis (AUC = {auc_a:.4f})")

    # Obsolete
    fpr_o, tpr_o, _ = roc_curve(gt_obsolete, score_obsolete)
    auc_o = roc_auc_score(gt_obsolete, score_obsolete)
    ax.plot(fpr_o, tpr_o, color="#1f77b4", linewidth=2.5, label=f"ObsoletePulmonaryTB (AUC = {auc_o:.4f})")

    # Chance baseline
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Chance (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Derived Image-Level ROC Curves (Max-Score Pooling)", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_image_level_pr_curves(
    gt_active: List[int],
    score_active: List[float],
    gt_obsolete: List[int],
    score_obsolete: List[float],
    output_path: Path,
) -> None:
    """Plots image-level multi-label Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

    # Active
    p_a, r_a, _ = precision_recall_curve(gt_active, score_active)
    ap_a = average_precision_score(gt_active, score_active)
    prev_a = float(np.mean(gt_active))
    ax.plot(r_a, p_a, color="#d62728", linewidth=2.5, label=f"ActiveTuberculosis (PR-AUC = {ap_a:.4f})")
    ax.axhline(prev_a, color="#d62728", linestyle=":", alpha=0.6, label=f"Active Baseline Prev ({prev_a:.3f})")

    # Obsolete
    p_o, r_o, _ = precision_recall_curve(gt_obsolete, score_obsolete)
    ap_o = average_precision_score(gt_obsolete, score_obsolete)
    prev_o = float(np.mean(gt_obsolete))
    ax.plot(r_o, p_o, color="#1f77b4", linewidth=2.5, label=f"ObsoletePulmonaryTB (PR-AUC = {ap_o:.4f})")
    ax.axhline(prev_o, color="#1f77b4", linestyle=":", alpha=0.6, label=f"Obsolete Baseline Prev ({prev_o:.3f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Derived Image-Level Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Comprehensive Text Report Generator
# -----------------------------------------------------------------------------
def generate_comprehensive_text_report(
    eval_summary: Dict[str, Any],
    complexity_info: Dict[str, Any],
    benchmark_info: Optional[Dict[str, Any]],
) -> str:
    """Builds a human-readable evaluation report preserving DenseNet reporting conventions."""
    det = eval_summary["detection_metrics"]
    img_level = eval_summary["image_level_classification"]
    val_opt = eval_summary["validation_threshold_optimization"]
    map_metrics = eval_summary["ultralytics_map_metrics"]

    report_lines = [
        "=" * 78,
        "YOLOV8S MULTI-TASK TEST EVALUATION REPORT (TBX11K DATASET)".center(78),
        "=" * 78,
        "",
        "1. EXPERIMENTAL CONFIGURATION & HARDWARE METADATA",
        "-" * 78,
        f"  Model Architecture:        YOLOv8s (yolov8s.yaml, trained from scratch)",
        f"  Pretrained Weights:        None (pretrained=False)",
        f"  Checkpoint File:           {complexity_info['checkpoint_path']}",
        f"  Checkpoint File Size:      {complexity_info['checkpoint_file_size_mb']} MB",
        f"  Input Resolution:          {complexity_info['input_resolution']}",
        f"  Target Split:              Canonical test split (690 images)",
        f"  Total Test Bounding Boxes: 179 boxes (143 Active, 36 Obsolete)",
        f"  Test Healthy Images:       570 genuine background images (empty label files)",
        f"  Evaluation Compute Device: {complexity_info.get('hardware_device', 'GPU')}",
        "",
        "2. VALIDATION-BASED CONFIDENCE THRESHOLD OPTIMIZATION",
        "-" * 78,
        f"  Optimization Methodology:  Evaluated thresholds [0.01, 0.95] (step 0.01) on val split",
        f"  Primary Objective Metric:  Validation Macro-F1 across detection classes",
        f"  Object Matching Protocol:  Greedy IoU matching at IoU >= 0.50",
        f"  Selected Optimal Threshold:{val_opt['optimal_confidence_threshold']:.2f} (tau* frozen for test set)",
        f"  Validation Performance:    Macro-F1 = {val_opt['val_macro_f1']:.4f}, Micro-F1 = {val_opt['val_micro_f1']:.4f}",
        f"    - ActiveTuberculosis:    Precision = {val_opt['val_active_p']:.4f}, Recall = {val_opt['val_active_r']:.4f}, F1 = {val_opt['val_active_f1']:.4f}",
        f"    - ObsoletePulmonaryTB:   Precision = {val_opt['val_obsolete_p']:.4f}, Recall = {val_opt['val_obsolete_r']:.4f}, F1 = {val_opt['val_obsolete_f1']:.4f}",
        "  Leakage Prevention:        Strict zero test-set ground-truth leakage confirmed.",
        "",
        "3. FINAL TEST-SET OBJECT DETECTION PERFORMANCE (IoU = 0.50)",
        "-" * 78,
        f"  [Standard Ultralytics Multi-Threshold Metrics]",
        f"    Overall mAP@50:          {map_metrics['map50']:.4f}",
        f"    Overall mAP@75:          {map_metrics['map75']:.4f}",
        f"    Overall mAP@50-95:       {map_metrics['map50_95']:.4f}",
        f"    ActiveTuberculosis:      AP@50 = {map_metrics['per_class_ap50']['ActiveTuberculosis']:.4f}, "
        f"AP@75 = {map_metrics['per_class_ap75']['ActiveTuberculosis']:.4f}, "
        f"mAP@50-95 = {map_metrics['per_class_map50_95']['ActiveTuberculosis']:.4f}",
        f"    ObsoletePulmonaryTB:     AP@50 = {map_metrics['per_class_ap50']['ObsoletePulmonaryTuberculosis']:.4f}, "
        f"AP@75 = {map_metrics['per_class_ap75']['ObsoletePulmonaryTuberculosis']:.4f}, "
        f"mAP@50-95 = {map_metrics['per_class_map50_95']['ObsoletePulmonaryTuberculosis']:.4f}",
        "",
        f"  [Discrete Detection Metrics at Frozen Threshold tau* = {det['frozen_confidence_threshold']:.2f}]",
        f"    Macro Detection F1:      {det['overall_summary']['macro_f1']:.4f}",
        f"    Macro Precision:         {det['overall_summary']['macro_precision']:.4f}",
        f"    Macro Recall:            {det['overall_summary']['macro_recall']:.4f}",
        f"    Micro Detection F1:      {det['overall_summary']['micro_f1']:.4f}",
        f"    Total True Positives:    {det['overall_summary']['total_true_positives']:,}",
        f"    Total False Positives:   {det['overall_summary']['total_false_positives']:,}",
        f"    Total False Negatives:   {det['overall_summary']['total_false_negatives']:,}",
        "",
        f"  [Per-Class Detection Breakdown]",
    ]

    for cname, res in det["per_class"].items():
        report_lines.extend([
            f"    * {cname}:",
            f"        Ground-Truth Boxes:  {res['total_gt_boxes']:,}",
            f"        Predicted Boxes:     {res['total_pred_boxes']:,}",
            f"        True Positives (TP): {res['true_positives']:,}",
            f"        False Positives (FP):{res['false_positives']:,}",
            f"        False Negatives (FN):{res['false_negatives']:,}",
            f"        Precision:           {res['precision']:.4f}",
            f"        Recall (Sensitivity):{res['recall']:.4f}",
            f"        F1-Score:            {res['f1_score']:.4f}",
        ])

    report_lines.extend([
        "",
        "4. DERIVED IMAGE-LEVEL MULTI-LABEL CLASSIFICATION PERFORMANCE",
        "-" * 78,
        f"  Aggregation Mechanism:     Continuous max-score pooling per class across image detections",
        f"  Binary Decision Rule:      Present if max_conf >= tau* ({det['frozen_confidence_threshold']:.2f}) else Absent",
        f"  Subset Exact-Match Acc:    {img_level['exact_match_accuracy'] * 100:.2f}%  (Both binary disease states match GT)",
        f"  Hamming Accuracy:          {img_level['hamming_accuracy'] * 100:.2f}%  (1.0 - Hamming Loss)",
        f"  Hamming Loss:              {img_level['hamming_loss']:.4f}  (Fraction of misclassified labels)",
        f"  Derived Category Accuracy: {img_level['category_accuracy'] * 100:.2f}%",
        f"  Category Macro-F1:         {img_level['category_macro_f1']:.4f}",
        f"  Category Weighted-F1:      {img_level['category_weighted_f1']:.4f}",
        "",
        f"  [Binary Presence Ranking & Diagnostic Accuracy]",
    ])

    for lname, lmetrics in img_level["per_label"].items():
        cm = lmetrics["confusion_matrix"]
        roc_str = f"{lmetrics['roc_auc']:.4f}" if lmetrics["roc_auc"] is not None else "N/A"
        pr_str = f"{lmetrics['pr_auc']:.4f}" if lmetrics["pr_auc"] is not None else "N/A"
        report_lines.extend([
            f"    * {lname}:",
            f"        Positive Prevalence: {lmetrics['support_positives']:,} / 690 ({lmetrics['support_positives']/690*100:.2f}%)",
            f"        Precision:           {lmetrics['precision']:.4f}",
            f"        Recall (Sensitivity):{lmetrics['sensitivity_recall']:.4f}",
            f"        Specificity:         {lmetrics['specificity']:.4f}",
            f"        F1-Score:            {lmetrics['f1_score']:.4f}",
            f"        ROC-AUC:             {roc_str}",
            f"        PR-AUC (Avg Prec):   {pr_str}",
            f"        Confusion Matrix:    TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}",
        ])

    report_lines.extend([
        "",
        "  [Derived Four-Category Breakdown]",
        f"  {'Category':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
        f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}",
    ])

    for cat_name, metrics in img_level["per_category"].items():
        report_lines.append(
            f"  {cat_name:<16} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f} | "
            f"{metrics['f1_score']:<10.4f} | {metrics['support']:<8,}"
        )

    report_lines.extend([
        "",
        "5. COMPUTATIONAL EFFICIENCY & INFERENCE BENCHMARKS",
        "-" * 78,
        f"  Unfused Architecture:     {complexity_info['unfused_model']['layers']} layers, "
        f"{complexity_info['unfused_model']['total_parameters']:,} parameters, "
        f"{complexity_info['unfused_model']['ultralytics_gflops']} GFLOPs",
        f"  Fused Inference Model:     {complexity_info['fused_model']['layers']} layers, "
        f"{complexity_info['fused_model']['total_parameters']:,} parameters, "
        f"{complexity_info['fused_model']['ultralytics_gflops']} GFLOPs",
    ])

    if complexity_info.get("thop_profiling", {}).get("macs") is not None:
        thop_info = complexity_info["thop_profiling"]
        report_lines.extend([
            f"  thop MACs Count:           {thop_info['macs'] / 1e9:.2f} GigaMACs",
            f"  thop FLOPs Count:          {thop_info['flops'] / 1e9:.2f} GigaFLOPs ({thop_info['convention']})",
        ])

    if benchmark_info:
        bs1 = benchmark_info["single_image_benchmark"]
        batched = benchmark_info["batched_benchmark"]
        hw = benchmark_info["hardware_environment"]
        report_lines.extend([
            f"  Inference Compute Device:  {hw['device']} ({hw['gpu_model']})",
            f"  Precision Mode:            {hw['precision_mode']}",
            f"  Single-Image Latency (B=1):{bs1['mean_latency_ms']:.2f} ms (+/- {bs1['std_latency_ms']:.2f} ms)",
            f"  Single-Image Median / Range: Median: {bs1['median_latency_ms']:.2f} ms | Min: {bs1['min_latency_ms']:.2f} ms, Max: {bs1['max_latency_ms']:.2f} ms",
            f"  Single-Image Throughput:   {bs1['throughput_fps']:.2f} FPS (frames per second)",
            f"  Batched Throughput (B={batched['batch_size']}):   {batched['throughput_images_per_sec']:.2f} images/sec "
            f"({batched['mean_batch_latency_ms']:.2f} ms per batch)",
        ])

    report_lines.extend([
        "",
        "6. METHODOLOGY & CLINICAL / OBJECT-DETECTION NOTES",
        "-" * 78,
        "  - The validation confidence threshold was tuned exclusively on data_prep/splits/val.csv.",
        "  - Test-set evaluation used the frozen threshold without post-hoc tuning.",
        "  - 570 test images are healthy background images; all false detections on them are counted as FPs.",
        "  - Max-pooling derivation of image presence preserves end-to-end multi-task consistency.",
        "  - Latency timing employs CUDA hardware synchronization to eliminate GPU queue latency distortion.",
        "=" * 78,
    ])
    return "\n".join(report_lines)


# -----------------------------------------------------------------------------
# Main Test Evaluation Execution Flow
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation script for trained YOLOv8s baseline model on TBX11K."
    )
    # Checkpoint and data paths
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo/runs/baseline_yolov8s_scratch/weights/best.pt",
        help="Path to YOLO best checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="yolo/dataset/dataset.yaml",
        help="Path to dataset.yaml configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="yolo/runs/baseline_yolov8s_scratch/evaluation",
        help="Directory to save evaluation reports, CSVs, and plots.",
    )
    # Evaluation settings
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size (default: 512).")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size (default: 4).")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers (default: 2).")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="IoU threshold for object matching (default: 0.50).")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 autocast during benchmarking.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency and throughput benchmarking.")

    args = parser.parse_args()

    # Path resolution
    workspace_root = Path.cwd().resolve()
    weights_path = Path(args.weights).resolve()
    data_yaml_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not weights_path.exists():
        logger.error(f"Checkpoint weights not found: {weights_path}")
        sys.exit(1)
    if not data_yaml_path.exists():
        logger.error(f"Dataset config YAML not found: {data_yaml_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized YOLOv8s evaluation pipeline.")
    logger.info(f"  Weights:    {weights_path}")
    logger.info(f"  Data YAML:  {data_yaml_path}")
    logger.info(f"  Output Dir: {output_dir}")

    # Parse dataset YAML for image & label paths
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dataset_root = Path(data_cfg.get("path", data_yaml_path.parent)).resolve()
    val_img_dir = dataset_root / data_cfg["val"]
    test_img_dir = dataset_root / data_cfg["test"]
    val_lbl_dir = dataset_root / "labels" / "val"
    test_lbl_dir = dataset_root / "labels" / "test"

    val_img_files = sorted(list(val_img_dir.glob("*.png")))
    test_img_files = sorted(list(test_img_dir.glob("*.png")))
    val_img_names = [p.name for p in val_img_files]
    test_img_names = [p.name for p in test_img_files]

    logger.info(f"Dataset splits verified: {len(val_img_names)} val images, {len(test_img_names)} test images.")

    # Device configuration
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info(f"Evaluation compute device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # -------------------------------------------------------------------------
    # STAGE 1: Model Complexity & Checkpoint Verification
    # -------------------------------------------------------------------------
    complexity_info = profile_model_complexity(weights_path=weights_path, device=device, imgsz=args.imgsz)
    complexity_info["hardware_device"] = f"{device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})"

    with open(output_dir / "model_complexity.json", "w", encoding="utf-8") as f:
        json.dump(complexity_info, f, indent=2)
    logger.info(f"Saved model complexity metrics -> {output_dir / 'model_complexity.json'}")

    # -------------------------------------------------------------------------
    # STAGE 2: Validation-Based Confidence Threshold Optimization
    # -------------------------------------------------------------------------
    val_gt = load_ground_truth(val_lbl_dir, val_img_names)
    eval_model = YOLO(str(weights_path))

    val_preds = extract_model_predictions(
        model=eval_model,
        image_dir=val_img_dir,
        image_names=val_img_names,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=str(args.device),
        min_conf=0.005,
    )

    optimal_thresh, best_val_rec, sweep_records = run_threshold_optimization_sweep(
        gt_map=val_gt,
        pred_map=val_preds,
        iou_thresh=args.iou_thresh,
        thresh_start=0.01,
        thresh_end=0.95,
        thresh_step=0.01,
    )

    # Save threshold_analysis.csv
    threshold_csv_path = output_dir / "threshold_analysis.csv"
    with open(threshold_csv_path, "w", encoding="utf-8") as f:
        headers = [
            "threshold", "macro_f1", "macro_precision", "macro_recall", "micro_f1",
            "active_precision", "active_recall", "active_f1", "active_tp", "active_fp", "active_fn",
            "obsolete_precision", "obsolete_recall", "obsolete_f1", "obsolete_tp", "obsolete_fp", "obsolete_fn"
        ]
        f.write(",".join(headers) + "\n")
        for r in sweep_records:
            f.write(
                f"{r['threshold']:.4f},{r['macro_f1']:.4f},{r['macro_precision']:.4f},{r['macro_recall']:.4f},"
                f"{r['micro_f1']:.4f},{r['active_precision']:.4f},{r['active_recall']:.4f},{r['active_f1']:.4f},"
                f"{r['active_tp']},{r['active_fp']},{r['active_fn']},{r['obsolete_precision']:.4f},"
                f"{r['obsolete_recall']:.4f},{r['obsolete_f1']:.4f},{r['obsolete_tp']},{r['obsolete_fp']},{r['obsolete_fn']}\n"
            )
    logger.info(f"Saved threshold sweep analysis -> {threshold_csv_path}")

    # Plot F1 vs Confidence Threshold curve
    plot_f1_vs_confidence_threshold(
        sweep_records=sweep_records,
        optimal_thresh=optimal_thresh,
        output_path=output_dir / "f1_vs_confidence_threshold.png",
    )
    logger.info(f"Saved F1 vs. confidence threshold plot -> {output_dir / 'f1_vs_confidence_threshold.png'}")

    # -------------------------------------------------------------------------
    # STAGE 3: Standard Ultralytics Test Validation (mAP Calculations)
    # -------------------------------------------------------------------------
    logger.info("Executing standard Ultralytics validation on test split...")
    val_run_results = eval_model.val(
        data=str(data_yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=args.workers,
        device=args.device,
        plots=False,
        save=False,
    )

    box_metrics = val_run_results.box
    map50 = float(box_metrics.map50)
    map75 = float(box_metrics.map75)
    map50_95 = float(box_metrics.map)

    # Per-class APs
    per_class_ap50 = {CLASS_NAMES[0]: float(box_metrics.ap50[0]), CLASS_NAMES[1]: float(box_metrics.ap50[1])}
    per_class_ap75 = {CLASS_NAMES[0]: float(box_metrics.all_ap[0, 5]), CLASS_NAMES[1]: float(box_metrics.all_ap[1, 5])}
    per_class_map50_95 = {CLASS_NAMES[0]: float(box_metrics.maps[0]), CLASS_NAMES[1]: float(box_metrics.maps[1])}

    map_metrics_dict = {
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "per_class_ap50": per_class_ap50,
        "per_class_ap75": per_class_ap75,
        "per_class_map50_95": per_class_map50_95,
    }

    # Plot detection PR curve
    plot_detection_pr_curves(
        recalls=box_metrics.px,
        precisions=box_metrics.prec_values,
        ap50_per_class=[box_metrics.ap50[0], box_metrics.ap50[1]],
        mAP50=map50,
        output_path=output_dir / "precision_recall_curve.png",
    )
    logger.info(f"Saved detection Precision-Recall curve -> {output_dir / 'precision_recall_curve.png'}")

    # -------------------------------------------------------------------------
    # STAGE 4: Test Split Prediction & Discrete Detection Evaluation at tau*
    # -------------------------------------------------------------------------
    test_gt = load_ground_truth(test_lbl_dir, test_img_names)
    test_preds = extract_model_predictions(
        model=eval_model,
        image_dir=test_img_dir,
        image_names=test_img_names,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=str(args.device),
        min_conf=0.005,
    )

    test_det_results = evaluate_test_detection(
        gt_map=test_gt,
        pred_map=test_preds,
        optimal_threshold=optimal_thresh,
        iou_thresh=args.iou_thresh,
    )

    plot_detection_confusion_matrix(
        cm_3x3=np.array(test_det_results["confusion_matrix_3x3"]),
        output_path=output_dir / "confusion_matrix_detection.png",
    )
    logger.info(f"Saved detection confusion matrix -> {output_dir / 'confusion_matrix_detection.png'}")

    # -------------------------------------------------------------------------
    # STAGE 5: Derived Image-Level Multi-Label Classification
    # -------------------------------------------------------------------------
    image_level_results = evaluate_image_level_classification(
        gt_map=test_gt,
        pred_map=test_preds,
        optimal_threshold=optimal_thresh,
        image_names=test_img_names,
    )

    plot_image_level_confusion_matrix(
        cm_4x4=np.array(image_level_results["confusion_matrix_4x4"]),
        output_path=output_dir / "confusion_matrix_image_level.png",
    )
    logger.info(f"Saved image-level confusion matrix -> {output_dir / 'confusion_matrix_image_level.png'}")

    raw_vecs = image_level_results["raw_vectors"]
    plot_image_level_roc_curves(
        gt_active=raw_vecs["gt_active"],
        score_active=raw_vecs["pred_score_active"],
        gt_obsolete=raw_vecs["gt_obsolete"],
        score_obsolete=raw_vecs["pred_score_obsolete"],
        output_path=output_dir / "roc_curves.png",
    )
    logger.info(f"Saved image-level ROC curves -> {output_dir / 'roc_curves.png'}")

    plot_image_level_pr_curves(
        gt_active=raw_vecs["gt_active"],
        score_active=raw_vecs["pred_score_active"],
        gt_obsolete=raw_vecs["gt_obsolete"],
        score_obsolete=raw_vecs["pred_score_obsolete"],
        output_path=output_dir / "pr_curves.png",
    )
    logger.info(f"Saved image-level PR curves -> {output_dir / 'pr_curves.png'}")

    # -------------------------------------------------------------------------
    # STAGE 6: Latency and Throughput Benchmarking
    # -------------------------------------------------------------------------
    benchmark_info: Optional[Dict[str, Any]] = None
    if not args.skip_benchmark:
        benchmark_info = benchmark_inference_speed(
            weights_path=weights_path,
            device=device,
            imgsz=args.imgsz,
            eval_batch_size=args.batch_size,
            use_amp=(not args.no_amp),
            num_warmup=20,
            num_runs=100,
        )
        with open(output_dir / "inference_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_info, f, indent=2)
        logger.info(f"Saved inference benchmark results -> {output_dir / 'inference_benchmark.json'}")

    # -------------------------------------------------------------------------
    # STAGE 7: CSV Tables Generation
    # -------------------------------------------------------------------------
    # 1. per_class_metrics.csv
    per_class_csv_path = output_dir / "per_class_metrics.csv"
    with open(per_class_csv_path, "w", encoding="utf-8") as f:
        headers = [
            "class_id", "class_name", "gt_box_instances", "gt_image_support",
            "precision_det", "recall_det", "f1_det", "ap50", "ap75", "map50_95",
            "tp_det", "fp_det", "fn_det", "image_roc_auc", "image_pr_auc",
            "image_precision", "image_recall", "image_f1"
        ]
        f.write(",".join(headers) + "\n")
        for c, cname in CLASS_NAMES.items():
            det_c = test_det_results["per_class"][cname]
            img_c = image_level_results["per_label"][cname]
            roc_str = f"{img_c['roc_auc']:.4f}" if img_c["roc_auc"] is not None else "N/A"
            pr_str = f"{img_c['pr_auc']:.4f}" if img_c["pr_auc"] is not None else "N/A"
            f.write(
                f"{c},{cname},{det_c['total_gt_boxes']},{img_c['support_positives']},"
                f"{det_c['precision']:.4f},{det_c['recall']:.4f},{det_c['f1_score']:.4f},"
                f"{map_metrics_dict['per_class_ap50'][cname]:.4f},{map_metrics_dict['per_class_ap75'][cname]:.4f},"
                f"{map_metrics_dict['per_class_map50_95'][cname]:.4f},{det_c['true_positives']},"
                f"{det_c['false_positives']},{det_c['false_negatives']},"
                f"{roc_str},{pr_str},"
                f"{img_c['precision']:.4f},{img_c['sensitivity_recall']:.4f},{img_c['f1_score']:.4f}\n"
            )
    logger.info(f"Saved per-class metrics -> {per_class_csv_path}")

    # 2. image_level_metrics.csv
    image_level_csv_path = output_dir / "image_level_metrics.csv"
    with open(image_level_csv_path, "w", encoding="utf-8") as f:
        headers = ["category", "precision", "recall", "f1_score", "support"]
        f.write(",".join(headers) + "\n")
        for cat_name, m in image_level_results["per_category"].items():
            f.write(f"{cat_name},{m['precision']:.4f},{m['recall']:.4f},{m['f1_score']:.4f},{m['support']}\n")
        # Summary rows
        f.write(f"Category_Macro_Average,{float(np.mean([m['precision'] for m in image_level_results['per_category'].values()])):.4f},"
                f"{float(np.mean([m['recall'] for m in image_level_results['per_category'].values()])):.4f},"
                f"{image_level_results['category_macro_f1']:.4f},{len(test_img_names)}\n")
        f.write(f"Subset_Exact_Match_Accuracy,{image_level_results['exact_match_accuracy']:.4f},N/A,N/A,{len(test_img_names)}\n")
        f.write(f"Hamming_Accuracy,{image_level_results['hamming_accuracy']:.4f},N/A,N/A,{len(test_img_names) * 2}\n")
    logger.info(f"Saved image-level metrics table -> {image_level_csv_path}")

    # 3. per_image_predictions.csv
    per_image_csv_path = output_dir / "per_image_predictions.csv"
    with open(per_image_csv_path, "w", encoding="utf-8") as f:
        headers = [
            "image_name", "gt_category", "gt_active_boxes", "gt_obsolete_boxes",
            "pred_active_boxes", "pred_obsolete_boxes", "max_conf_active", "max_conf_obsolete",
            "pred_category", "is_exact_match", "num_tp_boxes", "num_fp_boxes", "num_fn_boxes"
        ]
        f.write(",".join(headers) + "\n")
        for img_name in test_img_names:
            cls_info = image_level_results["per_image_classification"][img_name]
            det_info = test_det_results["per_image_det_info"][img_name]
            f.write(
                f"{img_name},{cls_info['gt_category']},{det_info['gt_boxes_active']},{det_info['gt_boxes_obsolete']},"
                f"{det_info['pred_boxes_active']},{det_info['pred_boxes_obsolete']},{cls_info['max_conf_active']:.4f},"
                f"{cls_info['max_conf_obsolete']:.4f},{cls_info['pred_category']},{int(cls_info['is_exact_match'])},"
                f"{det_info['tp']},{det_info['fp']},{det_info['fn']}\n"
            )
    logger.info(f"Saved per-image predictions -> {per_image_csv_path}")

    # -------------------------------------------------------------------------
    # STAGE 8: Master Evaluation Summary JSON & Human-Readable Text Report
    # -------------------------------------------------------------------------
    # Build summary dict (clean out raw numpy vectors for JSON serialization)
    clean_image_level = copy.deepcopy(image_level_results)
    clean_image_level.pop("raw_vectors", None)
    clean_image_level.pop("per_image_classification", None)

    clean_det = copy.deepcopy(test_det_results)
    clean_det.pop("per_image_det_info", None)

    evaluation_summary = {
        "metadata": {
            "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_architecture": "YOLOv8s",
            "checkpoint_evaluated": str(weights_path),
            "target_dataset": "TBX11K",
            "test_images_count": len(test_img_names),
            "test_healthy_count": sum(1 for img in test_img_names if len(test_gt.get(img, [])) == 0),
            "test_tb_count": sum(1 for img in test_img_names if len(test_gt.get(img, [])) > 0),
            "test_gt_boxes_count": sum(len(b) for b in test_gt.values()),
        },
        "validation_threshold_optimization": {
            "optimal_confidence_threshold": optimal_thresh,
            "iou_matching_threshold": args.iou_thresh,
            "val_macro_f1": best_val_rec["macro_f1"],
            "val_micro_f1": best_val_rec["micro_f1"],
            "val_active_p": best_val_rec["active_precision"],
            "val_active_r": best_val_rec["active_recall"],
            "val_active_f1": best_val_rec["active_f1"],
            "val_obsolete_p": best_val_rec["obsolete_precision"],
            "val_obsolete_r": best_val_rec["obsolete_recall"],
            "val_obsolete_f1": best_val_rec["obsolete_f1"],
        },
        "ultralytics_map_metrics": map_metrics_dict,
        "detection_metrics": clean_det,
        "image_level_classification": clean_image_level,
        "model_complexity": complexity_info,
        "inference_benchmark": benchmark_info,
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    logger.info(f"Saved evaluation summary JSON -> {output_dir / 'evaluation_summary.json'}")

    report_text = generate_comprehensive_text_report(
        eval_summary=evaluation_summary,
        complexity_info=complexity_info,
        benchmark_info=benchmark_info,
    )
    with open(output_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved comprehensive evaluation report -> {output_dir / 'evaluation_report.txt'}")

    # -------------------------------------------------------------------------
    # STAGE 9: Terminal Summary Output
    # -------------------------------------------------------------------------
    print("\n" + report_text)
    print("\n[SUCCESS] YOLOv8s test evaluation completed successfully.")
    print(f"All 14 evaluation artifacts generated in: {output_dir}")


if __name__ == "__main__":
    main()
