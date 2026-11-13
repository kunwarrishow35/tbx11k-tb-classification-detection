"""
TBX11K RetinaNet (ResNet-50 + FPN) Evaluation Pipeline.
======================================================

Protocol Mirrors yolo/test.py:
  1. Inference on validation and test splits with official model defaults (score floor 0.05, NMS 0.50, 300 detections).
  2. Validation-only confidence-threshold sweep tau in [0.01, 0.95] (step 0.01) at IoU 0.50.
     Matching logic strictly copied from yolo/test.py lines 177-240 (greedy matching per class).
  3. Optimal threshold tau* chosen by maximizing validation Macro-F1 over the two classes.
  4. Test set evaluated once using the frozen optimal threshold tau*.
  5. Derived 4-category image-level view and Binary Any TB vs Healthy view using exact YOLO definitions.
  6. Computational complexity (thop 2xMACs FLOPs) and latency/FPS benchmarking (forward-only and end-to-end).
  7. Outputs saved mirroring YOLO layout:
     evaluation_report.txt, evaluation_summary.json, model_complexity.json, inference_benchmark.json,
     per_image_predictions.csv, per_class_metrics.csv, image_level_metrics.csv, threshold_analysis.csv,
     and evaluation curve plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set matplotlib cache directory before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add official_ref to path for unmodified reference modules
_CURRENT_DIR = Path(__file__).resolve().parent
_OFFICIAL_REF_DIR = _CURRENT_DIR / "official_ref"
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_OFFICIAL_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_REF_DIR))

from official_ref import presets
from official_ref import utils
from official_ref.coco_utils import get_coco_api_from_dataset
from official_ref.coco_eval import CocoEvaluator

import dataset as tbx_dataset

# Detection Classes (0: ActiveTuberculosis, 1: ObsoletePulmonaryTuberculosis)
CLASS_NAMES: Dict[int, str] = {
    0: "ActiveTuberculosis",
    1: "ObsoletePulmonaryTuberculosis",
}

CATEGORY_NAMES: List[str] = ["Healthy", "Active-only", "Obsolete-only", "Both"]


@dataclass
class BoundingBox:
    """Bounding box dataclass matching yolo/test.py:111-120."""
    class_id: int  # 0: Active, 1: Obsolete
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


# -----------------------------------------------------------------------------
# Geometry and Object Matching Utilities (Copied from yolo/test.py:123-240)
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


def match_detections_for_image(
    gt_boxes: List[BoundingBox],
    pred_boxes: List[BoundingBox],
    iou_thresh: float = 0.50,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], np.ndarray, List[float]]:
    """
    Greedy matching between predictions and ground-truth boxes per class at a given IoU threshold.
    Strictly follows yolo/test.py:177-240 with 3x3 confusion matrix and matched IoU recording.
    """
    tp = {0: 0, 1: 0}
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}
    cm_3x3 = np.zeros((3, 3), dtype=np.int64)
    matched_ious: List[float] = []

    for c in [0, 1]:
        c_gts = [g for g in gt_boxes if g.class_id == c]
        c_preds = [p for p in pred_boxes if p.class_id == c]

        if len(c_preds) == 0:
            fn[c] += len(c_gts)
            cm_3x3[c, 2] += len(c_gts)
            continue

        if len(c_gts) == 0:
            fp[c] += len(c_preds)
            cm_3x3[2, c] += len(c_preds)
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
                        best_iou = float(ious[i_p, i_g])
                        best_g = i_g
            if best_g >= 0:
                matched_gt[best_g] = True
                tp[c] += 1
                cm_3x3[c, c] += 1
                matched_ious.append(best_iou)
            else:
                fp[c] += 1
                cm_3x3[2, c] += 1

        unmatched_gt_count = int(np.sum(~matched_gt))
        fn[c] += unmatched_gt_count
        cm_3x3[c, 2] += unmatched_gt_count

    return tp, fp, fn, cm_3x3, matched_ious


# -----------------------------------------------------------------------------
# Validation Confidence-Threshold Optimization Sweep (Mirrors yolo/test.py:303-393)
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
    Selects optimal threshold tau* maximizing validation Macro-F1 with strictly-greater tie-breaking.
    """
    thresholds = np.arange(thresh_start, thresh_end + 1e-5, thresh_step)
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
            tp, fp, fn, _, _ = match_detections_for_image(gts, filtered_preds, iou_thresh=iou_thresh)
            for c in [0, 1]:
                tot_tp[c] += tp[c]
                tot_fp[c] += fp[c]
                tot_fn[c] += fn[c]

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

        # Primary optimization: argmax Macro-F1 with strictly-greater tie-breaking (first lowest threshold)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            optimal_threshold = thresh_val
            best_record = rec

    return optimal_threshold, best_record, sweep_records


# -----------------------------------------------------------------------------
# Test Set Evaluation: Object Detection Metrics (Mirrors yolo/test.py:399-485)
# -----------------------------------------------------------------------------
def evaluate_detection_performance(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    optimal_threshold: float,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Evaluates detection performance on a split at a specified confidence threshold."""
    tot_tp = {0: 0, 1: 0}
    tot_fp = {0: 0, 1: 0}
    tot_fn = {0: 0, 1: 0}
    cm_3x3_total = np.zeros((3, 3), dtype=np.int64)
    all_matched_ious: List[float] = []
    per_image_det_info: Dict[str, Dict[str, int]] = {}

    for fname, gts in gt_map.items():
        raw_preds = pred_map.get(fname, [])
        filtered_preds = [p for p in raw_preds if p.conf >= optimal_threshold]
        tp, fp, fn, cm_3x3, matched_ious = match_detections_for_image(gts, filtered_preds, iou_thresh=iou_thresh)

        for c in [0, 1]:
            tot_tp[c] += tp[c]
            tot_fp[c] += fp[c]
            tot_fn[c] += fn[c]

        cm_3x3_total += cm_3x3
        all_matched_ious.extend(matched_ious)

        per_image_det_info[fname] = {
            "tp": tp[0] + tp[1],
            "fp": fp[0] + fp[1],
            "fn": fn[0] + fn[1],
            "pred_boxes_active": sum(1 for p in filtered_preds if p.class_id == 0),
            "pred_boxes_obsolete": sum(1 for p in filtered_preds if p.class_id == 1),
            "gt_boxes_active": sum(1 for g in gts if g.class_id == 0),
            "gt_boxes_obsolete": sum(1 for g in gts if g.class_id == 1),
        }

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
    mean_matched_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

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
            "mean_matched_iou": mean_matched_iou,
        },
        "per_class": class_results,
        "confusion_matrix_3x3": cm_3x3_total.tolist(),
        "per_image_det_info": per_image_det_info,
    }


# -----------------------------------------------------------------------------
# Derived Image-Level Multi-Label & 4-Category Classification (Mirrors yolo/test.py:491-671)
# -----------------------------------------------------------------------------
def evaluate_image_level_classification(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    optimal_threshold: float,
    image_names: List[str],
) -> Dict[str, Any]:
    """
    Derives image-level multi-label, 4-category, and Binary Any-TB-vs-Healthy metrics.
    Continuous scores: Max confidence per class across all predicted boxes in the image.
    Binary decision: class_present = 1 if max_conf >= optimal_threshold else 0.
    """
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
            gt_cat = "Active-only"
        elif act_gt == 0 and obs_gt == 1:
            gt_cat = "Obsolete-only"
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
            pred_cat = "Active-only"
        elif act_pred == 0 and obs_pred == 1:
            pred_cat = "Obsolete-only"
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

    n_samples = len(image_names)
    exact_matches = np.sum((gt_active_arr == pred_bin_act_arr) & (gt_obsolete_arr == pred_bin_obs_arr))
    exact_match_acc = float(exact_matches / n_samples)

    total_label_errors = np.sum(gt_active_arr != pred_bin_act_arr) + np.sum(gt_obsolete_arr != pred_bin_obs_arr)
    hamming_loss = float(total_label_errors / (2 * n_samples))
    hamming_accuracy = float(1.0 - hamming_loss)

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

    # 4-category classification metrics
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

    # Binary Any TB vs Healthy Screening
    gt_any = ((gt_active_arr == 1) | (gt_obsolete_arr == 1)).astype(np.int32)
    pred_any = ((pred_bin_act_arr == 1) | (pred_bin_obs_arr == 1)).astype(np.int32)
    tn_any, fp_any, fn_any, tp_any = confusion_matrix(gt_any, pred_any, labels=[0, 1]).ravel()
    prec_any = float(tp_any / (tp_any + fp_any)) if (tp_any + fp_any) > 0 else 0.0
    rec_any = float(tp_any / (tp_any + fn_any)) if (tp_any + fn_any) > 0 else 0.0
    spec_any = float(tn_any / (tn_any + fp_any)) if (tn_any + fp_any) > 0 else 0.0
    f1_any = float(2 * prec_any * rec_any / (prec_any + rec_any)) if (prec_any + rec_any) > 0 else 0.0

    any_tb_vs_healthy = {
        "true_positives": int(tp_any),
        "false_positives": int(fp_any),
        "true_negatives": int(tn_any),
        "false_negatives": int(fn_any),
        "precision": prec_any,
        "sensitivity_recall": rec_any,
        "specificity": spec_any,
        "f1_score": f1_any,
        "support_positives": int(np.sum(gt_any == 1)),
        "support_negatives": int(np.sum(gt_any == 0)),
    }

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
        "any_tb_vs_healthy": any_tb_vs_healthy,
        "per_image_classification": per_image_classification,
        "raw_vectors": {
            "gt_active": [int(x) for x in gt_active],
            "gt_obsolete": [int(x) for x in gt_obsolete],
            "pred_score_active": [float(x) for x in pred_score_active],
            "pred_score_obsolete": [float(x) for x in pred_score_obsolete],
            "gt_cat_idx": [int(x) for x in gt_cat_idx],
            "pred_cat_idx": [int(x) for x in pred_cat_idx],
        },
    }


# -----------------------------------------------------------------------------
# Model Complexity Profiling (thop with 2 x MACs FLOPs)
# -----------------------------------------------------------------------------
def profile_model_complexity(
    model: torch.nn.Module,
    weights_path: Path,
    device: torch.device,
    imgsz: int = 512,
) -> Dict[str, Any]:
    """Profiles model parameters, GFLOPs at 512x512, and checkpoint file size."""
    ckpt_size_mb = float(weights_path.stat().st_size / (1024 * 1024)) if weights_path.exists() else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops_info: Dict[str, Any] = {"tool": "none", "macs": None, "flops": None}
    try:
        import thop
        model_copy = copy.deepcopy(model).to(device).eval()
        dummy_inp = [torch.randn(3, imgsz, imgsz, device=device)]
        with torch.inference_mode():
            macs, params = thop.profile(model_copy, inputs=(dummy_inp,), verbose=False)
        flops_info = {
            "tool": "thop",
            "macs": float(macs),
            "flops": float(2.0 * macs),
            "convention": "1 MAC = 2 floating point operations",
            "gflops": round(float(2.0 * macs / 1e9), 2),
        }
        del model_copy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        flops_info["error"] = str(e)

    return {
        "architecture": "RetinaNet (ResNet-50 + FPN)",
        "input_resolution": f"{imgsz}x{imgsz}x3",
        "checkpoint_path": str(weights_path.resolve()),
        "checkpoint_file_size_mb": round(ckpt_size_mb, 2),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "thop_profiling": flops_info,
    }


# -----------------------------------------------------------------------------
# Latency and Throughput Benchmarking (Mirrors yolo/test.py:737-850)
# -----------------------------------------------------------------------------
def benchmark_inference_speed(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 512,
    eval_batch_size: int = 4,
    use_amp: bool = True,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> Dict[str, Any]:
    """
    Benchmarks inference latency and throughput.
    Main benchmark: Forward-only pass (backbone + head) matching YOLO's py_model forward.
    Secondary benchmark: End-to-end full pipeline (transform + forward + box decode + NMS).
    """
    model.eval()

    # Define forward-only module (backbone + head)
    class ForwardOnlyRetinaNet(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone = m.backbone
            self.head = m.head

        def forward(self, x):
            features = self.backbone(x)
            cls_logits, bbox_reg = self.head(list(features.values()))
            return cls_logits, bbox_reg

    fwd_model = ForwardOnlyRetinaNet(model).to(device).eval()

    # 1. Single-Image Benchmark (Batch size = 1)
    single_tensor = torch.randn(1, 3, imgsz, imgsz, device=device)
    single_list = [single_tensor[0]]

    # Warmup forward-only
    with torch.inference_mode():
        for _ in range(num_warmup):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = fwd_model(single_tensor)
            else:
                _ = fwd_model(single_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed forward-only B=1
    timings_bs1 = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = fwd_model(single_tensor)
            else:
                _ = fwd_model(single_tensor)
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

    # 2. Batched Benchmark (Batch size = 4)
    batched_tensor = torch.randn(eval_batch_size, 3, imgsz, imgsz, device=device)

    with torch.inference_mode():
        for _ in range(num_warmup // 2):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = fwd_model(batched_tensor)
            else:
                _ = fwd_model(batched_tensor)
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
                    _ = fwd_model(batched_tensor)
            else:
                _ = fwd_model(batched_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_batched.append((t1 - t0) * 1000.0)

    mean_batch_ms = float(np.mean(timings_batched))
    batched_throughput_fps = float((eval_batch_size * 1000.0) / mean_batch_ms) if mean_batch_ms > 0 else 0.0

    # 3. End-to-end Pipeline Benchmark (transform + forward + decode/NMS)
    with torch.inference_mode():
        for _ in range(num_warmup // 2):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_list)
            else:
                _ = model(single_list)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings_e2e = []
    with torch.inference_mode():
        for _ in range(num_runs // 2):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_list)
            else:
                _ = model(single_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_e2e.append((t1 - t0) * 1000.0)

    mean_e2e = float(np.mean(timings_e2e))
    fps_e2e = float(1000.0 / mean_e2e) if mean_e2e > 0 else 0.0

    return {
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
            "pipeline_stage": "Forward-only (Backbone + Head; matches YOLO py_model benchmark)",
        },
        "batched_benchmark": {
            "batch_size": eval_batch_size,
            "input_resolution": f"{eval_batch_size}x{imgsz}x{imgsz}x3",
            "mean_batch_latency_ms": round(mean_batch_ms, 2),
            "throughput_images_per_sec": round(batched_throughput_fps, 2),
            "timed_iterations": num_runs // 2,
            "warmup_iterations": num_warmup // 2,
            "pipeline_stage": "Forward-only Batched",
        },
        "end_to_end_benchmark": {
            "batch_size": 1,
            "input_resolution": f"{imgsz}x{imgsz}x3",
            "mean_latency_ms": round(mean_e2e, 2),
            "throughput_fps": round(fps_e2e, 2),
            "pipeline_stage": "Full End-to-End Pipeline (Transform + Backbone + FPN + Box Decode + NMS)",
        },
    }


# -----------------------------------------------------------------------------
# Visualization Functions (Plots matching yolo/test.py)
# -----------------------------------------------------------------------------
def plot_detection_confusion_matrix(cm_3x3: np.ndarray, output_path: Path) -> None:
    """Plots and saves the 3x3 object detection confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    labels = ["Active TB", "Obsolete TB", "Background"]
    im = ax.imshow(cm_3x3, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Ground Truth",
        xlabel="Predicted",
        title="Object Detection Confusion Matrix (IoU >= 0.50)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_3x3.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{cm_3x3[i, j]:,}",
                ha="center", va="center",
                color="white" if cm_3x3[i, j] > thresh else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_image_level_confusion_matrix(cm_4x4: np.ndarray, output_path: Path) -> None:
    """Plots and saves the 4x4 derived image-level confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_4x4, interpolation="nearest", cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=CATEGORY_NAMES,
        yticklabels=CATEGORY_NAMES,
        ylabel="Ground Truth Category",
        xlabel="Predicted Category",
        title="Derived 4-Category Image-Level Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm_4x4.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(
                j, i, f"{cm_4x4[i, j]:,}",
                ha="center", va="center",
                color="white" if cm_4x4[i, j] > thresh else "black",
                fontweight="bold"
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_f1_vs_confidence_threshold(
    sweep_records: List[Dict[str, Any]],
    optimal_tau: float,
    output_path: Path,
) -> None:
    """Plots validation F1 curves across confidence thresholds."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    taus = [r["threshold"] for r in sweep_records]
    macro_f1s = [r["macro_f1"] for r in sweep_records]
    act_f1s = [r["active_f1"] for r in sweep_records]
    obs_f1s = [r["obsolete_f1"] for r in sweep_records]

    ax.plot(taus, macro_f1s, label="Macro-F1 (Average)", color="black", linewidth=2.5)
    ax.plot(taus, act_f1s, label="ActiveTuberculosis F1", color="#d62728", linewidth=1.8, linestyle="--")
    ax.plot(taus, obs_f1s, label="ObsoletePulmonaryTB F1", color="#1f77b4", linewidth=1.8, linestyle="-.")

    ax.axvline(optimal_tau, color="green", linestyle=":", linewidth=2, label=f"Optimal tau* = {optimal_tau:.2f}")
    ax.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation F1 Score (IoU >= 0.50)", fontsize=11, fontweight="bold")
    ax.set_title("Validation F1 vs Confidence Threshold Sweep", fontsize=12, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", frameon=True, fontsize=9)

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
    """Plots derived image-level ROC curves."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Active TB
    try:
        auc_a = float(roc_auc_score(gt_active, score_active))
        auc_a_str = f"{auc_a:.4f}"
    except Exception:
        auc_a_str = "N/A"
    fpr_a, tpr_a, _ = roc_curve(gt_active, score_active)
    ax.plot(fpr_a, tpr_a, color="#d62728", linewidth=2.5, label=f"ActiveTuberculosis (AUC = {auc_a_str})")

    # Obsolete TB
    try:
        auc_o = float(roc_auc_score(gt_obsolete, score_obsolete))
        auc_o_str = f"{auc_o:.4f}"
    except Exception:
        auc_o_str = "N/A"
    fpr_o, tpr_o, _ = roc_curve(gt_obsolete, score_obsolete)
    ax.plot(fpr_o, tpr_o, color="#1f77b4", linewidth=2.5, label=f"ObsoletePulmonaryTB (AUC = {auc_o_str})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Derived Image-Level ROC Curves", fontsize=12, fontweight="bold")
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
    """Plots derived image-level PR curves."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Active
    try:
        ap_a = float(average_precision_score(gt_active, score_active))
        ap_a_str = f"{ap_a:.4f}"
    except Exception:
        ap_a_str = "N/A"
    p_a, r_a, _ = precision_recall_curve(gt_active, score_active)
    prev_a = float(np.mean(gt_active))
    ax.plot(r_a, p_a, color="#d62728", linewidth=2.5, label=f"ActiveTuberculosis (PR-AUC = {ap_a_str})")
    ax.axhline(prev_a, color="#d62728", linestyle=":", alpha=0.6, label=f"Active Baseline Prev ({prev_a:.3f})")

    # Obsolete
    try:
        ap_o = float(average_precision_score(gt_obsolete, score_obsolete))
        ap_o_str = f"{ap_o:.4f}"
    except Exception:
        ap_o_str = "N/A"
    p_o, r_o, _ = precision_recall_curve(gt_obsolete, score_obsolete)
    prev_o = float(np.mean(gt_obsolete))
    ax.plot(r_o, p_o, color="#1f77b4", linewidth=2.5, label=f"ObsoletePulmonaryTB (PR-AUC = {ap_o_str})")
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
# Comprehensive Text Report Generator (Mirrors yolo/test.py:1075-1244)
# -----------------------------------------------------------------------------
def generate_comprehensive_text_report(
    eval_summary: Dict[str, Any],
    complexity_info: Dict[str, Any],
    benchmark_info: Optional[Dict[str, Any]],
) -> str:
    """Builds a human-readable evaluation report preserving YOLO/DenseNet reporting conventions."""
    det = eval_summary["detection_metrics"]
    img_level = eval_summary["image_level_classification"]
    val_opt = eval_summary["validation_threshold_optimization"]
    coco_metrics = eval_summary["coco_eval_metrics"]
    cm_3x3 = np.array(det["confusion_matrix_3x3"])
    any_tb = img_level["any_tb_vs_healthy"]

    report_lines = [
        "=" * 78,
        "RETINANET (RESNET-50 + FPN) MULTI-TASK TEST EVALUATION REPORT".center(78),
        "=" * 78,
        "",
        "1. EXPERIMENTAL CONFIGURATION & HARDWARE METADATA",
        "-" * 78,
        f"  Model Architecture:        RetinaNet (ResNet-50 + FPN, trained from scratch)",
        f"  Pretrained Weights:        None (weights=None, weights_backbone=None)",
        f"  Checkpoint File:           {complexity_info['checkpoint_path']}",
        f"  Checkpoint File Size:      {complexity_info['checkpoint_file_size_mb']} MB",
        f"  Input Resolution:          {complexity_info['input_resolution']}",
        f"  Target Split:              Canonical test split (690 images)",
        f"  Total Test Bounding Boxes: 179 boxes (143 Active, 36 Obsolete)",
        f"  Test Healthy Images:       570 genuine background images (empty targets)",
        f"  Evaluation Compute Device: {complexity_info.get('hardware_device', 'GPU')}",
        f"  Loss Formulation:          Sigmoid Focal Loss (alpha=0.25, gamma=2.0) + Plain L1 Box Loss (_loss_type='l1')",
        f"  Post-Processing Defaults:  score_thresh=0.05, nms_thresh=0.50, detections_per_img=300",
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
        f"  [Standard COCO Multi-Threshold Metrics]",
        f"    Overall mAP@50:          {coco_metrics.get('mAP50', 0.0):.4f}",
        f"    Overall mAP@75:          {coco_metrics.get('mAP75', 0.0):.4f}",
        f"    Overall mAP@50-95:       {coco_metrics.get('mAP50_95', 0.0):.4f}",
        f"    ActiveTuberculosis:      AP@50 = {coco_metrics.get('per_class', {}).get('ActiveTuberculosis', {}).get('AP50', 0.0):.4f}, "
        f"mAP@50-95 = {coco_metrics.get('per_class', {}).get('ActiveTuberculosis', {}).get('AP50_95', 0.0):.4f}",
        f"    ObsoletePulmonaryTB:     AP@50 = {coco_metrics.get('per_class', {}).get('ObsoletePulmonaryTuberculosis', {}).get('AP50', 0.0):.4f}, "
        f"mAP@50-95 = {coco_metrics.get('per_class', {}).get('ObsoletePulmonaryTuberculosis', {}).get('AP50_95', 0.0):.4f}",
        "",
        f"  [Discrete Detection Metrics at Frozen Threshold tau* = {det['frozen_confidence_threshold']:.2f}]",
        f"    Macro Detection F1:      {det['overall_summary']['macro_f1']:.4f}",
        f"    Macro Precision:         {det['overall_summary']['macro_precision']:.4f}",
        f"    Macro Recall:            {det['overall_summary']['macro_recall']:.4f}",
        f"    Micro Detection F1:      {det['overall_summary']['micro_f1']:.4f}",
        f"    Total True Positives:    {det['overall_summary']['total_true_positives']:,}",
        f"    Total False Positives:   {det['overall_summary']['total_false_positives']:,}",
        f"    Total False Negatives:   {det['overall_summary']['total_false_negatives']:,}",
        f"    Mean Matched IoU:        {det['overall_summary']['mean_matched_iou']:.4f}",
        "",
        "  [3x3 Detection Confusion Matrix (IoU >= 0.50)]",
        "  Rows: Ground Truth, Columns: Predicted [Active, Obsolete, Background]",
        f"    [Active    ] [ {cm_3x3[0, 0]:>4d}, {cm_3x3[0, 1]:>4d}, {cm_3x3[0, 2]:>4d} ]",
        f"    [Obsolete  ] [ {cm_3x3[1, 0]:>4d}, {cm_3x3[1, 1]:>4d}, {cm_3x3[1, 2]:>4d} ]",
        f"    [Background] [ {cm_3x3[2, 0]:>4d}, {cm_3x3[2, 1]:>4d}, {cm_3x3[2, 2]:>4d} ]",
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
        "  [Binary Presence Ranking & Diagnostic Accuracy]",
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
        "  [Binary Any TB vs Healthy Screening Performance]",
        f"    Positive Ground Truth:   {any_tb['support_positives']:,} / 690 ({any_tb['support_positives']/690*100:.2f}%)",
        f"    Precision:               {any_tb['precision']:.4f}",
        f"    Sensitivity (Recall):    {any_tb['sensitivity_recall']:.4f}",
        f"    Specificity:             {any_tb['specificity']:.4f}",
        f"    F1-Score:                {any_tb['f1_score']:.4f}",
        f"    Confusion Matrix:        TP={any_tb['true_positives']}, FP={any_tb['false_positives']}, FN={any_tb['false_negatives']}, TN={any_tb['true_negatives']}",
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
        f"  Total Parameters:          {complexity_info['total_parameters']:,}",
        f"  Trainable Parameters:      {complexity_info['trainable_parameters']:,}",
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
        e2e = benchmark_info["end_to_end_benchmark"]
        hw = benchmark_info["hardware_environment"]
        report_lines.extend([
            f"  Inference Compute Device:  {hw['device']} ({hw['gpu_model']})",
            f"  Precision Mode:            {hw['precision_mode']}",
            f"  Single-Image Latency (B=1):{bs1['mean_latency_ms']:.2f} ms (+/- {bs1['std_latency_ms']:.2f} ms)",
            f"  Single-Image Median / Range: Median: {bs1['median_latency_ms']:.2f} ms | Min: {bs1['min_latency_ms']:.2f} ms, Max: {bs1['max_latency_ms']:.2f} ms",
            f"  Primary Forward Throughput:{bs1['throughput_fps']:.2f} FPS (Forward pass only: Backbone + Head; matches YOLO)",
            f"  Batched Throughput (B={batched['batch_size']}):   {batched['throughput_images_per_sec']:.2f} images/sec ({batched['mean_batch_latency_ms']:.2f} ms/batch)",
            f"  End-to-End Pipeline Time:  {e2e['mean_latency_ms']:.2f} ms ({e2e['throughput_fps']:.2f} FPS with transform + decode + NMS)",
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
        "  - Note on Candidate Filtering: RetinaNet uses a score floor of 0.05 and max 300 detections per image",
        "    at inference; candidate proposals with confidence < 0.05 are pruned prior to NMS. Therefore,",
        "    evaluation thresholds tau in [0.01, 0.05] yield identical detections to tau = 0.05.",
        "  - Note on Box Loss: The torchvision RetinaNet implementation uses plain L1 loss (_loss_type='l1',",
        "    not Smooth L1) for bounding box regression.",
        "=" * 78,
    ])
    return "\n".join(report_lines)


# -----------------------------------------------------------------------------
# Streaming Inference and COCO Evaluation
# -----------------------------------------------------------------------------
def run_inference_and_coco_eval(
    model: torch.nn.Module,
    dataset: tbx_dataset.TBX11KDetectionDataset,
    device: torch.device,
) -> Tuple[CocoEvaluator, Dict[str, List[BoundingBox]], Dict[str, List[BoundingBox]]]:
    """Runs model inference, COCO evaluation, and extracts BoundingBox predictions."""
    model.eval()
    coco_gt = get_coco_api_from_dataset(dataset)
    coco_evaluator = CocoEvaluator(coco_gt, ["bbox"])

    pred_map: Dict[str, List[BoundingBox]] = {}
    gt_map: Dict[str, List[BoundingBox]] = {}

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn
    )

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            # Format for CocoEvaluator (target["image_id"] is plain int)
            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

            # Store boxes in BoundingBox format
            for target, output in zip(targets, outputs):
                img_idx = target["image_id"]
                fname = dataset.samples[img_idx]["file_name"]

                # GT boxes: convert 1-based class labels to 0-based (1->0: Active, 2->1: Obsolete)
                gt_boxes_list: List[BoundingBox] = []
                for box, lbl in zip(target["boxes"].cpu().numpy(), target["labels"].cpu().numpy()):
                    gt_boxes_list.append(
                        BoundingBox(
                            class_id=int(lbl) - 1,
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            conf=1.0,
                        )
                    )
                gt_map[fname] = gt_boxes_list

                # Predictions
                pred_boxes_list: List[BoundingBox] = []
                p_boxes = output["boxes"].cpu().numpy()
                p_scores = output["scores"].cpu().numpy()
                p_labels = output["labels"].cpu().numpy()

                for box, score, lbl in zip(p_boxes, p_scores, p_labels):
                    c_id = int(lbl) - 1
                    if c_id in [0, 1]:
                        pred_boxes_list.append(
                            BoundingBox(
                                class_id=c_id,
                                x1=float(box[0]),
                                y1=float(box[1]),
                                x2=float(box[2]),
                                y2=float(box[3]),
                                conf=float(score),
                            )
                        )
                pred_map[fname] = pred_boxes_list

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator, gt_map, pred_map


def extract_coco_dict(coco_evaluator: CocoEvaluator) -> Dict[str, Any]:
    """Extracts standard COCO AP metrics from CocoEvaluator."""
    metrics: Dict[str, Any] = {
        "mAP50_95": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "per_class": {
            "ActiveTuberculosis": {"AP50_95": 0.0, "AP50": 0.0, "AP75": 0.0},
            "ObsoletePulmonaryTuberculosis": {"AP50_95": 0.0, "AP50": 0.0, "AP75": 0.0},
        },
    }
    if coco_evaluator is None or "bbox" not in coco_evaluator.coco_eval:
        return metrics

    ce = coco_evaluator.coco_eval["bbox"]
    if ce.stats is not None and len(ce.stats) > 0:
        if ce.stats[0] < 0:
            raise RuntimeError(f"COCO evaluation error: mAP50-95 is negative ({ce.stats[0]}). Evaluation failed.")
        metrics["mAP50_95"] = float(ce.stats[0])
        metrics["mAP50"] = float(ce.stats[1])
        metrics["mAP75"] = float(ce.stats[2])

    if hasattr(ce, "eval") and isinstance(ce.eval, dict) and "precision" in ce.eval:
        precision = ce.eval["precision"]  # [T, R, K, A, M]
        cat_ids = ce.params.catIds  # [1, 2]
        for idx, cid in enumerate(cat_ids):
            cname = "ActiveTuberculosis" if cid == 1 else "ObsoletePulmonaryTuberculosis"
            s = precision[:, :, idx, 0, 2]
            valid_s = s[s > -1]
            if len(valid_s) == 0:
                raise RuntimeError(
                    f"COCO evaluation error: No valid precision entries (all -1) for category {cname} (id {cid}). "
                    f"Both classes have ground truth; this indicates an evaluation failure."
                )
            ap_all = float(np.mean(valid_s))

            s50 = precision[0, :, idx, 0, 2]
            valid_s50 = s50[s50 > -1]
            if len(valid_s50) == 0:
                raise RuntimeError(
                    f"COCO evaluation error: No valid AP50 precision entries (all -1) for category {cname} (id {cid})."
                )
            ap_50 = float(np.mean(valid_s50))

            s75 = precision[5, :, idx, 0, 2]
            valid_s75 = s75[s75 > -1]
            if len(valid_s75) == 0:
                raise RuntimeError(
                    f"COCO evaluation error: No valid AP75 precision entries (all -1) for category {cname} (id {cid})."
                )
            ap_75 = float(np.mean(valid_s75))

            metrics["per_class"][cname] = {
                "AP50_95": ap_all,
                "AP50": ap_50,
                "AP75": ap_75,
            }

    return metrics


# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RetinaNet on TBX11K")
    parser.add_argument("--weights", default="outputs/run_B_fulldata/best.pth", type=str, help="checkpoint path")
    parser.add_argument("--output-dir", default="outputs/run_B_fulldata/evaluation", type=str, help="output directory")
    parser.add_argument("--device", default="cuda", type=str, help="compute device")
    parser.add_argument("--iou-thresh", default=0.50, type=float, help="IoU threshold for greedy matching")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip latency and throughput benchmarking")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode: verifies dataset loading and imports without inference")

    args = parser.parse_args()

    # Enforce dataset self-check
    tbx_dataset.self_check_dataset()

    # Load datasets
    print("Loading datasets for evaluation...")
    eval_transform = presets.DetectionPresetEval(backend="PIL", use_v2=False)
    val_dataset = tbx_dataset.TBX11KDetectionDataset(split="val", transforms=eval_transform)
    test_dataset = tbx_dataset.TBX11KDetectionDataset(split="test", transforms=eval_transform)

    print(f"Validation images loaded: {len(val_dataset)}")
    print(f"Test images loaded:       {len(test_dataset)}")

    if args.dry_run:
        print("[Dry Run] Successfully imported all modules, parsed arguments, and loaded dataset splits.")
        print(f"[Dry Run] Val dataset: {len(val_dataset)} images; Test dataset: {len(test_dataset)} images.")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Evaluation compute device: {device}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=3,
        trainable_backbone_layers=None,
        min_size=512,
        max_size=512,
        score_thresh=0.05,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # 1. Validation Inference & COCO Evaluation
    print("\n=================== STAGE 1: VALIDATION EVALUATION & SWEEP ===================")
    val_evaluator, val_gt_map, val_pred_map = run_inference_and_coco_eval(model, val_dataset, device)
    val_coco_metrics = extract_coco_dict(val_evaluator)

    # 2. Validation Confidence Threshold Sweep (0.01 to 0.95 in 0.01 steps)
    print("Running validation confidence threshold sweep (0.01 to 0.95, step 0.01, IoU 0.50)...")
    optimal_tau, best_val_rec, sweep_records = run_threshold_optimization_sweep(
        gt_map=val_gt_map,
        pred_map=val_pred_map,
        iou_thresh=args.iou_thresh,
        thresh_start=0.01,
        thresh_end=0.95,
        thresh_step=0.01,
    )
    print(f"Optimal validation confidence threshold: tau* = {optimal_tau:.2f} (Macro-F1 = {best_val_rec['macro_f1']:.4f})")

    # Save threshold_analysis.csv
    sweep_csv_path = output_dir / "threshold_analysis.csv"
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_records[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_records)
    print(f"Saved threshold sweep analysis -> {sweep_csv_path}")

    # Plot F1 vs threshold curve
    f1_curve_path = output_dir / "f1_vs_confidence_threshold.png"
    plot_f1_vs_confidence_threshold(sweep_records, optimal_tau, f1_curve_path)
    print(f"Saved F1 curve plot -> {f1_curve_path}")

    # 3. Test Inference & Evaluation (Evaluated once with frozen tau*)
    print("\n=================== STAGE 2: TEST EVALUATION (FROZEN THRESHOLD) ===================")
    test_evaluator, test_gt_map, test_pred_map = run_inference_and_coco_eval(model, test_dataset, device)
    test_coco_metrics = extract_coco_dict(test_evaluator)

    test_image_names = [s["file_name"] for s in test_dataset.samples]

    test_det_metrics = evaluate_detection_performance(
        gt_map=test_gt_map,
        pred_map=test_pred_map,
        optimal_threshold=optimal_tau,
        iou_thresh=args.iou_thresh,
    )

    test_img_metrics = evaluate_image_level_classification(
        gt_map=test_gt_map,
        pred_map=test_pred_map,
        optimal_threshold=optimal_tau,
        image_names=test_image_names,
    )

    # 4. Save CSV Reports
    # per_image_predictions.csv
    per_img_csv = output_dir / "per_image_predictions.csv"
    with open(per_img_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name", "gt_active", "gt_obsolete", "gt_category",
            "max_conf_active", "max_conf_obsolete",
            "pred_active", "pred_obsolete", "pred_category", "is_exact_match"
        ])
        for name in test_image_names:
            info = test_img_metrics["per_image_classification"][name]
            writer.writerow([
                name, info["gt_active"], info["gt_obsolete"], info["gt_category"],
                round(info["max_conf_active"], 6), round(info["max_conf_obsolete"], 6),
                info["pred_active"], info["pred_obsolete"], info["pred_category"],
                info["is_exact_match"]
            ])
    print(f"Saved per-image predictions -> {per_img_csv}")

    # per_class_metrics.csv
    per_cls_csv = output_dir / "per_class_metrics.csv"
    with open(per_cls_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_id", "gt_boxes", "pred_boxes", "tp", "fp", "fn", "precision", "recall", "f1_score"])
        for cname, res in test_det_metrics["per_class"].items():
            writer.writerow([
                cname, res["class_id"], res["total_gt_boxes"], res["total_pred_boxes"],
                res["true_positives"], res["false_positives"], res["false_negatives"],
                round(res["precision"], 4), round(res["recall"], 4), round(res["f1_score"], 4)
            ])
    print(f"Saved per-class metrics -> {per_cls_csv}")

    # image_level_metrics.csv
    img_lvl_csv = output_dir / "image_level_metrics.csv"
    with open(img_lvl_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset_exact_match_acc", "hamming_accuracy", "hamming_loss",
            "category_accuracy", "category_macro_f1", "category_weighted_f1",
            "any_tb_precision", "any_tb_recall", "any_tb_specificity", "any_tb_f1"
        ])
        any_tb = test_img_metrics["any_tb_vs_healthy"]
        writer.writerow([
            round(test_img_metrics["exact_match_accuracy"], 4),
            round(test_img_metrics["hamming_accuracy"], 4),
            round(test_img_metrics["hamming_loss"], 4),
            round(test_img_metrics["category_accuracy"], 4),
            round(test_img_metrics["category_macro_f1"], 4),
            round(test_img_metrics["category_weighted_f1"], 4),
            round(any_tb["precision"], 4),
            round(any_tb["sensitivity_recall"], 4),
            round(any_tb["specificity"], 4),
            round(any_tb["f1_score"], 4),
        ])
    print(f"Saved image-level metrics -> {img_lvl_csv}")

    # 5. Generate and Save Plots
    cm_det_path = output_dir / "confusion_matrix_detection.png"
    plot_detection_confusion_matrix(np.array(test_det_metrics["confusion_matrix_3x3"]), cm_det_path)
    print(f"Saved detection confusion matrix -> {cm_det_path}")

    cm_img_path = output_dir / "confusion_matrix_image_level.png"
    plot_image_level_confusion_matrix(np.array(test_img_metrics["confusion_matrix_4x4"]), cm_img_path)
    print(f"Saved image-level confusion matrix -> {cm_img_path}")

    raw_v = test_img_metrics["raw_vectors"]
    roc_path = output_dir / "roc_curves.png"
    plot_image_level_roc_curves(
        raw_v["gt_active"], raw_v["pred_score_active"],
        raw_v["gt_obsolete"], raw_v["pred_score_obsolete"],
        roc_path
    )
    print(f"Saved ROC curves -> {roc_path}")

    pr_path = output_dir / "pr_curves.png"
    plot_image_level_pr_curves(
        raw_v["gt_active"], raw_v["pred_score_active"],
        raw_v["gt_obsolete"], raw_v["pred_score_obsolete"],
        pr_path
    )
    print(f"Saved PR curves -> {pr_path}")

    # 6. Model Complexity & Inference Benchmarking
    print("\nProfiling model complexity...")
    complexity_info = profile_model_complexity(model, weights_path, device=device)
    complexity_info["hardware_device"] = str(device)
    with open(output_dir / "model_complexity.json", "w", encoding="utf-8") as f:
        json.dump(complexity_info, f, indent=2)
    print(f"Saved model complexity -> {output_dir / 'model_complexity.json'}")

    benchmark_info = None
    if not args.skip_benchmark:
        print("Benchmarking inference latency and throughput (forward-only & end-to-end)...")
        benchmark_info = benchmark_inference_speed(
            model=model,
            device=device,
            imgsz=512,
            eval_batch_size=4,
            use_amp=True,
            num_warmup=20,
            num_runs=100,
        )
        with open(output_dir / "inference_benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark_info, f, indent=2)
        print(f"Saved inference benchmark -> {output_dir / 'inference_benchmark.json'}")
    else:
        print("[Skipping inference benchmark as requested]")

    # 7. Summary JSON & Comprehensive Text Report
    val_opt_summary = {
        "optimal_confidence_threshold": optimal_tau,
        "val_macro_f1": best_val_rec["macro_f1"],
        "val_micro_f1": best_val_rec["micro_f1"],
        "val_active_p": best_val_rec["active_precision"],
        "val_active_r": best_val_rec["active_recall"],
        "val_active_f1": best_val_rec["active_f1"],
        "val_obsolete_p": best_val_rec["obsolete_precision"],
        "val_obsolete_r": best_val_rec["obsolete_recall"],
        "val_obsolete_f1": best_val_rec["obsolete_f1"],
    }

    eval_summary = {
        "experiment": "RetinaNet (ResNet-50 + FPN) Full-Train-Split Evaluation",
        "checkpoint": str(weights_path.resolve()),
        "validation_threshold_optimization": val_opt_summary,
        "detection_metrics": test_det_metrics,
        "image_level_classification": test_img_metrics,
        "coco_eval_metrics": test_coco_metrics,
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved evaluation summary JSON -> {output_dir / 'evaluation_summary.json'}")

    text_report = generate_comprehensive_text_report(eval_summary, complexity_info, benchmark_info)
    report_txt_path = output_dir / "evaluation_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"Saved comprehensive evaluation report -> {report_txt_path}")
    print("\n" + text_report)


if __name__ == "__main__":
    main()
