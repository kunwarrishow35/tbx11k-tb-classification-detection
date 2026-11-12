"""
TBX11K RetinaNet (ResNet-50 + FPN) Evaluation Pipeline.
======================================================

Protocol Mirrors yolo/test.py:
  1. Inference on validation and test splits with official model defaults (score floor 0.05, NMS 0.50, 300 detections).
  2. Validation-only confidence-threshold sweep tau in [0.01, 0.95] (step 0.01) at IoU 0.50.
     Matching logic strictly copied from yolo/test.py lines 177-240 (greedy matching per class).
  3. Optimal threshold tau* chosen by maximizing validation Macro-F1 over the two classes.
  4. Test set evaluated once using the frozen optimal threshold tau*.
  5. Official COCO metrics computed on val and test via official CocoEvaluator:
     mAP50-95, mAP50, mAP75, and per-class AP50-95 and AP50 for Active TB and Obsolete TB.
  6. Outputs saved to outputs/run_A_scratch/: val_results.json, test_results.json, results.txt.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image

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

# Detection Classes (0: Background, 1: ActiveTuberculosis, 2: ObsoletePulmonaryTuberculosis)
CLASS_NAMES = {
    0: "ActiveTuberculosis",
    1: "ObsoletePulmonaryTuberculosis",
}


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
    Copied from yolo/test.py:123-144.
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
    Copied strictly from yolo/test.py:177-240 with matched IoU recording.
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


def run_threshold_optimization_sweep(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    iou_thresh: float = 0.50,
    thresh_start: float = 0.01,
    thresh_end: float = 0.95,
    thresh_step: float = 0.01,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Validation score threshold sweep matching yolo/test.py:303-395.
    Selects optimal threshold tau* maximizing macro-F1 over the two classes.
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

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            optimal_threshold = thresh_val
            best_record = rec

    return optimal_threshold, best_record, sweep_records


def evaluate_test_detection(
    gt_map: Dict[str, List[BoundingBox]],
    pred_map: Dict[str, List[BoundingBox]],
    optimal_threshold: float,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates detection performance on the test set using the frozen validation threshold.
    Copied strictly from yolo/test.py:399-490.
    """
    tot_tp = {0: 0, 1: 0}
    tot_fp = {0: 0, 1: 0}
    tot_fn = {0: 0, 1: 0}
    cm_3x3_total = np.zeros((3, 3), dtype=np.int64)
    all_matched_ious: List[float] = []

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
        "per_class_results": class_results,
        "detection_confusion_matrix_3x3": cm_3x3_total.tolist(),
    }


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

            # Format for CocoEvaluator
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


def extract_coco_dict(coco_evaluator) -> Dict[str, Any]:
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate RetinaNet on TBX11K")
    parser.add_argument("--weights", default="outputs/run_A_scratch/best.pth", type=str, help="checkpoint path")
    parser.add_argument("--output-dir", default="outputs/run_A_scratch", type=str, help="output directory")
    parser.add_argument("--device", default="cuda", type=str, help="compute device")
    parser.add_argument("--iou-thresh", default=0.50, type=float, help="IoU threshold for greedy matching")
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

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint file not found: {args.weights}")

    # Load model
    print(f"Loading checkpoint: {args.weights}")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)

    model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=3,
        trainable_backbone_layers=5,
        min_size=512,
        max_size=512,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

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

    val_results = {
        "split": "val",
        "optimal_threshold": optimal_tau,
        "validation_sweep_best": best_val_rec,
        "coco_metrics": val_coco_metrics,
        "sweep_records": sweep_records,
    }
    with open(os.path.join(args.output_dir, "val_results.json"), "w", encoding="utf-8") as f:
        json.dump(val_results, f, indent=2)

    # 3. Test Inference & Evaluation (Evaluated once with frozen tau*)
    print("\n=================== STAGE 2: TEST EVALUATION (FROZEN THRESHOLD) ===================")
    test_evaluator, test_gt_map, test_pred_map = run_inference_and_coco_eval(model, test_dataset, device)
    test_coco_metrics = extract_coco_dict(test_evaluator)

    test_det_metrics = evaluate_test_detection(
        gt_map=test_gt_map,
        pred_map=test_pred_map,
        optimal_threshold=optimal_tau,
        iou_thresh=args.iou_thresh,
    )

    test_results = {
        "split": "test",
        "frozen_threshold": optimal_tau,
        "detection_metrics": test_det_metrics,
        "coco_metrics": test_coco_metrics,
    }
    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    # Required statement
    print("\nThresholds were tuned on val only; test evaluated once.")

    # Generate readable results.txt
    det_sum = test_det_metrics["overall_summary"]
    per_cls = test_det_metrics["per_class_results"]
    report_lines = [
        "================================================================================",
        "TBX11K RETINANET (RESNET-50 + FPN FROM SCRATCH) EVALUATION REPORT",
        "================================================================================",
        f"Weights Checkpoint:          {args.weights}",
        f"Validation Optimal Threshold: tau* = {optimal_tau:.2f} (Tuned on val.csv only)",
        f"Matching IoU Threshold:       {args.iou_thresh:.2f}",
        f"Validation Macro-F1 at tau*:  {best_val_rec['macro_f1']:.4f}",
        "",
        "Thresholds were tuned on val only; test evaluated once.",
        "",
        "--- DISCRETE DETECTION METRICS ON TEST SPLIT (tau* = {:.2f}, IoU = {:.2f}) ---".format(optimal_tau, args.iou_thresh),
        f"Macro Detection F1:          {det_sum['macro_f1']:.4f}",
        f"Macro Precision:             {det_sum['macro_precision']:.4f}",
        f"Macro Recall:                {det_sum['macro_recall']:.4f}",
        f"Micro Detection F1:          {det_sum['micro_f1']:.4f}",
        f"Micro Precision:             {det_sum['micro_precision']:.4f}",
        f"Micro Recall:                {det_sum['micro_recall']:.4f}",
        f"Mean Matched IoU:            {det_sum['mean_matched_iou']:.4f}",
        "",
        "Per-Class Detection Performance on Test Split:",
        "  Class Name                     TP    FP    FN    GT    Pred   Precision  Recall     F1",
        "  --------------------------------------------------------------------------------------",
    ]
    for cname, res in per_cls.items():
        report_lines.append(
            f"  {cname:<28s} {res['true_positives']:5d} {res['false_positives']:5d} {res['false_negatives']:5d} "
            f"{res['total_gt_boxes']:5d} {res['total_pred_boxes']:5d}   "
            f"{res['precision']:8.4f}  {res['recall']:8.4f}  {res['f1_score']:8.4f}"
        )
    report_lines.extend([
        "  --------------------------------------------------------------------------------------",
        f"  All classes (Macro)                                           {det_sum['macro_precision']:8.4f}  {det_sum['macro_recall']:8.4f}  {det_sum['macro_f1']:8.4f}",
        "",
        "--- COCO-STYLE METRICS (OFFICIAL COCOEVALUATOR) ---",
        f"Validation Set mAP @ IoU 0.50:0.95: {val_coco_metrics['mAP50_95']:.4f}",
        f"Validation Set mAP @ IoU 0.50:      {val_coco_metrics['mAP50']:.4f}",
        f"Validation Set mAP @ IoU 0.75:      {val_coco_metrics['mAP75']:.4f}",
        f"Test Set mAP @ IoU 0.50:0.95:        {test_coco_metrics['mAP50_95']:.4f}",
        f"Test Set mAP @ IoU 0.50:             {test_coco_metrics['mAP50']:.4f}",
        f"Test Set mAP @ IoU 0.75:             {test_coco_metrics['mAP75']:.4f}",
        "",
        "Per-Class COCO AP on Test Split:",
        f"  ActiveTuberculosis:            AP @ IoU 0.50:0.95 = {test_coco_metrics['per_class']['ActiveTuberculosis']['AP50_95']:.4f}, AP @ IoU 0.50 = {test_coco_metrics['per_class']['ActiveTuberculosis']['AP50']:.4f}, AP @ IoU 0.75 = {test_coco_metrics['per_class']['ActiveTuberculosis']['AP75']:.4f}",
        f"  ObsoletePulmonaryTuberculosis: AP @ IoU 0.50:0.95 = {test_coco_metrics['per_class']['ObsoletePulmonaryTuberculosis']['AP50_95']:.4f}, AP @ IoU 0.50 = {test_coco_metrics['per_class']['ObsoletePulmonaryTuberculosis']['AP50']:.4f}, AP @ IoU 0.75 = {test_coco_metrics['per_class']['ObsoletePulmonaryTuberculosis']['AP75']:.4f}",
        "================================================================================",
    ])

    results_txt_path = os.path.join(args.output_dir, "results.txt")
    with open(results_txt_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(report_lines) + "\n")
    print(f"Saved readable results summary to {results_txt_path}")


if __name__ == "__main__":
    main()
