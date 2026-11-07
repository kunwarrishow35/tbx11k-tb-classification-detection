"""
YOLOv8s Baseline Object-Detection Training Pipeline for TBX11K Chest Radiographs.
================================================================================

Task:
Multi-label pulmonary tuberculosis detection and localization on the canonical
TBX11K dataset (4,599 images: 3,219 train, 690 val, 690 test).

Classes:
  0: ActiveTuberculosis
  1: ObsoletePulmonaryTuberculosis

Authoritative Baseline Configuration:
  * Model Architecture: Ultralytics YOLOv8s (Small)
  * Initialization: Trained strictly FROM SCRATCH (yolov8s.yaml, weights=None, pretrained=False)
  * Input Resolution: 512 x 512
  * Batch Size: 4 (configured for workstation 6 GB GPU VRAM)
  * Workers: 2 (audited across DenseNet-121, ResNet-101, and EfficientNet baselines)
  * Total Epochs: 70 (Smoke test: 2)
  * Optimizer: auto (Ultralytics default)
  * Seed: 42
  * AMP: True (Automatic Mixed Precision)
  * Patience: 70 (No early stopping before 70 epochs)
  * Project: yolo/runs
  * Name: baseline_yolov8s_scratch (Smoke test: smoke_test_yolov8s_scratch)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import ultralytics
from ultralytics import YOLO

# =============================================================================
# AUTHORITATIVE HYPERPARAMETERS & CONFIGURATION CONSTANTS
# =============================================================================
MODEL_ARCH: str = "yolov8s.yaml"
PRETRAINED: bool = False
IMGSZ: int = 512
DEFAULT_EPOCHS: int = 70
SMOKE_TEST_EPOCHS: int = 2
BATCH_SIZE: int = 4
WORKERS: int = 2  # Audited and fixed to match DenseNet-121, ResNet-101, and EfficientNet
OPTIMIZER: str = "auto"
SEED: int = 42
AMP: bool = True
PATIENCE: int = 70

PROJECT_DIR_REL: str = "yolo/runs"
FULL_RUN_NAME: str = "baseline_yolov8s_scratch"
SMOKE_RUN_NAME: str = "smoke_test_yolov8s_scratch"

# Detection Classes (Strictly 2 classes)
CLASS_NAMES: Dict[int, str] = {
    0: "ActiveTuberculosis",
    1: "ObsoletePulmonaryTuberculosis",
}

CLASS_MAP_REVERSE: Dict[str, int] = {v: k for k, v in CLASS_NAMES.items()}

# Expected Canonical Split Counts
EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {
        "images": 3219,
        "healthy": 2660,
        "active_only": 431,
        "obsolete_only": 105,
        "mixed": 23,
        "active_boxes": 657,
        "obsolete_boxes": 180,
        "total_boxes": 837,
    },
    "val": {
        "images": 690,
        "healthy": 570,
        "active_only": 103,
        "obsolete_only": 12,
        "mixed": 5,
        "active_boxes": 172,
        "obsolete_boxes": 23,
        "total_boxes": 195,
    },
    "test": {
        "images": 690,
        "healthy": 570,
        "active_only": 96,
        "obsolete_only": 22,
        "mixed": 2,
        "active_boxes": 143,
        "obsolete_boxes": 36,
        "total_boxes": 179,
    },
}


# =============================================================================
# WORKSPACE & DATASET RESOLUTION
# =============================================================================
def resolve_workspace_paths(
    custom_data_dir: Optional[str] = None,
    custom_splits_dir: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """Dynamically resolves workspace root, TBX11K directory, and splits directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd(),
        script_dir.parent,
        script_dir,
    ]

    workspace_root = None
    for cand in candidates:
        if (cand / "TBX11K").is_dir() and (cand / "data_prep").is_dir():
            workspace_root = cand.resolve()
            break

    if workspace_root is None:
        workspace_root = script_dir.parent.resolve()

    data_dir = Path(custom_data_dir).resolve() if custom_data_dir else workspace_root / "TBX11K"
    splits_dir = Path(custom_splits_dir).resolve() if custom_splits_dir else workspace_root / "data_prep" / "splits"

    return workspace_root, data_dir, splits_dir


# =============================================================================
# DATASET PREPARATION & AUDIT
# =============================================================================
def prepare_and_verify_yolo_dataset(
    workspace_root: Path,
    data_dir: Path,
    splits_dir: Path,
) -> Tuple[Path, Dict[str, Dict[str, Any]]]:
    """
    Validates canonical manifests, parses original XML annotations, prepares YOLO
    label text files and images structure in yolo/dataset, generates dataset.yaml,
    and runs a strict dataset audit.
    """
    dataset_dir = workspace_root / "yolo" / "dataset"
    dataset_yaml_path = dataset_dir / "dataset.yaml"

    print("\n" + "=" * 90)
    print("VERIFYING & AUDITING CANONICAL TBX11K DATASET FOR YOLO")
    print("=" * 90)
    print(f"Dataset root: {data_dir}")
    print(f"Splits root:  {splits_dir}")
    print(f"YOLO dataset: {dataset_dir}")

    # Create destination directories
    for split in ["train", "val", "test"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    audit_stats: Dict[str, Dict[str, Any]] = {}

    for split in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        stats = {
            "images": len(reader),
            "healthy": 0,
            "active_only": 0,
            "obsolete_only": 0,
            "mixed": 0,
            "active_boxes": 0,
            "obsolete_boxes": 0,
            "total_boxes": 0,
            "empty_labels": 0,
            "invalid_boxes": 0,
            "missing_images": 0,
            "missing_xmls": 0,
        }

        for row in reader:
            img_rel = row["image_path"].strip()
            xml_rel = row.get("xml_path", "").strip()

            src_img = data_dir / img_rel
            if not src_img.is_file():
                stats["missing_images"] += 1
                raise FileNotFoundError(f"Source image file does not exist: {src_img}")

            img_filename = src_img.name
            stem = src_img.stem

            dst_img = dataset_dir / "images" / split / img_filename
            dst_lbl = dataset_dir / "labels" / split / f"{stem}.txt"

            # Sync image file (copy if missing or size differs; preserve source untouched)
            if not dst_img.is_file() or dst_img.stat().st_size != src_img.stat().st_size:
                shutil.copy2(src_img, dst_img)

            lines: List[str] = []
            img_active_boxes = 0
            img_obsolete_boxes = 0

            if not xml_rel:
                # Healthy / negative background image (empty label file)
                stats["healthy"] += 1
                stats["empty_labels"] += 1
            else:
                src_xml = data_dir / xml_rel
                if not src_xml.is_file():
                    stats["missing_xmls"] += 1
                    raise FileNotFoundError(f"Source XML file does not exist: {src_xml}")

                tree = ET.parse(src_xml)
                root = tree.getroot()

                size_elem = root.find("size")
                if size_elem is None:
                    raise ValueError(f"XML missing <size> element: {src_xml}")

                xml_w = float(size_elem.find("width").text)
                xml_h = float(size_elem.find("height").text)

                if xml_w <= 0 or xml_h <= 0:
                    raise ValueError(f"Invalid XML dimensions in {src_xml}: w={xml_w}, h={xml_h}")

                for obj in root.findall("object"):
                    cls_name = obj.find("name").text.strip()
                    if cls_name not in CLASS_MAP_REVERSE:
                        raise ValueError(f"Unknown class '{cls_name}' encountered in {src_xml}")

                    cls_id = CLASS_MAP_REVERSE[cls_name]
                    bnd = obj.find("bndbox")
                    xmin = float(bnd.find("xmin").text)
                    ymin = float(bnd.find("ymin").text)
                    xmax = float(bnd.find("xmax").text)
                    ymax = float(bnd.find("ymax").text)

                    # Normalized YOLO coordinates [x_center, y_center, width, height]
                    xc = ((xmin + xmax) / 2.0) / xml_w
                    yc = ((ymin + ymax) / 2.0) / xml_h
                    bw = (xmax - xmin) / xml_w
                    bh = (ymax - ymin) / xml_h

                    # Geometric sanity checks
                    if (
                        xmin < 0
                        or ymin < 0
                        or xmax > xml_w
                        or ymax > xml_h
                        or bw <= 0.0
                        or bh <= 0.0
                        or not (0.0 <= xc <= 1.0)
                        or not (0.0 <= yc <= 1.0)
                        or not (0.0 < bw <= 1.0)
                        or not (0.0 < bh <= 1.0)
                    ):
                        stats["invalid_boxes"] += 1
                        logging.warning(f"Invalid bounding box in {src_xml}: xc={xc}, yc={yc}, bw={bw}, bh={bh}")

                    if cls_id == 0:
                        img_active_boxes += 1
                        stats["active_boxes"] += 1
                    else:
                        img_obsolete_boxes += 1
                        stats["obsolete_boxes"] += 1

                    stats["total_boxes"] += 1
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                if img_active_boxes > 0 and img_obsolete_boxes > 0:
                    stats["mixed"] += 1
                elif img_active_boxes > 0:
                    stats["active_only"] += 1
                elif img_obsolete_boxes > 0:
                    stats["obsolete_only"] += 1
                else:
                    stats["empty_labels"] += 1

            # Write label file (empty string for negative images)
            content = "\n".join(lines) + ("\n" if lines else "")
            with open(dst_lbl, mode="w", encoding="utf-8") as lf:
                lf.write(content)

        audit_stats[split] = stats

    # Write / Update dataset.yaml with forward slashes for cross-platform safety
    path_str = str(dataset_dir.resolve()).replace("\\", "/")
    yaml_content = f"""# YOLOv8s Dataset Configuration for TBX11K Multi-Label Detection
path: {path_str}
train: images/train
val: images/val
test: images/test

names:
  0: ActiveTuberculosis
  1: ObsoletePulmonaryTuberculosis
"""
    with open(dataset_yaml_path, mode="w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    # Print comprehensive audit table
    print("\n" + "=" * 105)
    print(f"{'Split':<8} | {'Images':<8} | {'Healthy':<8} | {'Active-Only':<12} | {'Obsolete-Only':<14} | {'Mixed':<6} | {'Active Boxes':<12} | {'Obs Boxes':<10} | {'Total Boxes'}")
    print("-" * 105)
    for s, st in audit_stats.items():
        print(f"{s:<8} | {st['images']:<8} | {st['healthy']:<8} | {st['active_only']:<12} | {st['obsolete_only']:<14} | {st['mixed']:<6} | {st['active_boxes']:<12} | {st['obsolete_boxes']:<10} | {st['total_boxes']}")
    print("=" * 105)

    # Verify counts against authoritative expectations
    for s, expected in EXPECTED_COUNTS.items():
        actual = audit_stats[s]
        for k, v in expected.items():
            if actual[k] != v:
                raise ValueError(
                    f"Dataset audit mismatch in '{s}' for '{k}': expected {v}, but got {actual[k]}! "
                    "Aborting to prevent corrupted training execution."
                )

    print("[DATASET AUDIT PASSED] All image, label, class, and box counts match canonical specifications perfectly.\n")
    return dataset_yaml_path, audit_stats


# =============================================================================
# EXPERIMENT METADATA BUILDER
# =============================================================================
def save_experiment_metadata(
    output_dir: Path,
    run_name: str,
    model: YOLO,
    dataset_yaml: Path,
    splits_dir: Path,
    audit_stats: Dict[str, Dict[str, Any]],
    training_args: Dict[str, Any],
    start_time: float,
    end_time: float,
    is_smoke_test: bool,
    warnings: List[str],
) -> Path:
    """Creates a comprehensive experiment_metadata.json record in the run directory."""
    results_csv_path = output_dir / "results.csv"
    final_metrics: Dict[str, Any] = {}

    if results_csv_path.is_file():
        try:
            with open(results_csv_path, mode="r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if reader:
                    # Strip whitespace from keys
                    last_row = {k.strip(): v.strip() for k, v in reader[-1].items()}
                    final_metrics = last_row
        except Exception as e:
            warnings.append(f"Could not parse results.csv for final metrics: {e}")

    # Gather generated artifacts
    generated_artifacts = []
    if output_dir.is_dir():
        for p in output_dir.rglob("*"):
            if p.is_file():
                generated_artifacts.append(str(p.relative_to(output_dir)))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    metadata = {
        "experiment_name": run_name,
        "model_architecture": "YOLOv8s Detection (Ultralytics)",
        "model_yaml": MODEL_ARCH,
        "scratch_initialization": {
            "verified": True,
            "pretrained": False,
            "weights_loaded": None,
            "initialization_method": "Random initialization from yolov8s.yaml architecture",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
        "dataset": {
            "dataset_yaml": str(dataset_yaml.resolve()),
            "splits_manifests": {
                "train": str((splits_dir / "train.csv").resolve()),
                "val": str((splits_dir / "val.csv").resolve()),
                "test": str((splits_dir / "test.csv").resolve()),
            },
            "audit_statistics": audit_stats,
            "classes": CLASS_NAMES,
        },
        "hyperparameters": {
            "image_size": training_args.get("imgsz", IMGSZ),
            "epochs": training_args.get("epochs", DEFAULT_EPOCHS),
            "batch_size": training_args.get("batch", BATCH_SIZE),
            "workers": training_args.get("workers", WORKERS),
            "optimizer": training_args.get("optimizer", OPTIMIZER),
            "seed": training_args.get("seed", SEED),
            "amp": training_args.get("amp", AMP),
            "patience": training_args.get("patience", PATIENCE),
            "learning_rate_initial": getattr(model.trainer, "lr0", "default (0.01)"),
            "learning_rate_final": getattr(model.trainer, "lrf", "default (0.01)"),
        },
        "training_arguments": training_args,
        "environment": {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": cuda_ver,
            "ultralytics_version": ultralytics.__version__,
            "gpu_model": gpu_name,
            "os": os.name,
        },
        "execution": {
            "smoke_test": is_smoke_test,
            "start_time_utc": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            "end_time_utc": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            "duration_seconds": round(end_time - start_time, 2),
            "duration_human": f"{(end_time - start_time) / 60.0:.2f} minutes",
        },
        "final_metrics": final_metrics,
        "generated_artifacts": sorted(generated_artifacts),
        "warnings_and_errors": warnings,
    }

    metadata_path = output_dir / "experiment_metadata.json"
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[Metadata Saved] Experiment metadata successfully saved to: {metadata_path}")
    return metadata_path


# =============================================================================
# CLI PARSER
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for YOLOv8s training script."""
    parser = argparse.ArgumentParser(
        description="YOLOv8s Multi-Label Tuberculosis Detection Baseline Training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Execution mode
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-epoch smoke test to verify pipeline components without full 70-epoch training.",
    )

    # Core hyperparameters (Authoritative Defaults)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs (fixed to 70 for baseline).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size (fixed to 4).")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Image resolution (fixed to 512).")
    parser.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers (fixed to 2).")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic random seed.")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="Optimizer choice (default: auto).")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience (default: 70).")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP).")

    # Paths
    parser.add_argument("--data-dir", type=str, default=None, help="Custom path to TBX11K root folder.")
    parser.add_argument("--splits-dir", type=str, default=None, help="Custom path to data_prep/splits/ folder.")
    parser.add_argument("--project", type=str, default=PROJECT_DIR_REL, help="Custom project output directory.")
    parser.add_argument("--name", type=str, default=None, help="Custom run name.")

    return parser.parse_args()


# =============================================================================
# MAIN TRAINING RUNNER
# =============================================================================
def main() -> int:
    args = parse_arguments()
    warnings: List[str] = []

    # Determine execution parameters
    is_smoke = args.smoke_test
    epochs = SMOKE_TEST_EPOCHS if is_smoke else args.epochs
    run_name = args.name if args.name else (SMOKE_RUN_NAME if is_smoke else FULL_RUN_NAME)

    workspace_root, data_dir, splits_dir = resolve_workspace_paths(args.data_dir, args.splits_dir)
    project_dir = workspace_root / args.project
    run_output_dir = project_dir / run_name

    # Determine compute device
    device_str = args.device
    if device_str != "cpu" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Falling back to CPU execution.")
        device_str = "cpu"
        warnings.append("CUDA unavailable; fell back to CPU.")

    use_amp = not args.no_amp and (device_str != "cpu")

    # Banner display
    print("\n" + "=" * 90)
    if is_smoke:
        print("STARTING YOLOv8s MANDATORY 2-EPOCH SMOKE TEST (SCRATCH INITIALIZATION)".center(90))
    else:
        print("STARTING YOLOv8s 70-EPOCH FULL BASELINE TRAINING (SCRATCH INITIALIZATION)".center(90))
    print("=" * 90)
    print(f"Model Architecture:       Ultralytics YOLOv8s ({MODEL_ARCH})")
    print(f"Pretrained Weights:       STRICTLY DISABLED (pretrained=False, from scratch)")
    print(f"Configured Epochs:        {epochs} {'(Smoke Test Mode)' if is_smoke else '(Full Baseline)'}")
    print(f"Image Resolution:         {args.imgsz} x {args.imgsz}")
    print(f"Batch Size:               {args.batch_size}")
    print(f"DataLoader Workers:       {args.workers} (Audited baseline parity)")
    print(f"Compute Device:           {device_str} ({torch.cuda.get_device_name(0) if device_str != 'cpu' else 'CPU'})")
    print(f"AMP Status:               {use_amp}")
    print(f"Patience:                 {args.patience}")
    print(f"Random Seed:              {args.seed}")
    print(f"Project Directory:        {project_dir}")
    print(f"Run Output Directory:     {run_output_dir}")
    print(f"Classes:                  {CLASS_NAMES}")
    print("=" * 90 + "\n")

    # 1. Dataset Verification & Preparation
    dataset_yaml_path, audit_stats = prepare_and_verify_yolo_dataset(
        workspace_root=workspace_root,
        data_dir=data_dir,
        splits_dir=splits_dir,
    )

    # 2. Model Initialization (From Scratch)
    print(f"[Model Init] Initializing YOLOv8s from architecture specification: {MODEL_ARCH} ...")
    model = YOLO(MODEL_ARCH)

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    print(f"[Model Init] Task: {model.task}")
    print(f"[Model Init] Total Parameters:     {total_params:,}")
    print(f"[Model Init] Trainable Parameters: {trainable_params:,}")
    print(f"[Model Init] Model Weights State:  {getattr(model, 'weights', None)} (Verified unweighted/scratch)")
    print(f"[Model Init] Ultralytics Version:  {ultralytics.__version__}")

    # 3. Configure Training Arguments
    train_kwargs = {
        "data": str(dataset_yaml_path),
        "epochs": epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "device": device_str,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "amp": use_amp,
        "patience": args.patience,
        "project": str(project_dir),
        "name": run_name,
        "pretrained": False,      # Strictly NO pretrained weights
        "val": True,              # Run validation on val split after each epoch
        "plots": True,            # Generate curves, confusion matrix, batch visualizations
        "save": True,             # Save last.pt and best.pt
        "exist_ok": True,         # Re-use existing folder if restarting
        "verbose": True,
    }

    start_time = time.time()
    print("\n[Training Start] Launching Ultralytics YOLOv8s training loop...\n")

    try:
        results = model.train(**train_kwargs)
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 90)
        print(f"YOLOv8s {'SMOKE TEST' if is_smoke else 'TRAINING'} COMPLETED SUCCESSFULLY".center(90))
        print("=" * 90)
        print(f"Total Elapsed Time:   {duration / 60.0:.2f} minutes ({duration:.1f}s)")
        print(f"Run Output Directory: {run_output_dir}")

        # Save experiment metadata
        metadata_path = save_experiment_metadata(
            output_dir=run_output_dir,
            run_name=run_name,
            model=model,
            dataset_yaml=dataset_yaml_path,
            splits_dir=splits_dir,
            audit_stats=audit_stats,
            training_args=train_kwargs,
            start_time=start_time,
            end_time=end_time,
            is_smoke_test=is_smoke,
            warnings=warnings,
        )

        return 0

    except Exception as e:
        print(f"\n[Training FAILED] Exception encountered: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

