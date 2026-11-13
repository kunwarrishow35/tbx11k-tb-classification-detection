r"""PyTorch Detection Training - TBX11K From-Scratch RetinaNet (ResNet-50 + FPN).

Adapted from the official torchvision detection reference script (v0.29.1),
multi-task and distributed arguments pruned.
https://github.com/pytorch/vision/blob/v0.29.1/references/detection/train.py

Official Reference Recipe:
    In torchvision v0.29.1 references/detection/README.md, the RetinaNet baseline training command:
        torchrun --nproc_per_node=8 train.py --dataset coco --model retinanet_resnet50_fpn \
            --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 \
            --weights-backbone ResNet50_Weights.IMAGENET1K_V1
    Run A is strictly from-scratch and deliberately omits --weights-backbone.
    Official docstring guidance:
        "The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
            --lr 0.02 --batch-size 2 --world-size 8
        If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU."
    For RetinaNet, base lr is 0.01 on 8 GPUs x 2 images (effective batch size 16).
    On our single-GPU setup with batch size 4, the learning rate is scaled linearly according to
    Goyal et al. (2017, Section 2.1, page 2) and official docstring guidance:
        lr = 0.01 * (batch_size 4 / 16) = 0.0025.
    Warmup is 803 iterations: min(1000, len(data_loader) - 1). GroupedBatchSampler yields
    floor(3219 / 4) = 804 batches, so 3 images are skipped each epoch.
"""

import argparse
import csv
import datetime
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection

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
from official_ref.engine import evaluate, train_one_epoch
from official_ref.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

import dataset as tbx_dataset


def set_seed(seed: int = 42) -> None:
    """Seeds python, numpy and torch (seed 42)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(is_train: bool, args):
    split = "train" if is_train else "val"
    transforms = get_transform(is_train, args)
    ds = tbx_dataset.TBX11KDetectionDataset(
        split=split,
        workspace_root=args.workspace_root,
        transforms=transforms,
        is_train=is_train,
    )
    # num_classes = 3 (0: Background, 1: ActiveTuberculosis, 2: ObsoletePulmonaryTuberculosis)
    return ds, 3


def get_transform(is_train: bool, args):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=args.data_augmentation,
            backend=args.backend,
            use_v2=args.use_v2,
        )
    else:
        return presets.DetectionPresetEval(
            backend=args.backend,
            use_v2=args.use_v2,
        )


def extract_coco_metrics(coco_evaluator) -> Dict[str, float]:
    """Extracts standard COCO mAP and per-class AP from CocoEvaluator."""
    metrics: Dict[str, float] = {
        "mAP50_95": 0.0,
        "mAP50": 0.0,
        "mAP75": 0.0,
        "AP50_95_active": 0.0,
        "AP50_active": 0.0,
        "AP50_95_obsolete": 0.0,
        "AP50_obsolete": 0.0,
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
        cat_ids = ce.params.catIds  # Typically [1, 2]
        for idx, cid in enumerate(cat_ids):
            s = precision[:, :, idx, 0, 2]
            valid_s = s[s > -1]
            if len(valid_s) == 0:
                raise RuntimeError(
                    f"COCO evaluation error: No valid precision entries (all -1) for category id {cid}. "
                    f"Both classes have ground truth; this indicates an evaluation failure."
                )
            ap_all = float(np.mean(valid_s))

            s50 = precision[0, :, idx, 0, 2]
            valid_s50 = s50[s50 > -1]
            if len(valid_s50) == 0:
                raise RuntimeError(
                    f"COCO evaluation error: No valid AP50 precision entries (all -1) for category id {cid}."
                )
            ap_50 = float(np.mean(valid_s50))

            if cid == 1:
                metrics["AP50_95_active"] = ap_all
                metrics["AP50_active"] = ap_50
            elif cid == 2:
                metrics["AP50_95_obsolete"] = ap_all
                metrics["AP50_obsolete"] = ap_50

    return metrics


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch RetinaNet TBX11K Training", add_help=add_help)

    parser.add_argument("--workspace-root", default=None, type=str, help="workspace root path")
    parser.add_argument("--model", default="retinanet_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=4, type=int, help="batch size per GPU (default: 4 for RTX 4050 6GB)"
    )
    parser.add_argument("--epochs", default=70, type=int, metavar="N", help="number of total epochs to run (default: 70)")
    parser.add_argument(
        "-j", "--workers", default=2, type=int, metavar="N", help="number of data loading workers (default: 2)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer (default: sgd)")
    parser.add_argument(
        "--lr",
        default=0.0025,
        type=float,
        help="initial learning rate (0.0025 = 0.01 * 4/16: linear scaling with batch size 4 on 1 GPU per Goyal et al. / docstring)",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum (default: 0.9)")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[43, 59],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (scaled from official [16, 22] for 70 epochs: 16*70/26=43, 22*70/26=59)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (default: 0.1)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="outputs/run_B_fulldata", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int, help="aspect ratio group factor (default: 3)")
    parser.add_argument(
        "--trainable-backbone-layers",
        default=None,
        type=int,
        help="number of trainable layers of backbone (default None -> all 5 layers trainable from scratch without warning)",
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument("--weights", default=None, type=str, help="model weights (None for train from scratch)")
    parser.add_argument(
        "--weights-backbone", default=None, type=str, help="backbone weights (None for random backbone)"
    )
    parser.add_argument("--amp", action="store_true", default=True, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable mixed precision")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--seed", default=42, type=int, help="random seed (default: 42)")

    # Smoke testing flags (Phase 3 code checks)
    parser.add_argument("--smoke-iters", default=0, type=int, help="Limit training to N iterations (0 for full training)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip validation evaluation (used for smoke test)")

    return parser


def main(args):
    # Enforce dataset integrity self-check prior to any execution
    tbx_dataset.self_check_dataset()

    set_seed(args.seed)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Using compute device: {device}")

    # Data loading code
    print("Loading data")
    dataset, num_classes = get_dataset(is_train=True, args=args)
    dataset_test, _ = get_dataset(is_train=False, args=args)

    print(f"Training set: {len(dataset)} images (full canonical split: 559 TB + 2660 healthy)")
    print(f"Validation set: {len(dataset_test)} images (full canonical split matching yolo/test.py)")

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(args.seed)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        generator=train_loader_generator,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "min_size": 512,
        "max_size": 512,
    }

    model = torchvision.models.get_model(
        args.model,
        weights=args.weights,
        weights_backbone=args.weights_backbone,
        num_classes=num_classes,
        **kwargs,
    )
    model.to(device)

    # Print backbone BN type and trainable parameter counts
    bn_types = set(type(m).__name__ for m in model.backbone.modules() if "BatchNorm" in type(m).__name__)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"[Model Architecture] {args.model}")
    print(f"[Backbone Normalization] BN types: {bn_types}")
    print(f"[Parameter Counts] Total: {total_params:,d}, Trainable: {trainable_params:,d}, Frozen: {frozen_params:,d}")

    parameters = [p for p in model.parameters() if p.requires_grad]

    opt_name = args.opt.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD is supported in this recipe.")

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    best_map50_95 = -1.0
    metrics_log_path = os.path.join(args.output_dir, "metrics.csv")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_map50_95 = checkpoint.get("best_map50_95", -1.0)
        if args.amp and "scaler" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resumed checkpoint at epoch {checkpoint['epoch']} (next epoch: {args.start_epoch})")

    # Initialize metrics.csv if file does not exist; otherwise append on resume
    if not os.path.exists(metrics_log_path):
        with open(metrics_log_path, "w", newline="", encoding="utf-8") as mf:
            writer = csv.writer(mf)
            writer.writerow([
                "epoch",
                "lr",
                "total_loss",
                "loss_classification",
                "loss_bbox_regression",
                "val_mAP50_95",
                "val_mAP50",
                "val_mAP75",
                "val_AP50_95_active",
                "val_AP50_active",
                "val_AP50_95_obsolete",
                "val_AP50_obsolete",
                "epoch_time_seconds",
            ])

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()

        # Handle smoke testing iteration limit
        if args.smoke_iters > 0:
            print(f"[Smoke Test] Running strictly {args.smoke_iters} iterations for epoch {epoch}...")
            # Create a limited loader slice for code verification
            limited_batches = []
            for b_idx, batch in enumerate(data_loader):
                limited_batches.append(batch)
                if len(limited_batches) >= args.smoke_iters:
                    break
            train_metrics = train_one_epoch(model, optimizer, limited_batches, device, epoch, args.print_freq, scaler)
        else:
            train_metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)

        lr_scheduler.step()
        epoch_time = time.time() - epoch_start

        # Extract training losses from MetricLogger (fail fast if expected meters are missing)
        if "loss" not in train_metrics.meters:
            raise RuntimeError(
                f"Missing 'loss' meter in train_metrics.meters. Found: {list(train_metrics.meters.keys())}"
            )
        if "classification" not in train_metrics.meters:
            raise RuntimeError(
                f"Missing 'classification' meter in train_metrics.meters. Found: {list(train_metrics.meters.keys())}"
            )
        if "bbox_regression" not in train_metrics.meters:
            raise RuntimeError(
                f"Missing 'bbox_regression' meter in train_metrics.meters. Found: {list(train_metrics.meters.keys())}"
            )

        total_loss = train_metrics.meters["loss"].global_avg
        loss_cls = train_metrics.meters["classification"].global_avg
        loss_reg = train_metrics.meters["bbox_regression"].global_avg
        current_lr = train_metrics.meters["lr"].global_avg if "lr" in train_metrics.meters else args.lr

        # Validation evaluation
        if not args.skip_eval:
            print(f"Evaluating validation set for epoch {epoch}...")
            coco_evaluator = evaluate(model, data_loader_test, device=device)
            val_metrics = extract_coco_metrics(coco_evaluator)
            val_map = val_metrics["mAP50_95"]
            print(f"Epoch {epoch}: Val mAP50-95 = {val_map:.4f}, mAP50 = {val_metrics['mAP50']:.4f}")
        else:
            val_metrics = {
                "mAP50_95": 0.0,
                "mAP50": 0.0,
                "mAP75": 0.0,
                "AP50_95_active": 0.0,
                "AP50_active": 0.0,
                "AP50_95_obsolete": 0.0,
                "AP50_obsolete": 0.0,
            }
            val_map = 0.0

        # Save checkpoint.pth (for --resume) unconditionally each epoch
        if args.output_dir:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
                "best_map50_95": best_map50_95,
            }
            if args.amp and scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()

            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

            # Track and save best.pth (only during real runs with evaluation)
            if not args.skip_eval and val_map > best_map50_95:
                best_map50_95 = val_map
                checkpoint["best_map50_95"] = best_map50_95
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "best.pth"))
                print(f"--> Saved new best model checkpoint to {os.path.join(args.output_dir, 'best.pth')} (mAP50-95: {val_map:.4f})")

            # Record per-epoch metrics to CSV
            if not args.skip_eval:
                with open(metrics_log_path, "a", newline="", encoding="utf-8") as mf:
                    writer = csv.writer(mf)
                    writer.writerow([
                        epoch,
                        f"{current_lr:.6f}",
                        f"{total_loss:.4f}",
                        f"{loss_cls:.4f}",
                        f"{loss_reg:.4f}",
                        f"{val_metrics['mAP50_95']:.4f}",
                        f"{val_metrics['mAP50']:.4f}",
                        f"{val_metrics['mAP75']:.4f}",
                        f"{val_metrics['AP50_95_active']:.4f}",
                        f"{val_metrics['AP50_active']:.4f}",
                        f"{val_metrics['AP50_95_obsolete']:.4f}",
                        f"{val_metrics['AP50_obsolete']:.4f}",
                        f"{epoch_time:.2f}",
                    ])

        # If smoke testing, break after first epoch of limited iterations
        if args.smoke_iters > 0:
            print(f"[Smoke Test] Completed {args.smoke_iters} iterations successfully.")
            break

    # Save last.pth at training conclusion
    if args.output_dir and args.smoke_iters == 0:
        checkpoint["epoch"] = args.epochs - 1
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "last.pth"))
        print(f"Saved final model checkpoint to {os.path.join(args.output_dir, 'last.pth')}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
