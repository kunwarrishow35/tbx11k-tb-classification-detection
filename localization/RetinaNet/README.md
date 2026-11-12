# TBX11K RetinaNet (ResNet-50 + FPN) From-Scratch Localization Pipeline

This directory contains the from-scratch (Run A) lesion localization implementation for the TBX11K dataset, following torchvision's official `references/detection` recipe (`v0.29.1`).

---

## 1. Directory Structure

```text
localization/RetinaNet/
├── RetinaNet_R50_Official_vs_Ours (1).pdf # Authoritative specification PDF
├── official_ref/                          # Unmodified reference scripts (v0.29.1)
│   ├── engine.py
│   ├── utils.py
│   ├── coco_utils.py
│   ├── coco_eval.py
│   ├── transforms.py
│   ├── presets.py
│   ├── group_by_aspect_ratio.py
│   ├── SHA256SUMS.txt
│   └── SOURCE.txt
├── dataset.py                             # TBX11K detection dataset + self-check
├── train.py                               # Training pipeline adapted from the official script, multi-task and distributed arguments pruned
├── train_vs_official.diff                 # Unified diff of train.py against official
├── test.py                                # Evaluation pipeline mirroring yolo/test.py
├── config.json                            # Hyperparameters, versions & computed data counts
├── README.md                              # This documentation
└── outputs/
    └── run_A_scratch/                     # Checkpoints (best.pth, last.pth, checkpoint.pth)
        ├── metrics.csv                    # Per-epoch training losses (loss_classification, loss_bbox_regression) and COCO AP metrics
        ├── val_results.json               # Validation threshold sweep and metrics
        ├── test_results.json              # Final frozen test-set evaluation
        └── results.txt                    # Formatted evaluation report
```

---

## 2. Quickstart: How to Run

### Environment
Execute within the project virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Full 70-Epoch Training (Run A: From Scratch)
```powershell
python localization/RetinaNet/train.py --output-dir localization/RetinaNet/outputs/run_A_scratch
```
* Training runs for 70 epochs with batch size 4, base learning rate 0.0025, AMP enabled.
* Every epoch saves `checkpoint.pth` (resume state) and checks validation COCO mAP50-95.
* The model achieving the highest validation mAP50-95 is automatically saved to `best.pth`.
* Training conclues with `last.pth` saved at epoch 70.

### Resuming Training
```powershell
python localization/RetinaNet/train.py --output-dir localization/RetinaNet/outputs/run_A_scratch --resume localization/RetinaNet/outputs/run_A_scratch/checkpoint.pth
```

### Evaluation
```powershell
python localization/RetinaNet/test.py --weights localization/RetinaNet/outputs/run_A_scratch/best.pth --output-dir localization/RetinaNet/outputs/run_A_scratch
```
* Performs official COCO evaluation on validation and test splits.
* Optimizes the confidence threshold on the validation split over $\tau \in [0.01, 0.95]$ in 0.01 steps at IoU 0.50 (maximizing Macro-F1).
* Evaluates the test set once using the frozen optimal threshold $\tau^*$.
* Generates `results.txt`, `val_results.json`, and `test_results.json`.

### Evaluation Dry-Run
```powershell
python localization/RetinaNet/test.py --dry-run
```

---

## 3. What Changed vs. Official Torchvision Recipe

Every deviation from the official torchvision detection recipe (`pytorch/vision/references/detection/`) is enumerated and categorized:

| Recipe Component | Official Recipe Value | Our Implementation | Change Type | Reason |
| :--- | :--- | :--- | :---: | :--- |
| **Model** | `retinanet_resnet50_fpn` | `retinanet_resnet50_fpn` | **IDENTICAL** | Architecture preserved unchanged. |
| **Dataset** | COCO (91 classes) | TBX11K (`num_classes=3`: 0=Background, 1=Active TB, 2=Obsolete TB) | **DATASET** | Domain-specific classes. |
| **Training Images** | Images without annotations removed | TB images with $\ge 1$ lesion only (559 images) | **DATASET** | Preserves standard detection training policy. |
| **Validation / Test Images** | Full validation set | Full canonical sets (690 val, 690 test; healthy background included) | **DATASET** | Replicates `yolo/test.py` evaluation protocol. |
| **Weights** | ImageNet pretrained ResNet-50 (`--weights-backbone ResNet50_Weights.IMAGENET1K_V1`) | Random initialization (`weights=None, weights_backbone=None`) | **PROJECT** | The v0.29.1 README RetinaNet command passes `--weights-backbone ResNet50_Weights.IMAGENET1K_V1`; Run A deliberately omits that flag to train strictly from scratch. |
| **Trainable Layers** | 3 layers (stem & layer 1 frozen) | All 5 backbone layers trainable (`trainable_backbone_layers=None` default) | **PROJECT** | Builder default for from-scratch training enables all 5 stages without `UserWarning`. |
| **Normalization** | `FrozenBatchNorm2d` | Trainable `torch.nn.BatchNorm2d` | **PROJECT** | Builder default for from-scratch training. |
| **Input Resolution** | `min_size=800, max_size=1333` | `min_size=512, max_size=512` | **PROJECT** | Uniform project-wide input resolution. |
| **Image Normalization** | ImageNet mean and std | ImageNet mean and std | **IDENTICAL** | Kept inside model transform. |
| **Anchors & FPN** | 5 levels (P3–P7), 9 anchors/cell | 5 levels (P3–P7), 9 anchors/cell | **IDENTICAL** | Standard RetinaNet anchors. |
| **Loss Functions** | Sigmoid Focal Loss ($\alpha=0.25, \gamma=2$) + L1 Box Loss | Built-in torchvision RetinaNet loss | **IDENTICAL** | Official loss formulation untouched. |
| **Matching Thresholds** | Fg IoU $\ge 0.5$, Bg IoU $< 0.4$ | Fg IoU $\ge 0.5$, Bg IoU $< 0.4$ | **IDENTICAL** | Standard detection matching. |
| **Post-Processing** | Score floor 0.05, NMS 0.50, top 300 | Score floor 0.05, NMS 0.50, top 300 | **IDENTICAL** | Standard RetinaNet post-processing. |
| **Optimizer** | SGD (`momentum=0.9, weight_decay=1e-4`) | SGD (`momentum=0.9, weight_decay=1e-4`) | **IDENTICAL** | Official optimizer settings. |
| **Learning Rate** | 0.01 on 8 GPUs $\times$ 2 images (batch 16) | **0.0025** on 1 GPU $\times$ 4 images (batch 4) | **SYSTEM** | The v0.29.1 README command specifies `--lr 0.01`; scaled linearly for single GPU batch 4 via $0.01 \times (4 / 16) = 0.0025$ (Goyal et al., 2017). |
| **LR Scheduler** | `MultiStepLR` [16, 22], $\gamma=0.1$ (26 epochs) | `MultiStepLR` **[43, 59]**, $\gamma=0.1$ (70 epochs) | **PROJECT** | Milestones scaled linearly: $16, 22 \times 70/26$. |
| **Warmup** | Epoch 0 only, `LinearLR` (factor 1/1000) | Epoch 0 only, `LinearLR` (factor 1/1000, 138 iters) | **IDENTICAL** | Official warmup logic: $\min(1000, \text{len} - 1) = \min(1000, 139 - 1) = 138$ iters. GroupedBatchSampler yields $\lfloor 559 / 4 \rfloor = 139$ batches (3 images skipped per epoch). |
| **Epochs** | 26 | **70 epochs** | **PROJECT** | Project-wide standard training duration. |
| **Batch Size** | 16 (8 GPUs $\times$ 2 images) | **4** (1 GPU $\times$ 4 images) | **SYSTEM** | Workstation VRAM constraint (RTX 4050 6GB). |
| **Gradient Clipping** | None | None | **IDENTICAL** | Unchanged from official `engine.py`. |
| **Mixed Precision** | Off by default | **AMP enabled** (`torch.cuda.amp.GradScaler`) | **SYSTEM** | Hardware optimization for RTX 4050. |
| **Workers** | 4 | **2** | **SYSTEM** | CPU core allocation for workstation stability. |
| **Augmentation** | Random horizontal flip ($p=0.5$), boxes flipped | Random horizontal flip ($p=0.5$), boxes flipped | **IDENTICAL** | Official detection transforms. |
| **Distributed Training**| `DistributedDataParallel` (8 GPUs) | Single GPU | **SYSTEM** | Single-GPU execution environment. |
| **Random Seed** | None | **Seed 42** | **SYSTEM** | Seeds python, numpy and torch (seed 42). |
| **Early Stopping** | None | None | **IDENTICAL** | Trains full 70 epochs. |
| **Checkpointing** | Saves `model_{epoch}.pth` every epoch | Saves `checkpoint.pth` (resume), `last.pth`, and `best.pth` | **ADDITION** | Prevents disk overflow; tracks validation best. |
| **Metrics Logging** | Console logging | Console logging + `metrics.csv` (`total_loss`, `loss_classification`, `loss_bbox_regression`, COCO APs) | **ADDITION** | Machine-readable per-epoch tracking; fail-fast if meters are missing; appends cleanly on resume. |
| **Evaluation Protocol**| COCO evaluation | COCO evaluation + validation-tuned score sweep (IoU 0.50 greedy matching) | **PROJECT** | Mirrors `yolo/test.py` benchmark protocol. |

---

## 4. Threshold Sweep Behavior & Score Floor Note

During evaluation (`test.py`), a confidence threshold sweep is conducted over $\tau \in [0.01, 0.95]$ in 0.01 increments on the validation set.
* **Important Note:** In RetinaNet, the default post-processing parameter `score_thresh` is set to `0.05` in `GeneralizedRCNNTransform` / `RetinaNet.postprocess_detections`. Consequently, raw predictions below 0.05 are filtered out before output generation.
* **Effect on Sweep:** Candidate sweep thresholds $\tau < 0.05$ (e.g. 0.01, 0.02, 0.03, 0.04) evaluate on the identical set of candidate predictions as $\tau = 0.05$. This behavior is documented and expected.

---

## 5. Known Limitations

1. **Batch Size 4 Normalization Noise:** Training from scratch with standard trainable `nn.BatchNorm2d` at a batch size of 4 exhibits noisy batch normalization statistics compared to batch size 16 or 32. In accordance with the project specification, GroupNorm or SyncBatchNorm was not substituted to adhere strictly to torchvision defaults.
2. **Workstation Hardware:** Executed on a single NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM). Training throughput is approximately 1.5–2 minutes per epoch (~1.75–2.25 hours for 70 epochs).
