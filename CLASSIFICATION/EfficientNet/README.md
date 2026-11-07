# EfficientNet-B0 Multi-Label Chest X-Ray Classification Pipeline

## 1. Overview & Scope

This directory contains the authoritative, reproducible training, validation, and evaluation pipeline for **EfficientNet-B0** on the **TBX11K** chest X-ray benchmark dataset. It implements the exact official TensorFlow TPU EfficientNet reference recipe (`models/official/efficientnet`), adapting only the strictly necessary dataset-related and hardware-related boundaries for a single NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).

### Core Task
- **Dataset:** Canonical TBX11K Benchmark (4,599 CXR images: 3,219 train, 690 val, 690 test).
- **Task:** 2-output multi-label binary classification:
  - Output Index 0: `ActiveTuberculosis`
  - Output Index 1: `ObsoletePulmonaryTuberculosis`
- **Derived Categories:** Healthy `[0, 0]`, Active-only `[1, 0]`, Obsolete-only `[0, 1]`, Both `[1, 1]`.
- **Training Paradigm:** Trained strictly **from scratch** (`weights=None`). Zero transfer learning or pretrained weights.
- **Governing Specification:** `EfficientNet_Official_Hyperparameters.pdf` & Project Directives.

---

## 2. Order of Authority & Governing Rules

1. **Primary Authority (Model Optimization):** The official hyperparameters recorded in `EfficientNet_Official_Hyperparameters.pdf` (from `tensorflow/tpu/models/official/efficientnet`).
2. **Permitted Deviations (Dataset & Hardware):** Multi-label BCE on 2 outputs, $512 \times 512$ input resolution, 70-epoch budget, batch size 4 with linear LR scaling, PyTorch 2.14 / timm 1.0.30 AMP FP16.
3. **Template & Standards (Code Structure):** Inherited from `CLASSIFICATION/ResNet-101/` for artifact schemas, logging, resume protocols, and evaluation suite parity.

---

## 3. Official Recipe vs. Implemented Adaptation

| Feature / Hyperparameter | Official TPU Reference (Tan & Le, 2019) | Implemented PyTorch Recipe (RTX 4050 6 GB) | Deviation Classification | Rationale & Technical Notes |
|:---|:---|:---|:---|:---|
| **Architecture** | EfficientNet-B0 (width 1.0, depth 1.0) | `torchvision.models.efficientnet_b0` | **MATCHES** | Expected total parameters: exactly 4,010,110 (2 classes). |
| **Pretrained Weights** | None (Trained from scratch) | `weights=None` (Random initialization) | **MATCHES** | Fair comparison against DenseNet-121 and ResNet-101 baselines. |
| **Classification Head** | 1000-class Linear | 2-output Linear (`fc = nn.Linear(1280, 2)`) | **DATASET** | Maps features to independent Bernoulli logits. |
| **Loss Function** | Softmax Cross-Entropy | `nn.BCEWithLogitsLoss()` on 2 raw logits | **DATASET** | Evaluates multi-label co-occurrence without mutual exclusivity. |
| **Label Smoothing** | 0.1 on Softmax Cross-Entropy | None | **DATASET** | Label smoothing is mathematically undefined for binary cross-entropy. |
| **Input Resolution** | $224 \times 224$ | $512 \times 512$ | **DATASET** | Higher resolution preserves subtle cavitary and apical X-ray lesions. |
| **Optimizer** | TF RMSProp (`alpha=0.9, mom=0.9, eps=1e-3`) | `timm.optim.RMSpropTF(alpha=0.9, momentum=0.9, eps=1e-3)` | **MATCHES** | `torch.optim.RMSprop` is forbidden due to diverging epsilon placement. |
| **Base Learning Rate** | 0.016 (defined at batch size 256) | 0.016 base rate | **MATCHES** | Official linear scaling anchor point. |
| **Linear LR Scaling** | `scaled_lr = 0.016 * (batch / 256)` | `0.016 * (4 / 256) = 0.00025` at batch 4 | **SYSTEM** | Linear scaling rule applied to batch size 4. |
| **Warmup Duration** | 5 epochs linear from 0 to scaled_lr | 5 epochs = 4,025 optimizer steps | **MATCHES** | Evaluated per optimizer step: `scaled_lr * step / warmup_steps`. |
| **LR Decay Schedule** | Exponential, $\times 0.97$ every 2.4 epochs | Staircase: $0.00025 \times 0.97^{\lfloor \text{step} / 1932 \rfloor}$ | **MATCHES** | Decay count originates at step 0 per official TF implementation. |
| **Weight Decay** | $1\text{e-}5$ on all non-BatchNorm variables | $1\text{e-}5$ on Conv/Linear weights & biases | **MATCHES** | Strict parameter group separation; asserts no BN in decay group. |
| **BatchNorm WD** | 0.0 (strictly no decay) | 0.0 on all BatchNorm weights/biases | **MATCHES** | Prevents variance shrinkage in normalization layers. |
| **BatchNorm Stats** | Momentum 0.99, Epsilon $1\text{e-}3$ (TF) | Momentum 0.01, Epsilon $1\text{e-}3$ (PyTorch) | **MATCHES** | PyTorch convention $0.01$ is mathematically identical to TF $0.99$. |
| **Dropout Rate** | 0.2 before final projection | `dropout = 0.2` | **MATCHES** | Official B0 rate. |
| **Stochastic Depth** | Survival probability 0.8 ($p=0.2$) | `stochastic_depth_prob = 0.2` | **MATCHES** | Linearly ramps drop-path probability across blocks from 0.0 to ~0.2. |
| **Weight EMA** | Decay 0.9999, early ramp $\min(0.9999, \frac{1+t}{10+t})$ | Custom `ModelEMA` class | **MATCHES** | Shadow model averages all floats (including BN running mean/var). |
| **Training Crop** | Inception-style crop (0.08–1.0, 3/4–4/3) | Custom `OfficialInceptionCrop` transform | **MATCHES** | Matches TF `sample_distorted_bounding_box` with coverage $\ge 0.1$. |
| **Eval Preprocessing** | 0.875 center crop then resize | Bilinear Resize(512, 512), no center crop | **DATASET** | Retains full lung field; matches DenseNet and ResNet protocols. |
| **Batch Size** | 2048 (across TPU pod) | 4 (physical) | **SYSTEM** | 6 GB VRAM hardware boundary (with fallback to 2 on OOM). |
| **Precision** | float32 / bfloat16 | CUDA FP16 Autocast + GradScaler | **SYSTEM** | Mixed precision execution fitting in ~940 MB peak VRAM. |
| **Training Budget** | ~350 epochs (218,949 steps) | 70 epochs (56,350 optimizer steps) | **DATASET** | Fixed study budget; identical to DenseNet and ResNet. |
| **Early Stopping** | None | None | **MATCHES** | Executes all 70 epochs; no patience flag. |
| **Checkpoint Best** | Evaluated on EMA weights | Best EMA validation Macro-F1 at $\tau=0.5$ | **MATCHES** | Evaluates both EMA and raw; EMA model saved to `best_model.pth`. |

---

## 4. Algorithmic Implementation Details

### 4.1 Custom Transform: `OfficialInceptionCrop`
Torchvision's `RandomResizedCrop` samples aspect ratio log-uniformly, lacks a minimum bounding box coverage constraint, and uses a different fallback mechanism. `OfficialInceptionCrop` directly replicates TensorFlow's `sample_distorted_bounding_box`:
1. Up to 10 attempts to sample:
   - Area fraction $A \sim U(0.08, 1.0)$.
   - Coverage constraint: $A \ge 0.1$.
   - Aspect ratio $R \sim U(3/4, 4/3)$ (uniform, not log-uniform).
2. Derives $w = \sqrt{A \cdot \text{orig\_area} \cdot R}$ and $h = \sqrt{A \cdot \text{orig\_area} / R}$.
3. Verifies crop fits inside image bounds ($0 < w \le \text{orig\_w}$, $0 < h \le \text{orig\_h}$).
4. If all 10 attempts fail, falls back to the full image.
5. If the resulting crop equals the original image dimensions, applies official center-crop fallback: centered crop of $\min(h, w) \cdot \frac{512}{512 + 32} \approx 0.875 \cdot \min(h, w)$.

### 4.2 Learning Rate Trajectory
With 3,219 training images and physical batch size 4, one epoch requires $\lceil 3219 / 4 \rceil = 805$ optimizer steps.
- **Warmup:** $5 \times 805 = 4,025$ steps.
- **Decay Interval:** $2.4 \times 805 = 1,932$ steps.
- **Step Formula:**
  $$\text{LR}(t) = \begin{cases} 0.00025 \times \frac{t}{4025} & t < 4025 \\ 0.00025 \times 0.97^{\lfloor t / 1932 \rfloor} & t \ge 4025 \end{cases}$$
- **Key Reference Checkpoints:**
  - Step 0: $\text{LR} = 0.0$
  - Step 2,012 (Mid-warmup): $\text{LR} = 1.25 \times 10^{-4}$
  - Step 4,025 (End of warmup): $\text{LR} = 0.00025 \times 0.97^2 = 2.3523 \times 10^{-4}$
  - Step 56,349 (Final step, Epoch 70): $\text{LR} = 0.00025 \times 0.97^{29} = 1.0335 \times 10^{-4}$

### 4.3 Exponential Moving Average (ModelEMA)
- Decay factor ramps up early:
  $$\beta(t) = \min\left(0.9999, \frac{1 + t}{10 + t}\right)$$
  At $t=0$, $\beta = 0.1$. At $t \ge 99,990$, $\beta = 0.9999$.
- Updates all floating-point tensors in `state_dict()` (weights, biases, and BatchNorm running mean/variance).
- Copies integer buffers (`num_batches_tracked`).
- **GradScaler Synchronization:** EMA update is applied strictly after a successful optimizer step; if `GradScaler` skips an optimizer step due to inf/nan gradients, the EMA update is skipped.

---

## 5. Declared Limitations (UNVERIFIED List)

1. **RMSpropTF Kernel Implementation:** `timm.optim.RMSpropTF` replicates TensorFlow's epsilon placement and gradient momentum formulation, but execution is performed by PyTorch CUDA kernels.
2. **Torchvision Initialization & Stochastic Depth:** Default PyTorch initialization and linear stochastic-depth scaling are used; not verified line-by-line against TF graph builder ops.
3. **Official Snapshot Provenance:** Hyperparameters are based on the canonical snapshot of `models/official/efficientnet`; live repository updates may contain minor differences.
4. **Batch-4 Learning Rate:** Derived via the official linear scaling rule ($0.016 \times 4 / 256$), not via empirical grid search at batch size 4.
5. **Small-Batch Normalization:** BatchNorm statistics are noisier at batch size 4 than at batch 2048; damped using momentum 0.01.
6. **Aggressive Cropping on CXRs:** Random cropping down to 0.08 area fraction can cut peripheral lesions; an ablation switch (`--augment none`) is provided.
7. **TensorFlow Crop Internals:** Uniform aspect ratio and coverage thresholds reflect documented TF op specifications.
8. **Label Smoothing Omission:** Omitted because label smoothing is mathematically undefined for multi-label binary cross-entropy.
9. **Epoch Budget & Selection Metric:** The 70-epoch budget and model selection on validation Macro-F1 are project-level design choices for standardized comparison with DenseNet-121 and ResNet-101.

---

## 6. Execution Instructions

### 6.1 Two-Epoch Smoke Test
Verify pipeline integrity, data loading, forward/backward passes, EMA tracking, and checkpoint generation without running full training:
```powershell
.\.venv\Scripts\python.exe CLASSIFICATION/EfficientNet/train.py --smoke-test
```

### 6.2 Full 70-Epoch Training Run
Execute from-scratch training with official Inception crop augmentation, RMSpropTF, Step LR, and dual ModelEMA tracking:
```powershell
.\.venv\Scripts\python.exe CLASSIFICATION/EfficientNet/train.py
```

### 6.3 Resuming Training
Resume an interrupted training run seamlessly; restores `global_step`, learning rate, EMA shadow weights, optimizer state, and appends to history:
```powershell
.\.venv\Scripts\python.exe CLASSIFICATION/EfficientNet/train.py --resume CLASSIFICATION/EfficientNet/outputs/train/last_checkpoint.pth
```

### 6.4 Model Evaluation (test.py)
Evaluate the trained EMA checkpoint on the canonical 690-image test set:
```powershell
.\.venv\Scripts\python.exe CLASSIFICATION/EfficientNet/test.py
```
To evaluate the raw un-averaged weights for ablation comparison:
```powershell
.\.venv\Scripts\python.exe CLASSIFICATION/EfficientNet/test.py --eval-raw
```

---

## 7. Generated Artifacts & Directory Structure

```text
CLASSIFICATION/EfficientNet/
├── outputs/
│   ├── train/
│   │   ├── best_model.pth           # Checkpoint with highest validation Macro-F1 (contains EMA weights)
│   │   ├── last_checkpoint.pth      # Most recent checkpoint saved each epoch
│   │   ├── training.log             # Formatted execution and epoch progression log
│   │   ├── training_history.csv     # Complete per-epoch metrics (EMA and Raw)
│   │   ├── training_plots.png       # 6-panel trajectory visualization
│   │   ├── lr_schedule_preview.csv  # Full step-by-step learning rate preview
│   │   └── run_manifest.json        # Reproducibility manifest with SHA-256 dataset hashes
│   ├── test/
│   │   ├── test_metrics.json        # Machine-readable evaluation and complexity metrics
│   │   ├── test_metrics.csv         # Tabular summary of test metrics
│   │   ├── per_image_predictions.csv# Predictions, probabilities, and ground truth for all 690 test CXRs
│   │   ├── test_report.txt          # Comprehensive human-readable evaluation report
│   │   ├── confusion_matrix_active_tb.png
│   │   ├── confusion_matrix_obsolete_tb.png
│   │   └── confusion_matrix_4category.png
│   └── smoke_test/                  # Sanity run artifacts
├── config.json                      # Machine-readable architecture & hyperparameter specification
├── README.md                        # This authoritative documentation
├── test.py                          # Comprehensive test set evaluation script
└── train.py                         # From-scratch training pipeline
```

