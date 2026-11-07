# Comprehensive Scientific and Technical Forensic Audit Report: IEEE TB Research Project

**Audit Scope:** Forensic analysis of existing model pipelines (`DensNet-121`, `ResNet-101`, `YOLOv8s`), dataset manifests, training configurations, and evaluation methodologies in preparation for training **EfficientNet-B0** from scratch.

**Audit Directives & Protocol:**
- **Strict Read-Only Enforcement:** No source code, manifests, weights, or datasets modified, moved, or deleted. No training or testing executed.
- **Exact Real Disk Path Nomenclature:** Folders and files are cited with exact disk spelling (e.g., `CLASSIFICATION/DensNet-121/` omitting the first 'e').
- **Verbatim Code Evidence:** All requested algorithmic pipelines and configuration blocks are quoted verbatim with exact line ranges.
- **Strict Absence Protocol:** Any missing or unconfigured feature is explicitly marked **`NOT FOUND`** without speculative inference.

---

## 1. PROJECT MAP

### 1.1 Directory Tree (Depth 3)

Excluded directories per audit protocol: `.venv/`, `.cache/`, `__pycache__/`, `.git/`.
Large data directories (`TBX11K/`, `data_prep/splits/`, `runs/`, `weights/`) summarized with entry counts and representative examples.

```text
IEEE RESEARCH/
├── CLASSIFICATION/
│   ├── DensNet-121/                          [Real disk spelling: 'DensNet-121']
│   │   ├── outputs/
│   │   │   ├── audit/                        [1 file: DenseNet121_Pipeline_Audit_Report.md (30,309 B)]
│   │   │   ├── smoke_test/                   [3 files: best_model.pth, last_checkpoint.pth, training.log]
│   │   │   ├── test/                         [7 files: test_metrics.json, test_metrics.csv, test_report.txt, ...]
│   │   │   └── train/                        [4 files: best_model.pth, last_checkpoint.pth, training.log, training_history.csv]
│   │   ├── DenseNet121_Official_vs_RTX4050_Training_Settings.pdf (39,528 B)
│   │   ├── test.py                           [967 lines, 39,487 B]
│   │   └── train.py                          [1,076 lines, 40,446 B]
│   ├── EfficientNet/
│   │   └── EfficientNet_Official_Hyperparameters.pdf (11,093 B) [ONLY file present]
│   └── ResNet-101/
│       ├── outputs/
│       │   ├── audit/                        [1 file: ResNet101_Pipeline_Audit_Report.md (5,570 B)]
│       │   ├── smoke_test/                   [11 files: audit reports, checkpoints, logs, plots]
│       │   ├── test/                         [7 files: test_metrics.json, test_metrics.csv, test_report.txt, ...]
│       │   └── train/                        [5 files: best_model.pth, last_checkpoint.pth, training.log, training_history.csv, training_plots.png]
│       ├── audit_pipeline.py                 [442 lines, 17,256 B]
│       ├── config.json                       [91 lines, 3,415 B]
│       ├── README.md                         [279 lines, 9,339 B]
│       ├── ResNet101_Official_vs_RTX4050_Training_Settings (1).pdf (38,068 B)
│       ├── test.py                           [997 lines, 41,141 B]
│       └── train.py                          [1,230 lines, 48,182 B]
├── data_prep/
│   ├── splits/                               [5 files: train.csv, val.csv, test.csv, split_config.json, split_audit_report.md]
│   ├── dataset_audit.py                      [554 lines, 25,724 B]
│   └── generate_splits.py                    [658 lines, 27,737 B]
├── runs/                                     [Residual directory created by Ultralytics validation calls: runs/detect/val*, empty]
├── scratch/                                  [Directory for scratch/audit files; contains project_audit.md]
├── scripts/                                  [EMPTY / NOT FOUND: Contains 0 files]
├── TBX11K/                                   [Original benchmark dataset: 13,101 files total across annotations/, code/, imgs/, lists/]
│   ├── annotations/                          [xml/ (1,199 XMLs), json/ (COCO annotations)]
│   ├── code/                                 [make_json_anno.py, pycococreatortools.py]
│   ├── imgs/                                 [health/ (3,800 images), tb/ (800 images), sick/ (3,800 images, excluded)]
│   └── lists/                                [Official splits txt lists]
├── weights/                                  [1 file: yolo26n.pt (5,544,453 B, pretrained)]
├── yolo/
│   ├── dataset/
│   │   ├── images/                           [train/ (3,219 PNGs), val/ (690 PNGs), test/ (690 PNGs)]
│   │   ├── labels/                           [train/ (3,219 TXTs + cache), val/ (690 TXTs + cache), test/ (690 TXTs + cache)]
│   │   └── dataset.yaml                      [13 lines, 251 B]
│   ├── runs/
│   │   ├── baseline_yolov8s_scratch/         [Completed run: weights/, evaluation/, results.csv, plots, metadata]
│   │   └── smoke_test_yolov8s_scratch/       [Smoke test run outputs]
│   ├── test.py                               [1,635 lines, 72,739 B]
│   └── train.py                              [540 lines, 24,077 B]
└── requirements-lock.txt                     [151 lines, 6,422 B, UTF-16LE]
```

### 1.2 Purpose of Each Top-Level Folder

| Folder Path | Architectural Role & Purpose in Project |
|:---|:---|
| `CLASSIFICATION/*` | Houses deep CNN classification model families trained from scratch on TBX11K CXRs. Contains `DensNet-121/`, `ResNet-101/`, and `EfficientNet/`. Dedicated subfolders isolate code, checkpoints, and evaluation artifacts per architecture. |
| `CLASSIFICATION/DensNet-121/` | Full training, evaluation, and smoke testing pipeline for DenseNet-121 from scratch. *(Note disk spelling: `DensNet-121` without 'e')*. |
| `CLASSIFICATION/ResNet-101/` | Full training, evaluation, pre-flight audit, and smoke testing pipeline for ResNet-101 from scratch. |
| `CLASSIFICATION/EfficientNet/` | Dedicated directory for the upcoming EfficientNet-B0 training from scratch. Currently holds only the official hyperparameter reference PDF. |
| `data_prep/` | Ground-truth dataset preparation and auditing. Contains scripts that define the canonical 70:15:15 sequential split policy and output manifests in `data_prep/splits/`. |
| `scripts/` | **EMPTY / NOT FOUND**. Reserved for shared cross-model utility or aggregation scripts. No scripts currently exist inside this directory. |
| `scratch/` | Temporary scratchpad for audit documentation. Holds `project_audit.md` (this report). Prior to this audit, it contained 0 files. |
| `runs/` | Residual empty directory structure (`runs/detect/val*`) created as an artifact of Ultralytics YOLO validation invocations. Does not store primary project outputs. |
| `weights/` | Storage for external model weights. Contains only `yolo26n.pt` (5.5 MB pretrained YOLO nano anchor checkpoint). |
| `yolo/` | Object detection and localization pipeline for YOLOv8s. Contains the symlinked/copied YOLO format dataset (`images/`, `labels/`, `dataset.yaml`), training scripts, evaluation suite, and completed baseline training run folders. |
| `TBX11K/` | Authoritative raw dataset directory containing source CXR images (`health/`, `tb/`, `sick/`), PASCAL VOC XML annotations, COCO JSON annotations, and official challenge splits. |

### 1.3 Inventory of All Training, Testing, Auditing, and Utility Scripts

| Script Path | Domain | Category | Executable Role | Lines | Size |
|:---|:---|:---|:---|:---:|:---:|
| `CLASSIFICATION/DensNet-121/train.py` | Classification | Training | From-scratch multi-label training of DenseNet-121 with StepLR, AMP, and early stopping | 1,076 | 40,446 B |
| `CLASSIFICATION/DensNet-121/test.py` | Classification | Evaluation | Test set evaluation (fixed $\tau=0.5$), confusion matrices, efficiency benchmarking | 967 | 39,487 B |
| `CLASSIFICATION/ResNet-101/train.py` | Classification | Training | From-scratch multi-label training of ResNet-101 with StepLR, AMP, resume support, OOM recovery | 1,230 | 48,182 B |
| `CLASSIFICATION/ResNet-101/test.py` | Classification | Evaluation | Test set evaluation (fixed $\tau=0.5$), confusion matrices, efficiency benchmarking | 997 | 41,141 B |
| `CLASSIFICATION/ResNet-101/audit_pipeline.py` | Classification | Pre-Flight Audit | 20-point pre-flight sanity check for dataset integrity, tensors, shapes, backward pass | 442 | 17,256 B |
| `yolo/train.py` | Detection | Training | From-scratch training of YOLOv8s (`yolov8s.yaml`, `pretrained=False`) on TBX11K | 540 | 24,077 B |
| `yolo/test.py` | Detection | Evaluation | Multi-stage evaluation: val threshold sweep $\tau^*=0.21$, test detection at $\tau^*$, mAP suite, Max-Score Pooling | 1,635 | 72,739 B |
| `data_prep/generate_splits.py` | Data Prep | Manifest Generation | Generates canonical 70:15:15 integer manifests (`train.csv`, `val.csv`, `test.csv`, `split_config.json`) | 658 | 27,737 B |
| `data_prep/dataset_audit.py` | Data Prep | Forensic Audit | Strictly read-only audit of TBX11K source files, VOC XML bounding boxes, class distribution | 554 | 25,724 B |
| `TBX11K/code/make_json_anno.py` | Data Prep | Vendor Tool | Source TBX11K utility script to generate COCO JSON annotations from XML files | 134 | 5,471 B |
| `TBX11K/code/pycococreatortools.py`| Data Prep | Vendor Tool | Source TBX11K helper module supporting polygon masking and COCO dictionary creation | 196 | 7,725 B |

## 2. ENVIRONMENT

### 2.1 Dependencies in `requirements-lock.txt` and Runtime State

The file `requirements-lock.txt` is encoded in UTF-16LE. The table below compares the requested core machine learning libraries against the lockfile and verifies the active python runtime:

| Package / Tool | Version in `requirements-lock.txt` | Active Runtime Status | Notes / Significance |
|:---|:---|:---|:---|
| **Python** | *Unpinned in lockfile header* | `3.11.15` (64-bit AMD64) | Active environment located at `.venv/Scripts/python.exe` |
| **torch** | `torch==2.14.1+cu130` | `2.14.1+cu130` | Built with CUDA 13.0 acceleration support |
| **torchvision** | `torchvision==0.29.1+cu130` | `0.29.1+cu130` | PyTorch vision models and transforms library |
| **timm** | `timm==1.0.30` | **INSTALLED** (`1.0.30`) | PyTorch Image Models library available for model construction |
| **scikit-learn** | `scikit-learn==1.9.1` | `1.9.1` | Used for ROC-AUC, PR-AUC, confusion matrices, metrics |
| **numpy** | `numpy==2.4.6` | `2.4.6` | Numerical computing backend |
| **pandas** | `pandas==3.1.0` | `3.1.0` | CSV manifest and history dataframe manipulation |
| **matplotlib** | `matplotlib==3.11.2` | `3.11.2` | Plotting curves and confusion matrices (`Agg` headless backend) |
| **scipy** | `scipy==1.17.1` | `1.17.1` | Scientific computing support |
| **OpenCV** | `opencv-python==5.0.0.93`<br>`opencv-python-headless==5.0.0.93` | `5.0.0.93` | Computer vision and image processing backend |
| **Pillow** | `pillow==13.0.0` | `13.0.0` | Image loading and EXIF/RGB format conversion |
| **tensorboard** | `tensorboard==2.21.0`<br>`tensorboard-data-server==0.7.2` | `2.21.0` | Scalar and event logging support |
| **wandb** | **NOT FOUND** | **NOT FOUND** | Not present in lockfile and not installed in environment |
| **ultralytics** | `ultralytics==8.4.189`<br>`ultralytics-thop==2.2.3` | `8.4.189` | YOLO object detection framework and FLOPs profiler |

### 2.2 Hardware and Compute Capability

A direct non-allocating hardware audit via PyTorch confirms:
- `torch.cuda.is_available()`: `True`
- **Device Count:** `1`
- **Device Name:** `NVIDIA GeForce RTX 4050 Laptop GPU`
- **Total GPU VRAM:** `5.99658203125 GB` (~`6.0 GB` / 6,438,780,928 bytes)
- **Compute Capability:** `sm_89` (Ada Lovelace architecture)
- **Mixed Precision Support:** Native FP16 AMP via `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.

### 2.3 Operating System and Platform Conventions

- **Operating System:** Microsoft Windows 11 Home / Pro (`Windows-10-10.0.26300-SP0` / OS `nt`).
- **Path Separators:** Windows native backslash (`\`). All project scripts use `pathlib.Path` with `.resolve()` to maintain platform-agnostic path handling and avoid string concatenation pitfalls.
- **DataLoader Worker Handling:** On Windows, PyTorch spawns subprocesses rather than forking. To prevent shared-memory deadlocks and worker crashes, both `train.py` and `test.py` strictly set `num_workers = 2`.
- **Headless Matplotlib Rendering:** To avoid GUI thread lockups on Windows headless execution, scripts invoke `matplotlib.use('Agg')` immediately before importing `pyplot`.

## 3. DATA AND SPLITS (Classification)

### 3.1 Classification Manifest Architecture

All classification experiments read the canonical dataset manifests produced by `data_prep/generate_splits.py` located at:
- **Training Manifest:** `data_prep/splits/train.csv` (3,219 samples)
- **Validation Manifest:** `data_prep/splits/val.csv` (690 samples)
- **Test Manifest:** `data_prep/splits/test.csv` (690 samples)
- **Split Metadata:** `data_prep/splits/split_config.json` (authoritative partition specification)
- **Audit Log:** `data_prep/splits/split_audit_report.md` (detailed provenance document)

**Model Manifest Parity:**
Both `CLASSIFICATION/DensNet-121/train.py` and `CLASSIFICATION/ResNet-101/train.py` (and their respective `test.py` scripts) read the **EXACT SAME MANIFEST FILES**. DenseNet and ResNet share 100% split parity.

**Manifest Schema:**
Each CSV file contains exactly 6 columns:
```csv
image_path,cohort,split,ActiveTuberculosis,ObsoletePulmonaryTuberculosis,xml_path
```
Representative rows:
```csv
imgs/health/h0001.png,health,train,0,0,
imgs/tb/tb0003.png,tb,train,1,0,annotations/xml/tb0003.xml
```

### 3.2 Label Scheme and Encoding

- **Paradigm:** 2-output multi-label binary classification evaluated with `nn.BCEWithLogitsLoss()`.
- **Target Vector:** `[ActiveTuberculosis, ObsoletePulmonaryTuberculosis]` where each element is $\in \{0, 1\}$.
- **Target Column Order:** `['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']` (Index 0: Active TB, Index 1: Obsolete TB).
- **Healthy Images:** Encoded as `[0, 0]` (True negative for both targets).
- **Active Only:** Encoded as `[1, 0]`.
- **Obsolete Only:** Encoded as `[0, 1]`.
- **Mixed ('Both'):** Encoded as `[1, 1]` (Image contains both active and obsolete manifestations).
- **Sick Non-TB Cohort:** The 3,800 images in `TBX11K/imgs/sick/` are **COMPLETELY EXCLUDED** from the experimental pool. They do not appear in any manifest.

**Derived 4-Category Mapping:**
For secondary clinical classification and multi-class confusion matrices, the 2 binary outputs are mapped to 4 mutually exclusive categories:
```python
def derive_category(active: int, obsolete: int) -> int:
    if active == 0 and obsolete == 0: return 0  # Healthy
    elif active == 1 and obsolete == 0: return 1  # Active Only
    elif active == 0 and obsolete == 1: return 2  # Obsolete Only
    else: return 3                                # Both
```

### 3.3 Authoritative Image and Bounding Box Counts

The distribution computed directly from `data_prep/splits/*.csv` and `data_prep/splits/split_config.json` is summarized below:

| Split Partition | Healthy `(0, 0)` | Active Only `(1, 0)` | Obsolete Only `(0, 1)` | Both `(1, 1)` | Total Images | Active Positives (Marginal) | Obsolete Positives (Marginal) | Total Active Boxes | Total Obsolete Boxes | Total Boxes |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Train (70%)** | 2,660 | 431 | 105 | 23 | **3,219** | 454 (14.10%) | 128 (3.98%) | 657 | 180 | 837 |
| **Validation (15%)** | 570 | 103 | 12 | 5 | **690** | 108 (15.65%) | 17 (2.46%) | 172 | 23 | 195 |
| **Test (15%)** | 570 | 96 | 22 | 2 | **690** | 98 (14.20%) | 24 (3.48%) | 143 | 36 | 179 |
| **Total (100%)** | **3,800** | **630** | **139** | **30** | **4,599** | **660** (14.35%) | **169** (3.67%) | **972** | **239** | **1,211** |

### 3.4 Data Leakage Prevention and Splitting Methodology

- **Splitting Policy:** Deterministic integer slicing over lexicographically sorted filenames.
- **Seed Configuration:** `"None (pure deterministic sorted positional slicing)"` as recorded in `split_config.json`.
- **Patient Overlap:** TBX11K does not provide patient identifier metadata. Positional sequential slicing on sorted filenames ensures that contiguous blocks of image sequences cannot cross partitions.
- **Explicit Boundaries:**
  - **Health Cohort (3,800 total):**
    - Train (2,660 images): `h0001.png` to `h3523.png`
    - Validation (570 images): `h3525.png` to `h4266.png`
    - Test (570 images): `h4267.png` to `h5000.png`
  - **TB Cohort (799 eligible images):**
    - Train (559 images): `tb0003.png` to `tb0839.png`
    - Validation (120 images): `tb0841.png` to `tb1018.png`
    - Test (120 images): `tb1019.png` to `tb1197.png`
- **Excluded Image `tb1199.png`:** The XML file `annotations/xml/tb1199.xml` contains zero bounding-box annotations. It was excluded from `data_prep/splits/` to maintain 100% sample identity between classification and YOLO localization.
- **Disjoint Verification:** Mathematical intersection between Train, Validation, and Test manifests is strictly empty (0 overlapping filenames).

### 3.5 Inventory of `data_prep/` Scripts

1. **`data_prep/generate_splits.py`:**
   - **Role:** Generates canonical manifests `train.csv`, `val.csv`, `test.csv`, `split_config.json`, and `split_audit_report.md`.
   - **Execution Command:** `python data_prep/generate_splits.py` (No CLI arguments required; auto-resolves repository root).

2. **`data_prep/dataset_audit.py`:**
   - **Role:** Strictly read-only audit utility verifying image format integrity, XML parsing, coordinate sanity, and cohort distributions.
   - **Execution Command:** `python data_prep/dataset_audit.py [--dataset_dir ./TBX11K]`.

## 4. TRAINING CODE (DenseNet-121 and ResNet-101)

### 4.1 DenseNet-121 Implementation (`CLASSIFICATION/DensNet-121/train.py`)

- **Script Path:** `CLASSIFICATION/DensNet-121/train.py` (Real folder name: `DensNet-121` without 'e')
- **Primary Architecture:** DenseNet-121 instantiated from scratch (`weights=None`)
- **Default Output Folder:** `CLASSIFICATION/DensNet-121/outputs/train/`

#### 4.1.1 CLI Arguments and Defaults

| Argument Flag | Type | Default Value | Description |
|:---|:---|:---|:---|
| `--data-dir` | `str` | `None` (auto-detected `TBX11K`) | Path to dataset root |
| `--splits-dir` | `str` | `None` (auto-detected `data_prep/splits`) | Path to manifest split CSVs |
| `--output-dir` | `str` | `None` (`outputs/train/`) | Checkpoint & log output directory |
| `--epochs` | `int` | `70` (`NUM_EPOCHS`) | Maximum training epochs |
| `--batch-size` | `int` | `4` (`BATCH_SIZE`) | Physical batch size per iteration |
| `--accum-steps` | `int` | `1` (`GRADIENT_ACCUMULATION_STEPS`) | Gradient accumulation steps |
| `--lr` | `float` | `0.01` (`LEARNING_RATE`) | Initial learning rate |
| `--momentum` | `float` | `0.9` (`MOMENTUM`) | SGD momentum |
| `--weight-decay` | `float` | `0.0001` (`WEIGHT_DECAY`) | Weight decay (L2 penalty) |
| `--lr-step-size` | `int` | `30` (`LR_STEP_SIZE`) | StepLR drop period (epochs) |
| `--lr-gamma` | `float` | `0.1` (`LR_GAMMA`) | Multiplicative LR decay factor |
| `--patience` | `int` | `20` (`EARLY_STOPPING_PATIENCE`) | Early stopping patience (epochs) |
| `--threshold` | `float` | `0.5` | Binary decision threshold |
| `--num-workers` | `int` | `2` (`NUM_WORKERS`) | DataLoader worker processes |
| `--seed` | `int` | `42` (`SEED`) | Reproducibility seed |
| `--no-cuda` | `flag` | `False` | Force CPU execution |
| `--no-amp` | `flag` | `False` | Disable FP16 mixed precision |
| `--smoke-test` | `flag` | `False` | Execute 2-epoch sanity run |
| `--resume` | - | **`NOT FOUND`** | *DenseNet-121 does not support checkpoint resume* |

#### 4.1.2 Verbatim Code: Dataset Class (`TBX11KMultiLabelDataset`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 138–243*

```python
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

```

#### 4.1.3 Verbatim Code: Preprocessing & Normalization Transforms (`get_transforms`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 248–260*

```python
def get_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Standard preprocessing transform pipeline:
    - Resize to 512 x 512
    - Convert to float tensor [0.0, 1.0]
    - Normalize using standard ImageNet mean and std
    Consistent across train, validation, and test.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
```

#### 4.1.4 Verbatim Code: Model Construction (`build_densenet121`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 266–280*

```python
def build_densenet121(num_classes: int = 2) -> nn.Module:
    """
    Instantiates DenseNet-121 from scratch (weights=None).
    Replaces the classification head with nn.Linear(in_features, num_classes).
    Confirms all backbone and classifier parameters are trainable.
    """
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    # Ensure all parameters require gradients (from-scratch training)
    for name, param in model.named_parameters():
        param.requires_grad = True

    return model
```

#### 4.1.5 Verbatim Code: DataLoader Settings, Loss, Optimizer, and LR Scheduler
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 904–939*

```python
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
    logging.info("Instantiating DenseNet-121 from scratch (weights=None)...")
    model = build_densenet121(num_classes=2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=NESTEROV,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (device.type == "cuda") else None

```

#### 4.1.6 Verbatim Code: Training Loop with AMP and Gradient Accumulation (`train_one_epoch`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 387–447*

```python
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
```

#### 4.1.7 Verbatim Code: Validation Loop (`validate_one_epoch`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 450–488*

```python
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
```

#### 4.1.8 Verbatim Code: Metric Evaluation Functions
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 286–310 (`compute_binary_metrics`) and Lines 331–381 (`compute_epoch_metrics`)*

```python
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
```

#### 4.1.9 Verbatim Code: Checkpoint Saving Function (`save_checkpoint`)
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 494–525*

```python
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
            "mean": NORM_MEAN,
            "std": NORM_STD,
        },
        "target_columns": TARGET_COLUMNS,
        "category_names": CATEGORY_NAMES,
    }
    torch.save(state, path)
```

#### 4.1.10 Verbatim Code: Best-Model Criterion and Early Stopping Logic
*File: `CLASSIFICATION/DensNet-121/train.py`, Lines 982–1015 and Lines 1063–1068*

```python
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

        # Check early stopping
        if epochs_no_improve >= args.patience:
            logging.info(
                f"Early stopping triggered at epoch {epoch}: "
                f"Validation macro-F1 has not improved for {args.patience} consecutive epochs."
            )
            break
```

#### 4.1.11 Resume Logic and Training Curve Plots

- **Resume Logic:** **`NOT FOUND`**. DenseNet-121 does not provide resume functionality.
- **Training Curve Plotting:** **`NOT FOUND`**. DenseNet-121 does not generate a training plot PNG during training.

### 4.2 ResNet-101 Implementation (`CLASSIFICATION/ResNet-101/train.py`)

- **Script Path:** `CLASSIFICATION/ResNet-101/train.py`
- **Primary Architecture:** ResNet-101 instantiated from scratch (`weights=None`)
- **Default Output Folder:** `CLASSIFICATION/ResNet-101/outputs/train/`

#### 4.2.1 CLI Arguments and Defaults

| Argument Flag | Type | Default Value | Description |
|:---|:---|:---|:---|
| `--data-dir` | `str` | `None` (auto-detected `TBX11K`) | Path to dataset root |
| `--splits-dir` | `str` | `None` (auto-detected `data_prep/splits`) | Path to manifest split CSVs |
| `--output-dir` | `str` | `None` (`outputs/train/`) | Checkpoint & log output directory |
| `--epochs` | `int` | `70` (`NUM_EPOCHS`) | Maximum training epochs |
| `--batch-size` | `int` | `4` (`BATCH_SIZE`) | Physical batch size (fallback: 2) |
| `--accum-steps` | `int` | `1` (`GRADIENT_ACCUMULATION_STEPS`) | Gradient accumulation steps |
| `--lr` | `float` | `0.01` (`LEARNING_RATE`) | Initial learning rate |
| `--momentum` | `float` | `0.9` (`MOMENTUM`) | SGD momentum |
| `--weight-decay` | `float` | `0.0001` (`WEIGHT_DECAY`) | Weight decay (L2 penalty) |
| `--nesterov` | `flag` | `False` (`NESTEROV`) | Enable Nesterov momentum |
| `--lr-step-size` | `int` | `30` (`LR_STEP_SIZE`) | StepLR drop period (epochs) |
| `--lr-gamma` | `float` | `0.1` (`LR_GAMMA`) | Multiplicative LR decay factor |
| `--patience` | `int` | `20` (`EARLY_STOPPING_PATIENCE`) | Early stopping patience (epochs) |
| `--threshold` | `float` | `0.5` | Binary decision threshold |
| `--num-workers` | `int` | `2` (`NUM_WORKERS`) | DataLoader worker processes |
| `--seed` | `int` | `42` (`SEED`) | Reproducibility seed |
| `--no-cuda` | `flag` | `False` | Force CPU execution |
| `--no-amp` | `flag` | `False` | Disable FP16 mixed precision |
| `--smoke-test` | `flag` | `False` | Execute 2-epoch sanity run |
| `--resume` | `str` | `None` | Path to checkpoint file from which to resume training |

#### 4.2.2 Verbatim Code: Dataset Class (`TBX11KMultiLabelDataset`)
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 139–243*

```python
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
```

#### 4.2.3 Verbatim Code: Preprocessing Transforms (`get_transforms`) — *No ImageNet Normalization*
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 249–260*

```python
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

```

#### 4.2.4 Verbatim Code: Model Construction (`build_resnet101`)
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 265–279*

```python
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
```

#### 4.2.5 Verbatim Code: DataLoader Settings, Loss, Optimizer, and LR Scheduler
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 985–1028*

```python
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

```

#### 4.2.6 Verbatim Code: Checkpoint Resume Logic
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 1040–1068*

```python
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

```

#### 4.2.7 Verbatim Code: Dynamic OOM Recovery Handling
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 1084–1113*

```python
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

```

#### 4.2.8 Verbatim Code: Checkpoint Saving Function (`save_checkpoint`)
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 495–528*

```python
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
```

#### 4.2.9 Verbatim Code: Training Curve Plotting (`plot_training_curves`)
*File: `CLASSIFICATION/ResNet-101/train.py`, Lines 534–568*

```python
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
```

### 4.3 Detailed Technical Divergences: DenseNet-121 vs. ResNet-101

| Feature / Hyperparameter | DenseNet-121 Implementation | ResNet-101 Implementation | Architectural Significance for EfficientNet-B0 |
|:---|:---|:---|:---|
| **Pixel Normalization** | `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | Strictly **NO ImageNet normalization** (scaled to `[0.0, 1.0]`) | **CRITICAL CHOICE**: EfficientNet implementations typically expect standard ImageNet normalization. Must be explicitly specified. |
| **Nesterov Momentum** | `nesterov=True` hardcoded in `torch.optim.SGD` | `nesterov=False` by default; controllable via `--nesterov` CLI flag | Nesterov acceleration alters SGD momentum dynamics; ResNet makes it configurable, DenseNet hardcodes it True. |
| **Resume Capability** | **`NOT FOUND`** | Full `--resume` flag with optimizer, scheduler, scaler, and history restore | EfficientNet must implement ResNet-101's robust resume architecture. |
| **OOM Recovery** | **`NOT FOUND`** (Process crashes on OOM) | Dynamic `try...except torch.cuda.OutOfMemoryError` fallback to $B=2, \text{accum}=2$ | Protects long training runs from aborting on transient VRAM spikes. |
| **Training Curve Plots** | **`NOT FOUND`** | Automatically saves `training_plots.png` at end of training | ResNet plots Loss, Macro-F1, Exact Match, and Learning Rate curves automatically. |
| **Checkpoint Metadata** | Includes `mean` and `std` under `preprocessing` dict | Includes `normalization: 'none'` and `tensor_range: '[0.0, 1.0]'` | Ensures evaluation scripts can inspect preprocessing metadata. |
| **Folder Spelling** | `CLASSIFICATION/DensNet-121/` (missing 'e') | `CLASSIFICATION/ResNet-101/` | Folder name on disk has a typo; path resolution must accommodate it. |

## 5. EVALUATION CODE

### 5.1 DenseNet-121 Evaluation Pipeline (`CLASSIFICATION/DensNet-121/test.py`)

- **Script Path:** `CLASSIFICATION/DensNet-121/test.py`
- **Target Split Evaluated:** Canonical Test Set (`data_prep/splits/test.csv`, 690 images: 570 Healthy + 120 TB)
- **Batch Size:** `4` (DataLoader `batch_size=4`, `num_workers=2`, `shuffle=False`)
- **Test-Time Augmentation (TTA):** **`NOT FOUND`** (Strictly deterministic single-crop evaluation)
- **Threshold Protocol:** Fixed $\tau = 0.50$ across both labels (CLI default `--threshold 0.5`). **NO test-set threshold tuning** is performed.

#### 5.1.1 Verbatim Code: Checkpoint Loading & Architecture Initialization
*File: `CLASSIFICATION/DensNet-121/test.py`, Lines 758–795*

```python
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
    logging.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_densenet121(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info(f"Loaded DenseNet-121 model state from epoch {checkpoint.get('epoch', 'N/A')}.")
```

#### 5.1.2 Verbatim Code: Test Transforms (`get_test_transforms`)
*File: `CLASSIFICATION/DensNet-121/test.py`, Lines 202–207*

```python
def get_test_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
```

#### 5.1.3 Verbatim Code: Metric Computation Suite (`compute_test_metrics`)
*File: `CLASSIFICATION/DensNet-121/test.py`, Lines 241–360*

```python
def compute_test_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes complete classification metrics across the multi-label test set:
    - Subset exact match accuracy and Hamming metrics
    - Per-label precision, sensitivity/recall, specificity, F1, ROC-AUC, PR-AUC
    - Macro-F1 and Weighted-F1
    - 4-category confusion matrix and per-category metrics
    """
    y_pred = (y_probs >= threshold).astype(int)
    n_samples = len(y_true)

    # 1. Exact Match Accuracy: all labels must match for a sample
    exact_matches = (y_true[:, 0] == y_pred[:, 0]) & (y_true[:, 1] == y_pred[:, 1])
    exact_match_acc = float(np.mean(exact_matches))

    # 2. Hamming Loss & Hamming Accuracy
    hamming_loss = float(np.mean(y_true != y_pred))
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
    macro_f1 = float(np.mean(label_f1s))

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

    cat_accuracy = float(np.mean(true_cats == pred_cats))

    return {
        "classification_summary": {
            "exact_match_accuracy": exact_match_acc,
            "hamming_loss": hamming_loss,
            "hamming_accuracy": hamming_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "category_accuracy": cat_accuracy,
            "total_test_samples": n_samples,
            "probability_threshold": threshold,
        },
        "per_label_metrics": per_label_metrics,
        "per_category_metrics": per_category_metrics,
        "four_category_confusion_matrix": cat_cm,
        "derived_category_names": CATEGORY_NAMES,
    }
```

#### 5.1.4 Verbatim Code: Computational Efficiency Benchmarking (`benchmark_efficiency`)
*File: `CLASSIFICATION/DensNet-121/test.py`, Lines 366–430*

```python
def benchmark_efficiency(
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_warmup: int = 10,
    num_runs: int = 100,
    eval_batch_size: int = 4,
) -> Dict[str, Any]:
    """
    Measures computational efficiency:
    - Total parameters and trainable parameters
    - FLOPs and MACs via 'thop' (ultralytics_thop)
    - Single-image (B=1) latency: mean, std, min, max in ms
    - FPS at batch-size-1
    - Batched throughput (images/sec) at eval_batch_size
    - Hardware synchronization ensured
    """
    model.eval()

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs and MACs
    flops_info: Dict[str, Any] = {"tool": "none", "macs": None, "flops": None}
    try:
        import thop
        dummy_for_flops = torch.randn(1, 3, *input_size).to(device)
        macs, params = thop.profile(model, inputs=(dummy_for_flops,), verbose=False)
        flops_info = {
            "tool": "thop (ultralytics_thop)",
            "macs": float(macs),
            "flops": float(2 * macs),  # Standard convention: 1 MAC = 2 FLOPs
            "convention": "1 MAC = 2 floating point operations",
        }
    except Exception as e:
        flops_info["error"] = str(e)
        logging.warning(f"FLOPs counting via thop not available or failed: {e}")

    # Latency benchmarking (Batch size = 1)
    single_inp = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_inp)
            else:
                _ = model(single_inp)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark single-image latency
    timings = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    _ = model(single_inp)
```

#### 5.1.5 Verbatim Code: Output Files Written and Destination Directory
*File: `CLASSIFICATION/DensNet-121/test.py`, Lines 888–960*

```python
    # 3. Save per-image predictions CSV
    per_image_csv_path = output_dir / "per_image_predictions.csv"
    with open(per_image_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_records[0].keys()))
        writer.writeheader()
        writer.writerows(image_records)
    logging.info(f"Saved per-image predictions to {per_image_csv_path}")

    # 4. Save JSON metrics
    metrics_json_path = output_dir / "test_metrics.json"
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

    # 5. Save tabular summary CSV
    metrics_csv_path = output_dir / "test_metrics.csv"
    summary_rows = []
    for k, v in test_metrics["classification_summary"].items():
        summary_rows.append({"metric_group": "overall_classification", "metric_name": k, "value": v})
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

    # 6. Generate confusion matrix plots
    active_cm_path = output_dir / "confusion_matrix_active_tb.png"
    obsolete_cm_path = output_dir / "confusion_matrix_obsolete_tb.png"
    cat_cm_path = output_dir / "confusion_matrix_4category.png"

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
    logging.info(f"Saved confusion matrix figures to {output_dir}")

    # 7. Generate comprehensive text report
    report_text = generate_text_report(
        test_metrics=test_metrics,
        efficiency_metrics=efficiency_metrics,
        checkpoint_info=checkpoint,
    )
    report_path = output_dir / "test_report.txt"
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Saved text evaluation report to {report_path}")
```

### 5.2 ResNet-101 Evaluation Pipeline (`CLASSIFICATION/ResNet-101/test.py`)

- **Script Path:** `CLASSIFICATION/ResNet-101/test.py`
- **Target Split Evaluated:** Canonical Test Set (`data_prep/splits/test.csv`, 690 images: 570 Healthy + 120 TB)
- **Batch Size:** `4` (DataLoader `batch_size=4`, `num_workers=2`, `shuffle=False`)
- **Test-Time Augmentation (TTA):** **`NOT FOUND`**
- **Threshold Protocol:** Fixed $\tau = 0.50$ across both labels (CLI default `--threshold 0.5`). **NO test-set threshold tuning** is performed.

#### 5.2.1 Verbatim Code: Checkpoint Loading & Architecture Initialization
*File: `CLASSIFICATION/ResNet-101/test.py`, Lines 785–824*

```python
    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
    else:
        # Search candidate locations
        script_dir = Path(__file__).resolve().parent
        cand_ckpts = [
            output_dir.parent / "train" / "best_model.pth",
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
            ckpt_path = cand_ckpts[1]  # Default fallback for logging/error reporting

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at: {ckpt_path}. "
            f"Please run training first or specify --checkpoint <path>."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    logging.info(f"Loading checkpoint from: {ckpt_path}")
    logging.info(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_resnet101(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logging.info(f"Loaded ResNet-101 model state from epoch {checkpoint.get('epoch', 'N/A')}.")

```

#### 5.2.2 Verbatim Code: Test Transforms (`get_test_transforms`) — *No ImageNet Normalization*
*File: `CLASSIFICATION/ResNet-101/test.py`, Lines 208–218*

```python
def get_test_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """
    Mandatory ResNet-101 test preprocessing pipeline:
    1. Bilinear resize to 512 x 512.
    2. Convert to float32 tensor via ToTensor() scaled to [0.0, 1.0].
    Strictly NO ImageNet normalization and NO data augmentation.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
```

#### 5.2.3 Verbatim Code: Metric Computation Suite (`compute_test_metrics`)
*File: `CLASSIFICATION/ResNet-101/test.py`, Lines 253–374*

```python
def compute_test_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes complete classification metrics across the multi-label test set:
    - Subset exact match accuracy and Hamming metrics
    - Per-label precision, sensitivity/recall, specificity, F1, ROC-AUC, PR-AUC
    - Macro-F1 and Weighted-F1
    - 4-category confusion matrix and per-category metrics
    """
    y_pred = (y_probs >= threshold).astype(int)
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
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # ROC-AUC and PR-AUC from raw continuous probabilities
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
            "accuracy": float(acc),
            "precision": float(prec),
            "sensitivity_recall": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support_positives": support,
            "support_negatives": int(np.sum(t == 0)),
        }

    # Macro-F1 (unweighted arithmetic mean of Active and Obsolete F1)
    macro_f1 = float(np.mean(label_f1s)) if label_f1s else 0.0

    # Weighted-F1 (weighted by positive support)
    total_support = sum(label_supports)
    if total_support > 0:
        weighted_f1 = float(sum(f * s for f, s in zip(label_f1s, label_supports)) / total_support)
    else:
        weighted_f1 = 0.0

    # 4. Derived 4-Category Metrics (One-vs-Rest)
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

    return {
        "classification_summary": {
            "exact_match_accuracy": exact_match_acc,
            "hamming_loss": hamming_loss,
            "hamming_accuracy": hamming_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "category_accuracy": cat_accuracy,
            "total_test_samples": n_samples,
            "probability_threshold": threshold,
        },
        "per_label_metrics": per_label_metrics,
        "per_category_metrics": per_category_metrics,
        "four_category_confusion_matrix": cat_cm,
        "derived_category_names": CATEGORY_NAMES,
    }
```

#### 5.2.4 Verbatim Code: Output Files Written and Destination Directory
*File: `CLASSIFICATION/ResNet-101/test.py`, Lines 915–989*

```python
        }

    # 3. Save per-image predictions CSV
    per_image_csv_path = output_dir / "per_image_predictions.csv"
    with open(per_image_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_records[0].keys()))
        writer.writeheader()
        writer.writerows(image_records)
    logging.info(f"Saved per-image predictions to {per_image_csv_path}")

    # 4. Save JSON metrics
    metrics_json_path = output_dir / "test_metrics.json"
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

    # 5. Save tabular summary CSV
    metrics_csv_path = output_dir / "test_metrics.csv"
    summary_rows = []
    for k, v in test_metrics["classification_summary"].items():
        summary_rows.append({"metric_group": "overall_classification", "metric_name": k, "value": v})
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

    # 6. Generate confusion matrix plots
    active_cm_path = output_dir / "confusion_matrix_active_tb.png"
    obsolete_cm_path = output_dir / "confusion_matrix_obsolete_tb.png"
    cat_cm_path = output_dir / "confusion_matrix_4category.png"

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
    logging.info(f"Saved confusion matrix figures to {output_dir}")

    # 7. Generate comprehensive text report
    report_text = generate_text_report(
        test_metrics=test_metrics,
        efficiency_metrics=efficiency_metrics,
        checkpoint_info=checkpoint,
    )
    report_path = output_dir / "test_report.txt"
    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_text)
    logging.info(f"Saved text evaluation report to {report_path}")
```

### 5.3 YOLOv8s Evaluation Pipeline (`yolo/test.py`)

- **Script Path:** `yolo/test.py`
- **Checkpoint Evaluated:** `yolo/runs/baseline_yolov8s_scratch/weights/best.pt`
- **Checkpoint Loading:** Instantiated via Ultralytics API: `eval_model = YOLO(str(weights_path))`
- **Two-Split Evaluation Design:**
  1. **Validation Split (`yolo/dataset/images/val/`, 690 images):** Swept over confidence thresholds $[0.01, 0.95]$ with step $0.01$ at $\text{IoU}=0.50$ to locate optimal threshold $\tau^* = 0.21$ maximizing Validation Macro-F1.
  2. **Test Split (`yolo/dataset/images/test/`, 690 images):** Evaluated at frozen $\tau^* = 0.21$ for discrete detection metrics, continuous mAP suite via `eval_model.val()`, and derived image-level multi-label classification via Max-Score Pooling.
- **Test-Time Augmentation (TTA):** **`NOT FOUND`**

#### 5.3.1 Verbatim Code: Model Complexity Profiling & Checkpoint Inspection (`profile_model_complexity`)
*File: `yolo/test.py`, Lines 676–734*

```python
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
```

#### 5.3.2 Verbatim Code: Ground Truth Loading and Bounding Box IoU Matching
*File: `yolo/test.py`, Lines 123–143 (`compute_box_iou`) and Lines 177–238 (`match_detections_for_image`)*

```python
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
```

#### 5.3.3 Verbatim Code: Validation Confidence Threshold Sweep (`run_threshold_optimization_sweep`)
*File: `yolo/test.py`, Lines 303–393*

```python
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
```

#### 5.3.4 Verbatim Code: Discrete Test Detection Evaluation at Optimal Threshold (`evaluate_test_detection`)
*File: `yolo/test.py`, Lines 399–485*

```python
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
```

#### 5.3.5 Verbatim Code: Standard Ultralytics Validation for Continuous mAP Metrics
*File: `yolo/test.py`, Lines 1387–1416*

```python
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
```

#### 5.3.6 Verbatim Code: Derived Image-Level Multi-Label Classification via Max-Score Pooling
*File: `yolo/test.py`, Lines 491–615*

```python
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
```

#### 5.3.7 Verbatim Code: Artifact Generation and Output Writing in `yolo/test.py`
*File: `yolo/test.py`, Lines 1583–1625*

```python
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

```

#### 5.3.8 Cross-Model Imports

- **Cross-Model Utilities in `scripts/`:** **`NOT FOUND`**
- All three evaluation scripts (`CLASSIFICATION/DensNet-121/test.py`, `CLASSIFICATION/ResNet-101/test.py`, `yolo/test.py`) are self-contained executable scripts and do not import functions or classes from external project scripts.

## 6. SCRIPTS AND RUNS

### 6.1 Status of `scripts/` Directory

- **Inventory:** The directory `scripts/` is **COMPLETELY EMPTY / NOT FOUND** (0 files, 0 subdirectories).
- **Cross-Model Aggregation Scripts:** **`NOT FOUND`**
- **Cross-Model Comparison / Plotting Scripts:** **`NOT FOUND`**
- **Results-File Schema Expectation:** Because no aggregation scripts currently exist, there is no shared programmatic schema enforcing unified cross-model reporting. Each model architecture currently produces its own isolated outputs under `outputs/test/` or `yolo/runs/.../evaluation/`.

### 6.2 Output and Run Directories: Naming Conventions and Completed Runs

#### 6.2.1 Classification Run Naming Conventions
Classification pipelines do not write to the top-level `runs/` folder. Instead, they isolate run outputs directly inside their architectural subfolder:
- `CLASSIFICATION/<ModelName>/outputs/train/` (Checkpoints, logs, training history)
- `CLASSIFICATION/<ModelName>/outputs/test/` (Evaluation metrics, confusion matrices, text reports)
- `CLASSIFICATION/<ModelName>/outputs/smoke_test/` (2-epoch sanity runs)
- `CLASSIFICATION/<ModelName>/outputs/audit/` (Pre-flight audit markdown reports)

**File List of One Completed Classification Run (`CLASSIFICATION/DensNet-121/outputs/train/`):**
- `best_model.pth` (56,366,767 B)
- `last_checkpoint.pth` (56,372,252 B)
- `training.log` (19,014 B)
- `training_history.csv` (19,876 B)

**Corresponding Test Evaluation Output (`CLASSIFICATION/DensNet-121/outputs/test/`):**
- `confusion_matrix_4category.png` (185,556 B)
- `confusion_matrix_active_tb.png` (133,086 B)
- `confusion_matrix_obsolete_tb.png` (134,813 B)
- `per_image_predictions.csv` (42,652 B)
- `test_metrics.csv` (1,496 B)
- `test_metrics.json` (3,410 B)
- `test_report.txt` (6,219 B)

#### 6.2.2 YOLO Run Naming Conventions and Completed Run
YOLO training outputs are directed to `yolo/runs/<experiment_name>/` (e.g., `baseline_yolov8s_scratch/`).

**File List of One Completed YOLO Run (`yolo/runs/baseline_yolov8s_scratch/`):**
- `args.yaml` (1,903 B)
- `BoxF1_curve.png` (112,583 B)
- `BoxP_curve.png` (119,707 B)
- `BoxPR_curve.png` (116,958 B)
- `BoxR_curve.png` (117,921 B)
- `confusion_matrix.png` (121,249 B)
- `confusion_matrix_normalized.png` (129,067 B)
- `experiment_metadata.json` (4,616 B)
- `labels.jpg` (120,846 B)
- `results.csv` (8,940 B)
- `results.png` (310,027 B)
- `train_batch0.jpg` (113,319 B)
- `train_batch1.jpg` (98,796 B)
- `train_batch2.jpg` (94,393 B)
- `train_batch48300.jpg` (91,982 B)
- `train_batch48301.jpg` (89,745 B)
- `train_batch48302.jpg` (91,988 B)
- `val_batch0_labels.jpg` (236,658 B)
- `val_batch0_pred.jpg` (236,658 B)
- `val_batch1_labels.jpg` (230,858 B)
- `val_batch1_pred.jpg` (230,858 B)
- `val_batch2_labels.jpg` (234,027 B)
- `val_batch2_pred.jpg` (234,027 B)
- `weights/best.pt` (22,572,403 B)
- `weights/last.pt` (22,572,403 B)
- `evaluation/` (contains 14 evaluation artifacts, including `evaluation_summary.json`, `evaluation_report.txt`, `per_class_metrics.csv`, `image_level_metrics.csv`, `per_image_predictions.csv`, `threshold_analysis.csv`, and 6 plot figures)

### 6.3 Weights Directory Inventory

The repository root `weights/` directory contains:
- **`weights/yolo26n.pt`** | Size: `5,544,453 bytes` (~`5.5 MB`) | **Pretrained** anchor weight binary (external download).

*Note: All trained model checkpoints are strictly stored within their respective run output directories (`CLASSIFICATION/DensNet-121/outputs/train/`, `CLASSIFICATION/ResNet-101/outputs/train/`, and `yolo/runs/baseline_yolov8s_scratch/weights/`).*

### 6.4 Scratch Directory Status

- **Status:** The `scratch/` folder was completely empty (0 files) prior to this audit. It now contains exactly one file: `scratch/project_audit.md` (this audit document).

## 7. CONVENTIONS AND PITFALLS

### 7.1 Coding Conventions to Follow for EfficientNet-B0 Implementation

1. **Argparse Style:**
   - All hyperparameter defaults must be declared as module-level constants at the top of `train.py` (e.g., `NUM_EPOCHS = 70`, `BATCH_SIZE = 4`, `LEARNING_RATE = 0.01`).
   - Command-line arguments should provide explicit help strings, explicit data types, and reference the top-level constants as defaults.
   - Paths (`--data-dir`, `--splits-dir`, `--output-dir`) must default to `None` and trigger dynamic workspace path resolution via a `resolve_workspace_paths()` helper.
   - CLI flags must include `--no-cuda`, `--no-amp`, `--smoke-test`, and `--resume`.

2. **Logging Conventions:**
   - Use Python's built-in `logging` module configured to stream concurrently to `sys.stdout` and an output file (`outputs/train/training.log`).
   - Authoritative format string: `%(asctime)s [%(levelname)s] %(message)s`.
   - Per-epoch summary line must output: Epoch number, Learning Rate, Train Loss, Val Loss, Val Macro-F1, Val Exact Match, Active F1, Obsolete F1, Peak VRAM (MB), Epoch Duration (s), and `[BEST]` tag.

3. **Config and Metadata Persistence:**
   - Embed the full runtime argument configuration dictionary (`config: vars(args)`) directly inside the checkpoint dictionary saved to `best_model.pth` and `last_checkpoint.pth`.
   - Save preprocessing metadata in the checkpoint: `image_size`, `channels`, `format`, `mean`, `std`, `normalization`.
   - Also persist `config.json` at the root of the model's directory or in `outputs/train/` for external inspection without loading PyTorch checkpoint binaries.

4. **Seed and Reproducibility:**
   - Provide a `set_seed(seed: int = 42)` helper invoked before dataset instantiation or model creation.
   - Explicitly seed `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, set `torch.backends.cudnn.deterministic = True`, and `torch.backends.cudnn.benchmark = False`.

5. **Type Hints and Documentation:**
   - Full Python 3.10+ typing annotations on all function signatures (`Tuple`, `Dict[str, Any]`, `Optional[Path]`, `List[str]`).
   - Complete module docstrings and Google-style function docstrings explaining input arguments and return contracts.

6. **DataLoader and Multiprocessing on Windows:**
   - Set `num_workers = 2` by default. Do not use higher worker counts on Windows to prevent IPC shared-memory heap fragmentation.
   - Enable `pin_memory = (device.type == 'cuda')`.

### 7.2 Forensic Audit of Bugs, Inconsistencies, and Risks in Existing Codebase

*(Audit-only findings; reported with evidence, no fixes applied)*

1. **CRITICAL DIVERGENCE: Preprocessing Normalization Discrepancy:**
   - **Evidence:** `CLASSIFICATION/DensNet-121/train.py` (lines 256–260) applies standard ImageNet normalization (`Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`). In contrast, `CLASSIFICATION/ResNet-101/train.py` (lines 256–259) explicitly applies **NO ImageNet normalization**, scaling raw tensors to `[0.0, 1.0]` via `ToTensor()`. 
   - **Risk for EfficientNet-B0:** Training EfficientNet from scratch requires deciding whether to follow DenseNet's ImageNet normalization or ResNet's `[0, 1]` unnormalized scaling. EfficientNet-B0 architectures typically train more stably with ImageNet normalization or standard z-score normalization. This divergence must be harmonized.

2. **HIGH RISK: Hardcoded Nesterov Momentum in DenseNet-121:**
   - **Evidence:** DenseNet `train.py` hardcodes `nesterov=True` in optimizer creation (`torch.optim.SGD(..., nesterov=NESTEROV)`) and lacks a `--nesterov` CLI argument. ResNet `train.py` defaults `--nesterov` to `False` but makes it configurable via CLI.
   - **Risk:** DenseNet and ResNet use different SGD update rules, confounding pure architectural comparison.

3. **HIGH RISK: Missing Resume Support in DenseNet-121:**
   - **Evidence:** DenseNet `train.py` completely lacks a `--resume` argument and has no checkpoint reloading logic. If an RTX 4050 Laptop GPU enters thermal throttling, reboots, or encounters a power interruption, the run cannot be resumed and must restart from epoch 1.
   - **Required Action for EfficientNet-B0:** Must adopt ResNet-101's full checkpoint resume logic.

4. **MEDIUM RISK: OOM Exception Handling Divergence:**
   - **Evidence:** ResNet `train.py` wraps `train_one_epoch` in `try...except torch.cuda.OutOfMemoryError` and falls back dynamically to physical batch size 2 with gradient accumulation steps 2. DenseNet `train.py` has no OOM recovery.
   - **Risk:** EfficientNet models with MBConv blocks can experience peak memory spikes during backpropagation. Dynamic OOM recovery should be included.

5. **MEDIUM RISK: Training Curves Plot Generation:**
   - **Evidence:** ResNet-101 includes `plot_training_curves()` generating `training_plots.png` automatically. DenseNet-121 only generates CSV history without graphical curves.

6. **LOW RISK: Directory Naming Typo (`DensNet-121`):**
   - **Evidence:** The directory on disk is literally named `CLASSIFICATION/DensNet-121/` (missing the second 'e'). Scripts expecting `DenseNet-121` fail with `FileNotFoundError`.

7. **LOW RISK: Residual Root Directories Created by Ultralytics:**
   - **Evidence:** Running `yolo/test.py` leaves empty directories in `runs/detect/val*` and `runs/detect/yolo/runs/...` in the repository root due to default Ultralytics val caching.

### 7.3 Status of `CLASSIFICATION/EfficientNet/`

A forensic inspection of `CLASSIFICATION/EfficientNet/` confirms:
- **Folder Exists:** `True`
- **Files Present:** Exactly 1 file: `EfficientNet_Official_Hyperparameters.pdf` (11,093 bytes).
- **Source Code / Scripts:** **`NOT FOUND`** (No `train.py`, `test.py`, `config.json`, or weights exist in this folder yet).

## 8. SUMMARY TABLE FOR EFFICIENTNET-B0 IMPLEMENTATION

The comprehensive master comparison table below summarizes the operational configuration of DenseNet-121, ResNet-101, and YOLOv8s. Every row where DenseNet-121 and ResNet-101 **DIFFER** is explicitly marked with **`[DIFFERENCE]`**.

| Architectural / Pipeline Feature | DenseNet-121 | ResNet-101 | YOLOv8s Baseline (`yolo/test.py`) | Divergence Status & Guidance for EfficientNet-B0 |
|:---|:---|:---|:---|:---|
| **Model Paradigm** | Multi-label 2-head binary classification | Multi-label 2-head binary classification | Object detection & localization (2 classes) | **Identical**; EfficientNet-B0 must use 2-head binary classification |
| **Folder Name on Disk** | `CLASSIFICATION/DensNet-121/` | `CLASSIFICATION/ResNet-101/` | `yolo/` | `[DIFFERENCE]` (DenseNet folder has typo; EfficientNet folder is `CLASSIFICATION/EfficientNet/`) |
| **Target Labels** | `['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']` | `['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']` | `['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']` | **Identical** across all models |
| **Split Manifest Files** | `data_prep/splits/train.csv`<br>`val.csv`<br>`test.csv` | `data_prep/splits/train.csv`<br>`val.csv`<br>`test.csv` | `yolo/dataset/images/{train,val,test}` & `dataset.yaml` | **Identical** split manifests (3,219 / 690 / 690 images) |
| **Input Resolution** | $512 \times 512$ (Bilinear resize) | $512 \times 512$ (Bilinear resize) | $512 \times 512$ (Letterbox resize) | **Identical** $512 \times 512$ resolution |
| **Pixel Normalization** | `mean=[0.485, 0.456, 0.406]`<br>`std=[0.229, 0.224, 0.225]` | Strictly **NO Normalization** (`[0.0, 1.0]` tensor) | Normalization handled internally by Ultralytics (`[0.0, 1.0]`) | `[DIFFERENCE]` **CRITICAL**: Decide whether EfficientNet uses ImageNet norm or unnormalized scaling |
| **Data Augmentation** | None (Deterministic single-crop) | None (Deterministic single-crop) | Training augmentations (HSV, flip, mosaic) | **Identical** for classification (No augmentation) |
| **Physical Batch Size** | $B = 4$ | $B = 4$ (fallback to 2 on OOM) | $B = 4$ | **Identical** nominal batch size $B=4$ |
| **Gradient Accumulation** | $1$ step (Effective $B=4$) | $1$ step (Effective $B=4$, fallback to 2) | Ultralytics nominal accumulate | **Identical** effective batch size 4 |
| **Maximum Epochs** | 70 epochs | 70 epochs | 70 epochs | **Identical** max epochs (70) |
| **Early Stopping Metric** | Validation Macro-F1 | Validation Macro-F1 | Validation loss / mAP | **Identical** criterion for classification |
| **Early Stopping Patience** | 20 epochs | 20 epochs | Not configured | **Identical** patience (20 epochs) |
| **Optimizer** | SGD (`momentum=0.9`, `weight_decay=1e-4`) | SGD (`momentum=0.9`, `weight_decay=1e-4`) | AdamW (Ultralytics auto) | **Identical** base SGD settings |
| **Nesterov Momentum** | `nesterov=True` (hardcoded) | `nesterov=False` (default, CLI switchable) | N/A | `[DIFFERENCE]`: ResNet has `--nesterov` flag, DenseNet hardcodes True |
| **Learning Rate Schedule** | `StepLR(step_size=30, gamma=0.1)` | `StepLR(step_size=30, gamma=0.1)` | Linear / Cosine decay | **Identical** StepLR schedule |
| **Loss Function** | `nn.BCEWithLogitsLoss()` | `nn.BCEWithLogitsLoss()` | CIoU + DFL + BCE | **Identical** loss for classification |
| **Automatic Mixed Precision (AMP)** | Enabled (`torch.amp.autocast('cuda')`, `GradScaler`) | Enabled (`torch.amp.autocast('cuda')`, `GradScaler`) | Enabled (native Ultralytics AMP) | **Identical** AMP FP16 precision |
| **DataLoader Settings** | `num_workers=2`, `pin_memory=True` | `num_workers=2`, `pin_memory=True` | `workers=2` | **Identical** worker count (2) for Windows stability |
| **Resume Logic** | **`NOT FOUND`** | Supported (`--resume <ckpt>`) | Supported (`resume=True`) | `[DIFFERENCE]`: EfficientNet must implement ResNet's resume logic |
| **Dynamic OOM Recovery** | **`NOT FOUND`** | Implemented (auto-fallback to $B=2, \text{accum}=2$) | N/A | `[DIFFERENCE]`: EfficientNet should adopt ResNet's OOM recovery |
| **Training Curve Plots** | **`NOT FOUND`** | Saves `training_plots.png` automatically | Saves `results.png` automatically | `[DIFFERENCE]`: ResNet generates plots, DenseNet does not |
| **Best Checkpoint Filename** | `outputs/train/best_model.pth` | `outputs/train/best_model.pth` | `weights/best.pt` | **Identical** checkpoint filename and path layout |
| **Checkpoint Dictionary Keys** | `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `val_macro_f1`, `val_metrics`, `config`, `preprocessing`, `target_columns`, `category_names` | Identical keys as DenseNet | Native Ultralytics checkpoint dictionary | **Identical** state dictionary keys |
| **Evaluation Target Split** | Canonical Test Split (`test.csv`, 690 CXRs) | Canonical Test Split (`test.csv`, 690 CXRs) | Validation split (sweep) + Test split | **Identical** test split (690 images) |
| **Threshold Protocol** | Fixed $\tau = 0.50$ (No test-set tuning) | Fixed $\tau = 0.50$ (No test-set tuning) | Tuned on validation split ($\tau^*=0.21$), frozen for test | **Identical** fixed $\tau=0.50$ for classification |
| **Evaluation Metric Suite** | Exact Match Acc, Hamming Loss, Hamming Acc, Per-Label (Precision, Sensitivity/Recall, Specificity, F1, ROC-AUC, PR-AUC), Macro-F1, Weighted-F1, 4-Category Confusion Matrix & Metrics | Identical to DenseNet plus Per-Label Accuracy key | Discrete Det metrics at $\tau^*$, continuous mAP50, mAP75, mAP50-95, image-level Max-Score Pooling metrics | `[DIFFERENCE]`: ResNet adds per-label `accuracy` in JSON |
| **Evaluation Outputs Written** | 7 files (`test_metrics.json`, `test_metrics.csv`, `per_image_predictions.csv`, `test_report.txt`, 3 CM plots) | 7 files (same file naming and structure as DenseNet) | 14 files (`evaluation_summary.json`, `evaluation_report.txt`, 4 CSVs, 6 plot figures) | **Identical** artifact bundle between DenseNet and ResNet |
| **Run Output Folder Naming** | `CLASSIFICATION/DensNet-121/outputs/train/`<br>`CLASSIFICATION/DensNet-121/outputs/test/` | `CLASSIFICATION/ResNet-101/outputs/train/`<br>`CLASSIFICATION/ResNet-101/outputs/test/` | `yolo/runs/baseline_yolov8s_scratch/` | `[DIFFERENCE]`: Model-specific folder naming under `CLASSIFICATION/` |

---
### End of Audit Report
