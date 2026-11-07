# Canonical Dataset Split Verification & Audit Report

**Date of Execution:** 2026-10-26  
**Environment / Workspace:** `C:\Users\kunwa\Desktop\IEEE RESEARCH`  
**Dataset Reference:** TBX11K Benchmark Dataset  
**Generation Script:** `data_prep/generate_splits.py`  
**Manifest Target Directory:** `data_prep/splits/`  

---

## 1. Executive Summary & Verification Matrix

This report independently verifies the canonical 70:15:15 integer dataset split implementation against the prior dataset audit findings (`data_prep/dataset_audit.py`). All checks have been performed by directly reading physical files and source XML annotations.

| # | Verification Check Item | Target Requirement | Verified Result | Status |
|---|---|---|---|:---:|
| **1** | **Source Image Counts** | 3,800 Healthy, 800 physical TB in `TBX11K/imgs/` | 3,800 Healthy, 800 TB | **PASS** |
| **2** | **Exclusion of `tb1199.png`** | XML has 0 objects; explicit exclusion from eligible pool | Verified 0 objects; excluded from pool (799 remaining) | **PASS** |
| **3** | **Split Counts by Cohort** | Train: 2,660 H / 559 TB; Val: 570 H / 120 TB; Test: 570 H / 120 TB | 2,660/559, 570/120, 570/120 exactly matched | **PASS** |
| **4** | **Boundary Filenames & Positional Order** | Derived from sorted filenames; boundaries match reference | Positional indices 0:2660:3230:3800 and 0:559:679:799 | **PASS** |
| **5** | **Zero Overlap & Zero Loss/Duplication** | Split intersection = $\emptyset$; total union = 4,599 | Disjoint sets verified; 0 duplicates, 0 missing | **PASS** |
| **6** | **XML Annotation & Box Validity** | 100% of 799 TB images have valid XMLs & coordinates | 799/799 valid XMLs; valid coordinates ($x_{max}>x_{min}, y_{max}>y_{min}$) | **PASS** |
| **7** | **Multi-Label Image Categorization** | Active-only: 630, Obsolete-only: 139, Mixed: 30 | 630 Active-only, 139 Obsolete-only, 30 Mixed | **PASS** |
| **8** | **Bounding Box Totals & Class Counts** | Active: 972 boxes, Obsolete: 239 boxes, Total: 1,211 | 972 Active, 239 Obsolete, 1,211 Total | **PASS** |
| **9** | **Mixed Images Accounted For** | All 30 mixed images preserved with explicit split assignment | Train: 23, Val: 5, Test: 2 (Total: 30) | **PASS** |
| **10** | **Dataset Immutability & Test Set Isolation** | Source images/XMLs and official `test/` untouched | Read-only verification; 0 file modifications to raw data | **PASS** |

---

## 2. Source Image Counts & Finalized Eligibility Pool

### 2.1 Source Discovery vs. Audit Findings
- **Healthy Pool (`imgs/health/`):** Exactly 3,800 PNG images discovered (`h0001.png` to `h5000.png` with non-contiguous indexing). All 3,800 images are eligible and included.
- **Physical TB Pool (`imgs/tb/`):** Exactly 800 PNG images discovered (`tb0003.png` to `tb1199.png`).
- **Excluded Image (`tb1199.png`):** Source annotation `annotations/xml/tb1199.xml` was inspected and verified to contain exactly 0 `<object>` elements. In accordance with the study policy:
  - `tb1199.png` is explicitly **excluded** from training, validation, and test manifests.
  - It is **not** converted into a healthy image.
  - It is **not** used as a negative detection sample.
  - The physical file and its XML annotation remain unmodified.
- **Eligible TB Pool:** Exactly 800 - 1 = **799 images**.
- **Excluded Study Cohorts:** `sick/` (2,000 images) and `extra/` (1,800 images) are excluded from this study.
- **Official Competition Test Set (`test/`):** 1,800 images kept strictly isolated and untouched.

---

## 3. Split Counts and Boundary Filenames

The split policy uses a deterministic integer partition over lexicographically sorted eligible filenames. Filename gaps in the original dataset do not cause count drift because slicing is performed directly on the sorted list.

### 3.1 Split Summary Table

| Split | Healthy Images | Eligible TB Images | Total Images | Ratio | Healthy Filename Range | TB Filename Range |
|---|---:|---:|---:|---:|---|---|
| **Train** | 2,660 | 559 | **3,219** | 70.0% | `h0001.png` $\rightarrow$ `h3523.png` | `tb0003.png` $\rightarrow$ `tb0839.png` |
| **Validation** | 570 | 120 | **690** | 15.0% | `h3525.png` $\rightarrow$ `h4266.png` | `tb0841.png` $\rightarrow$ `tb1018.png` |
| **Test** | 570 | 120 | **690** | 15.0% | `h4267.png` $\rightarrow$ `h5000.png` | `tb1019.png` $\rightarrow$ `tb1197.png` |
| **Total** | **3,800** | **799** | **4,599** | **100.0%** | `h0001.png` $\rightarrow$ `h5000.png` | `tb0003.png` $\rightarrow$ `tb1197.png` |

### 3.2 Positional Allocation Verification
- **Healthy Index Boundaries:**
  - Train: `[0:2660]` (2,660 images) — First: `h0001.png`, Last: `h3523.png`
  - Validation: `[2660:3230]` (570 images) — First: `h3525.png`, Last: `h4266.png`
  - Test: `[3230:3800]` (570 images) — First: `h4267.png`, Last: `h5000.png`
- **TB Filtered Pool Index Boundaries:**
  - Train: `[0:559]` (559 images) — First: `tb0003.png`, Last: `tb0839.png`
  - Validation: `[559:679]` (120 images) — First: `tb0841.png`, Last: `tb1018.png`
  - Test: `[679:799]` (120 images) — First: `tb1019.png`, Last: `tb1197.png`

---

## 4. Multi-Label Classification Targets & Category Breakdown

Classification targets are derived strictly from the ground-truth XML annotations without name alteration:
1. `ActiveTuberculosis`
2. `ObsoletePulmonaryTuberculosis`

### 4.1 Image-Level Category Counts by Split

| Split | Healthy `(0, 0)` | Active Only `(1, 0)` | Obsolete Only `(0, 1)` | Mixed `(1, 1)` | Total TB | Total Split |
|---|---:|---:|---:|---:|---:|---:|
| **Train** | 2,660 | 431 | 105 | 23 | 559 | **3,219** |
| **Validation** | 570 | 103 | 12 | 5 | 120 | **690** |
| **Test** | 570 | 96 | 22 | 2 | 120 | **690** |
| **Total** | **3,800** | **630** | **139** | **30** | **799** | **4,599** |

### 4.2 Classification Label Prevalence (Positive Binary Flags)

| Target Label | Train Positives (%) | Val Positives (%) | Test Positives (%) | Overall Positives (%) |
|---|---:|---:|---:|---:|
| **`ActiveTuberculosis`** | 454 (14.10%) | 108 (15.65%) | 98 (14.20%) | 660 (14.35%) |
| **`ObsoletePulmonaryTuberculosis`** | 128 (3.98%) | 17 (2.46%) | 24 (3.48%) | 169 (3.67%) |

---

## 5. Bounding-Box Counts for Downstream Localization

For compatibility with the Faster R-CNN detection experiment, bounding boxes were extracted and validated across all 799 eligible TB images.

| Split | `ActiveTuberculosis` Boxes | `ObsoletePulmonaryTuberculosis` Boxes | Total Bounding Boxes | Mean Boxes / TB Image |
|---|---:|---:|---:|---:|
| **Train** | 657 | 180 | **837** | 1.50 |
| **Validation** | 172 | 23 | **195** | 1.62 |
| **Test** | 143 | 36 | **179** | 1.49 |
| **Total** | **972** | **239** | **1,211** | **1.52** |

---

## 6. Audit of the 30 Mixed Images

All 30 mixed images containing co-occurring `ActiveTuberculosis` and `ObsoletePulmonaryTuberculosis` lesions have been retained and accounted for without omission.

### 6.1 Split Membership Breakdown

- **Train Split (23 images):**  
  `tb0135.png`, `tb0142.png`, `tb0154.png`, `tb0167.png`, `tb0190.png`, `tb0246.png`, `tb0255.png`, `tb0279.png`, `tb0284.png`, `tb0350.png`, `tb0378.png`, `tb0392.png`, `tb0395.png`, `tb0501.png`, `tb0506.png`, `tb0526.png`, `tb0543.png`, `tb0639.png`, `tb0640.png`, `tb0667.png`, `tb0676.png`, `tb0713.png`, `tb0786.png`
- **Validation Split (5 images):**  
  `tb0870.png`, `tb0875.png`, `tb0945.png`, `tb0949.png`, `tb0968.png`
- **Test Split (2 images):**  
  `tb1104.png`, `tb1143.png`
- **Total Mixed Images Accounted For:** **30 / 30 (100.0%)**

---

## 7. Split Manifests Specification

The manifests generated in `data_prep/splits/` conform to the canonical research specification and serve both classification and localization pipelines:

- `train.csv` (3,219 rows + header)
- `val.csv` (690 rows + header)
- `test.csv` (690 rows + header)
- `split_config.json` (Machine-readable configuration and index bounds)

### 7.1 CSV Manifest Schema

| Column Name | Type | Description | Sample Values |
|---|---|---|---|
| `image_path` | `str` | Relative path to image from `TBX11K/` | `imgs/health/h0001.png`, `imgs/tb/tb0003.png` |
| `cohort` | `str` | Source cohort name | `health`, `tb` |
| `split` | `str` | Partition name | `train`, `val`, `test` |
| `ActiveTuberculosis` | `int` | Binary classification target | `0`, `1` |
| `ObsoletePulmonaryTuberculosis` | `int` | Binary classification target | `0`, `1` |
| `xml_path` | `str` | Relative path to XML annotation (TB only) | ``, `annotations/xml/tb0003.xml` |

---

## 8. Dataset Integrity and Immutability Attestation

1. **Original Dataset Untouched:** All 3,800 healthy images, 800 physical TB images, and raw XML annotations remain in their original directories with original permissions and checksums. No image or annotation file was moved, copied, renamed, deleted, or modified.
2. **`tb1199.png` Preserved Unchanged:** `tb1199.png` and `tb1199.xml` remain exactly in place.
3. **Official Competition Test Set Isolated:** `TBX11K/test/` (1,800 images) remains completely separate and untouched.
4. **No Training Executed:** The task strictly generated dataset partition artifacts and verified data integrity.

**Final Certification:** The dataset split implementation is **CANONICAL, FULLY VERIFIED, AND READY FOR MODEL EXPERIMENTS**.
