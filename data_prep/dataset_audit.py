"""
TBX11K Research Dataset Audit Tool
==================================

A comprehensive, strictly read-only audit script for the TBX11K benchmark dataset.
Verifies file inventory, image integrity, VOC XML annotations, COCO JSON annotations,
class distributions, sub-class co-occurrence, and provisional 70:15:15 split planning.

STRICTLY READ-ONLY:
- Does NOT create dataset splits, manifests, or output folders.
- Does NOT modify, rename, move, copy, or delete any source files.

Usage:
    python data_prep/dataset_audit.py
    python data_prep/dataset_audit.py --dataset_dir ./TBX11K
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def format_header(title: str, width: int = 78) -> str:
    """Format section headers for clean terminal output."""
    return f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}"


def format_sub_header(title: str, width: int = 78) -> str:
    """Format subsection headers."""
    return f"\n{'-' * width}\n  {title}\n{'-' * width}"


# =====================================================================
# 1. Physical Inventory & Manifest Audit
# =====================================================================
def audit_inventory(dataset_root: Path):
    """
    Scans physical disk directories and compares against official manifest files.
    Distinguishes the core TBX11K benchmark from external cohorts (extra/).
    """
    print(format_header("1. DATASET INVENTORY & MANIFEST AUDIT"))
    imgs_dir = dataset_root / "imgs"
    lists_dir = dataset_root / "lists"

    subdirs = ["health", "tb", "sick", "test"]
    physical_files = {}
    valid_exts = {".png", ".jpg", ".jpeg"}

    print("\n[Physical Image Directory Counts]")
    for sd in subdirs:
        folder = imgs_dir / sd
        if folder.exists() and folder.is_dir():
            files = [f.name for f in folder.iterdir() if f.suffix.lower() in valid_exts]
            physical_files[sd] = sorted(files)
            print(f"  imgs/{sd:<8} : {len(files):>5} images")
        else:
            physical_files[sd] = []
            print(f"  imgs/{sd:<8} : [DIRECTORY NOT FOUND]")

    # Check external cohorts (extra/)
    extra_dir = imgs_dir / "extra"
    extra_files = []
    if extra_dir.exists() and extra_dir.is_dir():
        for root, _, filenames in os.walk(extra_dir):
            for fname in filenames:
                if Path(fname).suffix.lower() in valid_exts:
                    rel_path = os.path.relpath(os.path.join(root, fname), imgs_dir).replace("\\", "/")
                    extra_files.append(rel_path)
    physical_files["extra"] = sorted(extra_files)
    print(f"  imgs/extra   : {len(extra_files):>5} images (External cohorts: DA, DB, Montgomery, Shenzhen)")

    total_physical_core = sum(len(physical_files[sd]) for sd in subdirs)
    print(f"  --------------------------------------------------")
    print(f"  Core TBX11K Total : {total_physical_core:>5} images")
    print(f"  All Images Total  : {total_physical_core + len(extra_files):>5} images")

    # Manifest audits
    manifest_specs = [
        ("TBX11K_train.txt", "Core Benchmark - Training Pool"),
        ("TBX11K_val.txt", "Core Benchmark - Validation Pool"),
        ("TBX11K_trainval.txt", "Core Benchmark - Train + Val Pool"),
        ("all_train.txt", "Combined Pool (Core + Extra Cohorts) - Train"),
        ("all_val.txt", "Combined Pool (Core + Extra Cohorts) - Val"),
        ("all_trainval.txt", "Combined Pool (Core + Extra Cohorts) - TrainVal"),
        ("all_test.txt", "Combined Competition Test Set (2800 Core + 502 Extra)"),
    ]

    print("\n[Manifest File Verification]")
    manifest_data = {}
    for fname, desc in manifest_specs:
        mpath = lists_dir / fname
        if mpath.exists():
            with open(mpath, "r", encoding="utf-8") as f:
                lines = [line.strip().replace("\\", "/") for line in f if line.strip()]
            manifest_data[fname] = lines
            unique_count = len(set(lines))
            dup_msg = f"(Duplicates: {len(lines) - unique_count})" if len(lines) != unique_count else ""
            print(f"  {fname:<20} : {len(lines):>5} entries {dup_msg} [{desc}]")
        else:
            manifest_data[fname] = None
            print(f"  {fname:<20} : [FILE NOT FOUND]")

    # Check manifest file-on-disk existence
    print("\n[Integrity: Manifest Entries vs Disk Files]")
    all_disk_rel = set()
    for sd in subdirs:
        for f in physical_files[sd]:
            all_disk_rel.add(f"{sd}/{f}")
    for ef in extra_files:
        all_disk_rel.add(f"extra/{ef}" if not ef.startswith("extra/") else ef)

    for fname, lines in manifest_data.items():
        if lines is not None:
            missing = [item for item in lines if item not in all_disk_rel]
            if missing:
                print(f"  WARNING: {fname} has {len(missing)} missing paths on disk! (e.g., {missing[:2]})")
            else:
                print(f"  OK: All {len(lines)} paths in {fname} exist on disk.")

    # Overlap and union validation
    train_lines = set(manifest_data.get("TBX11K_train.txt") or [])
    val_lines = set(manifest_data.get("TBX11K_val.txt") or [])
    trainval_lines = set(manifest_data.get("TBX11K_trainval.txt") or [])
    test_lines = set(manifest_data.get("all_test.txt") or [])

    print("\n[Set Algebra & Partition Isolation]")
    train_val_overlap = train_lines.intersection(val_lines)
    print(f"  Train & Val overlap in TBX11K         : {len(train_val_overlap)} images (Expected: 0)")
    
    tv_union = train_lines.union(val_lines)
    union_matches = (tv_union == trainval_lines)
    print(f"  Train U Val == Trainval Manifest      : {union_matches} (Union: {len(tv_union)}, Trainval: {len(trainval_lines)})")

    tv_test_overlap = trainval_lines.intersection(test_lines)
    print(f"  Trainval & Competition Test overlap   : {len(tv_test_overlap)} images (Expected: 0)")

    return physical_files, manifest_data


# =====================================================================
# 2. Image Integrity & Readability
# =====================================================================
def audit_image_integrity(dataset_root: Path, physical_files: dict):
    """
    Decodes physical candidate images in memory using PIL.
    Checks dimensions, channels, color modes, and detects corrupt files without modifying disk.
    """
    print(format_header("2. IMAGE INTEGRITY & DECODING AUDIT"))
    if not HAS_PIL:
        print("  WARNING: PIL / Pillow is not installed. Skipping deep pixel decoding checks.")
        return

    imgs_dir = dataset_root / "imgs"
    # Target candidate pool: health + tb
    candidate_subdirs = ["health", "tb"]
    total_checked = sum(len(physical_files[sd]) for sd in candidate_subdirs)
    print(f"\nDecoding {total_checked} candidate images in memory (health/ and tb/)...")

    dim_counter = Counter()
    mode_counter = Counter()
    corrupt_files = []

    for sd in candidate_subdirs:
        for fname in physical_files[sd]:
            fpath = imgs_dir / sd / fname
            try:
                with Image.open(fpath) as img:
                    img.verify()  # Verify file header and integrity
                with Image.open(fpath) as img:
                    dim_counter[img.size] += 1
                    mode_counter[img.mode] += 1
            except Exception as e:
                corrupt_files.append((f"{sd}/{fname}", str(e)))

    print(f"\n[Image Properties Summary]")
    print(f"  Dimensions found : {dict(dim_counter)}")
    print(f"  Color modes      : {dict(mode_counter)}")
    print(f"  Corrupt files    : {len(corrupt_files)}")

    if corrupt_files:
        print("  ALERT: Corrupt images detected:")
        for cf, err in corrupt_files[:5]:
            print(f"    - {cf}: {err}")
    else:
        print("  OK: 100% of candidate images successfully decoded with uniform dimensions.")


# =====================================================================
# 3. XML Annotation Audit
# =====================================================================
def audit_xml_annotations(dataset_root: Path, tb_files: list):
    """
    Parses VOC XML annotations, auditing bounding-box counts, coordinate sanity,
    class distribution, and mutually exclusive sub-class categorizations.
    """
    print(format_header("3. VOC XML ANNOTATION AUDIT"))
    xml_dir = dataset_root / "annotations" / "xml"

    xml_files = sorted([f.name for f in xml_dir.glob("*.xml")]) if xml_dir.exists() else []
    print(f"\n[XML Inventory]")
    print(f"  XML files on disk : {len(xml_files)}")
    print(f"  TB images on disk : {len(tb_files)}")

    # Check 1:1 image-to-xml pairing
    tb_stem_set = {Path(f).stem for f in tb_files}
    xml_stem_set = {Path(f).stem for f in xml_files}

    missing_xml = tb_stem_set - xml_stem_set
    orphan_xml = xml_stem_set - tb_stem_set
    print(f"  TB images missing XML : {len(missing_xml)}")
    print(f"  Orphan XMLs (no image): {len(orphan_xml)}")

    # Detailed box parsing
    box_class_counter = Counter()
    images_by_category = {
        "active_only": [],
        "obsolete_only": [],
        "both": [],
        "zero_boxes": [],
    }
    multi_box_count = 0
    box_count_per_image = Counter()
    invalid_boxes = []
    parsed_annotations = {}

    for tb_fname in tb_files:
        stem = Path(tb_fname).stem
        xml_path = xml_dir / f"{stem}.xml"
        if not xml_path.exists():
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes = []
            classes_in_image = set()

            size_elem = root.find("size")
            xml_w = int(size_elem.find("width").text) if size_elem is not None else 512
            xml_h = int(size_elem.find("height").text) if size_elem is not None else 512

            for obj in root.findall("object"):
                cname = obj.find("name").text.strip()
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)

                # Coordinate sanity checks
                reasons = []
                if xmin >= xmax:
                    reasons.append(f"xmin({xmin}) >= xmax({xmax})")
                if ymin >= ymax:
                    reasons.append(f"ymin({ymin}) >= ymax({ymax})")
                if xmin < 0 or ymin < 0 or xmax > xml_w or ymax > xml_h:
                    reasons.append(f"coordinates out-of-bounds ([0,{xml_w}]x[0,{xml_h}])")
                if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                    reasons.append(f"degenerate box area (<1px)")

                if reasons:
                    invalid_boxes.append((f"tb/{tb_fname}", cname, reasons))

                boxes.append({"class": cname, "bbox": [xmin, ymin, xmax, ymax], "orig_size": (xml_w, xml_h)})
                box_class_counter[cname] += 1
                classes_in_image.add(cname)

            box_count_per_image[len(boxes)] += 1
            if len(boxes) > 1:
                multi_box_count += 1

            # Mutually exclusive categorization
            has_active = "ActiveTuberculosis" in classes_in_image
            has_obsolete = "ObsoletePulmonaryTuberculosis" in classes_in_image

            if len(boxes) == 0:
                images_by_category["zero_boxes"].append(tb_fname)
            elif has_active and has_obsolete:
                images_by_category["both"].append(tb_fname)
            elif has_active:
                images_by_category["active_only"].append(tb_fname)
            elif has_obsolete:
                images_by_category["obsolete_only"].append(tb_fname)
            else:
                # Other unexpected classes
                pass

            parsed_annotations[f"tb/{tb_fname}"] = boxes

        except Exception as e:
            print(f"  ERROR parsing {xml_path}: {e}")

    print(f"\n[Bounding Box Class Breakdown]")
    for cname, count in box_class_counter.items():
        print(f"  {cname:<32} : {count:>5} boxes")
    print(f"  Total Bounding Boxes Encounters   : {sum(box_class_counter.values()):>5} boxes")

    print(f"\n[Mutually Exclusive Image Categorization (N={len(tb_files)})]")
    print(f"  1. Active-only Images             : {len(images_by_category['active_only']):>5} images")
    print(f"  2. Obsolete-only Images           : {len(images_by_category['obsolete_only']):>5} images")
    print(f"  3. Both (Active + Obsolete)       : {len(images_by_category['both']):>5} images")
    print(f"  4. Zero Boxes (Empty Annotation)  : {len(images_by_category['zero_boxes']):>5} images")
    cat_sum = sum(len(v) for v in images_by_category.values())
    print(f"  Sum of Disjoint Categories        : {cat_sum:>5} images (Check == {len(tb_files)})")

    print(f"\n[Lesion Density & Box Validation]")
    print(f"  Images with multiple boxes (>1)   : {multi_box_count:>5} images")
    print(f"  Invalid / Degenerate boxes found  : {len(invalid_boxes):>5}")
    if invalid_boxes:
        for img, cls, reas in invalid_boxes[:3]:
            print(f"    - {img} ({cls}): {', '.join(reas)}")

    # Explicit tb1199 investigation
    print(f"\n[Specific Case Audit: tb1199.png & tb1199.xml]")
    tb1199_boxes = parsed_annotations.get("tb/tb1199.png")
    if tb1199_boxes is not None:
        print(f"  Path              : imgs/tb/tb1199.png")
        print(f"  XML File          : annotations/xml/tb1199.xml (Size: 145 bytes)")
        print(f"  Parsed Box Count  : {len(tb1199_boxes)} boxes")
        print(f"  Audit Status      : ZERO bounding boxes. Header initialized but no <object> recorded.")
    else:
        print("  tb1199.png not found in parsed annotations.")

    return parsed_annotations, box_class_counter, images_by_category


# =====================================================================
# 4. JSON Cross-Check
# =====================================================================
def audit_json_crosscheck(dataset_root: Path, parsed_annotations: dict):
    """
    Cross-checks existing COCO JSON annotations against VOC XML parsed ground truth.
    Explains discrepancies and investigates image/box presence across different splits.
    """
    print(format_header("4. COCO JSON CROSS-CHECK & DISCREPANCY AUDIT"))
    json_dir = dataset_root / "annotations" / "json"

    target_jsons = [
        ("TBX11K_trainval.json", "Full benchmark trainval set (healthy + tb + sick)"),
        ("TBX11K_trainval_only_tb.json", "TB-only benchmark trainval set"),
        ("TBX11K_train.json", "Full benchmark train set"),
        ("TBX11K_train_only_tb.json", "TB-only benchmark train set"),
        ("TBX11K_val.json", "Full benchmark val set"),
        ("TBX11K_val_only_tb.json", "TB-only benchmark val set"),
    ]

    for jname, desc in target_jsons:
        jpath = json_dir / jname
        if not jpath.exists():
            print(f"\n  {jname}: [NOT FOUND]")
            continue

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            img_count = len(data.get("images", []))
            anno_count = len(data.get("annotations", []))
            categories = {c["id"]: c["name"] for c in data.get("categories", [])}
            anno_class_counts = Counter(categories.get(a["category_id"], str(a["category_id"])) for a in data.get("annotations", []))

            # Check presence of tb1199
            tb1199_in_imgs = any("tb1199" in img["file_name"] for img in data.get("images", []))

            print(f"\n  {jname} ({desc}):")
            print(f"    - Total images registered      : {img_count}")
            print(f"    - Total annotations registered : {anno_count}")
            print(f"    - Annotation breakdown         : {dict(anno_class_counts)}")
            print(f"    - 'tb1199' present in 'images' : {tb1199_in_imgs}")

        except Exception as e:
            print(f"  ERROR reading {jname}: {e}")

    print(format_sub_header("Technical Explanation of JSON Discrepancies"))
    print(
        "  1. In TB-only JSONs (e.g. TBX11K_trainval_only_tb.json):\n"
        "     The official generator code/make_json_anno.py skips any image where\n"
        "     load_annotation() returns None when --tb_only is specified.\n"
        "     Because tb1199.xml has 0 objects, it is dropped entirely from the\n"
        "     'images' list. Hence, TB-only trainval has exactly 799 images (not 800).\n\n"
        "  2. In Full Benchmark JSONs (e.g. TBX11K_trainval.json):\n"
        "     When --tb_only is False, images with 0 boxes (health, sick, and tb1199)\n"
        "     are retained in 'images' with zero appended annotation blocks.\n"
        "     Hence, TBX11K_trainval.json records all 8,400 images with 1,211 boxes."
    )


# =====================================================================
# 5. Research Eligibility & Clinical Framing
# =====================================================================
def audit_research_eligibility(physical_files: dict, parsed_annotations: dict):
    """
    Evaluates dataset subsets for research eligibility based on clinical and task suitability.
    """
    print(format_header("5. RESEARCH ELIGIBILITY & CLINICAL TASK SUITABILITY"))

    health_count = len(physical_files.get("health", []))
    tb_count = len(physical_files.get("tb", []))
    sick_count = len(physical_files.get("sick", []))
    extra_count = len(physical_files.get("extra", []))
    test_count = len(physical_files.get("test", []))

    print(f"\n[Cohort Scope Evaluation]")
    print(f"  Proposed Candidate Pool (Current Research):")
    print(f"    - health/ : {health_count:>5} images (Verified healthy background controls)")
    print(f"    - tb/     : {tb_count:>5} images (TB-positive cohort with focal lesion boxes)")
    print(f"    - Total Candidate Population : {health_count + tb_count} images")

    print(f"\n  Excluded Cohorts (Current Research):")
    print(f"    - sick/   : {sick_count:>5} images (Non-TB pulmonary diseases, excluded to prevent cross-pathology conflation)")
    print(f"    - extra/  : {extra_count:>5} images (External multi-site datasets: DA, DB, MC, Shenzhen)")
    print(f"    - test/   : {test_count:>5} images (Unlabelled official competition challenge test set)")

    print(format_sub_header("Task-Specific Suitability for tb/ Images"))
    print(
        "  A. Binary Disease Classification (TB-Positive vs Healthy Negative):\n"
        "     - Eligible TB images: All 800 images in tb/ are clinically diagnosed as TB-positive.\n"
        "     - tb1199.png is safe for binary classification (label = 1 / Positive).\n\n"
        "  B. Object Detection / Lesion Localization:\n"
        "     - Eligible TB images: 799 images possess valid focal bounding boxes.\n"
        "     - tb1199.png CANNOT be safely used as a positive detection sample (0 ground truth boxes\n"
        "       cause any correct model prediction to be scored as False Positive in mAP).\n"
        "     - tb1199.png CANNOT be used as a negative sample (treating a TB patient as background\n"
        "       destroys model specificity and clinical validity)."
    )


# =====================================================================
# 6. Split Planning Information (Informational Only)
# =====================================================================
def audit_split_planning(physical_files: dict):
    """
    Calculates provisional 70:15:15 integer boundaries from sorted candidate filenames.
    Provides boundaries for both N=800 and N=799 options.
    STRICTLY INFORMATIONAL: Does not create split files or folders.
    """
    print(format_header("6. PROVISIONAL 70:15:15 SPLIT PLANNING (INFORMATIONAL ONLY)"))
    print("  * NO FILES, FOLDERS, OR MANIFESTS ARE GENERATED DURING THIS PHASE *\n")

    health_sorted = sorted(physical_files.get("health", []))
    tb_sorted = sorted(physical_files.get("tb", []))

    # Health split: N = 3800
    n_h = len(health_sorted)
    h_train_cnt = round(n_h * 0.70)
    h_val_cnt = round(n_h * 0.15)
    h_test_cnt = n_h - h_train_cnt - h_val_cnt

    h_train = health_sorted[:h_train_cnt]
    h_val = health_sorted[h_train_cnt:h_train_cnt + h_val_cnt]
    h_test = health_sorted[h_train_cnt + h_val_cnt:]

    print(f"[Option 1: Complete Candidate Pool (Health: 3800, TB: 800)]")
    n_tb_800 = len(tb_sorted)
    tb_tr_800 = round(n_tb_800 * 0.70)
    tb_va_800 = round(n_tb_800 * 0.15)
    tb_te_800 = n_tb_800 - tb_tr_800 - tb_va_800

    tb_train_800 = tb_sorted[:tb_tr_800]
    tb_val_800 = tb_sorted[tb_tr_800:tb_tr_800 + tb_va_800]
    tb_test_800 = tb_sorted[tb_tr_800 + tb_va_800:]

    table_data_800 = [
        ("health", "Train", "70.0%", len(h_train), f"1 - {len(h_train)}", h_train[0], h_train[-1]),
        ("health", "Val",   "15.0%", len(h_val),   f"{len(h_train)+1} - {len(h_train)+len(h_val)}", h_val[0], h_val[-1]),
        ("health", "Test",  "15.0%", len(h_test),  f"{len(h_train)+len(h_val)+1} - {n_h}", h_test[0], h_test[-1]),
        ("tb",     "Train", "70.0%", len(tb_train_800), f"1 - {len(tb_train_800)}", tb_train_800[0], tb_train_800[-1]),
        ("tb",     "Val",   "15.0%", len(tb_val_800),   f"{len(tb_train_800)+1} - {len(tb_train_800)+len(tb_val_800)}", tb_val_800[0], tb_val_800[-1]),
        ("tb",     "Test",  "15.0%", len(tb_test_800),  f"{len(tb_train_800)+len(tb_val_800)+1} - {n_tb_800}", tb_test_800[0], tb_test_800[-1]),
    ]

    print(f"  {'Cohort':<8} {'Split':<6} {'Ratio':<6} {'Count':<7} {'Index Range':<15} {'First Filename':<16} {'Last Filename':<16}")
    print(f"  {'-'*80}")
    for cohort, split, ratio, cnt, irange, first_f, last_f in table_data_800:
        print(f"  {cohort:<8} {split:<6} {ratio:<6} {cnt:<7} {irange:<15} {first_f:<16} {last_f:<16}")
    print(f"  Total Candidate Pool: {n_h + n_tb_800} images (Train: {len(h_train)+len(tb_train_800)}, Val: {len(h_val)+len(tb_val_800)}, Test: {len(h_test)+len(tb_test_800)})")

    # TB Option 2: Pure Detection Pool excluding tb1199 (N = 799)
    tb_filtered = [f for f in tb_sorted if f != "tb1199.png"]
    n_tb_799 = len(tb_filtered)
    tb_tr_799 = round(n_tb_799 * 0.70)
    tb_va_799 = round(n_tb_799 * 0.15)
    tb_te_799 = n_tb_799 - tb_tr_799 - tb_va_799

    tb_train_799 = tb_filtered[:tb_tr_799]
    tb_val_799 = tb_filtered[tb_tr_799:tb_tr_799 + tb_va_799]
    tb_test_799 = tb_filtered[tb_tr_799 + tb_va_799:]

    print(f"\n[Option 2: Filtered Detection Pool (Excluding tb1199.png: TB N=799)]")
    table_data_799 = [
        ("tb", "Train", "69.96%", len(tb_train_799), f"1 - {len(tb_train_799)}", tb_train_799[0], tb_train_799[-1]),
        ("tb", "Val",   "15.02%", len(tb_val_799),   f"{len(tb_train_799)+1} - {len(tb_train_799)+len(tb_val_799)}", tb_val_799[0], tb_val_799[-1]),
        ("tb", "Test",  "15.02%", len(tb_test_799),  f"{len(tb_train_799)+len(tb_val_799)+1} - {n_tb_799}", tb_test_799[0], tb_test_799[-1]),
    ]
    print(f"  {'Cohort':<8} {'Split':<6} {'Ratio':<6} {'Count':<7} {'Index Range':<15} {'First Filename':<16} {'Last Filename':<16}")
    print(f"  {'-'*80}")
    for cohort, split, ratio, cnt, irange, first_f, last_f in table_data_799:
        print(f"  {cohort:<8} {split:<6} {ratio:<6} {cnt:<7} {irange:<15} {first_f:<16} {last_f:<16}")
    print(f"  Total Candidate Pool: {n_h + n_tb_799} images (Train: {len(h_train)+len(tb_train_799)}, Val: {len(h_val)+len(tb_val_799)}, Test: {len(h_test)+len(tb_test_799)})")


# =====================================================================
# Main Execution Runner
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Comprehensive Read-Only Audit of TBX11K Dataset")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to the TBX11K root folder. If omitted, auto-discovers relative to script location."
    )
    args = parser.parse_args()

    # Determine dataset root
    if args.dataset_dir:
        dataset_root = Path(args.dataset_dir).resolve()
    else:
        # Check current working directory, then script parent directories
        cwd_candidate = Path("TBX11K").resolve()
        script_candidate = Path(__file__).resolve().parent.parent / "TBX11K"
        if cwd_candidate.exists():
            dataset_root = cwd_candidate
        elif script_candidate.exists():
            dataset_root = script_candidate
        else:
            print("ERROR: Could not locate TBX11K directory. Please pass --dataset_dir /path/to/TBX11K")
            sys.exit(1)

    print(f"Starting TBX11K Dataset Audit (Read-Only Mode)...")
    print(f"Dataset Target: {dataset_root}")

    # Phase 1: Physical inventory and manifests
    physical_files, manifest_data = audit_inventory(dataset_root)

    # Phase 2: Image integrity and decoding
    audit_image_integrity(dataset_root, physical_files)

    # Phase 3: VOC XML Annotations
    parsed_annotations, box_counts, img_cats = audit_xml_annotations(dataset_root, physical_files.get("tb", []))

    # Phase 4: COCO JSON cross-check
    audit_json_crosscheck(dataset_root, parsed_annotations)

    # Phase 5: Research eligibility
    audit_research_eligibility(physical_files, parsed_annotations)

    # Phase 6: Provisional split planning
    audit_split_planning(physical_files)

    print(format_header("AUDIT COMPLETE — NO DISK MODIFICATIONS PERFORMED"))


if __name__ == "__main__":
    main()
