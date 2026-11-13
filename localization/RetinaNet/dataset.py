"""
TBX11K Torchvision Detection Dataset for RetinaNet
==================================================

Dataset class for TBX11K pulmonary tuberculosis lesion localization
using torchvision detection reference structures.

Classes:
  0: Background
  1: ActiveTuberculosis
  2: ObsoletePulmonaryTuberculosis

Canonical Splits:
  - train: 3219 images (559 TB images with >= 1 box + 2660 healthy images, 837 boxes: 657 Active, 180 Obsolete)
  - val:   690 images (120 TB with boxes + 570 healthy background, 195 boxes: 172 Active, 23 Obsolete)
  - test:  690 images (120 TB with boxes + 570 healthy background, 179 boxes: 143 Active, 36 Obsolete)
"""

import os
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

# Canonical Audit Reference Counts (from Phase 1 Data Audit)
CANONICAL_AUDIT_COUNTS = {
    "train": {
        "images": 3219,
        "images_with_boxes": 559,
        "empty_images": 2660,
        "boxes": 837,
        "active_boxes": 657,
        "obsolete_boxes": 180,
    },
    "val": {
        "images": 690,
        "images_with_boxes": 120,
        "empty_images": 570,
        "boxes": 195,
        "active_boxes": 172,
        "obsolete_boxes": 23,
    },
    "test": {
        "images": 690,
        "images_with_boxes": 120,
        "empty_images": 570,
        "boxes": 179,
        "active_boxes": 143,
        "obsolete_boxes": 36,
    },
}


def find_workspace_root() -> Path:
    """Resolves the IEEE RESEARCH workspace root dynamically."""
    candidate = Path(__file__).resolve().parent
    for _ in range(5):
        if (candidate / "TBX11K").is_dir() and (candidate / "data_prep").is_dir():
            return candidate
        candidate = candidate.parent
    return Path.cwd()


class TBX11KDetectionDataset(Dataset):
    """
    Torchvision-compatible detection dataset for TBX11K.
    """

    def __init__(
        self,
        split: str,
        workspace_root: Optional[Path] = None,
        transforms: Optional[Callable] = None,
        is_train: Optional[bool] = None,
    ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")

        self.split = split
        self.transforms = transforms
        self.workspace_root = Path(workspace_root) if workspace_root else find_workspace_root()
        self.data_dir = self.workspace_root / "TBX11K"
        self.splits_dir = self.workspace_root / "data_prep" / "splits"
        self.is_train = (split == "train") if is_train is None else is_train

        # Load split CSV
        split_csv = self.splits_dir / f"{split}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        with open(split_csv, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

        # Load canonical COCO annotations JSON
        json_path = self.data_dir / "annotations" / "json" / "TBX11K_trainval.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation JSON: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Build image_id -> file_name dictionary lookup
        id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # Index annotations by image basename
        base_to_annos: Dict[str, List[Dict[str, Any]]] = {}
        for img_info in coco_data["images"]:
            base_name = Path(img_info["file_name"]).name
            base_to_annos[base_name] = []

        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            img_file = id_to_file[img_id]
            base_name = Path(img_file).name
            base_to_annos.setdefault(base_name, []).append(ann)

        self.samples: List[Dict[str, Any]] = []
        for idx, row in enumerate(csv_rows):
            img_rel = row["image_path"].strip()
            base_name = Path(img_rel).name
            full_img_path = self.data_dir / img_rel
            if not full_img_path.is_file():
                raise FileNotFoundError(f"Image not found: {full_img_path}")

            annos = base_to_annos.get(base_name, [])

            boxes = []
            labels = []
            areas = []
            for a in annos:
                cid = a["category_id"]  # 1: ActiveTuberculosis, 2: ObsoletePulmonaryTuberculosis
                x, y, w, h = a["bbox"]
                # Convert COCO xywh to absolute xyxy float32
                x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cid))
                areas.append(float(w * h))

            self.samples.append({
                "image_id": idx,
                "image_path": full_img_path,
                "file_name": base_name,
                "cohort": row["cohort"],
                "raw_boxes": boxes,
                "raw_labels": labels,
                "raw_areas": areas,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def get_height_and_width(self, idx: int) -> Tuple[int, int]:
        """Fast path for aspect-ratio grouping without loading images."""
        return (512, 512)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Tensor]]:
        sample = self.samples[idx]
        img_path = sample["image_path"]

        # Read image from file using PIL
        with Image.open(img_path) as im:
            image = im.convert("RGB")
            width, height = image.size

        boxes = sample["raw_boxes"]
        labels = sample["raw_labels"]
        areas = sample["raw_areas"]

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": int(sample["image_id"]),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def self_check_dataset() -> bool:
    """
    Self-check function verifying per-split image and box counts against
    the Phase 1 canonical audit. Raises RuntimeError on mismatch.
    """
    workspace_root = find_workspace_root()
    print("Running TBX11KDetectionDataset self-check against canonical audit...")

    for split in ["train", "val", "test"]:
        ds = TBX11KDetectionDataset(split=split, workspace_root=workspace_root)
        ref = CANONICAL_AUDIT_COUNTS[split]

        total_imgs = len(ds)
        total_boxes = 0
        active_boxes = 0
        obsolete_boxes = 0

        for s in ds.samples:
            for l in s["raw_labels"]:
                total_boxes += 1
                if l == 1:
                    active_boxes += 1
                elif l == 2:
                    obsolete_boxes += 1
                else:
                    raise ValueError(f"Unknown label {l} in {split}")

        print(f"  [{split:5s}] Images: {total_imgs:4d} (ref: {ref['images']:4d}), "
              f"Boxes: {total_boxes:4d} (ref: {ref['boxes']:4d}), "
              f"Active: {active_boxes:4d} (ref: {ref['active_boxes']:4d}), "
              f"Obsolete: {obsolete_boxes:4d} (ref: {ref['obsolete_boxes']:4d})")

        if total_imgs != ref["images"]:
            raise RuntimeError(f"Self-check FAILED on {split} image count: {total_imgs} != {ref['images']}")
        if total_boxes != ref["boxes"]:
            raise RuntimeError(f"Self-check FAILED on {split} total boxes: {total_boxes} != {ref['boxes']}")
        if active_boxes != ref["active_boxes"]:
            raise RuntimeError(f"Self-check FAILED on {split} active boxes: {active_boxes} != {ref['active_boxes']}")
        if obsolete_boxes != ref["obsolete_boxes"]:
            raise RuntimeError(f"Self-check FAILED on {split} obsolete boxes: {obsolete_boxes} != {ref['obsolete_boxes']}")

    print("All dataset self-checks PASSED successfully.")
    return True


if __name__ == "__main__":
    self_check_dataset()
