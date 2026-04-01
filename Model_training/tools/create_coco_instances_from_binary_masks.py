import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def build_coco(images_dir: Path, labels_dir: Path, output_json: Path, category_name: str) -> None:
    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    label_files = sorted([p for p in labels_dir.glob("*.png")])
    for label_path in label_files:
        image_path = images_dir / label_path.name
        if not image_path.exists():
            continue

        mask = np.array(Image.open(label_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 0).astype(np.uint8)

        h, w = binary.shape
        images.append(
            {
                "id": image_id,
                "file_name": str(image_path.name),
                "height": int(h),
                "width": int(w),
            }
        )

        if binary.sum() > 0:
            rle = binary_mask_to_rle(binary)
            area = int(mask_utils.area(rle))
            bbox = mask_utils.toBbox(rle).tolist()
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name}],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, help="Foot Ulcer Segmentation Challenge root path")
    parser.add_argument("--out-dir", required=True, help="Output directory for COCO json")
    parser.add_argument("--category", default="wound", help="Category name for the single class")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)

    for split in ("train", "validation", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        output_json = out_dir / f"instances_{split}.json"
        build_coco(images_dir, labels_dir, output_json, args.category)


if __name__ == "__main__":
    main()
