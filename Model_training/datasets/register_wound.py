import os
from pathlib import Path

from detectron2.data.datasets import register_coco_instances


def register_wound() -> None:
    root = os.getenv("WOUND_DATASET_ROOT", "")
    ann_dir = os.getenv("WOUND_COCO_DIR", "")
    if not root or not ann_dir:
        return

    root_path = Path(root)
    ann_path = Path(ann_dir)

    register_coco_instances(
        "wound_train",
        {},
        str(ann_path / "instances_train.json"),
        str(root_path / "train" / "images"),
    )
    register_coco_instances(
        "wound_val",
        {},
        str(ann_path / "instances_validation.json"),
        str(root_path / "validation" / "images"),
    )
    register_coco_instances(
        "wound_test",
        {},
        str(ann_path / "instances_test.json"),
        str(root_path / "test" / "images"),
    )


if __name__ == "__main__":
    register_wound()
