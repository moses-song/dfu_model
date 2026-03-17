from __future__ import annotations

# Simple, beginner-friendly Mask2Former inference code (Transformers).
# This file is intentionally verbose with comments to explain each step.

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .settings import HUGGINGFACE_HUB_TOKEN


@dataclass(frozen=True)
class Mask2FormerConfig:
    # Use a public Mask2Former checkpoint for now.
    # You can replace this with your fine-tuned checkpoint later.
    model_id: str = "facebook/mask2former-swin-base-coco-panoptic"
    image_size: int = 512
    device: str = "cpu"


class Mask2FormerRunner:
    """
    Simple wrapper for Mask2Former inference.

    Flow:
    1) Load processor + model (once)
    2) Preprocess image
    3) Run model
    4) Postprocess to get masks
    """

    def __init__(self, cfg: Mask2FormerConfig) -> None:
        self.cfg = cfg

        # The processor handles resizing/normalization the same way as training.
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.model_id,
            token=HUGGINGFACE_HUB_TOKEN or None,
        )

        # The model does the segmentation.
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.model_id,
            token=HUGGINGFACE_HUB_TOKEN or None,
        ).to(cfg.device)

        # Turn off training behaviors (dropout, etc.).
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, np.ndarray]:
        # 1) Preprocess input image into tensors for the model
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        # 2) Run model forward pass
        outputs = self.model(**inputs)

        # 3) Convert model outputs into readable segmentation
        # target_sizes expects (height, width)
        target_sizes = [image.size[::-1]]
        result = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes
        )[0]

        # result contains:
        # - segmentation: (H, W) int map, each pixel is a segment id
        # - segments_info: metadata for each segment
        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        return {
            "segmentation": segmentation,
            "segments_info": np.array(segments_info, dtype=object),
        }


def load_mask2former(
    model_id: str = "facebook/mask2former-swin-base-coco-panoptic",
    image_size: int = 512,
    device: str = "cpu",
) -> Mask2FormerRunner:
    # Convenience function so beginners can do:
    # runner = load_mask2former()
    cfg = Mask2FormerConfig(model_id=model_id, image_size=image_size, device=device)
    return Mask2FormerRunner(cfg)
