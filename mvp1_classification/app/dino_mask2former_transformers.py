from __future__ import annotations

# Custom DINOv3 backbone wired into Transformers Mask2Former.
# This is beginner-friendly and intentionally verbose.

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers.modeling_outputs import BackboneOutput

from .settings import HUGGINGFACE_HUB_TOKEN


@dataclass(frozen=True)
class DinoV3Mask2FormerConfig:
    # DINOv3 backbone (ViT-B/16, LVD-1689M)
    dino_model_name: str = "vit_base_patch16_dinov3.lvd1689m"

    # Base Mask2Former checkpoint (we will keep its decoder weights)
    base_mask2former_id: str = "facebook/mask2former-swin-base-coco-panoptic"

    # Input image size for the DINOv3 backbone
    image_size: int = 512

    # Mask2Former expects feature maps with 256 channels
    feature_dim: int = 256

    device: str = "cpu"


class DinoV3FeaturePyramidBackbone(nn.Module):
    """
    DINOv3 ViT backbone that outputs a 4-level feature pyramid.

    Mask2Former expects multi-scale feature maps (stride 4/8/16/32),
    but ViT outputs a token sequence. We convert tokens -> 2D map,
    then build a pyramid by up/down-sampling.
    """

    def __init__(self, cfg: DinoV3Mask2FormerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 1) Load DINOv3 ViT from timm
        self.vit = timm.create_model(
            cfg.dino_model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.vit.eval()

        # 2) Resolve expected mean/std for this model
        data_cfg = timm.data.resolve_model_data_config(self.vit)
        self.mean = torch.tensor(data_cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std = torch.tensor(data_cfg.get("std", (0.229, 0.224, 0.225)))

        # 3) Projection + pyramid layers
        in_dim = getattr(self.vit, "num_features", None)
        if in_dim is None:
            raise ValueError("Unable to infer DINOv3 feature dimension (num_features).")

        self.proj = nn.Conv2d(in_dim, cfg.feature_dim, kernel_size=1)
        self.p2 = nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, padding=1)
        self.p5 = nn.Conv2d(cfg.feature_dim, cfg.feature_dim, kernel_size=3, padding=1)

        # DINOv3 uses class + register tokens
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Resize + normalize image for DINOv3.
        Returns tensor shape: (1, 3, H, W)
        """
        img = image.convert("RGB").resize((self.cfg.image_size, self.cfg.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)

        # Normalize with model-specific mean/std
        mean = self.mean[:, None, None]
        std = self.std[:, None, None]
        x = (x - mean) / std

        return x.unsqueeze(0)

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        """
        Returns BackboneOutput(feature_maps=[P2, P3, P4, P5]).
        """
        # 1) ViT token output
        tokens = self.vit.forward_features(pixel_values)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        # 2) Convert tokens to 2D feature map (stride=16)
        b, n, c = tokens.shape
        h = pixel_values.shape[-2] // 16
        w = pixel_values.shape[-1] // 16

        patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        if patch_tokens.shape[1] != h * w:
            raise ValueError(
                f"Token count mismatch. expected={h*w}, got={patch_tokens.shape[1]}"
            )

        feat16 = patch_tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        feat16 = self.proj(feat16)

        # 3) Build feature pyramid
        p4 = self.p4(feat16)  # stride 16
        p3 = self.p3(F.interpolate(feat16, scale_factor=2, mode="bilinear", align_corners=False))  # stride 8
        p2 = self.p2(F.interpolate(feat16, scale_factor=4, mode="bilinear", align_corners=False))  # stride 4
        p5 = self.p5(F.max_pool2d(feat16, kernel_size=2, stride=2))  # stride 32

        return BackboneOutput(feature_maps=[p2, p3, p4, p5])


class DinoV3Mask2FormerRunner:
    """
    Mask2Former with custom DINOv3 backbone.

    We load a pretrained Mask2Former checkpoint (decoder weights),
    then replace its backbone with DINOv3 + pyramid adapter.
    """

    def __init__(self, cfg: DinoV3Mask2FormerConfig) -> None:
        self.cfg = cfg

        # Load Mask2Former (keeps decoder weights)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.base_mask2former_id,
            token=HUGGINGFACE_HUB_TOKEN or None,
        ).to(cfg.device)

        # Replace backbone with DINOv3 pyramid backbone
        self.backbone = DinoV3FeaturePyramidBackbone(cfg).to(cfg.device)
        self.model.model.backbone = self.backbone

        # Update config to match our pyramid (stride 4/8/16/32)
        self.model.config.feature_strides = [4, 8, 16, 32]
        self.model.config.feature_size = cfg.feature_dim
        self.model.config.mask_feature_size = cfg.feature_dim

        # Image processor for post-processing only
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            cfg.base_mask2former_id,
            token=HUGGINGFACE_HUB_TOKEN or None,
        )

        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, np.ndarray]:
        # 1) Preprocess for DINOv3
        pixel_values = self.backbone.preprocess(image).to(self.cfg.device)

        # 2) Run model
        outputs = self.model(pixel_values=pixel_values)

        # 3) Postprocess with HF utility
        target_sizes = [image.size[::-1]]
        result = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=target_sizes
        )[0]

        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        return {
            "segmentation": segmentation,
            "segments_info": np.array(segments_info, dtype=object),
        }


def load_dinov3_mask2former(
    dino_model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    base_mask2former_id: str = "facebook/mask2former-swin-base-coco-panoptic",
    image_size: int = 512,
    feature_dim: int = 256,
    device: str = "cpu",
) -> DinoV3Mask2FormerRunner:
    cfg = DinoV3Mask2FormerConfig(
        dino_model_name=dino_model_name,
        base_mask2former_id=base_mask2former_id,
        image_size=image_size,
        feature_dim=feature_dim,
        device=device,
    )
    return DinoV3Mask2FormerRunner(cfg)
