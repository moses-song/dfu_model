from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm


@dataclass(frozen=True)
class DinoV3Config:
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    patch_size: int = 16
    image_size: int = 512
    out_dim: int = 256  # Mask2Former pixel decoder expects 256-dim features


class DinoV3Backbone(nn.Module):
    """
    Wraps a DINOv3 ViT (timm) and returns patch tokens + spatial grid size.
    """

    def __init__(self, cfg: DinoV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.model.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int], int]:
        """
        Returns:
            tokens: (B, N, C)
            grid_hw: (H, W) of patch grid
            num_prefix_tokens: number of non-patch tokens to strip
        """
        # timm ViT returns (B, N, C) from forward_features
        tokens = self.model.forward_features(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]

        # Compute grid size from input
        h = x.shape[-2] // self.cfg.patch_size
        w = x.shape[-1] // self.cfg.patch_size

        # DINOv3 uses class + register tokens; timm exposes num_prefix_tokens
        num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)
        return tokens, (h, w), num_prefix_tokens


class ViTTokenPyramidAdapter(nn.Module):
    """
    Adapts ViT patch tokens into multi-scale feature maps for Mask2Former.
    Produces a 4-level pyramid at strides [4, 8, 16, 32].
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.p2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.p3 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.p4 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.p5 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(
        self, tokens: torch.Tensor, grid_hw: Tuple[int, int], num_prefix_tokens: int
    ) -> Dict[str, torch.Tensor]:
        b, n, c = tokens.shape
        h, w = grid_hw

        # Strip class/register tokens
        patch_tokens = tokens[:, num_prefix_tokens:, :]
        if patch_tokens.shape[1] != h * w:
            raise ValueError(
                f"Token count mismatch. expected={h*w}, got={patch_tokens.shape[1]}"
            )

        # (B, N, C) -> (B, C, H, W) at stride=16
        feat16 = patch_tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        feat16 = self.proj(feat16)

        # Build pyramid
        p4 = self.p4(feat16)  # stride 16
        p3 = self.p3(F.interpolate(feat16, scale_factor=2, mode="bilinear", align_corners=False))  # stride 8
        p2 = self.p2(F.interpolate(feat16, scale_factor=4, mode="bilinear", align_corners=False))  # stride 4
        p5 = self.p5(F.max_pool2d(feat16, kernel_size=2, stride=2))  # stride 32

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}


class DinoV3Mask2FormerBridge(nn.Module):
    """
    Returns a Mask2Former-ready feature pyramid.
    If a Mask2Former head is provided, it will be called with the pyramid.
    """

    def __init__(
        self,
        cfg: DinoV3Config,
        mask2former_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoV3Backbone(cfg)
        in_dim = getattr(self.backbone.model, "num_features", None)
        if in_dim is None:
            raise ValueError("Unable to infer backbone feature dim (num_features).")
        self.adapter = ViTTokenPyramidAdapter(in_dim=in_dim, out_dim=cfg.out_dim)
        self.mask2former_head = mask2former_head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens, grid_hw, num_prefix = self.backbone(x)
        pyramid = self.adapter(tokens, grid_hw, num_prefix)
        if self.mask2former_head is None:
            return pyramid
        return self.mask2former_head(pyramid)


def _set_hf_token_env(token: Optional[str]) -> None:
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)


def build_bridge(
    token: Optional[str] = None,
    model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    image_size: int = 512,
) -> DinoV3Mask2FormerBridge:
    _set_hf_token_env(token)
    cfg = DinoV3Config(model_name=model_name, image_size=image_size)
    return DinoV3Mask2FormerBridge(cfg)


def preprocess_image(image: Image.Image, image_size: int = 512) -> torch.Tensor:
    img = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1)
    return arr.unsqueeze(0)


@torch.no_grad()
def infer_pyramid(
    image: Image.Image,
    token: Optional[str] = None,
    model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    image_size: int = 512,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    bridge = build_bridge(token=token, model_name=model_name, image_size=image_size).to(device)
    x = preprocess_image(image, image_size=image_size).to(device)
    return bridge(x)
