from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ..image_utils import blend_images
from ..settings import (
    DINO_WEIGHTS_PATH,
    DINO_IMAGE_SIZE,
    DINO_DEVICE,
    PCA_BLEND,
    PCA_BRIGHTNESS,
)
from .dinov3_loader import load_dinov3_vitb16_backbone


class DinoPcaVisualizer:
    backend_name = "dinov3_vitb16"

    def __init__(self) -> None:
        self.model = load_dinov3_vitb16_backbone(DINO_WEIGHTS_PATH)

        self.model.eval()
        self.device = DINO_DEVICE
        self.model.to(self.device)

        self.mean = torch.tensor((0.485, 0.456, 0.406))
        self.std = torch.tensor((0.229, 0.224, 0.225))

        self.num_prefix_tokens = getattr(self.model, "n_storage_tokens", 0) + 1
        self.patch_size = self._infer_patch_size()

    def _infer_patch_size(self) -> int:
        patch = getattr(self.model, "patch_embed", None)
        if patch is None or not hasattr(patch, "patch_size"):
            return 16
        patch_size = patch.patch_size
        if isinstance(patch_size, (tuple, list)):
            return int(patch_size[0])
        return int(patch_size)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        mean = self.mean[:, None, None]
        std = self.std[:, None, None]
        x = (x - mean) / std
        return x.unsqueeze(0)

    def _get_patch_tokens(self, image: Image.Image) -> Tuple[torch.Tensor, int, int]:
        if DINO_IMAGE_SIZE % self.patch_size != 0:
            raise RuntimeError("DINO_IMAGE_SIZE must be divisible by patch size")

        x = self._preprocess(image).to(self.device)
        tokens = self.model.forward_features(x)
        if isinstance(tokens, dict):
            tokens = tokens["x_norm_patchtokens"]
        elif isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
            if isinstance(tokens, dict):
                tokens = tokens["x_norm_patchtokens"]

        if tokens.shape[0] != 1:
            raise RuntimeError("expected single-image batch")

        h = DINO_IMAGE_SIZE // self.patch_size
        w = DINO_IMAGE_SIZE // self.patch_size
        if tokens.shape[1] != h * w:
            raise RuntimeError("token count mismatch for visualization")

        return tokens[0], h, w

    @torch.no_grad()
    def extract_feature_matrix(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int], int]:
        patch_tokens, h, w = self._get_patch_tokens(image)
        return patch_tokens.detach().cpu().numpy(), (h, w), self.patch_size

    @torch.no_grad()
    def visualize(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        patch_tokens, h, w = self._get_patch_tokens(image)
        patch_tokens = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
        _, _, v = torch.pca_lowrank(patch_tokens, q=3, niter=2)
        comps = patch_tokens @ v[:, :3]

        comps = comps.view(h, w, 3)

        # Use a single continuous palette based on intensity (higher = brighter)
        intensity = torch.linalg.norm(comps, dim=-1)
        intensity = intensity - intensity.min()
        intensity = intensity / (intensity.max() + 1e-6)
        intensity = (intensity * PCA_BRIGHTNESS).clamp(0.0, 1.0)

        # Single continuous palette (black -> yellow)
        c1 = torch.tensor([0.0, 0.0, 0.0])  # dark
        c2 = torch.tensor([1.0, 0.95, 0.0])  # bright

        t = intensity.unsqueeze(-1)
        rgb = (c1 + (c2 - c1) * t).clamp(0.0, 1.0)

        rgb = (rgb * 255.0).to(torch.uint8).cpu().numpy()
        pca_img = Image.fromarray(rgb, mode="RGB").resize(image.size, Image.BILINEAR)
        overlay = blend_images(image, pca_img, PCA_BLEND)
        return pca_img, overlay

    @torch.no_grad()
    def cosine_similarity(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        patch_tokens, h, w = self._get_patch_tokens(image)
        tokens = torch.nn.functional.normalize(patch_tokens, dim=-1)
        center_idx = (h // 2) * w + (w // 2)
        anchor = tokens[center_idx]
        sim = tokens @ anchor

        sim = sim.view(h, w)
        sim = sim - sim.min()
        sim = sim / (sim.max() + 1e-6)

        rgb = self._apply_colormap(sim)
        cosine_img = Image.fromarray(rgb, mode="RGB").resize(image.size, Image.BILINEAR)
        overlay = blend_images(image, cosine_img, PCA_BLEND)
        return cosine_img, overlay

    def _apply_colormap(self, t: torch.Tensor) -> np.ndarray:
        # simple viridis-like palette: purple -> teal -> yellow
        c1 = torch.tensor([68, 1, 84], device=t.device, dtype=torch.float32)
        c2 = torch.tensor([32, 144, 140], device=t.device, dtype=torch.float32)
        c3 = torch.tensor([253, 231, 37], device=t.device, dtype=torch.float32)

        t = t.clamp(0.0, 1.0)
        t = t.unsqueeze(-1)
        mid = 0.5
        low_mask = (t <= mid).float()
        high_mask = 1.0 - low_mask

        t_low = (t / mid).clamp(0.0, 1.0)
        t_high = ((t - mid) / mid).clamp(0.0, 1.0)

        low = c1 + (c2 - c1) * t_low
        high = c2 + (c3 - c2) * t_high
        rgb = low * low_mask + high * high_mask
        return rgb.byte().cpu().numpy()


class ImagePatchFallbackVisualizer:
    backend_name = "image_patch_fallback"

    def __init__(self) -> None:
        self.patch = 16

    def _patch_matrix(self, image: Image.Image) -> Tuple[np.ndarray, int, int]:
        img = image.convert("RGB").resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        h = DINO_IMAGE_SIZE // self.patch
        w = DINO_IMAGE_SIZE // self.patch
        arr = arr.reshape(h, self.patch, w, self.patch, 3).mean(axis=(1, 3))
        return arr.reshape(h * w, 3), h, w

    def extract_feature_matrix(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int], int]:
        matrix, h, w = self._patch_matrix(image)
        return matrix, (h, w), self.patch

    def visualize(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        matrix, h, w = self._patch_matrix(image)
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        comps = matrix @ vh[:3].T
        intensity = np.linalg.norm(comps, axis=-1).reshape(h, w)
        intensity = intensity - intensity.min()
        intensity = intensity / (intensity.max() + 1e-6)
        rgb = np.stack(
            [
                intensity,
                intensity * 0.95,
                np.zeros_like(intensity),
            ],
            axis=-1,
        )
        pca_img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").resize(
            image.size, Image.BILINEAR
        )
        overlay = blend_images(image, pca_img, PCA_BLEND)
        return pca_img, overlay

    def cosine_similarity(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        matrix, h, w = self._patch_matrix(image)
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-6
        matrix = matrix / norms
        center_idx = (h // 2) * w + (w // 2)
        sim = matrix @ matrix[center_idx]
        sim = sim.reshape(h, w)
        sim = sim - sim.min()
        sim = sim / (sim.max() + 1e-6)
        rgb = np.stack([sim * 0.25, sim * 0.7, sim], axis=-1)
        cosine_img = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB").resize(
            image.size, Image.BILINEAR
        )
        overlay = blend_images(image, cosine_img, PCA_BLEND)
        return cosine_img, overlay


_pca: Optional[DinoPcaVisualizer] = None


def get_pca_visualizer():
    global _pca
    if _pca is None:
        try:
            _pca = DinoPcaVisualizer()
        except Exception:
            _pca = ImagePatchFallbackVisualizer()
    return _pca
