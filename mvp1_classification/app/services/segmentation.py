from __future__ import annotations

from typing import Dict, Optional, Tuple
import importlib

import numpy as np
from PIL import Image

from ..settings import (
    SEG_DEFAULT_BACKEND,
    SEG_CONFIG_PATH,
    SEG_WEIGHTS_PATH,
    SEG_DEVICE,
    SEG_SCORE_THRESH,
    SEG_AREA_THRESHOLD,
    DINO_MODEL_NAME,
    DINO_WEIGHTS_PATH,
    SEG_DINO_IMAGE_SIZE,
    DINO_DEVICE,
    DINO_SEG_THRESHOLD,
    CUSTOM_SEGMENTER,
    DINO_M2F_CONFIG_PATH,
    DINO_M2F_WEIGHTS_PATH,
)

  
class BaseSegmenter:
    name = "base"

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float, bool]:
        raise NotImplementedError


class DemoSegmenter(BaseSegmenter):
    name = "demo"

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float, bool]:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]
        brightness = rgb.mean(axis=2)
        redness = red - (green + blue) / 2.0

        red_cutoff = max(18.0, float(np.percentile(redness, 82)))
        dark_cutoff = float(np.percentile(brightness, 45))
        mask = (redness >= red_cutoff) & (brightness <= dark_cutoff + 45.0)

        mask = self._smooth(mask)
        area_ratio = float(mask.mean())
        wound_present = area_ratio >= SEG_AREA_THRESHOLD
        return mask.astype(np.uint8), area_ratio, wound_present

    def _smooth(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask

        padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
        neighbors = np.zeros(mask.shape, dtype=np.uint8)
        for y in range(3):
            for x in range(3):
                neighbors += padded[y : y + mask.shape[0], x : x + mask.shape[1]]
        return neighbors >= 5


class SwinMask2FormerSegmenter(BaseSegmenter):
    name = "swin_m2f"

    def __init__(self) -> None:
        if not SEG_CONFIG_PATH or not SEG_WEIGHTS_PATH:
            raise RuntimeError("SEG_CONFIG_PATH and SEG_WEIGHTS_PATH must be set")

        # Lazy imports to avoid hard dependency unless used.
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(SEG_CONFIG_PATH)
        cfg.MODEL.WEIGHTS = SEG_WEIGHTS_PATH
        cfg.MODEL.DEVICE = SEG_DEVICE
        # Set a generic score threshold for instance predictions
        if hasattr(cfg.MODEL, "ROI_HEADS"):
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SEG_SCORE_THRESH
        if hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = SEG_SCORE_THRESH
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float, bool]:
        # Detectron2 expects BGR numpy image by default
        np_img = np.asarray(image.convert("RGB"))[:, :, ::-1]
        outputs = self.predictor(np_img)

        mask = None
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                if masks.size > 0:
                    mask = masks.any(axis=0)

        if mask is None and "sem_seg" in outputs:
            sem = outputs["sem_seg"].to("cpu").numpy()
            # take argmax to create a single mask
            mask = sem.argmax(axis=0) > 0

        if mask is None:
            mask = np.zeros(np_img.shape[:2], dtype=bool)

        area_ratio = float(mask.mean())
        wound_present = area_ratio >= SEG_AREA_THRESHOLD
        return mask.astype(np.uint8), area_ratio, wound_present


_DINO_BACKBONE_REGISTERED = False


def _register_dino_backbone() -> None:
    global _DINO_BACKBONE_REGISTERED
    if _DINO_BACKBONE_REGISTERED:
        return

    import torch
    from torch import nn
    from torch.nn import functional as F
    import timm
    from detectron2.modeling import BACKBONE_REGISTRY, Backbone
    from detectron2.layers import ShapeSpec

    @BACKBONE_REGISTRY.register()
    class DinoV3Backbone(Backbone):
        def __init__(self, cfg, input_shape):
            super().__init__()
            model_name = cfg.MODEL.DINO_V3.MODEL_NAME
            weights_path = cfg.MODEL.DINO_V3.WEIGHTS_PATH
            pretrained = False if weights_path else True
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )
            if weights_path:
                self._load_weights(weights_path)

            self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)
            self.patch_size = self._infer_patch_size()
            self.embed_dim = getattr(self.model, "num_features", None) or self.model.embed_dim

            self.out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

        def _infer_patch_size(self) -> int:
            patch = getattr(self.model, "patch_embed", None)
            if patch is None or not hasattr(patch, "patch_size"):
                return 16
            patch_size = patch.patch_size
            if isinstance(patch_size, (tuple, list)):
                return int(patch_size[0])
            return int(patch_size)

        def _load_weights(self, path: str) -> None:
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]

            if isinstance(state, dict):
                cleaned = {}
                for key, value in state.items():
                    cleaned[key.replace("module.", "")] = value
                state = cleaned

            self.model.load_state_dict(state, strict=False)

        def forward(self, x):
            tokens = self.model.forward_features(x)
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[-1]

            tokens = tokens[:, self.num_prefix_tokens :, :]
            b, n, c = tokens.shape
            h = x.shape[-2] // self.patch_size
            w = x.shape[-1] // self.patch_size
            if n != h * w:
                raise RuntimeError("token count mismatch for DINO backbone")

            feat = tokens.transpose(1, 2).reshape(b, c, h, w)
            res4 = feat
            res3 = F.interpolate(res4, scale_factor=2.0, mode="bilinear", align_corners=False)
            res2 = F.interpolate(res4, scale_factor=4.0, mode="bilinear", align_corners=False)
            res5 = F.avg_pool2d(res4, kernel_size=2, stride=2)

            outputs = {
                "res2": res2,
                "res3": res3,
                "res4": res4,
                "res5": res5,
            }
            return {k: outputs[k] for k in self.out_features}

        def output_shape(self):
            shapes = {
                "res2": ShapeSpec(channels=self.embed_dim, stride=4),
                "res3": ShapeSpec(channels=self.embed_dim, stride=8),
                "res4": ShapeSpec(channels=self.embed_dim, stride=16),
                "res5": ShapeSpec(channels=self.embed_dim, stride=32),
            }
            return {k: shapes[k] for k in self.out_features}

    _DINO_BACKBONE_REGISTERED = True


class DinoMask2FormerSegmenter(BaseSegmenter):
    name = "dino_m2f"

    def __init__(self) -> None:
        if not DINO_M2F_CONFIG_PATH or not DINO_M2F_WEIGHTS_PATH:
            raise RuntimeError("DINO_M2F_CONFIG_PATH and DINO_M2F_WEIGHTS_PATH must be set")

        _register_dino_backbone()

        from detectron2.config import get_cfg, CfgNode as CN
        from detectron2.engine import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config
        from mask2former import add_maskformer2_config

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        if not hasattr(cfg.MODEL, "DINO_V3"):
            cfg.MODEL.DINO_V3 = CN()
        cfg.MODEL.DINO_V3.MODEL_NAME = DINO_MODEL_NAME
        cfg.MODEL.DINO_V3.WEIGHTS_PATH = DINO_WEIGHTS_PATH

        cfg.merge_from_file(DINO_M2F_CONFIG_PATH)
        cfg.MODEL.WEIGHTS = DINO_M2F_WEIGHTS_PATH
        cfg.MODEL.DEVICE = SEG_DEVICE
        cfg.MODEL.BACKBONE.NAME = "DinoV3Backbone"
        cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]

        # ensure RGB input for DINO backbone and fixed resize
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.MIN_SIZE_TEST = SEG_DINO_IMAGE_SIZE
        cfg.INPUT.MAX_SIZE_TEST = SEG_DINO_IMAGE_SIZE

        # DINOv3 uses ImageNet stats; detectron2 expects 0-255 scale
        dino_mean = [0.485, 0.456, 0.406]
        dino_std = [0.229, 0.224, 0.225]
        cfg.MODEL.PIXEL_MEAN = [m * 255.0 for m in dino_mean]
        cfg.MODEL.PIXEL_STD = [s * 255.0 for s in dino_std]

        if hasattr(cfg.MODEL, "MASK_FORMER"):
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = SEG_SCORE_THRESH

        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float, bool]:
        np_img = np.asarray(image.convert("RGB"))
        outputs = self.predictor(np_img)

        mask = None
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                if masks.size > 0:
                    mask = masks.any(axis=0)

        if mask is None and "sem_seg" in outputs:
            sem = outputs["sem_seg"].to("cpu").numpy()
            mask = sem.argmax(axis=0) > 0

        if mask is None:
            mask = np.zeros(np_img.shape[:2], dtype=bool)

        area_ratio = float(mask.mean())
        wound_present = area_ratio >= SEG_AREA_THRESHOLD
        return mask.astype(np.uint8), area_ratio, wound_present


class CustomHeadSegmenter(BaseSegmenter):
    name = "custom_head"

    def __init__(self) -> None:
        if not CUSTOM_SEGMENTER:
            raise RuntimeError(
                "CUSTOM_SEGMENTER is not set. Provide module:Class for the custom head."
            )

        module_name, _, class_name = CUSTOM_SEGMENTER.partition(":")
        if not module_name or not class_name:
            raise RuntimeError("CUSTOM_SEGMENTER must be in the form module:Class")

        module = importlib.import_module(module_name)
        segmenter_cls = getattr(module, class_name, None)
        if segmenter_cls is None:
            raise RuntimeError(f"Custom segmenter class not found: {CUSTOM_SEGMENTER}")

        self.impl = segmenter_cls()

    def predict(self, image: Image.Image) -> Tuple[np.ndarray, float, bool]:
        return self.impl.predict(image)


_SEGMENTERS: Dict[str, BaseSegmenter] = {}
_BACKEND_FACTORIES = {
    "demo": DemoSegmenter,
    "dino_m2f": DinoMask2FormerSegmenter,
    "swin_m2f": SwinMask2FormerSegmenter,
    "detectron2": SwinMask2FormerSegmenter,
    "custom_head": CustomHeadSegmenter,
}


def get_segmenter(backend: Optional[str] = None) -> BaseSegmenter:
    key = (backend or SEG_DEFAULT_BACKEND).strip().lower()
    if key not in _BACKEND_FACTORIES:
        raise ValueError(f"unknown segmentation backend: {key}")

    if key not in _SEGMENTERS:
        _SEGMENTERS[key] = _BACKEND_FACTORIES[key]()

    return _SEGMENTERS[key]


def render_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    overlay = rgb.copy()
    # red overlay
    overlay[mask > 0] = (255, 0, 0)
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)
    return Image.fromarray(blended)


def render_mask(mask: np.ndarray) -> Image.Image:
    mask_img = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_img, mode="L")
