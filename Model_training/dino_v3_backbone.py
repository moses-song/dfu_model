from __future__ import annotations

from typing import Dict


_REGISTERED = False


def register_dino_v3_backbone() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import torch
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
                raise RuntimeError("token count mismatch for DINOv3 backbone")

            feat = tokens.transpose(1, 2).reshape(b, c, h, w)
            res4 = feat
            res3 = F.interpolate(res4, scale_factor=2.0, mode="bilinear", align_corners=False)
            res2 = F.interpolate(res4, scale_factor=4.0, mode="bilinear", align_corners=False)
            res5 = F.avg_pool2d(res4, kernel_size=2, stride=2)

            outputs: Dict[str, torch.Tensor] = {
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

    _REGISTERED = True


register_dino_v3_backbone()
