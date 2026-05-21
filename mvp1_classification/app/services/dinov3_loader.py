from __future__ import annotations

from pathlib import Path
import sys
import torch

from ..settings import DINO_WEIGHTS_PATH, PROJECT_ROOT


def _ensure_repo_on_path() -> None:
    repo_dir = Path(PROJECT_ROOT) / "dinov3"
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_dinov3_vitb16_backbone(weights_path: str = DINO_WEIGHTS_PATH):
    _ensure_repo_on_path()
    from dinov3.hub.backbones import dinov3_vitb16

    model = dinov3_vitb16(pretrained=False)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
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
        model.load_state_dict(state, strict=False)
    return model
