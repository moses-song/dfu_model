from __future__ import annotations

from pathlib import Path
import sys

from ..settings import DINO_WEIGHTS_PATH, PROJECT_ROOT


def _ensure_repo_on_path() -> None:
    repo_dir = Path(PROJECT_ROOT) / "dinov3"
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_dinov3_vitb16_backbone(weights_path: str = DINO_WEIGHTS_PATH):
    _ensure_repo_on_path()
    from dinov3.hub.backbones import dinov3_vitb16

    if weights_path:
        return dinov3_vitb16(pretrained=True, weights=weights_path)
    return dinov3_vitb16(pretrained=True)
