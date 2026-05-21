from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from ..schemas import ModelCard
from ..settings import (
    DINO_M2F_CONFIG_PATH,
    DINO_M2F_WEIGHTS_PATH,
    DINO_WEIGHTS_PATH,
    DFU_MODEL_PATH,
    FOOT_MODEL_PATH,
    SINBAD_MODEL_PATH,
    WAGNER_MODEL_PATH,
)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    title: str
    summary: str
    kind: str
    backend: str
    task: str
    config_path: str = ""
    model_path: str = ""
    note: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "dinov3_backbone_pca": ModelSpec(
        id="dinov3_backbone_pca",
        title="DINOv3 Backbone PCA",
        summary="Run the original DINOv3 backbone and visualize patch-token focus with PCA and cosine maps.",
        kind="visualization",
        backend="dinov3_backbone",
        task="backbone",
        model_path=DINO_WEIGHTS_PATH,
        note="Uses the original DINOv3 ViT-B/16 pretrained checkpoint.",
    ),
    "dinov3_fastinst_d3_segmentation": ModelSpec(
        id="dinov3_fastinst_d3_segmentation",
        title="DINOv3 + fastinst_D3",
        summary="Compare wound segmentation output using the DINOv3-based segmentation path.",
        kind="segmentation",
        backend="dino_m2f",
        task="segmentation",
        config_path=DINO_M2F_CONFIG_PATH,
        model_path=DINO_M2F_WEIGHTS_PATH,
        note="The current backend is wired to the DINOv3 segmentation adapter. Replace its weights/config with fastinst_D3 artifacts when ready.",
    ),
    "dinov3_linear_foot": ModelSpec(
        id="dinov3_linear_foot",
        title="DINOv3 + Linear Head (Foot / Non-foot)",
        summary="Binary classification to check whether the uploaded image is a foot image.",
        kind="classification",
        backend="classifier",
        task="foot",
        model_path=FOOT_MODEL_PATH,
    ),
    "dinov3_linear_dfu": ModelSpec(
        id="dinov3_linear_dfu",
        title="DINOv3 + Linear Head (DFU / Other wound)",
        summary="Binary classification for DFU versus non-DFU wound images.",
        kind="classification",
        backend="classifier",
        task="dfu",
        model_path=DFU_MODEL_PATH,
    ),
    "dinov3_linear_wagner": ModelSpec(
        id="dinov3_linear_wagner",
        title="DINOv3 + Linear Head (Wagner 0-5)",
        summary="Classify the Wagner grade from 0 to 5.",
        kind="classification",
        backend="classifier",
        task="wagner",
        model_path=WAGNER_MODEL_PATH,
    ),
    "dinov3_linear_sinbad": ModelSpec(
        id="dinov3_linear_sinbad",
        title="DINOv3 + Linear Head (SINBAD)",
        summary="Show SINBAD-related output and reserve space for S/I/N/B/A/D component scores.",
        kind="classification",
        backend="classifier",
        task="sinbad",
        model_path=SINBAD_MODEL_PATH,
        note="Current adapter is single-head. Replace with a multi-label head to emit per-domain S/I/N/B/A/D scores.",
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model id: {model_id}") from exc


def iter_model_specs() -> Iterable[ModelSpec]:
    return MODEL_SPECS.values()


def build_model_catalog() -> List[ModelCard]:
    catalog: List[ModelCard] = []
    for spec in iter_model_specs():
        model_path = Path(spec.model_path) if spec.model_path else None
        weights_found = bool(model_path and model_path.exists())
        catalog.append(
            ModelCard(
                id=spec.id,
                title=spec.title,
                summary=spec.summary,
                kind=spec.kind,
                backend=spec.backend,
                task=spec.task,
                config_path=spec.config_path,
                model_path=spec.model_path,
                weights_found=weights_found,
                enabled=True,
                note=spec.note,
            )
        )
    return catalog
