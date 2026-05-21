from __future__ import annotations

from pathlib import Path
import time

from PIL import Image

from ..image_utils import image_to_data_url
from ..schemas import (
    ComponentScore,
    EvalMetric,
    ImageArtifact,
    MetricItem,
    ModelCard,
    ModelRunResult,
)
from .classifier import predict_task_with_features
from .feature_store import get_feature_context
from .model_catalog import get_model_spec
from .pca_focus import get_pca_visualizer
from .segmentation import get_segmenter, mask_to_bbox, render_mask, render_overlay


def _model_card(model_id: str) -> ModelCard:
    spec = get_model_spec(model_id)
    model_path = Path(spec.model_path) if spec.model_path else None
    return ModelCard(
        id=spec.id,
        title=spec.title,
        summary=spec.summary,
        kind=spec.kind,
        backend=spec.backend,
        task=spec.task,
        config_path=spec.config_path,
        model_path=spec.model_path,
        weights_found=bool(model_path and model_path.exists()),
        enabled=True,
        note=spec.note,
    )


def _artifacts(items: list[tuple[str, Image.Image]]) -> list[ImageArtifact]:
    return [ImageArtifact(label=label, data_url=image_to_data_url(image)) for label, image in items]


def _timing(start_time: float) -> tuple[float, float]:
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
    return round(elapsed_ms, 2), round(fps, 2)


def _base_eval_metrics() -> list[EvalMetric]:
    note = "Ground truth annotation is not provided in single-image inference mode."
    return [
        EvalMetric(label="DICE", display_value="N/A", available=False, note=note),
        EvalMetric(label="F1", display_value="N/A", available=False, note=note),
        EvalMetric(label="Precision", display_value="N/A", available=False, note=note),
        EvalMetric(label="Recall", display_value="N/A", available=False, note=note),
    ]


def _bbox_strings(bbox: dict | None) -> dict[str, str]:
    if not bbox:
        return {
            "bbox_xyxy": "-",
            "bbox_xywh": "-",
            "bbox_center": "-",
        }
    return {
        "bbox_xyxy": f"({bbox['x_min']}, {bbox['y_min']}) - ({bbox['x_max']}, {bbox['y_max']})",
        "bbox_xywh": f"x={bbox['x_min']}, y={bbox['y_min']}, w={bbox['width']}, h={bbox['height']}",
        "bbox_center": f"({bbox['center_x']}, {bbox['center_y']})",
    }


def run_model(model_id: str, image: Image.Image) -> ModelRunResult:
    rgb = image.convert("RGB")
    card = _model_card(model_id)
    feature_context = get_feature_context(rgb)

    if model_id == "dinov3_backbone_pca":
        start_time = time.perf_counter()
        visualizer = get_pca_visualizer()
        pca_map, pca_overlay = visualizer.visualize(rgb)
        cosine_map, cosine_overlay = visualizer.cosine_similarity(rgb)
        timing_ms, fps = _timing(start_time)
        backend_name = getattr(visualizer, "backend_name", "unknown")
        metrics = [
            MetricItem(label="Inference result", value="feature focus visualization"),
            MetricItem(label="Backbone", value="ViT-B/16"),
            MetricItem(label="Runtime", value=backend_name),
            MetricItem(label="Feature cache", value="hit" if feature_context.cache_hit else "miss"),
            MetricItem(label="Feature grid", value=f"{feature_context.grid_shape[0]}x{feature_context.grid_shape[1]}"),
            MetricItem(label="Inference time", value=f"{timing_ms} ms"),
            MetricItem(label="FPS", value=str(fps)),
        ]
        return ModelRunResult(
            model=card,
            status="completed",
            primary_label="focus visualization",
            metrics=metrics,
            eval_metrics=_base_eval_metrics(),
            artifacts=_artifacts(
                [
                    ("original", rgb),
                    ("pca_map", pca_map),
                    ("pca_overlay", pca_overlay),
                    ("cosine_map", cosine_map),
                    ("cosine_overlay", cosine_overlay),
                ]
            ),
            raw_outputs={
                "feature_backend": backend_name,
                "feature_vector_shape": f"{feature_context.feature_matrix.shape}",
            },
            timing_ms=timing_ms,
            fps=fps,
            feature_backend=feature_context.feature_backend,
            feature_cache_hit=feature_context.cache_hit,
            note=(
                "Bright regions indicate stronger patch-token variation. Compare PCA and cosine maps together."
                if backend_name == "dinov3_vitb16"
                else "The local DINOv3 package could not be loaded in this Python runtime, so the visualization fell back to image-patch PCA."
            ),
        )

    if model_id == "dinov3_fastinst_d3_segmentation":
        start_time = time.perf_counter()
        note = card.note
        backend = "dino_m2f"
        try:
            segmenter = get_segmenter("dino_m2f")
        except Exception:
            segmenter = get_segmenter("demo")
            backend = segmenter.name
            note = (
                f"{card.note} Fine-tuned segmentation weights are not available yet, "
                "so the button fell back to demo segmentation."
            ).strip()
        mask, area_ratio, wound_present = segmenter.predict(rgb)
        overlay = render_overlay(rgb, mask)
        mask_image = render_mask(mask)
        bbox = mask_to_bbox(mask)
        bbox_strings = _bbox_strings(bbox)
        timing_ms, fps = _timing(start_time)
        detection_label = "wound detected" if wound_present else "no wound detected"
        return ModelRunResult(
            model=card,
            status="completed",
            primary_label=detection_label,
            metrics=[
                MetricItem(label="Inference result", value=detection_label),
                MetricItem(label="Segmentation area", value=f"{area_ratio * 100:.2f}%"),
                MetricItem(label="Bounding box", value=bbox_strings["bbox_xywh"]),
                MetricItem(label="Inference time", value=f"{timing_ms} ms"),
                MetricItem(label="FPS", value=str(fps)),
                MetricItem(label="Feature backend", value=feature_context.feature_backend),
                MetricItem(label="Feature cache", value="hit" if feature_context.cache_hit else "miss"),
                MetricItem(label="Weights", value="found" if card.weights_found else "missing"),
                MetricItem(label="Backend", value=backend),
            ],
            eval_metrics=_base_eval_metrics(),
            artifacts=_artifacts(
                [
                    ("original", rgb),
                    ("overlay", overlay),
                    ("mask", mask_image),
                ]
            ),
            detections=[
                {
                    "label": "wound_region",
                    "bbox_xyxy": bbox_strings["bbox_xyxy"],
                    "bbox_xywh": bbox_strings["bbox_xywh"],
                    "bbox_center": bbox_strings["bbox_center"],
                    "mask_detected": "true" if wound_present else "false",
                }
            ],
            raw_outputs={
                "wound_present": str(wound_present).lower(),
                "area_ratio": f"{area_ratio:.6f}",
                "bbox_xyxy": bbox_strings["bbox_xyxy"],
            },
            timing_ms=timing_ms,
            fps=fps,
            feature_backend=feature_context.feature_backend,
            feature_cache_hit=feature_context.cache_hit,
            note=note,
        )

    start_time = time.perf_counter()
    prediction = predict_task_with_features(card.task, rgb, feature_context)
    timing_ms, fps = _timing(start_time)
    normal_skin_hint = "normal_skin" if card.task == "dfu" and prediction.class_label == "other_injury" else ""
    metrics = [
        MetricItem(label="Inference result", value=prediction.class_label),
        MetricItem(label="Confidence", value=f"{prediction.score * 100:.2f}%"),
        MetricItem(label="Inference time", value=f"{timing_ms} ms"),
        MetricItem(label="FPS", value=str(fps)),
        MetricItem(label="Feature backend", value=feature_context.feature_backend),
        MetricItem(label="Feature cache", value="hit" if feature_context.cache_hit else "miss"),
        MetricItem(label="Feature shape", value=f"{feature_context.feature_matrix.shape[0]}x{feature_context.feature_matrix.shape[1]}"),
        MetricItem(label="Weights", value="found" if prediction.weights_found else "missing"),
        MetricItem(label="Backend", value=prediction.backend),
    ]
    component_scores: list[ComponentScore] = []
    note = prediction.note or card.note

    if model_id == "dinov3_linear_sinbad":
        for label in ["S", "I", "N", "B", "A", "D"]:
            component_scores.append(
                ComponentScore(
                    label=label,
                    score=None,
                    note="Pending multi-label SINBAD head integration.",
                )
            )

    raw_outputs = {
        "class_label": prediction.class_label,
        "class_index": str(prediction.class_index),
        "score": f"{prediction.score:.6f}",
    }
    if normal_skin_hint:
        raw_outputs["normal_skin_hint"] = normal_skin_hint

    detections = [
        {
            "label": prediction.class_label,
            "status": "detected" if prediction.class_label not in {"not_foot"} else "rejected",
            "normal_skin_hint": normal_skin_hint or "-",
        }
    ]

    return ModelRunResult(
        model=card,
        status="completed",
        primary_label=prediction.class_label,
        score=prediction.score,
        metrics=metrics,
        eval_metrics=_base_eval_metrics(),
        artifacts=_artifacts([("original", rgb)]),
        component_scores=component_scores,
        detections=detections,
        raw_outputs=raw_outputs,
        timing_ms=timing_ms,
        fps=fps,
        feature_backend=feature_context.feature_backend,
        feature_cache_hit=feature_context.cache_hit,
        note=note,
    )
