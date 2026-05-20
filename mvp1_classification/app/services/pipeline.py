from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional
import uuid

from PIL import Image

from ..image_utils import image_to_data_url
from ..schemas import (
    AnalysisResult,
    ImageArtifact,
    PredictionResult,
    SegmentationResult,
)
from ..settings import SEG_CONFIG_PATH, SEG_WEIGHTS_PATH
from .classifier import Prediction, predict_task
from .segmentation import get_segmenter, render_mask, render_overlay


def _prediction_schema(prediction: Prediction, status: str = "completed") -> PredictionResult:
    payload = asdict(prediction)
    payload["labels"] = list(prediction.labels)
    payload["status"] = status
    return PredictionResult(**payload)


def _should_continue_foot(prediction: Prediction) -> bool:
    return prediction.class_label.lower() in {"foot", "foot_image", "valid_foot"}


def _is_dfu(prediction: Prediction) -> bool:
    return prediction.class_label.lower() in {"dfu", "diabetic_foot_ulcer"}


def analyze_image(
    image: Image.Image,
    image_name: str,
    clinical_inputs: Optional[Dict[str, str]] = None,
) -> AnalysisResult:
    rgb = image.convert("RGB")
    clinical = {
        key: value
        for key, value in (clinical_inputs or {}).items()
        if value is not None and str(value).strip()
    }

    foot_prediction = predict_task("foot", rgb)
    foot_result = _prediction_schema(foot_prediction)

    segmenter = get_segmenter()
    mask, area_ratio, wound_present = segmenter.predict(rgb)
    overlay = render_overlay(rgb, mask)
    mask_image = render_mask(mask)

    segmentation = SegmentationResult(
        backend=segmenter.name,
        status="completed",
        wound_present=wound_present,
        area_ratio=area_ratio,
        weights_found=bool(SEG_WEIGHTS_PATH) and Path(SEG_WEIGHTS_PATH).exists(),
        config_path=SEG_CONFIG_PATH,
        weights_path=SEG_WEIGHTS_PATH,
        original=ImageArtifact(label="original", data_url=image_to_data_url(rgb)),
        overlay=ImageArtifact(label="overlay", data_url=image_to_data_url(overlay)),
        mask=ImageArtifact(label="binary_mask", data_url=image_to_data_url(mask_image)),
        note=(
            "demo segmentation; switch SEG_DEFAULT_BACKEND after trained weights are ready"
            if segmenter.name == "demo"
            else ""
        ),
    )

    foot_ok = _should_continue_foot(foot_prediction)
    dfu_result = None
    wagner_result = None
    sinbad_result = None
    next_action = (
        "발 이미지와 상처 segmentation 결과를 확인했습니다. "
        "DFU 여부와 grade/score 분류 결과를 함께 검토하세요."
    )

    if not foot_ok:
        next_action = (
            "발 이미지로 판단되지 않았습니다. 발 전체 또는 상처 주변이 잘 보이도록 "
            "다시 촬영한 이미지를 업로드하세요."
        )
    elif not wound_present:
        next_action = (
            "현재 이미지에서 상처 영역이 기준 이상으로 감지되지 않았습니다. "
            "임상적으로 의심되는 상처가 있다면 더 가까운 초점의 이미지를 추가로 확인하세요."
        )
    else:
        dfu_prediction = predict_task("dfu", rgb)
        dfu_result = _prediction_schema(dfu_prediction)

        if _is_dfu(dfu_prediction):
            wagner_result = _prediction_schema(predict_task("wagner", rgb))
            sinbad_result = _prediction_schema(predict_task("sinbad", rgb))
        else:
            next_action = (
                "상처는 감지되었지만 DFU로 분류되지는 않았습니다. "
                "비당뇨성 상처 분기 또는 추적 관찰 워크플로로 확장할 수 있습니다."
            )

    if clinical and wagner_result and sinbad_result:
        next_action = (
            "이미지 분석과 입력된 임상 정보를 함께 수집했습니다. "
            "다음 단계에서는 멀티모달 모델 또는 RAG 기반 위험도 산출로 grade/score 결과를 보정할 수 있습니다."
        )

    return AnalysisResult(
        request_id=str(uuid.uuid4()),
        image_name=image_name,
        image_width=rgb.width,
        image_height=rgb.height,
        foot=foot_result,
        segmentation=segmentation,
        dfu=dfu_result,
        wagner=wagner_result,
        sinbad=sinbad_result,
        clinical_inputs=clinical,
        next_action=next_action,
        disclaimer=(
            "본 결과는 연구/개발용 보조 정보입니다. 실제 진단, 치료, 내원 주기 결정에는 "
            "의료진 판단과 별도 검증이 필요합니다."
        ),
    )
