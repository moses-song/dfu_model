from typing import Dict, List, Optional

from pydantic import BaseModel


class ClassificationResult(BaseModel):
    class_label: str
    class_index: int
    score: float
    labels: List[str]


class PredictionResult(BaseModel):
    task: str
    class_label: str
    class_index: int
    score: float
    labels: List[str]
    backend: str
    model_path: str
    weights_found: bool
    status: str = "completed"
    note: str = ""


class ImageArtifact(BaseModel):
    label: str
    data_url: str


class SegmentationResult(BaseModel):
    backend: str
    status: str
    wound_present: bool
    area_ratio: float
    weights_found: bool
    config_path: str
    weights_path: str
    original: ImageArtifact
    overlay: ImageArtifact
    mask: ImageArtifact
    note: str = ""


class AnalysisResult(BaseModel):
    request_id: str
    image_name: str
    image_width: int
    image_height: int
    foot: PredictionResult
    segmentation: SegmentationResult
    dfu: Optional[PredictionResult]
    wagner: Optional[PredictionResult]
    sinbad: Optional[PredictionResult]
    clinical_inputs: Dict[str, str]
    next_action: str
    disclaimer: str
