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


class MetricItem(BaseModel):
    label: str
    value: str


class EvalMetric(BaseModel):
    label: str
    value: Optional[float] = None
    display_value: str = ""
    available: bool = False
    note: str = ""


class ComponentScore(BaseModel):
    label: str
    score: Optional[int] = None
    note: str = ""


class ModelCard(BaseModel):
    id: str
    title: str
    summary: str
    kind: str
    backend: str
    task: str
    config_path: str = ""
    model_path: str = ""
    weights_found: bool
    enabled: bool = True
    note: str = ""


class ModelRunResult(BaseModel):
    model: ModelCard
    status: str
    primary_label: str = ""
    score: Optional[float] = None
    metrics: List[MetricItem] = []
    eval_metrics: List[EvalMetric] = []
    artifacts: List[ImageArtifact] = []
    component_scores: List[ComponentScore] = []
    detections: List[Dict[str, str]] = []
    raw_outputs: Dict[str, str] = {}
    timing_ms: float = 0.0
    fps: float = 0.0
    feature_backend: str = ""
    feature_cache_hit: bool = False
    note: str = ""


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


class ModelCatalogResponse(BaseModel):
    models: List[ModelCard]
