from typing import Tuple
from PIL import Image
import numpy as np

from .settings import MODEL_PATH, MODEL_LABELS, DEFAULT_CLASS

class BaseClassifier:
    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image: Image.Image) -> Tuple[int, float]:
        raise NotImplementedError

class DummyClassifier(BaseClassifier):
    def load(self) -> None:
        # No-op; replace with model load logic.
        pass

    def predict(self, image: Image.Image) -> Tuple[int, float]:
        # Stable placeholder for UI and API wiring.
        # Returns DEFAULT_CLASS with a fake score.
        idx = MODEL_LABELS.index(DEFAULT_CLASS) if DEFAULT_CLASS in MODEL_LABELS else 0
        return idx, 0.50

_classifier: BaseClassifier | None = None

def get_classifier() -> BaseClassifier:
    global _classifier
    if _classifier is None:
        # Swap DummyClassifier to your real implementation.
        _classifier = DummyClassifier()
        _classifier.load()
    return _classifier
