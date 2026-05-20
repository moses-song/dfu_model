from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import importlib

from PIL import Image

from ..settings import (
    CLASSIFIER_BACKEND,
    CUSTOM_CLASSIFIER,
    DEFAULT_CLASS,
    DFU_DEFAULT_CLASS,
    DFU_LABELS,
    DFU_MODEL_PATH,
    FOOT_DEFAULT_CLASS,
    FOOT_LABELS,
    FOOT_MODEL_PATH,
    MODEL_LABELS,
    MODEL_PATH,
    SINBAD_DEFAULT_CLASS,
    SINBAD_LABELS,
    SINBAD_MODEL_PATH,
    WAGNER_DEFAULT_CLASS,
    WAGNER_LABELS,
    WAGNER_MODEL_PATH,
)


@dataclass(frozen=True)
class ClassifierTask:
    name: str
    labels: Sequence[str]
    default_class: str
    model_path: str


@dataclass(frozen=True)
class Prediction:
    task: str
    class_label: str
    class_index: int
    score: float
    labels: Sequence[str]
    backend: str
    model_path: str
    weights_found: bool
    status: str = "completed"
    note: str = ""


TASKS: Dict[str, ClassifierTask] = {
    "legacy": ClassifierTask("legacy", MODEL_LABELS, DEFAULT_CLASS, MODEL_PATH),
    "foot": ClassifierTask("foot", FOOT_LABELS, FOOT_DEFAULT_CLASS, FOOT_MODEL_PATH),
    "dfu": ClassifierTask("dfu", DFU_LABELS, DFU_DEFAULT_CLASS, DFU_MODEL_PATH),
    "wagner": ClassifierTask(
        "wagner",
        WAGNER_LABELS,
        WAGNER_DEFAULT_CLASS,
        WAGNER_MODEL_PATH,
    ),
    "sinbad": ClassifierTask(
        "sinbad",
        SINBAD_LABELS,
        SINBAD_DEFAULT_CLASS,
        SINBAD_MODEL_PATH,
    ),
}


class BaseClassifier:
    backend = "base"

    def __init__(self, task: ClassifierTask) -> None:
        self.task = task

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, image: Image.Image) -> Tuple[int, float]:
        raise NotImplementedError

    def predict_full(self, image: Image.Image) -> Prediction:
        class_index, score = self.predict(image)
        labels = list(self.task.labels)
        if class_index < 0 or class_index >= len(labels):
            class_index = 0
            score = 0.0
        return Prediction(
            task=self.task.name,
            class_label=labels[class_index],
            class_index=class_index,
            score=float(score),
            labels=labels,
            backend=self.backend,
            model_path=self.task.model_path,
            weights_found=bool(self.task.model_path)
            and Path(self.task.model_path).exists(),
            note="demo classifier; replace backend before clinical use"
            if self.backend == "dummy"
            else "",
        )


class DummyClassifier(BaseClassifier):
    backend = "dummy"

    def load(self) -> None:
        pass

    def predict(self, image: Image.Image) -> Tuple[int, float]:
        labels = list(self.task.labels)
        if not labels:
            return 0, 0.0
        idx = labels.index(self.task.default_class) if self.task.default_class in labels else 0
        return idx, 0.50


class CustomClassifier(BaseClassifier):
    backend = "custom"

    def load(self) -> None:
        if not CUSTOM_CLASSIFIER:
            raise RuntimeError("CUSTOM_CLASSIFIER is not set")

        module_name, _, class_name = CUSTOM_CLASSIFIER.partition(":")
        if not module_name or not class_name:
            raise RuntimeError("CUSTOM_CLASSIFIER must be in the form module:Class")

        module = importlib.import_module(module_name)
        classifier_cls = getattr(module, class_name, None)
        if classifier_cls is None:
            raise RuntimeError(f"Custom classifier class not found: {CUSTOM_CLASSIFIER}")

        self.impl = classifier_cls(self.task)
        self.impl.load()

    def predict(self, image: Image.Image) -> Tuple[int, float]:
        return self.impl.predict(image)


_CLASSIFIERS: Dict[str, BaseClassifier] = {}
_BACKEND_FACTORIES = {
    "dummy": DummyClassifier,
    "custom": CustomClassifier,
}


def get_classifier(task: str = "legacy") -> BaseClassifier:
    task_key = task.strip().lower()
    if task_key not in TASKS:
        raise ValueError(f"unknown classifier task: {task}")

    backend_key = CLASSIFIER_BACKEND
    if backend_key not in _BACKEND_FACTORIES:
        raise ValueError(f"unknown classifier backend: {backend_key}")

    cache_key = f"{backend_key}:{task_key}"
    if cache_key not in _CLASSIFIERS:
        classifier = _BACKEND_FACTORIES[backend_key](TASKS[task_key])
        classifier.load()
        _CLASSIFIERS[cache_key] = classifier

    return _CLASSIFIERS[cache_key]


def predict_task(task: str, image: Image.Image) -> Prediction:
    return get_classifier(task).predict_full(image)
