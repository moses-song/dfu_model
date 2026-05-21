from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha1
from typing import Optional, Tuple
import io
import time

import numpy as np
from PIL import Image

from .pca_focus import get_pca_visualizer


@dataclass
class FeatureContext:
    image_key: str
    image_size: Tuple[int, int]
    feature_backend: str
    feature_matrix: np.ndarray
    grid_shape: Tuple[int, int]
    patch_size: int
    created_at: float
    cache_hit: bool = False


class SharedFeatureStore:
    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, FeatureContext] = OrderedDict()

    def _make_key(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return sha1(buffer.getvalue()).hexdigest()

    def get(self, image: Image.Image) -> FeatureContext:
        key = self._make_key(image)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return FeatureContext(
                image_key=cached.image_key,
                image_size=cached.image_size,
                feature_backend=cached.feature_backend,
                feature_matrix=cached.feature_matrix,
                grid_shape=cached.grid_shape,
                patch_size=cached.patch_size,
                created_at=cached.created_at,
                cache_hit=True,
            )

        visualizer = get_pca_visualizer()
        feature_backend = getattr(visualizer, "backend_name", "unknown")
        matrix, grid_shape, patch_size = visualizer.extract_feature_matrix(image)
        context = FeatureContext(
            image_key=key,
            image_size=image.size,
            feature_backend=feature_backend,
            feature_matrix=matrix,
            grid_shape=grid_shape,
            patch_size=patch_size,
            created_at=time.perf_counter(),
            cache_hit=False,
        )
        self._cache[key] = context
        self._cache.move_to_end(key)
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return context


_STORE: Optional[SharedFeatureStore] = None


def get_feature_store() -> SharedFeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = SharedFeatureStore()
    return _STORE


def get_feature_context(image: Image.Image) -> FeatureContext:
    return get_feature_store().get(image)
