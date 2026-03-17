from pydantic import BaseModel
from typing import List

class ClassificationResult(BaseModel):
    class_label: str
    class_index: int
    score: float
    labels: List[str]
