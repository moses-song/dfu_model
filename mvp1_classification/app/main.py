from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from .model import get_classifier
from .schemas import ClassificationResult
from .settings import MODEL_LABELS

app = FastAPI(title="DFU MVP1 Classification")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/classify", response_model=ClassificationResult)
def classify(file: UploadFile = File(...)) -> ClassificationResult:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file required")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    classifier = get_classifier()
    class_index, score = classifier.predict(image)

    if class_index < 0 or class_index >= len(MODEL_LABELS):
        raise HTTPException(status_code=500, detail="model returned invalid class")

    return ClassificationResult(
        class_label=MODEL_LABELS[class_index],
        class_index=class_index,
        score=float(score),
        labels=MODEL_LABELS,
    )
