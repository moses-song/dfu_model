from pathlib import Path
from typing import Optional
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .model import get_classifier
from .schemas import AnalysisResult, ClassificationResult, ModelCatalogResponse, ModelRunResult
from .services.model_catalog import build_model_catalog, get_model_spec
from .services.model_runner import run_model
from .services.pipeline import analyze_image
from .settings import CORS_ORIGINS, MODEL_LABELS


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="DFU Mobile Web MVP")

if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")


def _read_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image file required")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models", response_model=ModelCatalogResponse)
def list_models() -> ModelCatalogResponse:
    return ModelCatalogResponse(models=build_model_catalog())


@app.post("/classify", response_model=ClassificationResult)
def classify(file: UploadFile = File(...)) -> ClassificationResult:
    image = _read_image(file)
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


@app.post("/api/analyze", response_model=AnalysisResult)
def analyze(
    file: UploadFile = File(...),
    glucose: Optional[str] = Form(None),
    hba1c: Optional[str] = Form(None),
    memo: Optional[str] = Form(None),
) -> AnalysisResult:
    image = _read_image(file)
    return analyze_image(
        image=image,
        image_name=file.filename or "upload",
        clinical_inputs={
            "glucose": glucose or "",
            "hba1c": hba1c or "",
            "memo": memo or "",
        },
    )


@app.post("/api/models/{model_id}/run", response_model=ModelRunResult)
def run_model_endpoint(
    model_id: str,
    file: UploadFile = File(...),
) -> ModelRunResult:
    image = _read_image(file)
    try:
        get_model_spec(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        return run_model(model_id, image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"model run failed: {exc}") from exc
