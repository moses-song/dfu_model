# MVP1 Classification (pluggable model)

Goal: image upload -> model inference -> Wagner class result.
Model weights are not included; this is a stub that is easy to swap.

## Run

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@sample.jpg"
```

## Swap the model

1. Put your model file path in `MODEL_PATH` env var.
2. Implement `load()` and `predict()` in `app/model.py`.
3. Update `MODEL_LABELS` if needed.

The API response shape is stable so the frontend does not break.

## Transformers Mask2Former (beginner example)

### 1) Set Hugging Face token (do not hardcode)

PowerShell:
```powershell
$env:HUGGINGFACE_HUB_TOKEN = "YOUR_TOKEN"
```

### 2) Quick test in Python

```python
from PIL import Image
from app.transformers_mask2former import load_mask2former

runner = load_mask2former()
img = Image.open("sample.jpg")
result = runner.predict(img)

print(result["segmentation"].shape)
print(result["segments_info"])
```

## DINOv3 + Transformers Mask2Former (custom backbone)

### 1) Set Hugging Face token (do not hardcode)

PowerShell:
```powershell
$env:HUGGINGFACE_HUB_TOKEN = "YOUR_TOKEN"
```

### 2) Quick test in Python

```python
from PIL import Image
from app.dino_mask2former_transformers import load_dinov3_mask2former

runner = load_dinov3_mask2former()
img = Image.open("sample.jpg")
result = runner.predict(img)

print(result["segmentation"].shape)
print(result["segments_info"])
```
