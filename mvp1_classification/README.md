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

## Internal deployment

### Recommended shape

For "everyone inside the company can access it", the simplest path is:

1. Run this app on one internal server or VM.
2. Bind it to `0.0.0.0:8000`.
3. Expose that server on the company network with a private IP or internal DNS such as `dfu-mvp.company.local`.
4. If needed, put Nginx or the company reverse proxy in front of it.

Current app structure fits this well because one FastAPI process serves both:

- static web UI
- inference APIs

### Docker build

```bash
cd ..
docker build -f mvp1_classification/Dockerfile -t dfu-mvp:latest .
```

### Docker run

```bash
docker run --rm -p 8000:8000 \
  -e DFU_PARAMETERS_DIR=/srv/parameters \
  -v ./parameters:/srv/parameters:ro \
  dfu-mvp:latest
```

### Docker Compose

Recommended for internal deployment because the run configuration is fixed in one file.

```bash
cd ..
docker compose up -d --build
```

Stop:

```bash
docker compose down
```

Logs:

```bash
docker compose logs -f web
```

Then users inside the company can access:

- `http://<internal-server-ip>:8000`

or, after internal DNS is connected:

- `http://dfu-mvp.company.local:8000`

### Production-style internal deployment

Recommended internal production path:

1. Internal Linux VM or Windows Server
2. `docker compose` based app deployment
3. Internal reverse proxy or Nginx
4. Internal DNS name
5. Firewall rule allowing the company subnet to the service port

### Notes

- If real model inference is enabled, CPU-only deployment may be slow. Use a GPU server if inference latency matters.
- `parameters/` is not baked into the image. It is mounted from the host so large model weights do not make image builds slow.
- The current app has no DB yet, so uploads and results are not persisted.
- Once DB is added, the Web Server should remain the only component that talks directly to the DB.

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
