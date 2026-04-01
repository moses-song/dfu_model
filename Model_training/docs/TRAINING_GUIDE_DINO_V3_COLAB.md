# DINOv3 + Mask2Former Training (Colab/Kaggle)

This guide assumes the repo root contains:
- `Mask2formers/` (clean upstream repo)
- `Model_training/` (custom configs/tools/train script)

## 0) Summary
- DFU wound segmentation
- Labels are binary PNG (0/255)
- Single class: `wound`

## 1) GPU Check
### Colab
```bash
!nvidia-smi
!python -V
```

### Kaggle
- Notebook Settings -> GPU = On

## 2) Repo + Dependencies
```bash
# Clone repo
!git clone <YOUR_REPO_URL> dfu_model

# Install deps (inside Mask2formers)
%cd dfu_model/Mask2formers
!pip install -U pip
!pip install "numpy==1.26.4" "opencv-python==4.9.0.80"
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install -r requirements.txt
!pip install timm

# Build MSDeformAttn
%cd mask2former/modeling/pixel_decoder/ops
!sh make.sh
%cd ../../../
```

## 3) Dataset (COCO conversion)
### Colab
```bash
from google.colab import drive
drive.mount('/content/drive')
```

### Kaggle
Attach dataset in the notebook

### Convert to COCO
```bash
%cd /content/dfu_model/Model_training

export WOUND_DATASET_ROOT="/content/drive/MyDrive/DFU/Foot Ulcer Segmentation Challenge"
export WOUND_COCO_DIR="/content/drive/MyDrive/DFU/annotations"

python tools/create_coco_instances_from_binary_masks.py \
  --dataset-root "$WOUND_DATASET_ROOT" \
  --out-dir "$WOUND_COCO_DIR" \
  --category wound
```

## 4) DINOv3 Backbone Weights
Example:
```bash
export DINO_V3_WEIGHTS="/content/drive/MyDrive/DFU/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
```

## 5) Training
Ensure Mask2formers is on PYTHONPATH:
```bash
export PYTHONPATH=/content/dfu_model/Mask2formers
```

Run training:
```bash
%cd /content/dfu_model/Model_training

python train_net.py \
  --config-file configs/custom/dino_v3_mask2former_wound_instance.yaml \
  --num-gpus 1 \
  OUTPUT_DIR /content/drive/MyDrive/DFU/outputs/dino_m2f \
  MODEL.DINO_V3.WEIGHTS_PATH "$DINO_V3_WEIGHTS"
```

### Resume
```bash
python train_net.py \
  --config-file configs/custom/dino_v3_mask2former_wound_instance.yaml \
  --num-gpus 1 \
  --resume \
  OUTPUT_DIR /content/drive/MyDrive/DFU/outputs/dino_m2f \
  MODEL.DINO_V3.WEIGHTS_PATH "$DINO_V3_WEIGHTS"
```

## 6) Outputs
Example:
```
/content/drive/MyDrive/DFU/outputs/dino_m2f/model_final.pth
```
Use this as `DINO_M2F_WEIGHTS_PATH` in the FastAPI app.

## 7) Notes
- Tune `SOLVER.MAX_ITER`, `IMS_PER_BATCH`, `INPUT.IMAGE_SIZE` inside
  `configs/custom/dino_v3_mask2former_wound_instance.yaml`.
- Iteration 계산:
  `iter_per_epoch = ceil(num_images / IMS_PER_BATCH)`
