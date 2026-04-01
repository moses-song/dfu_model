# Mask2Former Wound Segmentation Training Guide (Server)

## 1) Repo Layout (Clean + Custom)
- Clean upstream repo: `.../1st_mvp/Mask2formers`
- Custom training assets: `.../1st_mvp/Model_training`
  - `train_net.py`, `dino_v3_backbone.py`
  - `configs/custom/*.yaml`
  - `datasets/register_wound.py`
  - `tools/*.py`

## 2) Dataset
- Dataset root (local):
  `C:\Users\RexSoft\Desktop\Project\당뇨발과제\Dataset\wound-segmentation_uwm-bigdata\wound-segmentation\data\Foot Ulcer Segmentation Challenge`
- Structure:
  - `train/images/*.png`
  - `train/labels/*.png`
  - `validation/images/*.png`
  - `validation/labels/*.png`
  - `test/images/*.png`
  - `test/labels/*.png`
- Label format: PNG masks with values {0,255} (binary). Mask images are RGB but only 0/255 values.

## 3) What Was Added / Modified (now under Model_training)

### A) COCO conversion script
**File:** `Model_training/tools/create_coco_instances_from_binary_masks.py`
- Converts binary masks to COCO instance segmentation format (RLE).
- One class: `wound`.
- Output files:
  - `instances_train.json`
  - `instances_validation.json`
  - `instances_test.json`

**Run example:**
```
python Model_training/tools/create_coco_instances_from_binary_masks.py \
  --dataset-root /path/to/Foot\ Ulcer\ Segmentation\ Challenge \
  --out-dir /path/to/annotations \
  --category wound
```

### B) Dataset registration
**File:** `Model_training/datasets/register_wound.py`
- Registers COCO datasets using env vars:
  - `WOUND_DATASET_ROOT`
  - `WOUND_COCO_DIR`

### C) Auto-registration hook
**File:** `Model_training/train_net.py`
- Includes a safe import hook to auto-register `wound_*` datasets at runtime:
```
try:
    from datasets.register_wound import register_wound
    register_wound()
except Exception:
    pass
```

### D) Custom training config
**File:** `Model_training/configs/custom/wound_instance_swinb.yaml`
- Base: `configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml`
- Key overrides:
  - `MODEL.SEM_SEG_HEAD.NUM_CLASSES: 1`
  - `DATASETS.TRAIN: ("wound_train",)`
  - `DATASETS.TEST: ("wound_val",)`
  - `INPUT.IMAGE_SIZE: 512`
  - `INPUT.MIN_SCALE/MAX_SCALE: 1.0`
  - `INPUT.DATASET_MAPPER_NAME: "mask_former_instance"`
  - `SOLVER.IMS_PER_BATCH: 2`
  - `SOLVER.MAX_ITER: 10000` (adjustable)
  - `SOLVER.CHECKPOINT_PERIOD: 500` (adjustable)

## 4) Swin Pretrained Weights (Required)
Mask2Former expects a Detectron2-format Swin checkpoint.

### Download Swin Base (official)
```
wget -O swin_base_patch4_window12_384_22k.pth \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
```

### Convert to Detectron2 format
```
python Mask2formers/tools/convert-pretrained-swin-model-to-d2.py \
  swin_base_patch4_window12_384_22k.pth \
  swin_base_patch4_window12_384_22k_d2.pkl
```

Update config to point to the converted `*_d2.pkl`.

## 5) Environment (Server)
Recommended (tested on Colab, should translate to server):
- Python 3.10
- PyTorch 2.2.1 + CUDA 12.1
- Detectron2 from source
- numpy 1.26.4 (avoid numpy 2.x compatibility issues)
- OpenCV 4.9.0.80

### Install outline (Linux)
```
# install PyTorch + CUDA (match your CUDA version)
# install detectron2 (source)
# pip install -r Mask2formers/requirements.txt
# build MSDeformAttn:
cd Mask2formers/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

## 6) Training
Set dataset env vars:
```
export WOUND_DATASET_ROOT=/path/to/Foot\ Ulcer\ Segmentation\ Challenge
export WOUND_COCO_DIR=/path/to/annotations
```

Make sure Mask2formers is on PYTHONPATH:
```
export PYTHONPATH=/path/to/1st_mvp/Mask2formers
```

Start training:
```
python Model_training/train_net.py \
  --config-file Model_training/configs/custom/wound_instance_swinb.yaml \
  --num-gpus 1
```

Resume training:
```
python Model_training/train_net.py \
  --config-file Model_training/configs/custom/wound_instance_swinb.yaml \
  --num-gpus 1 \
  --resume
```

## 7) Epoch Calculation
Detectron2 uses iterations. Approximate:
```
iter_per_epoch = ceil(num_images / IMS_PER_BATCH)
max_iter = iter_per_epoch * epochs
```
With 791 images and batch=2:
- 1 epoch ≈ 396 iters
- 100 epochs ≈ 39,600 iters

## 8) Notes
- If you see `Checkpoint ... not found`, ensure `MODEL.WEIGHTS` points to a valid file.
- If you see NumPy-related errors, pin numpy to 1.26.4 and OpenCV 4.9.0.80.
- For persistence across interruptions, keep `OUTPUT_DIR` on a mounted disk.
