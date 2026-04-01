# Kaggle Training: DINOv3 + Mask2Former (DFU)

This guide assumes the repo root contains:
- `Mask2formers/` (clean upstream repo)
- `Model_training/` (custom configs/tools/train script)

It uses `Model_training/tools/kaggle_train_dino_m2f.py` and:
`Model_training/configs/custom/dino_v3_mask2former_wound_instance.yaml`.

## 1) Prepare Kaggle Inputs
Upload two Kaggle Datasets (or one dataset with both):
- Foot Ulcer Segmentation Challenge dataset
- Optional pretrained weights

Example paths after attaching datasets in a Kaggle notebook:
- Dataset root: `/kaggle/input/dfu-foot-ulcer/Foot Ulcer Segmentation Challenge`
- Weights: `/kaggle/input/dfu-weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`

## 2) Environment Setup (Kaggle)
```bash
!nvidia-smi
!python -V

# Clone repo
!git clone <YOUR_REPO_URL> dfu_model

# Python deps (install while inside Mask2formers)
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

## 3) Run Training
```bash
%cd /kaggle/working/dfu_model/Model_training

python tools/kaggle_train_dino_m2f.py \
  --dataset-root "/kaggle/input/dfu-foot-ulcer/Foot Ulcer Segmentation Challenge" \
  --coco-out "/kaggle/working/dfu_annotations" \
  --output-dir "/kaggle/working/dfu_outputs/dino_m2f" \
  --model-weights "/kaggle/input/dfu-weights/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth" \
  --num-gpus 1
```

## 4) Outputs
The trained checkpoints will appear in:
```
/kaggle/working/dfu_outputs/dino_m2f
```
Download `model_final.pth` and use it as `DINO_M2F_WEIGHTS_PATH` in the FastAPI app.
