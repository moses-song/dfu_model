# === Colab 안내 ===
# 1) Colab에서 Google Drive 마운트
#    from google.colab import drive
#    drive.mount('/content/drive')
#
# 2) 이 파일과 dinov3 폴더를 같은 위치에 두거나, 아래 REPO_ROOT/DINOV3_ROOT를 맞춰 주세요.
#    예) REPO_ROOT = Path("/content/drive/MyDrive/1st_mvp")
#
# 3) 아래 기본 경로를 Colab 환경 경로로 수정하세요.
#    - DEFAULT_WEIGHTS
#    - DEFAULT_IMAGE
#
# 4) GPU 런타임 사용 (Runtime > Change runtime type > GPU)
# ==================
#쓰는 방법
# python C:\Users\RexSoft\Desktop\Project\당뇨발과제\PM업무_송모세\1st_mvp\dinov3_m2f_infer_single.py ^
#   --weights "C:\Users\RexSoft\Desktop\Project\당뇨발과제\PM업무_송모세\1st_mvp\parameters\Fine-tuned_pth\260406\model_final_from_dinov3_vit7b16_ade20k_m2f_head-bf307cb1_260406.pth" ^
#   --image "C:\Users\RexSoft\Desktop\Project\당뇨발과제\Dataset\wound-segmentation_uwm-bigdata\wound-segmentation\data\Foot Ulcer Segmentation Challenge\test\images\1011.png" ^
#   --output-dir "C:\Users\RexSoft\Desktop\Project\당뇨발과제\PM업무_송모세\1st_mvp\outputs"

import argparse
import sys
from functools import partial
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parent
DINOV3_ROOT = REPO_ROOT / "dinov3"
if str(DINOV3_ROOT) not in sys.path:
    sys.path.insert(0, str(DINOV3_ROOT))


from dinov3.hub.backbones import dinov3_vit7b16  # noqa: E402
from dinov3.eval.segmentation.inference import make_inference  # noqa: E402
from dinov3.eval.segmentation.models import build_segmentation_decoder  # noqa: E402
from dinov3.eval.segmentation.transforms import make_segmentation_eval_transforms  # noqa: E402


DEFAULT_WEIGHTS = (
    r"C:\Users\RexSoft\Desktop\Project\당뇨발과제\PM업무_송모세\1st_mvp\parameters"
    r"\Fine-tuned_pth\260406\model_final_from_dinov3_vit7b16_ade20k_m2f_head-bf307cb1_260406.pth"
)
DEFAULT_IMAGE = (
    r"C:\Users\RexSoft\Desktop\Project\당뇨발과제\Dataset\wound-segmentation_uwm-bigdata"
    r"\wound-segmentation\data\Foot Ulcer Segmentation Challenge\test\images\1011.png"
)


def _render_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    overlay = rgb.copy()
    overlay[mask > 0] = (255, 0, 0)
    blended = (0.6 * rgb + 0.4 * overlay).astype(np.uint8)
    return Image.fromarray(blended)


def _load_model(
    weights_path: str,
    device: torch.device,
    num_classes: int = 2,
    hidden_dim: int = 2048,
) -> torch.nn.Module:
    backbone = dinov3_vit7b16(pretrained=False)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = build_segmentation_decoder(
        backbone_model=backbone,
        decoder_type="m2f",
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        autocast_dtype=autocast_dtype,
    )

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (showing up to 10) -> {missing[:10]}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (showing up to 10) -> {unexpected[:10]}")

    model.to(device)
    model.eval()
    return model


def _prepare_images(
    image: Image.Image,
    img_size: int,
    inference_mode: str,
    use_tta: bool,
    tta_ratios: list[float],
) -> list[torch.Tensor]:
    dummy_label = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
    transforms = make_segmentation_eval_transforms(
        img_size=img_size,
        inference_mode=inference_mode,
        use_tta=use_tta,
        tta_ratios=tta_ratios,
    )
    img_list, _ = transforms(image, dummy_label)
    if not isinstance(img_list, list):
        img_list = [img_list]
    return img_list


def run_inference(
    weights_path: str,
    image_path: str,
    output_dir: str,
    num_classes: int,
    img_size: int,
    inference_mode: str,
    crop_size: int,
    stride: int,
    use_tta: bool,
    tta_ratios: list[float],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DINOv3-7B inference. No CUDA device found.")

    device = torch.device("cuda")
    model = _load_model(weights_path, device=device, num_classes=num_classes)

    image = Image.open(image_path).convert("RGB")
    h, w = image.height, image.width
    img_list = _prepare_images(
        image=image,
        img_size=img_size,
        inference_mode=inference_mode,
        use_tta=use_tta,
        tta_ratios=tta_ratios,
    )

    aggregated = torch.zeros(1, num_classes, h, w, device=device)
    softmax_fn = partial(torch.nn.functional.softmax, dim=1)
    for idx, img_tensor in enumerate(img_list):
        x = img_tensor.unsqueeze(0).to(device)
        pred = make_inference(
            x,
            model,
            inference_mode=inference_mode,
            decoder_head_type="m2f",
            rescale_to=(h, w),
            n_output_channels=num_classes,
            crop_size=(crop_size, crop_size) if inference_mode == "slide" else None,
            stride=(stride, stride) if inference_mode == "slide" else None,
            apply_horizontal_flip=(use_tta and idx >= len(img_list) / 2),
            output_activation=softmax_fn,
        )
        aggregated += pred

    pred_mask = (aggregated / len(img_list)).argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    mask_img = Image.fromarray(pred_mask * 255, mode="L")
    overlay_img = _render_overlay(image, pred_mask)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    mask_path = out_dir / f"{stem}_mask.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    mask_img.save(mask_path)
    overlay_img.save(overlay_path)

    print(f"Saved mask: {mask_path}")
    print(f"Saved overlay: {overlay_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 + Mask2Former adapter inference (single image).")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    default_output = str(REPO_ROOT / "outputs" / datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=896)
    parser.add_argument("--inference-mode", choices=["whole", "slide"], default="slide")
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-ratios", type=float, nargs="+", default=[0.9, 0.95, 1.0, 1.05, 1.1])
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_inference(
        weights_path=args.weights,
        image_path=args.image,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        img_size=args.img_size,
        inference_mode=args.inference_mode,
        crop_size=args.crop_size,
        stride=args.stride,
        use_tta=args.use_tta,
        tta_ratios=args.tta_ratios,
    )


