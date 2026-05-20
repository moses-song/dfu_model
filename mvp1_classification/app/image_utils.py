from __future__ import annotations

import base64
import io

from PIL import Image


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_to_data_url(img: Image.Image) -> str:
    return f"data:image/png;base64,{image_to_base64(img)}"


def blend_images(base: Image.Image, overlay: Image.Image, alpha: float) -> Image.Image:
    base_rgb = base.convert("RGB")
    overlay_rgb = overlay.convert("RGB")
    return Image.blend(base_rgb, overlay_rgb, max(0.0, min(1.0, alpha)))
