"""On-the-fly JPEG re-compression (Task 6).

Re-encoding to disk at five quality levels would cost a lot of space we don't
have, so instead we round-trip each image through an in-memory JPEG encoder at a
chosen quality. This reproduces exactly the DCT-domain quantisation artefacts the
robustness test is about.
"""

from __future__ import annotations

import io

from PIL import Image

__all__ = ["jpeg_recompress", "JpegDegrade"]


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Return ``img`` after a JPEG encode/decode round-trip at ``quality`` (1-100).

    ``quality >= 100`` is treated as "no degradation" and returns the image
    unchanged (PIL still quantises at q=100, which we don't want for the
    best-case reference point).
    """
    if quality >= 100:
        return img.convert("RGB")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JpegDegrade:
    """Transform wrapper: apply :func:`jpeg_recompress` before the rest of the
    pipeline. Insert as the first element of a ``torchvision.transforms.Compose``
    that operates on PIL images."""

    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        return jpeg_recompress(img, self.quality)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(quality={self.quality})"
