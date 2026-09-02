"""Real-world perturbations for the robustness suite (Task 12).

Each perturbation is a callable ``severity (1..4) -> (PIL.Image -> PIL.Image)``.
Severity 0 is always the identity (clean reference). These slot in as the first
element of the evaluation transform, before Resize/CenterCrop/ToTensor, so a model
sees exactly the degraded image a deployed system would receive.

Covers the standard deepfake-robustness axes (cf. the DFDC perturbation protocol):
compression, blur, additive noise, downscaling, and contrast loss.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .transforms_jpeg import jpeg_recompress

__all__ = ["PERTURBATIONS", "SEVERITIES", "make_perturbation", "identity"]

SEVERITIES = (1, 2, 3, 4)


def identity(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _jpeg(sev: int):
    q = {1: 90, 2: 70, 3: 50, 4: 30}[sev]
    return lambda img: jpeg_recompress(img, q)


def _blur(sev: int):
    radius = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.5}[sev]
    return lambda img: img.convert("RGB").filter(ImageFilter.GaussianBlur(radius))


def _noise(sev: int):
    std = {1: 8, 2: 16, 3: 28, 4: 44}[sev]

    def f(img: Image.Image) -> Image.Image:
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        a = a + np.random.normal(0, std, a.shape).astype(np.float32)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    return f


def _resize(sev: int):
    scale = {1: 0.75, 2: 0.5, 3: 0.35, 4: 0.25}[sev]

    def f(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((max(int(w * scale), 8), max(int(h * scale), 8)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    return f


def _contrast(sev: int):
    factor = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.35}[sev]
    return lambda img: ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


PERTURBATIONS = {
    "jpeg": _jpeg,
    "blur": _blur,
    "noise": _noise,
    "resize": _resize,
    "contrast": _contrast,
}


def make_perturbation(name: str, severity: int):
    """Return the PIL->PIL transform for ``name`` at ``severity`` (0 = identity)."""
    if severity == 0:
        return identity
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; choices: {sorted(PERTURBATIONS)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 0 or one of {SEVERITIES}")
    return PERTURBATIONS[name](severity)
