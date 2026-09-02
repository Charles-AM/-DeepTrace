"""Task 10 — Spectral Artifact Simulation (SAS).

The generalization contribution. Instead of only training on the handful of real
forgery methods in FaceForensics++, we synthesize the *operation-level* spectral
signatures that every generative face pipeline leaves behind and inject them into
pristine faces. The detector then learns a generator-agnostic fingerprint that
transfers to unseen methods (GAN *and* diffusion) and survives compression.

Two artifact families, following the forensics literature:

* **Resampling / upsampling artifacts** — transpose-conv and interpolation in
  generators create periodic spectral replicas and a Nyquist "checkerboard".
  Simulated by (a) a random down-up resample chain and (b) an explicit faint
  periodic grid that plants energy at a chosen frequency.
  (cf. Zhang et al., AutoGAN, WIFS 2019; Tan et al., NPR, CVPR 2024.)
* **Blending-boundary artifacts** — compositing a synthesized face into a real
  frame leaves a ring of localized high-frequency energy along the mask edge.
  Simulated by a self-blend of a mildly perturbed copy through a soft irregular
  mask. (cf. Li et al., Face X-ray, CVPR 2020; Shiohara & Yamasaki, SBI, CVPR 2022.)

Our extension over that prior work: the blend is coupled end-to-end to the
differentiable DCT + learnable frequency mask, and it models the generative
*operation class* rather than one face-swap method, so the learned mask is both
taught and free to discover the discriminative bands.

Input/output: a float RGB tensor ``(3, H, W)`` in ``[0, 1]`` (pre-normalisation).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

__all__ = ["SpectralArtifactSimulator", "SASTransform"]

_RESAMPLE_MODES = ("nearest", "bilinear", "bicubic", "area")


def _rand(gen: torch.Generator | None, *shape) -> torch.Tensor:
    return torch.rand(*shape, generator=gen)


def _u(gen, lo, hi) -> float:
    return lo + (hi - lo) * float(_rand(gen, 1))


def _resample_chain(img: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """Down-sample then up-sample with random kernels — the tell-tale of generator
    upsampling stacks."""
    _, h, w = img.shape
    factor = int(_rand(gen, 1).mul(3).floor().item()) + 2          # 2, 3 or 4
    down = _RESAMPLE_MODES[int(_rand(gen, 1).mul(4).floor().item())]
    up = _RESAMPLE_MODES[int(_rand(gen, 1).mul(4).floor().item())]

    def _interp(x, size, mode):
        kw = {"antialias": True} if mode in ("bilinear", "bicubic") else {}
        return F.interpolate(x, size=size, mode=mode, **kw)

    x = img.unsqueeze(0)
    x = _interp(x, (max(h // factor, 8), max(w // factor, 8)), down)
    x = _interp(x, (h, w), up)
    return x.squeeze(0).clamp(0, 1)


def _periodic_grid(img: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """Multiply by a faint periodic pattern -> plants a spectral peak at freq k."""
    _, h, w = img.shape
    k = (h // 2, h // 3, h // 4)[int(_rand(gen, 1).mul(3).floor().item())]
    amp = _u(gen, 0.015, 0.06)
    ys = torch.arange(h).view(h, 1)
    xs = torch.arange(w).view(1, w)
    patt = 1.0 + amp * torch.cos(2 * math.pi * k * xs / w) * torch.cos(2 * math.pi * k * ys / h)
    return (img * patt).clamp(0, 1)


def _soft_blob_mask(h: int, w: int, gen: torch.Generator | None) -> torch.Tensor:
    """Irregular soft mask over the (roughly centred) face region -> its boundary
    is where the blending high-frequency energy lands."""
    cy, cx = _u(gen, 0.35, 0.65) * h, _u(gen, 0.35, 0.65) * w
    ry, rx = _u(gen, 0.28, 0.46) * h, _u(gen, 0.24, 0.42) * w
    ys = torch.arange(h).view(h, 1).float()
    xs = torch.arange(w).view(1, w).float()
    d = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
    m = (d < 1.0).float().view(1, 1, h, w)

    # low-frequency irregularity + soft edge
    lf = F.interpolate(_rand(gen, 1, 1, 4, 4), size=(h, w), mode="bilinear", align_corners=False)
    m = (m * (0.6 + 0.8 * lf)).clamp(0, 1)
    ksize = int(max(3, (min(h, w) // 8) | 1))
    m = gaussian_blur(m, kernel_size=[ksize, ksize], sigma=[_u(gen, ksize / 4, ksize / 2)] * 2)
    return m.clamp(0, 1).squeeze(0)                                # (1, H, W)


def _perturb_foreground(img: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """A mild, label-preserving perturbation of the copy that gets blended in."""
    x = img * (0.9 + 0.2 * _rand(gen, 3, 1, 1))
    x = x + 0.04 * (_rand(gen, 3, 1, 1) - 0.5)
    if float(_rand(gen, 1)) < 0.5:
        s = _u(gen, 0.4, 1.6)
        ks = int(max(3, round(s * 3)) | 1)
        x = gaussian_blur(x.clamp(0, 1), kernel_size=[ks, ks], sigma=[s, s])
    else:
        x = _resample_chain(x.clamp(0, 1), gen)
    # small translation via pad + crop
    _, h, w = img.shape
    dy = int(_u(gen, -0.03, 0.03) * h)
    dx = int(_u(gen, -0.03, 0.03) * w)
    x = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    return x.clamp(0, 1)


class SpectralArtifactSimulator:
    """Turn a pristine face tensor into a pseudo-fake carrying simulated generative
    spectral artifacts.

    Args:
        blend_prob: probability of applying the self-blend (boundary artifact).
        resample_prob: probability of applying the down-up resample chain.
        grid_prob: probability of applying the explicit periodic spectral peak.

    At least one of the three is always applied.
    """

    def __init__(
        self, blend_prob: float = 0.7, resample_prob: float = 0.6, grid_prob: float = 0.4
    ) -> None:
        for name, v in dict(blend_prob=blend_prob, resample_prob=resample_prob, grid_prob=grid_prob).items():
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        self.blend_prob = blend_prob
        self.resample_prob = resample_prob
        self.grid_prob = grid_prob

    def __call__(self, img: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"expected (3, H, W), got {tuple(img.shape)}")
        img = img.clamp(0, 1)
        _, h, w = img.shape
        fake = img.clone()
        applied = False

        if float(_rand(generator, 1)) < self.blend_prob:
            fg = _perturb_foreground(img, generator)
            m = _soft_blob_mask(h, w, generator)
            fake = m * fg + (1.0 - m) * fake
            applied = True
        if float(_rand(generator, 1)) < self.resample_prob:
            fake = _resample_chain(fake, generator)
            applied = True
        if float(_rand(generator, 1)) < self.grid_prob or not applied:
            fake = _periodic_grid(fake, generator)

        return fake.clamp(0, 1)


class SASTransform:
    """Dataset-level transform: with probability ``fake_ratio`` returns a SAS
    pseudo-fake (and label 1), otherwise the untouched real image (label 0).

    Expects a ``[0, 1]`` tensor; returns ``(tensor, label)``. Normalisation is
    applied by the caller afterwards.
    """

    def __init__(self, simulator: SpectralArtifactSimulator | None = None, fake_ratio: float = 0.5) -> None:
        self.sim = simulator or SpectralArtifactSimulator()
        self.fake_ratio = fake_ratio

    def __call__(self, img: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, int]:
        if float(_rand(generator, 1)) < self.fake_ratio:
            return self.sim(img, generator), 1
        return img.clamp(0, 1), 0
