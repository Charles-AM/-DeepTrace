"""Frequency branch: DCT -> (optional) learnable mask -> small CNN -> embedding.

Part of Task 3. Kept separate so it can be trained/evaluated on its own for the
"frequency only" ablation (Task 5).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dct import DifferentiableDCT2D
from .freq_dropout import FrequencyBandDropout
from .frequency_mask import LearnableFrequencyMask

__all__ = ["FrequencyBranch"]


class FrequencyBranch(nn.Module):
    """RGB image -> spectral feature vector.

    Pipeline: custom DCT  ->  learnable frequency mask (optional)  ->
    log-magnitude compression  ->  2-3 conv blocks  ->  global average pool  ->
    linear projection to ``embed_dim``.

    Args:
        image_size: spatial size of the (square) input.
        embed_dim: output feature dimension.
        use_mask: include the ``LearnableFrequencyMask`` (Task 2). ``False`` gives
            the "hybrid, no mask" ablation.
        mask_channels: ``None`` for a shared mask, or ``3`` for per-RGB-channel.
        log_scale: apply ``sign(x)*log1p(|x|)`` to the (masked) coefficients before
            the CNN. DCT coefficients span many orders of magnitude; this keeps the
            conv inputs well-conditioned. On by default.
        band_dropout_p: probability of applying :class:`FrequencyBandDropout` per
            sample during training (Task 9). ``0`` disables it.
        widths: channel counts of the conv blocks. The first block is stride-1,
            the rest stride-2.
    """

    def __init__(
        self,
        image_size: int,
        embed_dim: int = 256,
        use_mask: bool = True,
        mask_channels: int | None = None,
        log_scale: bool = True,
        band_dropout_p: float = 0.0,
        widths: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.dct = DifferentiableDCT2D(image_size)
        self.use_mask = use_mask
        self.log_scale = log_scale
        self.band_dropout = (
            FrequencyBandDropout(image_size, p=band_dropout_p) if band_dropout_p > 0 else None
        )
        self.mask = (
            LearnableFrequencyMask(image_size, channels=mask_channels) if use_mask else None
        )

        blocks: list[nn.Module] = []
        prev = 3
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            blocks += [
                nn.Conv2d(prev, w, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
            ]
            prev = w
        self.cnn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(prev, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = self.dct(x)                       # (B, 3, H, W) frequency coefficients
        if self.band_dropout is not None:
            coeff = self.band_dropout(coeff)      # zero a random radial band (train only)
        if self.mask is not None:
            coeff = self.mask(coeff)              # element-wise gate on raw coefficients
        if self.log_scale:
            coeff = torch.sign(coeff) * torch.log1p(coeff.abs())
        f = self.cnn(coeff)
        f = self.pool(f).flatten(1)
        return self.proj(f)
