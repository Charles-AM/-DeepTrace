"""Frequency-band dropout (Task 9).

A dropout-style regulariser that operates on radial *bands* of the DCT coefficient
map instead of individual units. During training, each sample independently has a
random contiguous ring of frequencies (measured by distance from the DC term)
zeroed with probability ``p``. This stops the detector from betting everything on
one band — e.g. the high-frequency GAN fingerprint that JPEG compression and many
diffusion models simply do not provide — and empirically flattens the learned
frequency mask toward a more distributed, transfer-friendly solution.

Inactive at eval time (like ``nn.Dropout``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FrequencyBandDropout"]


class FrequencyBandDropout(nn.Module):
    """Zero a random radial frequency band per sample during training.

    Args:
        size: spatial size of the (square) coefficient map.
        p: probability that a given sample gets a band dropped.
        max_width: maximum band width as a fraction of the maximum radius.
        min_width: minimum band width as a fraction of the maximum radius.
    """

    def __init__(
        self, size: int, p: float = 0.5, max_width: float = 0.25, min_width: float = 0.05
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if not 0.0 < min_width <= max_width <= 1.0:
            raise ValueError("require 0 < min_width <= max_width <= 1")
        self.p = p
        self.max_width = max_width
        self.min_width = min_width

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        radius = torch.sqrt(xx**2 + yy**2)
        self.register_buffer("radius_norm", radius / radius.max(), persistent=False)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return coeff

        b = coeff.shape[0]
        device = coeff.device
        r = self.radius_norm.to(device)

        apply = torch.rand(b, device=device) < self.p                 # (B,)
        width = self.min_width + torch.rand(b, device=device) * (self.max_width - self.min_width)
        lo = torch.rand(b, device=device) * (1.0 - width)             # band start
        hi = lo + width

        # per-sample keep mask: 0 inside the dropped band, 1 elsewhere
        r_ = r.unsqueeze(0)                                           # (1, H, W)
        in_band = (r_ >= lo.view(b, 1, 1)) & (r_ < hi.view(b, 1, 1))  # (B, H, W)
        drop = in_band & apply.view(b, 1, 1)
        keep = (~drop).to(coeff.dtype).unsqueeze(1)                   # (B, 1, H, W)
        return coeff * keep

    def extra_repr(self) -> str:
        return f"p={self.p}, width=[{self.min_width}, {self.max_width}]"
