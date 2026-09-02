"""F3-Net FAD baseline (Task 11) — the standard frequency-method comparison.

Faithful re-implementation of the **Frequency-Aware Decomposition** branch of
F3-Net (Qian et al., *Thinking in Frequency*, ECCV 2020): the image is split into
three frequency bands by partly-learnable band-pass filters, each band is mapped
back to the spatial domain, the three filtered images are stacked, and an Xception
backbone classifies the 9-channel stack. (We use the FAD branch alone — the most
widely reported F3-Net baseline — not the full FAD+LFS+MixBlock model.)

The contrast with our method is the point: F3-Net's filters partition the spectrum
into **hand-designed radial bands** with only a small learnable perturbation; our
`LearnableFrequencyMask` learns a full **per-coefficient** importance map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dct import DifferentiableDCT2D

__all__ = ["FADModule", "F3NetFAD"]

# radial band edges (fraction of max frequency) — low / middle / high
_BANDS = ((0.0, 0.10), (0.10, 0.35), (0.35, 1.01))


class FADModule(nn.Module):
    """Frequency-Aware Decomposition: RGB image -> 9-channel band-filtered stack."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.dct = DifferentiableDCT2D(image_size)

        yy, xx = torch.meshgrid(
            torch.arange(image_size, dtype=torch.float32),
            torch.arange(image_size, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt(xx**2 + yy**2)
        r = r / r.max()
        base = torch.stack([((r >= lo) & (r < hi)).float() for lo, hi in _BANDS])  # (3, H, W)
        self.register_buffer("base_filters", base, persistent=False)
        # small learnable perturbation per band, per location (F3-Net's f_w)
        self.filter_weights = nn.Parameter(torch.zeros_like(base))

    def effective_filters(self) -> torch.Tensor:
        return self.base_filters + torch.tanh(self.filter_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = self.dct(x)                                    # (B, 3, H, W)
        filt = self.effective_filters()                        # (3, H, W)
        bands = [self.dct.inverse(coeff * filt[i]) for i in range(3)]
        return torch.cat(bands, dim=1)                         # (B, 9, H, W)


class F3NetFAD(nn.Module):
    """FAD decomposition + Xception classifier."""

    def __init__(
        self, image_size: int = 128, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2
    ) -> None:
        super().__init__()
        import timm

        self.fad = FADModule(image_size)
        self.backbone = timm.create_model(
            "legacy_xception", pretrained=pretrained, num_classes=num_classes,
            in_chans=9, drop_rate=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.fad(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.forward_features(self.fad(x))
        return f.mean(dim=(-2, -1)) if f.dim() == 4 else f

    @torch.no_grad()
    def get_alpha(self):
        return None

    @torch.no_grad()
    def get_frequency_mask(self):
        # expose the learned band filters for the Task 8 comparison figure
        return self.fad.effective_filters().detach().cpu()
