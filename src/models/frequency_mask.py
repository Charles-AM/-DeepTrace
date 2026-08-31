"""Task 2 — Learnable frequency mask.

A trainable gate over DCT coefficients. The module holds raw learnable *logits*
of the same spatial shape as the coefficient map; a sigmoid squashes them to a
mask in ``(0, 1)`` which multiplies the coefficients element-wise. During training
the mask learns to keep informative frequency bands and suppress the rest.

Storing logits (not the post-sigmoid values) keeps the parameter unconstrained
and the 0..1 bound exact and automatic — no clamping, no projection step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["LearnableFrequencyMask"]


class LearnableFrequencyMask(nn.Module):
    """Element-wise learnable gate on a DCT coefficient map.

    Args:
        size: spatial size of the coefficient map — ``int`` (square) or ``(H, W)``.
        channels: if ``None`` (default) a single ``(1, H, W)`` mask is shared
            across all channels; set to ``C`` for an independent mask per channel.
        init_keep_prob: initial value of every mask entry, in ``(0, 1)``.
            Default ``0.9`` — the branch starts as a near-pass-through of the DCT
            and learns what to suppress.

    Shape:
        - input:  ``(B, C, H, W)`` DCT coefficients
        - output: ``(B, C, H, W)`` masked coefficients
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        channels: int | None = None,
        init_keep_prob: float = 0.9,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size
        if not 0.0 < init_keep_prob < 1.0:
            raise ValueError(f"init_keep_prob must be in (0, 1), got {init_keep_prob}")
        c = 1 if channels is None else int(channels)
        if c < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        init_logit = math.log(init_keep_prob / (1.0 - init_keep_prob))
        self.logits = nn.Parameter(torch.full((c, h, w), init_logit))

    @property
    def mask(self) -> torch.Tensor:
        """Differentiable ``(C, H, W)`` mask in ``(0, 1)``."""
        return torch.sigmoid(self.logits)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        if coeff.dim() != 4:
            raise ValueError(f"expected 4D (B, C, H, W), got {coeff.dim()}D")
        mc, mh, mw = self.logits.shape
        if coeff.shape[-2:] != (mh, mw):
            raise ValueError(
                f"coefficient spatial size {tuple(coeff.shape[-2:])} != mask size {(mh, mw)}"
            )
        if mc != 1 and mc != coeff.shape[1]:
            raise ValueError(
                f"mask has {mc} channels, incompatible with input's {coeff.shape[1]}"
            )
        return coeff * self.mask

    @torch.no_grad()
    def get_mask(self) -> torch.Tensor:
        """Detached ``(C, H, W)`` mask in ``[0, 1]`` on CPU — for heatmaps / Task 8."""
        return torch.sigmoid(self.logits).detach().cpu()

    @torch.no_grad()
    def get_logits(self) -> torch.Tensor:
        """Detached raw pre-sigmoid weights on CPU."""
        return self.logits.detach().cpu()

    def sparsity_penalty(self) -> torch.Tensor:
        """Mean mask activation. Add ``lambda * mask.sparsity_penalty()`` to the loss
        to push the mask toward keeping fewer frequencies (optional; used in ablations)."""
        return self.mask.mean()

    def extra_repr(self) -> str:
        c, h, w = self.logits.shape
        return f"channels={c}, h={h}, w={w}"
