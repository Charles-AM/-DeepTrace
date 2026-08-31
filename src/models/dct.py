"""Task 1 — Custom differentiable 2D DCT layer.

The 2D DCT-II is separable and linear, so it can be written as a pair of matrix
multiplications with a *fixed* orthonormal basis matrix ``A``::

    X = A_h @ x @ A_w^T

Because ``A`` is orthonormal, the inverse transform is simply::

    x = A_h^T @ X @ A_w

Implementing it this way (rather than calling an FFT/DCT library) keeps the whole
operation a differentiable ``matmul``, so gradients flow cleanly from the frequency
coefficients back to the input pixels. The basis matrices are stored as buffers,
not parameters — they are constants, never updated by the optimizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["DifferentiableDCT2D", "dct_matrix"]


def dct_matrix(n: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``A`` of shape ``(n, n)``.

    Row ``k`` is the ``k``-th cosine basis vector. This matches
    ``scipy.fftpack.dct(x, type=2, norm="ortho")`` applied along an axis:
    ``A @ x == scipy.fftpack.dct(x, type=2, norm="ortho")``.
    """
    k = torch.arange(n, dtype=dtype).view(-1, 1)   # frequency index (rows)
    m = torch.arange(n, dtype=dtype).view(1, -1)   # sample index    (cols)
    a = torch.cos(math.pi * (2.0 * m + 1.0) * k / (2.0 * n))
    a *= math.sqrt(2.0 / n)
    a[0] *= 1.0 / math.sqrt(2.0)                   # DC row scaling for orthonormality
    return a


class DifferentiableDCT2D(nn.Module):
    """Batched, differentiable 2D DCT-II (and its inverse).

    Args:
        size: spatial size of the input. An ``int`` for square inputs, or a
            ``(H, W)`` tuple for non-square inputs.
        dtype: dtype of the precomputed basis buffers (default ``float32``).
            The basis is always *computed* in ``float64`` for accuracy and then
            cast to this dtype.

    Shape:
        - input:  ``(B, C, H, W)``
        - output: ``(B, C, H, W)`` — DCT coefficients, same index layout
          (``[..., 0, 0]`` is the DC term).
    """

    def __init__(self, size: int | tuple[int, int], dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if isinstance(size, int):
            h = w = size
        else:
            h, w = size
        if h < 1 or w < 1:
            raise ValueError(f"size must be positive, got {(h, w)}")
        self.h, self.w = h, w

        # Both are registered as buffers (even when identical for square inputs) so
        # that ``.to(device)`` / ``.cuda()`` move *both* — an alias attribute would
        # be left behind on the original device.
        self.register_buffer("basis_h", dct_matrix(h).to(dtype), persistent=False)
        self.register_buffer("basis_w", dct_matrix(w).to(dtype), persistent=False)

    def _check(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"expected a 4D (B, C, H, W) tensor, got {x.dim()}D")
        if x.shape[-2:] != (self.h, self.w):
            raise ValueError(
                f"expected spatial size {(self.h, self.w)}, got {tuple(x.shape[-2:])}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial image -> DCT coefficients. ``X = A_h @ x @ A_w^T``."""
        self._check(x)
        basis_h = self.basis_h.to(x.dtype)
        basis_w = self.basis_w.to(x.dtype)
        x = torch.matmul(basis_h, x)                       # transform columns
        x = torch.matmul(x, basis_w.transpose(-2, -1))     # transform rows
        return x

    def inverse(self, coeff: torch.Tensor) -> torch.Tensor:
        """DCT coefficients -> spatial image. ``x = A_h^T @ X @ A_w``."""
        self._check(coeff)
        basis_h = self.basis_h.to(coeff.dtype)
        basis_w = self.basis_w.to(coeff.dtype)
        x = torch.matmul(basis_h.transpose(-2, -1), coeff)
        x = torch.matmul(x, basis_w)
        return x

    def extra_repr(self) -> str:
        return f"h={self.h}, w={self.w}"
