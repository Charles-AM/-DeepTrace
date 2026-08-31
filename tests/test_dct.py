"""Validation for the custom differentiable DCT layer (Task 1).

Checks:
  1. numerical equivalence with scipy.fftpack.dct (the "black box" reference)
  2. inverse round-trip (A is orthonormal -> perfect reconstruction)
  3. gradient flow (the whole point of building it from matmul)
  4. Parseval / energy preservation
  5. non-square inputs
  6. device + dtype handling
"""

import numpy as np
import pytest
import torch
from scipy.fftpack import dct

from src.models.dct import DifferentiableDCT2D, dct_matrix


def scipy_dct2(img: np.ndarray) -> np.ndarray:
    """Reference 2D DCT-II: apply along both axes, orthonormal norm."""
    return dct(dct(img, axis=0, type=2, norm="ortho"), axis=1, type=2, norm="ortho")


def test_matches_scipy_square():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    layer = DifferentiableDCT2D(32, dtype=torch.float64)
    out = layer(x).numpy()
    ref = np.stack(
        [[scipy_dct2(x[b, c].numpy()) for c in range(x.shape[1])] for b in range(x.shape[0])]
    )
    assert np.allclose(out, ref, atol=1e-10)


def test_matches_scipy_nonsquare():
    x = torch.randn(1, 1, 24, 40, dtype=torch.float64)
    layer = DifferentiableDCT2D((24, 40), dtype=torch.float64)
    out = layer(x).numpy()[0, 0]
    ref = scipy_dct2(x.numpy()[0, 0])
    assert np.allclose(out, ref, atol=1e-10)


def test_inverse_roundtrip():
    x = torch.randn(4, 3, 48, 48, dtype=torch.float64)
    layer = DifferentiableDCT2D(48, dtype=torch.float64)
    rec = layer.inverse(layer(x))
    assert torch.allclose(rec, x, atol=1e-10)


def test_gradients_flow():
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    layer = DifferentiableDCT2D(16)
    loss = layer(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_gradcheck():
    x = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
    layer = DifferentiableDCT2D(8, dtype=torch.float64)
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-4)


def test_parseval_energy_preserved():
    x = torch.randn(2, 3, 24, 24, dtype=torch.float64)
    layer = DifferentiableDCT2D(24, dtype=torch.float64)
    X = layer(x)
    assert torch.allclose(x.pow(2).sum(), X.pow(2).sum(), rtol=1e-9)


def test_dc_term_is_mean():
    # DCT[0, 0] of an N x N block == N * mean(block) for the orthonormal DCT-II
    n = 16
    x = torch.rand(1, 1, n, n, dtype=torch.float64)
    layer = DifferentiableDCT2D(n, dtype=torch.float64)
    dc = layer(x)[0, 0, 0, 0]
    assert torch.allclose(dc, n * x.mean(), atol=1e-10)


def test_basis_is_orthonormal():
    a = dct_matrix(20)
    assert torch.allclose(a @ a.t(), torch.eye(20, dtype=a.dtype), atol=1e-10)


def test_buffers_not_parameters():
    layer = DifferentiableDCT2D(16)
    assert len(list(layer.parameters())) == 0
    names = {n for n, _ in layer.named_buffers()}
    assert "basis_h" in names


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_passthrough(dtype):
    x = torch.randn(1, 1, 12, 12, dtype=dtype)
    layer = DifferentiableDCT2D(12)
    assert layer(x).dtype == dtype


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_runs_on_mps():
    x = torch.randn(2, 3, 32, 32, device="mps")
    layer = DifferentiableDCT2D(32).to("mps")
    out = layer(x)
    assert out.device.type == "mps"
    assert torch.isfinite(out).all()
