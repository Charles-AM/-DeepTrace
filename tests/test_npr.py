"""Parity and invariant checks for the NPR residual.

The reference is re-implemented here verbatim from
chuangchuangtan/NPR-DeepfakeDetection `networks/resnet.py`, so the test fails if
our version ever drifts from it.
"""
import torch
import torch.nn.functional as F

from src.models.npr import NPRModule, npr_residual, SCALE


def _reference(x, factor=0.5):
    """Verbatim from the published implementation, including the 2/3 scaling."""
    def interpolate(img, f):
        return F.interpolate(
            F.interpolate(img, scale_factor=f, mode="nearest",
                          recompute_scale_factor=True),
            scale_factor=1 / f, mode="nearest", recompute_scale_factor=True)
    return (x - interpolate(x, factor)) * 2.0 / 3.0


def test_matches_reference_implementation():
    torch.manual_seed(0)
    for size in (128, 160, 256):
        x = torch.randn(4, 3, size, size)
        assert torch.allclose(npr_residual(x), _reference(x), atol=1e-6), size


def test_shape_and_channels_unchanged():
    x = torch.randn(2, 3, 128, 128)
    out = npr_residual(x)
    assert out.shape == x.shape, out.shape


def test_no_learned_parameters():
    assert sum(p.numel() for p in NPRModule().parameters()) == 0


def test_no_batch_structure_dependence():
    """Each image is transformed independently: batching must not change a row.

    This is what UCF's reconstruction objective fails, since it chunks the batch
    into real and fake halves.
    """
    torch.manual_seed(1)
    x = torch.randn(8, 3, 128, 128)
    assert torch.allclose(npr_residual(x)[3], npr_residual(x[3:4])[0], atol=1e-6)
    perm = torch.randperm(8)
    assert torch.allclose(npr_residual(x)[perm], npr_residual(x[perm]), atol=1e-6)


def test_gradients_flow_through():
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    npr_residual(x).pow(2).mean().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_output_is_not_constant():
    """A collapsed transform would look like a convergence failure downstream."""
    out = npr_residual(torch.randn(4, 3, 128, 128))
    assert out.std() > 1e-4, out.std().item()


def test_flat_input_gives_zero_residual():
    """A constant image has nothing a down/up round trip can lose."""
    out = npr_residual(torch.full((1, 3, 128, 128), 0.7))
    assert out.abs().max() < 1e-6


def test_scale_constant():
    assert abs(SCALE - 2.0 / 3.0) < 1e-12
