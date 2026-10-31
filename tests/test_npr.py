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


# --- training-loop interface -------------------------------------------------
# The transform tests above all exercise npr_residual directly. They passed while
# XceptionNPR was missing get_alpha, which src/train.py calls every epoch, so the
# campaign would have crashed after the first epoch of the first NPR run. These
# build through build_model and assert the contract the training loop relies on.

def test_builds_through_build_model_and_satisfies_training_interface():
    from src.config import build_model

    model = build_model("xception_npr", image_size=128, pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 2)
    assert model.get_alpha() is None
    assert model.get_frequency_mask() is None
    out.sum().backward()


def test_exposes_the_same_interface_as_the_baseline_arm():
    """Both campaign arms go through one training loop, so any attribute the loop
    touches on the baseline must exist on the NPR arm too."""
    from src.config import build_model

    base = build_model("xception", image_size=128, pretrained=False)
    npr = build_model("xception_npr", image_size=128, pretrained=False)
    missing = [a for a in ("get_alpha", "get_frequency_mask", "forward",
                           "forward_features")
               if hasattr(base, a) and not hasattr(npr, a)]
    assert not missing, f"NPR arm is missing {missing}"


def test_forward_features_pools_to_a_vector():
    from src.config import build_model

    model = build_model("xception_npr", image_size=128, pretrained=False)
    f = model.forward_features(torch.randn(2, 3, 128, 128))
    assert f.dim() == 2 and f.shape[0] == 2
