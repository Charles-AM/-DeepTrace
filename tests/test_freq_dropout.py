"""Validation for frequency-band dropout (Task 9)."""

import pytest
import torch

from src.config import build_model
from src.models import FrequencyBandDropout


def test_noop_in_eval_mode():
    m = FrequencyBandDropout(32, p=1.0).eval()
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(m(x), x)


def test_noop_when_p_zero():
    m = FrequencyBandDropout(32, p=0.0).train()
    x = torch.randn(4, 3, 32, 32)
    assert torch.equal(m(x), x)


def test_train_mode_zeros_a_contiguous_radial_band():
    torch.manual_seed(0)
    m = FrequencyBandDropout(48, p=1.0, min_width=0.1, max_width=0.3).train()
    x = torch.ones(1, 1, 48, 48)
    out = m(x)[0, 0]
    zeroed = out == 0
    assert zeroed.any() and (~zeroed).any()                 # something dropped, something kept

    # the zeroed set must be a contiguous ring in radius
    r = m.radius_norm
    rz = r[zeroed]
    kept_between = ((r >= rz.min()) & (r <= rz.max()) & ~zeroed)
    assert not kept_between.any()


def test_per_sample_independence():
    torch.manual_seed(1)
    m = FrequencyBandDropout(32, p=1.0).train()
    out = m(torch.ones(8, 1, 32, 32))
    patterns = {tuple((out[i, 0] == 0).flatten().tolist()) for i in range(8)}
    assert len(patterns) > 1                                 # different band per sample


def test_gradient_flows_through_kept_coefficients():
    m = FrequencyBandDropout(16, p=1.0).train()
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        FrequencyBandDropout(16, p=1.5)
    with pytest.raises(ValueError):
        FrequencyBandDropout(16, min_width=0.5, max_width=0.2)


def test_wired_into_detector_config():
    model = build_model("full_banddrop", image_size=64, pretrained=False)
    assert model.frequency.band_dropout is not None
    assert model.frequency.band_dropout.p == 0.5

    plain = build_model("full", image_size=64, pretrained=False)
    assert plain.frequency.band_dropout is None


def test_detector_trains_with_band_dropout():
    model = build_model("full_banddrop", image_size=64, pretrained=False).train()
    x = torch.randn(3, 3, 64, 64)
    model(x).sum().backward()
    assert model.frequency.mask.logits.grad is not None
