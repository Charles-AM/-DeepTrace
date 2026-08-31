"""Validation for the learnable frequency mask (Task 2)."""

import math

import pytest
import torch

from src.models.frequency_mask import LearnableFrequencyMask


def test_output_shape_preserved():
    m = LearnableFrequencyMask(32)
    x = torch.randn(4, 3, 32, 32)
    assert m(x).shape == x.shape


def test_mask_stays_bounded_for_any_logits():
    # The mask is always in the closed [0, 1] and finite, no matter the logits.
    # (float32 sigmoid saturates to exactly 0.0 / 1.0 for |logit| >~ 16 -- that's
    # expected and harmless; we don't assert a strict open interval.)
    m = LearnableFrequencyMask(16)
    for scale in (1.0, 8.0, 100.0):
        with torch.no_grad():
            m.logits.copy_(torch.randn_like(m.logits) * scale)
        mask = m.mask
        assert torch.isfinite(mask).all()
        assert torch.all(mask >= 0) and torch.all(mask <= 1)


def test_mask_is_between_zero_and_one_at_realistic_init():
    m = LearnableFrequencyMask(16, init_keep_prob=0.9)
    with torch.no_grad():
        m.logits.add_(torch.randn_like(m.logits) * 1.5)  # a few steps of training
    mask = m.mask
    assert torch.all(mask > 0) and torch.all(mask < 1)


def test_elementwise_multiply_is_exact():
    m = LearnableFrequencyMask(8)
    with torch.no_grad():
        m.logits.copy_(torch.randn_like(m.logits))
    x = torch.randn(2, 3, 8, 8)
    expected = x * torch.sigmoid(m.logits)
    assert torch.allclose(m(x), expected)


def test_init_keep_prob_controls_starting_mask():
    for p in (0.5, 0.7, 0.9, 0.99):
        m = LearnableFrequencyMask(12, init_keep_prob=p)
        assert torch.allclose(m.mask, torch.full_like(m.mask, p), atol=1e-6)


def test_gradients_reach_logits_and_input():
    m = LearnableFrequencyMask(16)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    m(x).pow(2).sum().backward()
    assert m.logits.grad is not None and m.logits.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_single_trainable_parameter():
    m = LearnableFrequencyMask(16)
    params = list(m.parameters())
    assert len(params) == 1
    assert params[0].shape == (1, 16, 16)
    assert params[0].requires_grad


def test_per_channel_mask():
    m = LearnableFrequencyMask(16, channels=3)
    assert m.logits.shape == (3, 16, 16)
    x = torch.randn(2, 3, 16, 16)
    assert m(x).shape == x.shape


def test_get_mask_is_detached_cpu():
    m = LearnableFrequencyMask((8, 10))
    mask = m.get_mask()
    assert mask.shape == (1, 8, 10)
    assert not mask.requires_grad
    assert mask.device.type == "cpu"
    assert torch.all((mask >= 0) & (mask <= 1))


def test_non_square():
    m = LearnableFrequencyMask((24, 40))
    x = torch.randn(1, 3, 24, 40)
    assert m(x).shape == x.shape


def test_rejects_size_mismatch():
    m = LearnableFrequencyMask(16)
    with pytest.raises(ValueError):
        m(torch.randn(1, 3, 32, 32))


def test_rejects_bad_init_prob():
    with pytest.raises(ValueError):
        LearnableFrequencyMask(8, init_keep_prob=1.0)
    with pytest.raises(ValueError):
        LearnableFrequencyMask(8, init_keep_prob=0.0)


def test_sparsity_penalty_matches_mean():
    m = LearnableFrequencyMask(16, init_keep_prob=0.8)
    assert torch.allclose(m.sparsity_penalty(), m.mask.mean())


def test_integration_with_dct():
    from src.models.dct import DifferentiableDCT2D

    dct = DifferentiableDCT2D(32)
    mask = LearnableFrequencyMask(32)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = mask(dct(x))
    out.pow(2).sum().backward()
    assert out.shape == x.shape
    assert mask.logits.grad is not None
    assert x.grad is not None and torch.isfinite(x.grad).all()
