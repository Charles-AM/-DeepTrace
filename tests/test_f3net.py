"""Validation for the F3-Net FAD baseline (Task 11)."""

import torch

from src.config import build_model
from src.models.f3net import FADModule


def test_fad_produces_nine_channels():
    fad = FADModule(64)
    out = fad(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 9, 64, 64)
    assert torch.isfinite(out).all()


def test_fad_bands_are_complementary_at_init():
    # at init (zero perturbation) the three base filters partition the spectrum:
    # summing the reconstructed bands ~= reconstructing with an all-pass filter
    fad = FADModule(48)
    x = torch.randn(1, 3, 48, 48)
    bands = fad(x)
    recon = bands[:, 0:3] + bands[:, 3:6] + bands[:, 6:9]
    assert torch.allclose(recon, x, atol=1e-3)


def test_fad_filter_weights_are_trainable():
    fad = FADModule(32)
    x = torch.randn(2, 3, 32, 32)
    fad(x).pow(2).sum().backward()
    assert fad.filter_weights.grad is not None
    assert fad.filter_weights.grad.abs().sum() > 0


def test_f3net_builds_and_matches_interface():
    model = build_model("f3net", image_size=128, pretrained=False)
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 2)
    assert model.get_alpha() is None
    assert model.get_frequency_mask().shape == (3, 128, 128)
    out.sum().backward()


def test_f3net_forward_features():
    model = build_model("f3net", image_size=128, pretrained=False)
    feats = model.forward_features(torch.randn(2, 3, 128, 128))
    assert feats.dim() == 2 and feats.shape[0] == 2
