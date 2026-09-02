"""Validation for Spectral Artifact Simulation (Task 10)."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.sas import SASTransform, SpectralArtifactSimulator
from src.sas import _periodic_grid, _resample_chain, _soft_blob_mask


def _img(h=96, w=96):
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, h, w, generator=g)


def test_simulator_output_contract():
    out = SpectralArtifactSimulator()(_img())
    assert out.shape == (3, 96, 96)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.isfinite(out).all()


def test_simulator_changes_the_image():
    x = _img()
    out = SpectralArtifactSimulator(blend_prob=1.0, resample_prob=1.0, grid_prob=1.0)(x)
    assert (out - x).abs().mean() > 1e-3


def test_simulator_deterministic_with_generator():
    x = _img()
    a = SpectralArtifactSimulator()(x, torch.Generator().manual_seed(42))
    b = SpectralArtifactSimulator()(x, torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
    c = SpectralArtifactSimulator()(x, torch.Generator().manual_seed(43))
    assert not torch.equal(a, c)


def test_always_applies_at_least_one_artifact():
    x = _img()
    sim = SpectralArtifactSimulator(blend_prob=0.0, resample_prob=0.0, grid_prob=0.0)
    out = sim(x, torch.Generator().manual_seed(1))
    assert (out - x).abs().mean() > 1e-4          # grid fallback fires


def test_periodic_grid_injects_a_spectral_peak():
    x = torch.full((3, 64, 64), 0.5)
    out = _periodic_grid(x, torch.Generator().manual_seed(0))
    spec_in = torch.fft.rfft2(x[0]).abs()
    spec_out = torch.fft.rfft2(out[0]).abs()
    # energy away from DC increases
    assert spec_out[1:, 1:].sum() > spec_in[1:, 1:].sum() + 1e-3


def test_resample_chain_preserves_shape_and_range():
    x = _img(80, 100)
    out = _resample_chain(x, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1


def test_blob_mask_is_soft_and_bounded():
    m = _soft_blob_mask(64, 64, torch.Generator().manual_seed(0))
    assert m.shape == (1, 64, 64)
    assert m.min() >= 0 and m.max() <= 1
    assert ((m > 0.05) & (m < 0.95)).float().mean() > 0.02      # a real soft transition band


def test_sas_transform_labels():
    x = _img()
    fakes = reals = 0
    for s in range(60):
        _, lab = SASTransform(fake_ratio=0.5)(x, torch.Generator().manual_seed(s))
        fakes += lab == 1
        reals += lab == 0
    assert fakes > 10 and reals > 10                            # both produced

    _, lab = SASTransform(fake_ratio=0.0)(x, torch.Generator().manual_seed(0))
    assert lab == 0


def test_sas_shifts_the_spectral_profile():
    # the point of SAS is a *different* spectral signature, not simply "more HF"
    rng = torch.Generator().manual_seed(0)
    sim = SpectralArtifactSimulator()
    prof = lambda t: torch.fft.rfft2(t.mean(0)).abs().flatten()
    diffs = []
    for _ in range(8):
        b = torch.rand(3, 96, 96, generator=rng)
        f = sim(b, rng)
        diffs.append(((prof(f) - prof(b)).abs().sum() / prof(b).abs().sum()).item())
    assert np.mean(diffs) > 0.02                  # measurable spectral change


@pytest.fixture
def real_fake_dir(tmp_path):
    rng = np.random.default_rng(0)
    for cls, n in (("real", 20), ("fake", 20)):
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(d / f"{i}.png")
    return tmp_path


def test_sas_dataset_shapes_and_labels(real_fake_dir):
    from src.data import SASDataset, scan_images

    ds = SASDataset(scan_images(real_fake_dir), image_size=32, fake_ratio=0.5)
    assert len(ds) == 20                                        # reals only
    labels = [ds[i][1] for i in range(len(ds))]
    x, _ = ds[0]
    assert tuple(x.shape) == (3, 32, 32)
    assert 0 in labels and 1 in labels


def test_build_dataloaders_sas_enlarges_training_set(real_fake_dir, tmp_path):
    from src.data import build_dataloaders

    _, base = build_dataloaders(real_fake_dir, image_size=32, batch_size=4, num_workers=0,
                                seed=0, manifest=tmp_path / "m.csv")
    loaders_sas, _ = build_dataloaders(real_fake_dir, image_size=32, batch_size=4, num_workers=0,
                                       seed=0, manifest=tmp_path / "m.csv", sas={"fake_ratio": 0.5})
    n_base = len(base["train"])
    n_sas = len(loaders_sas["train"].dataset)
    assert n_sas > n_base
    x, y = next(iter(loaders_sas["train"]))
    assert x.shape[1:] == (3, 32, 32)
