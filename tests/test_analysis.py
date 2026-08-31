"""Tests for the Task 5/6/8 analysis utilities (logic only, no training)."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.models import SpatialFrequencyDetector


# --- Task 6: JPEG transform --------------------------------------------------- -

def test_jpeg_recompress_changes_image_at_low_quality():
    from src.transforms_jpeg import jpeg_recompress

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    q30 = np.asarray(jpeg_recompress(img, 30))
    assert q30.shape == (64, 64, 3)
    assert not np.array_equal(q30, np.asarray(img))          # compression altered pixels


def test_jpeg_recompress_passthrough_at_100():
    from src.transforms_jpeg import JpegDegrade

    img = Image.fromarray(np.full((16, 16, 3), 123, dtype=np.uint8))
    out = JpegDegrade(100)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


# --- Task 8: mask visualisations -------------------------------------------- -

def test_radial_profile_shape_and_axis(tmp_path):
    from src.visualize import radial_profile

    mask = np.random.rand(1, 32, 32)
    out = radial_profile(mask, tmp_path / "r.png", n_bins=16)
    assert (tmp_path / "r.png").exists()
    assert out["radius"].shape == out["weight"].shape == (16,)
    assert np.all(np.diff(out["radius"]) > 0)


def test_radial_profile_detects_lowpass_bias(tmp_path):
    from src.visualize import radial_profile

    # a mask that keeps only low frequencies (near DC at 0,0)
    yy, xx = np.mgrid[0:32, 0:32]
    mask = (np.sqrt(xx**2 + yy**2) < 8).astype(float)
    out = radial_profile(mask, tmp_path / "r.png", n_bins=16)
    assert np.nanmean(out["weight"][:4]) > np.nanmean(out["weight"][-4:])


def test_mask_heatmap_writes_file(tmp_path):
    from src.visualize import mask_heatmap

    m = mask_heatmap(np.random.rand(3, 24, 24), tmp_path / "h.png")
    assert (tmp_path / "h.png").exists() and m.shape == (24, 24)


def test_gradcam_outputs_normalised_maps():
    from src.visualize import GradCAM

    model = SpatialFrequencyDetector(image_size=64, pretrained=False)
    cam = GradCAM(model)(torch.randn(2, 3, 64, 64))
    assert cam.shape[0] == 2 and cam.ndim == 3
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5


def test_feature_projection_returns_2d(tmp_path):
    from src.visualize import feature_projection

    feats = np.random.randn(40, 32)
    labels = np.array([0, 1] * 20)
    emb = feature_projection(feats, labels, tmp_path / "p.png", method="tsne")
    assert emb.shape == (40, 2) and (tmp_path / "p.png").exists()


# --- Task 5: ablation aggregation ------------------------------------------- -

def test_ablation_aggregate_builds_table_and_plot(tmp_path):
    from src.run_ablation import aggregate

    rng = np.random.default_rng(0)
    rows = []
    for cfg, base in [("baseline_spatial", 0.80), ("no_mask", 0.86), ("full", 0.90)]:
        for seed in (0, 1, 2):
            m = {k: float(np.clip(base + rng.normal(0, 0.01), 0, 1)) for k in
                 ["accuracy", "precision", "recall", "f1", "roc_auc", "eer"]}
            rows.append({"run": f"rvf_{cfg}_seed{seed}", "config": cfg, "dataset": "rvf",
                         "seed": seed, "image_size": 128, "epochs": 15, **m})
    (tmp_path / "summary.csv").write_text(pd.DataFrame(rows).to_csv(index=False))

    stats = aggregate(tmp_path / "summary.csv", "rvf",
                      ["baseline_spatial", "no_mask", "full"], [0, 1, 2], tmp_path)
    assert (tmp_path / "ablation_table.md").exists()
    assert (tmp_path / "ablation_auc.png").exists()
    assert ("roc_auc", "mean") in stats.columns
    assert stats.loc["full", ("roc_auc", "mean")] > stats.loc["baseline_spatial", ("roc_auc", "mean")]
