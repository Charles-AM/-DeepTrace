"""Validation for dataset scanning, splitting, and dataloaders (Task 4)."""

import numpy as np
import pytest
from PIL import Image

from src.data import (
    FaceCropDataset,
    build_dataloaders,
    label_from_path,
    make_splits,
    read_manifest,
    scan_images,
)


@pytest.fixture
def fake_dataset(tmp_path):
    """A tiny on-disk dataset: <root>/real/*.png and <root>/fake/*.png."""
    rng = np.random.default_rng(0)
    for cls, n in (("real", 12), ("fake", 18)):
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            arr = rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i:03d}.png")
    return tmp_path


def test_label_from_path():
    assert label_from_path("data/fake/x.png") == 1
    assert label_from_path("data/original_sequences/y.png") == 0
    assert label_from_path("manipulated_sequences/deepfakes/z.png") == 1
    assert label_from_path("data/unknownfolder/w.png") is None


def test_scan_finds_and_labels_all(fake_dataset):
    items = scan_images(fake_dataset)
    assert len(items) == 30
    labels = sorted(lab for _, lab in items)
    assert labels.count(0) == 12 and labels.count(1) == 18


def test_scan_raises_on_empty(tmp_path):
    (tmp_path / "misc").mkdir()
    with pytest.raises(RuntimeError):
        scan_images(tmp_path)


def test_splits_deterministic_disjoint_and_complete(fake_dataset):
    items = scan_images(fake_dataset)
    a = make_splits(items, seed=42)
    b = make_splits(items, seed=42)
    assert a == b                                    # deterministic
    assert make_splits(items, seed=1) != a           # seed matters

    paths = {s: {p for p, _ in rows} for s, rows in a.items()}
    assert paths["train"] & paths["val"] == set()
    assert paths["train"] & paths["test"] == set()
    assert paths["val"] & paths["test"] == set()
    assert sum(len(v) for v in a.values()) == 30


def test_split_ratios_roughly_hold(fake_dataset):
    items = scan_images(fake_dataset)
    s = make_splits(items, ratios=(0.7, 0.15, 0.15), seed=0)
    assert abs(len(s["train"]) / 30 - 0.7) < 0.1


def test_group_by_keeps_group_together(tmp_path):
    items = [(f"/x/vid{v:02d}/frame{f}.png", v % 2) for v in range(20) for f in range(5)]
    splits = make_splits(items, seed=0, group_by=r"(vid\d+)")
    where = {}
    for split, rows in splits.items():
        for p, _ in rows:
            vid = p.split("/")[2]
            where.setdefault(vid, set()).add(split)
    assert all(len(s) == 1 for s in where.values())


def test_dataset_getitem_shapes(fake_dataset):
    items = scan_images(fake_dataset)
    ds = FaceCropDataset(items, image_size=32, train=False)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 32, 32)
    assert y in (0, 1)
    assert len(ds) == 30
    assert ds.class_counts() == (12, 18)


def test_train_transform_is_stochastic(fake_dataset):
    items = scan_images(fake_dataset)
    ds = FaceCropDataset(items, image_size=32, train=True)
    assert not np.allclose(ds[0][0].numpy(), ds[0][0].numpy())  # random crop/flip each call


def test_build_dataloaders_writes_and_reuses_manifest(fake_dataset, tmp_path):
    manifest = tmp_path / "m.csv"
    loaders1, ds1 = build_dataloaders(
        fake_dataset, image_size=32, batch_size=4, num_workers=0, seed=7, manifest=manifest
    )
    assert manifest.exists()
    x, y = next(iter(loaders1["train"]))
    assert x.shape[1:] == (3, 32, 32) and x.shape[0] == 4

    # a second call with a different seed must still reuse the saved split
    _, ds2 = build_dataloaders(
        fake_dataset, image_size=32, batch_size=4, num_workers=0, seed=999, manifest=manifest
    )
    assert [p for p, _ in ds1["test"].entries] == [p for p, _ in ds2["test"].entries]


def test_limit_truncates_splits(fake_dataset, tmp_path):
    _, ds = build_dataloaders(
        fake_dataset, image_size=32, batch_size=2, num_workers=0, seed=0,
        manifest=tmp_path / "m.csv", limit=3,
    )
    assert all(len(d) <= 3 for d in ds.values())
