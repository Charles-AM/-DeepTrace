"""Face-crop dataset, deterministic splits, and dataloader construction (Task 4).

The dataset is layout-agnostic: it recursively scans ``data_root`` for images and
labels each by whether its path contains a real-ish or fake-ish keyword. This
works for the Kaggle stand-in ("real"/"fake" folders) and for FaceForensics++
("original_sequences" / "manipulated_sequences").

Splits are written to a manifest CSV the first time and reused afterwards, so
**every model, ablation and baseline evaluates on exactly the same images** — the
comparability the verification framework requires. Optional ``group_by`` keeps all
crops from one video/identity in a single split to avoid frame-level leakage.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    import torchvision.transforms as T
except Exception:  # torchvision optional at import time (tests that don't need it)
    T = None

__all__ = [
    "FaceCropDataset",
    "scan_images",
    "make_splits",
    "build_dataloaders",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_FAKE_KEYS = ("fake", "manipulated", "deepfake", "synthesis", "df_", "faceswap", "neuraltextures", "face2face")
_REAL_KEYS = ("real", "original", "pristine", "genuine", "youtube")


def label_from_path(path: str | Path) -> int | None:
    """1 = fake, 0 = real, None = undecidable from the path."""
    s = str(path).lower()
    if any(k in s for k in _FAKE_KEYS):
        return 1
    if any(k in s for k in _REAL_KEYS):
        return 0
    return None


def scan_images(root: str | Path) -> list[tuple[str, int]]:
    """Recursively collect ``(path, label)`` for every labellable image under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    items: list[tuple[str, int]] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            lab = label_from_path(p.relative_to(root))
            if lab is not None:
                items.append((str(p), lab))
    if not items:
        raise RuntimeError(
            f"no labellable images under {root} — expected 'real'/'fake' (or FF++) keywords in paths"
        )
    return items


def make_splits(
    items: list[tuple[str, int]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    group_by: str | None = None,
) -> dict[str, list[tuple[str, int]]]:
    """Deterministically partition ``items`` into train/val/test.

    ``group_by`` is a regex with one capture group extracted from each path; all
    items sharing a group key land in the same split.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    rng = np.random.default_rng(seed)

    if group_by:
        pat = re.compile(group_by)

        def key(path: str) -> str:
            m = pat.search(path)
            return m.group(1) if m else path

        groups: dict[str, list[tuple[str, int]]] = {}
        for path, lab in items:
            groups.setdefault(key(path), []).append((path, lab))
        units = list(groups.values())
    else:
        units = [[it] for it in items]

    order = rng.permutation(len(units))
    n = len(units)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    buckets = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    out: dict[str, list[tuple[str, int]]] = {}
    for split, idxs in buckets.items():
        rows: list[tuple[str, int]] = []
        for i in idxs:
            rows.extend(units[i])
        out[split] = sorted(rows)
    return out


def write_manifest(splits: dict[str, list[tuple[str, int]]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "split"])
        for split, rows in splits.items():
            for p, lab in rows:
                w.writerow([p, lab, split])


def read_manifest(path: str | Path) -> dict[str, list[tuple[str, int]]]:
    splits: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            splits[row["split"]].append((row["path"], int(row["label"])))
    return splits


def _subsample(rows: list[tuple[str, int]], n: int, seed: int) -> list[tuple[str, int]]:
    """Seeded, roughly class-balanced subsample of at most ``n`` rows."""
    if n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[tuple[str, int]]] = {}
    for r in rows:
        by_label.setdefault(r[1], []).append(r)
    per = max(1, n // max(len(by_label), 1))
    picked: list[tuple[str, int]] = []
    for group in by_label.values():
        idx = rng.permutation(len(group))[:per]
        picked.extend(group[i] for i in idx)
    rng.shuffle(picked)
    return sorted(picked[:n])


def _transforms(image_size: int, train: bool):
    if T is None:
        raise ImportError("torchvision is required for image transforms")
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                norm,
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            norm,
        ]
    )


class FaceCropDataset(Dataset):
    """``(image_tensor, label)`` pairs from a list of ``(path, label)`` entries."""

    def __init__(
        self,
        entries: list[tuple[str, int]],
        image_size: int = 128,
        train: bool = False,
        transform=None,
    ) -> None:
        self.entries = list(entries)
        self.transform = transform if transform is not None else _transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return x, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([lab for _, lab in self.entries], dtype=int)

    def class_counts(self) -> tuple[int, int]:
        labs = self.labels
        return int((labs == 0).sum()), int((labs == 1).sum())


def build_dataloaders(
    data_root: str | Path,
    image_size: int = 128,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    group_by: str | None = None,
    manifest: str | Path | None = None,
    limit: int | None = None,
) -> tuple[dict[str, DataLoader], dict[str, FaceCropDataset]]:
    """Return ``({split: DataLoader}, {split: Dataset})``.

    If ``manifest`` exists it is loaded; otherwise the split is computed from
    ``data_root`` and saved there. ``limit`` caps each split to a class-balanced,
    seeded random subsample of that many items (debug runs) — the full manifest on
    disk is never truncated.
    """
    if manifest and Path(manifest).exists():
        splits = read_manifest(manifest)
    else:
        splits = make_splits(scan_images(data_root), ratios=ratios, seed=seed, group_by=group_by)
        if manifest:
            write_manifest(splits, manifest)

    if limit:
        splits = {k: _subsample(v, limit, seed) for k, v in splits.items()}

    datasets = {
        split: FaceCropDataset(rows, image_size=image_size, train=(split == "train"))
        for split, rows in splits.items()
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=(num_workers > 0),
        )
        for split, ds in datasets.items()
    }
    return loaders, datasets
