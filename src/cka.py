"""C4 — linear CKA between the spatial-branch and frequency-branch features.

Predictive redundancy (spatial ~= hybrid, C1; late fusion adds nothing, C1b) does
NOT by itself establish that the spatial CNN encodes the same information as the
frequency branch -- that's a representational claim and needs this check. High CKA
=> the two branches encode highly aligned/redundant representations. Low CKA =>
they encode different things (predictive redundancy would then have to be
explained some other way, e.g. the frequency signal is just too weak to matter,
not that it duplicates the spatial one).

    python -m src.cka --run ffpp_full_seed0 --results-root results \
        --dataset-name ffpp --seed 0 --image-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, read_manifest
from .utils import get_device


@torch.no_grad()
def _branch_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    s_all, f_all = [], []
    for x, _ in loader:
        x = x.to(device)
        s_all.append(model.spatial(x).float().cpu())
        f_all.append(model.frequency(x).float().cpu())
    return torch.cat(s_all).numpy(), torch.cat(f_all).numpy()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xty = x.T @ y
    num = np.linalg.norm(xty, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(num / max(denom, 1e-12))


def run(run_name: str, results_root: Path, dataset_name: str, seed: int,
        image_size: int, limit: int | None = None) -> dict:
    device = get_device()
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    entries = read_manifest(manifest)["test"]
    if limit:
        entries = entries[:limit]

    ckpt = torch.load(results_root / run_name / "best.pt", map_location=device)
    model = build_model(ckpt["config"], image_size=ckpt["args"]["image_size"],
                         pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    if model.spatial is None or model.frequency is None:
        raise ValueError(f"{run_name} (config={ckpt['config']}) has only one branch -- CKA needs both")

    ds = FaceCropDataset(entries, transform=_transforms(ckpt["args"]["image_size"], train=False))
    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)
    s, f = _branch_features(model, loader, device)

    cka = linear_cka(s, f)
    # random projection of s as a null baseline -- CKA(s, random) should be ~0
    rng = np.random.default_rng(0)
    null = linear_cka(s, rng.standard_normal(f.shape).astype(np.float32))

    out = {"run": run_name, "n": len(entries), "cka_spatial_freq": round(cka, 4),
           "cka_null_baseline": round(null, 4)}
    for k, v in out.items():
        print(f"  {k:20s} {v}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset-name", default="ffpp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.run, Path(a.results_root), a.dataset_name, a.seed, a.image_size, a.limit)


if __name__ == "__main__":
    main()
