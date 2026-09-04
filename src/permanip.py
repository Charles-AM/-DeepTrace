"""Per-manipulation-method breakdown on the FaceForensics++ test split.

For every trained checkpoint, score the test split, then compute metrics separately
for each forgery method (real vs that method only). Answers: does a frequency branch
help on any specific manipulation (e.g. NeuralTextures, the subtlest)?

    python -m src.permanip --runs ffpp_xception_seed0 ffpp_f3net_seed0 ... \
        --results-root results --dataset-name ffpp --seed 0 --image-size 128

Writes ``<out-dir>/permanip.csv`` (long form) and ``permanip_auc.csv`` (run x method).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, read_manifest
from .engine import predict_scores
from .metrics import compute_metrics
from .utils import get_device

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _method_of(path: str) -> str:
    for m in METHODS:
        if m.lower() in path.lower():
            return m
    return "real"


def run(
    runs: list[str],
    results_root: Path,
    dataset_name: str,
    seed: int,
    image_size: int,
    batch_size: int = 128,
    num_workers: int = 2,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    device = get_device()
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    entries = read_manifest(manifest)["test"]
    methods = np.array([_method_of(p) for p, _ in entries])

    rows = []
    for run_name in runs:
        ckpt_path = results_root / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"skip {run_name}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        sz = ckpt["args"]["image_size"]
        model = build_model(ckpt["config"], image_size=sz, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        ds = FaceCropDataset(entries, transform=_transforms(sz, train=False))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        y_true, y_score = predict_scores(model, loader, device)

        real = methods == "real"
        for m in METHODS:
            sel = real | (methods == m)
            mt = compute_metrics(y_true[sel], y_score[sel])
            rows.append({"run": run_name, "config": ckpt["config"], "method": m,
                         "n": int(sel.sum()), "roc_auc": round(mt["roc_auc"], 4),
                         "eer": round(mt["eer"], 4), "accuracy": round(mt["accuracy"], 4)})
            print(f"  {run_name}  {m:15s} auc={mt['roc_auc']:.4f} eer={mt['eer']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "permanip.csv", index=False)
    pivot = df.pivot_table(index="run", columns="method", values="roc_auc")
    pivot.to_csv(out_dir / "permanip_auc.csv")
    print(f"\nwrote {out_dir/'permanip.csv'}\n{pivot.to_string()}")
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset-name", default="ffpp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
