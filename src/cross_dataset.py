"""Task 7 — cross-dataset generalization.

Take checkpoints trained on dataset A (e.g. FaceForensics++) and evaluate them,
with no retraining, on the test data of dataset B (e.g. Celeb-DF v2). Report the
in-domain AUC, the cross-domain AUC, and the drop (Delta AUC).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, scan_images
from .engine import evaluate
from .utils import get_device


def _cross_loader(data_root: str, image_size: int, batch_size: int, num_workers: int, limit):
    entries = scan_images(data_root)
    if limit:
        # keep class balance in the truncation
        import numpy as np

        rng = np.random.default_rng(0)
        by = {0: [], 1: []}
        for e in entries:
            by[e[1]].append(e)
        per = limit // 2
        entries = sorted(
            [by[0][i] for i in rng.permutation(len(by[0]))[:per]]
            + [by[1][i] for i in rng.permutation(len(by[1]))[:per]]
        )
    ds = FaceCropDataset(entries, transform=_transforms(image_size, train=False))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers), ds


def run(
    runs: list[str],
    cross_root: str,
    results_root: Path,
    cross_name: str = "celebdf",
    batch_size: int = 128,
    num_workers: int = 2,
    limit: int | None = None,
) -> pd.DataFrame:
    device = get_device()
    summary = pd.read_csv(results_root / "summary.csv")

    rows = []
    for run_name in runs:
        ckpt_path = results_root / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"skip {run_name}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        image_size = ckpt["args"]["image_size"]
        model = build_model(ckpt["config"], image_size=image_size, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])

        loader, ds = _cross_loader(cross_root, image_size, batch_size, num_workers, limit)
        cross_metrics, _, _ = evaluate(model, loader, device)

        in_row = summary[summary.run == run_name]
        auc_in = float(in_row["roc_auc"].iloc[0]) if not in_row.empty else float("nan")
        auc_cross = cross_metrics["roc_auc"]
        rows.append(
            {
                "run": run_name,
                "config": ckpt["config"],
                "n_cross": len(ds),
                "auc_in": round(auc_in, 4),
                "auc_cross": round(auc_cross, 4),
                "delta_auc": round(auc_in - auc_cross, 4),
                "acc_cross": round(cross_metrics["accuracy"], 4),
                "eer_cross": round(cross_metrics["eer"], 4),
            }
        )
        print(f"  {run_name}: in={auc_in:.4f}  {cross_name}={auc_cross:.4f}  Δ={auc_in - auc_cross:+.4f}")

    df = pd.DataFrame(rows)
    out = results_root / f"cross_dataset_{cross_name}.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}\n{df.to_string(index=False)}")
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-dataset evaluation of trained checkpoints")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cross-root", required=True, help="root of the unseen dataset's images")
    p.add_argument("--cross-name", default="celebdf")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--results-root", default="results")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, a.cross_root, Path(a.results_root), cross_name=a.cross_name, limit=a.limit)


if __name__ == "__main__":
    main()
