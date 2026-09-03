"""Task 7 / Task 13 — cross-dataset generalization.

Take checkpoints trained on dataset A (FaceForensics++) and evaluate them, with no
retraining, on one or more unseen datasets B (Celeb-DF v2, DFDC, DF40, a diffusion
set, ...). Report in-domain AUC, cross-domain AUC per target, the drop (ΔAUC), and
the mean ΔAUC — the headline generalization number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import build_model
from .data import FaceCropDataset, _transforms, scan_images
from .engine import evaluate
from .utils import get_device


def _cross_loader(data_root: str, image_size: int, batch_size: int, num_workers: int, limit):
    entries = scan_images(data_root)
    if limit:
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


def _parse_targets(specs: list[str]) -> dict[str, str]:
    """``["celebdf=/path", "dfdc=/path"]`` -> ``{"celebdf": "/path", ...}``."""
    out: dict[str, str] = {}
    for s in specs:
        if "=" not in s:
            raise ValueError(f"target spec must be NAME=PATH, got {s!r}")
        name, path = s.split("=", 1)
        out[name] = path
    return out


def run(
    runs: list[str],
    targets: dict[str, str],
    results_root: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    limit: int | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    device = get_device()
    # results_root may be a read-only mounted dataset; write CSVs elsewhere
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results_root / "summary.csv")
    loader_cache: dict[tuple[str, int], tuple] = {}

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

        in_row = summary[summary.run == run_name]
        auc_in = float(in_row["roc_auc"].iloc[0]) if not in_row.empty else float("nan")

        deltas = []
        for name, path in targets.items():
            key = (path, image_size)
            if key not in loader_cache:
                loader_cache[key] = _cross_loader(path, image_size, batch_size, num_workers, limit)
            loader, ds = loader_cache[key]
            m, _, _ = evaluate(model, loader, device)
            delta = auc_in - m["roc_auc"]
            deltas.append(delta)
            rows.append(
                {
                    "run": run_name,
                    "config": ckpt["config"],
                    "target": name,
                    "n_cross": len(ds),
                    "auc_in": round(auc_in, 4),
                    "auc_cross": round(m["roc_auc"], 4),
                    "delta_auc": round(delta, 4),
                    "acc_cross": round(m["accuracy"], 4),
                    "eer_cross": round(m["eer"], 4),
                }
            )
            print(f"  {run_name}  {name}: in={auc_in:.4f} cross={m['roc_auc']:.4f} Δ={delta:+.4f}")
        if deltas:
            rows.append(
                {
                    "run": run_name, "config": ckpt["config"], "target": "MEAN",
                    "n_cross": 0, "auc_in": round(auc_in, 4),
                    "auc_cross": round(auc_in - float(np.mean(deltas)), 4),
                    "delta_auc": round(float(np.mean(deltas)), 4),
                    "acc_cross": float("nan"), "eer_cross": float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    out = out_dir / "cross_dataset.csv"
    df.to_csv(out, index=False)
    pivot = df[df.target != "MEAN"].pivot_table(index="run", columns="target", values="auc_cross")
    pivot["mean_delta"] = df[df.target == "MEAN"].set_index("run")["delta_auc"]
    pivot.to_csv(out_dir / "cross_dataset_pivot.csv")
    print(f"\nwrote {out}\n{pivot.to_string()}")
    return df


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-dataset evaluation of trained checkpoints")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--targets", nargs="+", required=True,
        help="one or more NAME=PATH, e.g. celebdf=/kaggle/working/celebdf_crops dfdc=/path",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--results-root", default="results", help="dir with summary.csv + <run>/best.pt (may be read-only)")
    p.add_argument("--out-dir", default=None, help="where to write cross_dataset.csv (default: cwd)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, _parse_targets(a.targets), Path(a.results_root), limit=a.limit,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
