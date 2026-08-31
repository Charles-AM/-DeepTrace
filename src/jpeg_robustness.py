"""Task 6 — JPEG compression robustness.

For each trained checkpoint, re-evaluate the held-out test split with every image
JPEG-recompressed (in memory) at a range of quality levels, then plot ROC-AUC vs
quality for all models on one axis and report a robustness score
``AUC(q=30) / AUC(original)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import build_model
from .data import FaceCropDataset, IMAGENET_MEAN, IMAGENET_STD, read_manifest
from .engine import evaluate
from .transforms_jpeg import JpegDegrade
from .utils import get_device

DEFAULT_QUALITIES = [100, 90, 70, 50, 30]


def _eval_transform(image_size: int, quality: int):
    import torchvision.transforms as T

    return T.Compose(
        [
            JpegDegrade(quality),
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluate_checkpoint_over_qualities(
    ckpt_path: Path,
    manifest: Path,
    device: torch.device,
    qualities=DEFAULT_QUALITIES,
    batch_size: int = 128,
    num_workers: int = 2,
    limit: int | None = None,
) -> dict[int, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    model = build_model(ckpt["config"], image_size=args["image_size"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_entries = read_manifest(manifest)["test"]
    if limit:
        test_entries = test_entries[:limit]

    out: dict[int, dict] = {}
    for q in qualities:
        ds = FaceCropDataset(
            test_entries, image_size=args["image_size"], transform=_eval_transform(args["image_size"], q)
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        metrics, _, _ = evaluate(model, loader, device)
        out[q] = metrics
        print(f"  {ckpt_path.parent.name}  q={q:3d}  auc={metrics['roc_auc']:.4f}")
    return out


def run(
    runs: list[str],
    results_root: Path,
    dataset_name: str,
    seed: int,
    image_size: int,
    qualities=DEFAULT_QUALITIES,
    limit: int | None = None,
) -> pd.DataFrame:
    device = get_device()
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing split manifest {manifest} — train a model first")

    rows = []
    for run_name in runs:
        ckpt = results_root / run_name / "best.pt"
        if not ckpt.exists():
            print(f"skip {run_name}: no checkpoint")
            continue
        per_q = evaluate_checkpoint_over_qualities(
            ckpt, manifest, device, qualities=qualities, limit=limit
        )
        base = per_q[max(qualities)]["roc_auc"]
        for q, m in per_q.items():
            rows.append(
                {
                    "run": run_name,
                    "config": run_name.split("_")[1] if "_" in run_name else run_name,
                    "quality": q,
                    "roc_auc": m["roc_auc"],
                    "accuracy": m["accuracy"],
                    "eer": m["eer"],
                    "robustness_vs_best": m["roc_auc"] / base if base else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = results_root / "jpeg_robustness.csv"
    df.to_csv(out_csv, index=False)
    _plot(df, results_root / "jpeg_robustness.png")

    q_lo, q_hi = min(qualities), max(qualities)
    wide = df.pivot_table(index="run", columns="quality", values="roc_auc")
    score = (wide[q_lo] / wide[q_hi]).rename(f"robustness_score(q{q_lo}/q{q_hi})")
    score.to_csv(results_root / "jpeg_robustness_score.csv")
    print(f"\nwrote {out_csv}\n{score}")
    return df


def _plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for run_name, g in df.groupby("run"):
        g = g.sort_values("quality")
        ax.plot(g["quality"], g["roc_auc"], marker="o", label=run_name)
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Robustness to JPEG compression")
    ax.invert_xaxis()  # heavier compression to the right
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JPEG robustness sweep over trained checkpoints")
    p.add_argument("--runs", nargs="+", required=True, help="run dir names under results/")
    p.add_argument("--dataset-name", default="rvf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--qualities", nargs="+", type=int, default=DEFAULT_QUALITIES)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--results-root", default="results")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        qualities=a.qualities, limit=a.limit)


if __name__ == "__main__":
    main()
