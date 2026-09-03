"""Task 12 — robustness suite.

For every trained checkpoint, re-evaluate the held-out test split under a battery
of real-world perturbations (JPEG, blur, noise, downscale, contrast loss) at four
severities, then produce:

  * ``results/robustness.csv``            — long-form (run, perturbation, severity, AUC, ...)
  * ``results/robustness_<pert>.png``     — AUC vs severity per perturbation, all models
  * ``results/robustness_scores.csv``     — AUC(severity 4) / AUC(clean) per (run, perturbation)

Supersedes ``jpeg_robustness.py`` (kept as a thin alias).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .config import build_model
from .data import IMAGENET_MEAN, IMAGENET_STD, read_manifest
from .engine import evaluate
from .transforms_perturb import PERTURBATIONS, SEVERITIES, make_perturbation
from .utils import get_device


def _eval_transform(image_size: int, perturbation: str, severity: int):
    import torchvision.transforms as T

    return T.Compose(
        [
            make_perturbation(perturbation, severity),
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluate_checkpoint(
    ckpt_path: Path,
    manifest: Path,
    device: torch.device,
    perturbations=tuple(PERTURBATIONS),
    severities=SEVERITIES,
    batch_size: int = 128,
    num_workers: int = 2,
    limit: int | None = None,
) -> list[dict]:
    from .data import FaceCropDataset

    ckpt = torch.load(ckpt_path, map_location=device)
    image_size = ckpt["args"]["image_size"]
    model = build_model(ckpt["config"], image_size=image_size, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    entries = read_manifest(manifest)["test"]
    if limit:
        entries = entries[:limit]

    run = ckpt_path.parent.name
    rows: list[dict] = []

    def _auc(pert: str, sev: int) -> dict:
        ds = FaceCropDataset(entries, transform=_eval_transform(image_size, pert, sev))
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
        m, _, _ = evaluate(model, loader, device)
        return m

    clean = _auc(perturbations[0], 0)
    for pert in perturbations:
        rows.append({"run": run, "perturbation": pert, "severity": 0,
                     "roc_auc": clean["roc_auc"], "accuracy": clean["accuracy"], "eer": clean["eer"]})
        for sev in severities:
            m = _auc(pert, sev)
            rows.append({"run": run, "perturbation": pert, "severity": sev,
                         "roc_auc": m["roc_auc"], "accuracy": m["accuracy"], "eer": m["eer"]})
            print(f"  {run}  {pert} sev{sev}  auc={m['roc_auc']:.4f}")
    return rows


def run(
    runs: list[str],
    results_root: Path,
    dataset_name: str,
    seed: int,
    image_size: int,
    perturbations=tuple(PERTURBATIONS),
    limit: int | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    device = get_device()
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()  # results_root may be read-only
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"missing split manifest {manifest} — train a model first")

    all_rows: list[dict] = []
    for run_name in runs:
        ckpt = results_root / run_name / "best.pt"
        if not ckpt.exists():
            print(f"skip {run_name}: no checkpoint")
            continue
        all_rows += evaluate_checkpoint(
            ckpt, manifest, device, perturbations=perturbations, limit=limit
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "robustness.csv", index=False)
    for pert in perturbations:
        _plot(df[df.perturbation == pert], pert, out_dir / f"robustness_{pert}.png")

    scores = (
        df.pivot_table(index=["run", "perturbation"], columns="severity", values="roc_auc")
        .assign(score=lambda t: t[max(SEVERITIES)] / t[0])
        .reset_index()[["run", "perturbation", "score"]]
    )
    scores.to_csv(out_dir / "robustness_scores.csv", index=False)
    print(f"\nwrote {out_dir/'robustness.csv'}\n{scores.to_string(index=False)}")
    return df


def _plot(df: pd.DataFrame, pert: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for run_name, g in df.groupby("run"):
        g = g.sort_values("severity")
        ax.plot(g["severity"], g["roc_auc"], marker="o", label=run_name)
    ax.set_xlabel(f"{pert} severity  (0 = clean)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title(f"Robustness to {pert}")
    ax.set_xticks([0, *SEVERITIES])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Robustness sweep over trained checkpoints")
    p.add_argument("--runs", nargs="+", required=True, help="run dir names under results/")
    p.add_argument("--dataset-name", default="rvf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--results-root", default="results", help="dir with manifests/ + <run>/best.pt (may be read-only)")
    p.add_argument("--out-dir", default=None, help="where to write robustness outputs (default: cwd)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        perturbations=tuple(a.perturbations), limit=a.limit,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
