"""C4(c) + C4b — test-time radial DCT band ablation.

One script, two questions, depending on which checkpoint you point it at:

  * **C4(c) — spatial models** (`baseline_spatial`, `xception`): does a purely
    spatial CNN actually *rely* on the frequency bands the forensics literature
    calls informative? If zeroing a band drops its AUC, it was using that band
    implicitly, without any explicit frequency machinery.
  * **C4b — frequency models** (`frequency_only`, `full`): is the frequency
    pathway's performance concentrated in one narrow band? That is FreqDebias's
    (CVPR 2025) "spectral bias" — over-reliance on specific bands as a
    generalisation liability. A sharply peaked drop profile = spectral bias
    present; a flat profile = reliance spread across the spectrum.

Method: for each radial band, take the model's *exact* input image (post
Resize/CenterCrop, so at the model's native resolution — never before, or the
resize's interpolation smears the zeroed band and contaminates the measurement,
the same trap that broke the first version of `src/spectra.py`), 2-D DCT it per
channel, zero the coefficients in that band, inverse-DCT back, then evaluate.

Radial convention: the DCT has DC at ``[0, 0]`` (corner origin, unlike the
centre-origin FFT in `src/spectra.py`), so radius is distance from that corner,
normalised so the highest-frequency corner is 1.0.

    python -m src.band_ablation --runs ffpp_baseline_spatial_seed0 ffpp_full_seed0 \
        --results-root results --dataset-name ffpp --seed 0 --image-size 128 \
        --n-bands 8 --limit 3000 --out-dir results/analysis/band_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .config import build_model
from .data import IMAGENET_MEAN, IMAGENET_STD, FaceCropDataset, read_manifest
from .engine import evaluate
from .utils import get_device


def radial_band_mask(size: int, band: int, n_bands: int) -> np.ndarray:
    """Boolean mask of DCT coefficients whose normalised corner-radius falls in
    ``band`` of ``n_bands`` equal-width radial bins over [0, 1]."""
    u, v = np.indices((size, size))
    r = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)
    r /= np.sqrt(2.0) * (size - 1)
    lo, hi = band / n_bands, (band + 1) / n_bands
    return (r >= lo) & (r <= hi if band == n_bands - 1 else r < hi)


class ZeroDCTBand:
    """PIL -> PIL: null out one radial DCT band, per colour channel.

    Applied AFTER Resize/CenterCrop so the image is exactly ``size`` and what the
    model sees is exactly what we ablated.
    """

    def __init__(self, mask: np.ndarray | None) -> None:
        self.mask = mask

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mask is None:  # clean reference
            return img.convert("RGB")
        from scipy.fft import dctn, idctn

        a = np.asarray(img.convert("RGB"), dtype=np.float64)
        for c in range(a.shape[2]):
            d = dctn(a[..., c], type=2, norm="ortho")
            d[self.mask] = 0.0
            a[..., c] = idctn(d, type=2, norm="ortho")
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _transform(image_size: int, mask: np.ndarray | None):
    import torchvision.transforms as T

    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        ZeroDCTBand(mask),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run(runs: list[str], results_root: Path, dataset_name: str, seed: int,
        image_size: int, n_bands: int = 8, limit: int | None = None,
        batch_size: int = 128, num_workers: int = 2,
        out_dir: Path | None = None) -> pd.DataFrame:
    device = get_device()
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = results_root / "manifests" / f"{dataset_name}_seed{seed}_sz{image_size}.csv"
    entries = read_manifest(manifest)["test"]
    if limit:
        entries = entries[:limit]

    rows: list[dict] = []
    for run_name in runs:
        ckpt_path = results_root / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"skip {run_name}: no checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        sz = ckpt["args"]["image_size"]
        model = build_model(ckpt["config"], image_size=sz, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        def _auc(mask):
            ds = FaceCropDataset(entries, transform=_transform(sz, mask))
            loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
            m, _, _ = evaluate(model, loader, device)
            return m["roc_auc"]

        clean = _auc(None)
        print(f"{run_name}  ({ckpt['config']})  clean auc={clean:.4f}")
        for b in range(n_bands):
            auc = _auc(radial_band_mask(sz, b, n_bands))
            rows.append({
                "run": run_name, "config": ckpt["config"], "band": b,
                "band_lo": round(b / n_bands, 3), "band_hi": round((b + 1) / n_bands, 3),
                "auc_clean": round(clean, 4), "auc_ablated": round(auc, 4),
                "delta_auc": round(auc - clean, 4),
            })
            print(f"    band {b} [{b/n_bands:.2f}-{(b+1)/n_bands:.2f}]  "
                  f"auc={auc:.4f}  delta={auc - clean:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "band_ablation.csv", index=False)
    pivot = df.pivot_table(index="run", columns="band", values="delta_auc")
    pivot.to_csv(out_dir / "band_ablation_delta.csv")
    print(f"\nwrote {out_dir/'band_ablation.csv'}\n{pivot.to_string()}")
    _plot(df, out_dir / "band_ablation.png")
    return df


def _plot(df: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run_name, g in df.groupby("run"):
        g = g.sort_values("band")
        ax.plot(g["band_lo"], g["delta_auc"], marker="o", label=run_name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("normalised radial DCT frequency (band lower edge)")
    ax.set_ylabel("AUC change when band is zeroed")
    ax.set_title("Which frequency bands does each model rely on?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--results-root", default="results")
    p.add_argument("--dataset-name", default="ffpp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-bands", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.runs, Path(a.results_root), a.dataset_name, a.seed, a.image_size,
        n_bands=a.n_bands, limit=a.limit,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
