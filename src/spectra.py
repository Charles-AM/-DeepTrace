"""Mean radial power spectrum of real vs fake face crops.

Motivation figure for the analysis paper: frequency-based detection assumes a
measurable real-vs-fake gap in the image spectrum. This computes the azimuthally
averaged power spectral density (via 2-D FFT) for real and fake crops, then repeats
it after re-compressing every crop to JPEG quality 30. If the gap that exists at
c23 collapses under heavy JPEG, that explains why frequency methods fail on the
compressed media deployed systems actually see.

    python -m src.spectra --manifest results/manifests/ffpp_seed0_sz128.csv \
        --limit 1500 --jpeg 30 --out-dir results/spectra
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _radial_psd(gray: np.ndarray) -> np.ndarray:
    """Azimuthally averaged power spectrum of a 2-D array, length = min(H, W)//2."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    psd = np.abs(f) ** 2
    h, w = psd.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    nbins = min(cy, cx)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin[:nbins] / np.maximum(nr[:nbins], 1)
    return radial


def _load_gray(path: str, jpeg: int | None, size: int = 128) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        if jpeg is not None:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        return np.asarray(im.convert("L"), dtype=np.float64) / 255.0


def run(manifest: Path, limit: int, jpeg: int, out_dir: Path, size: int = 128) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    df = df[df.split == "test"] if "split" in df.columns else df
    rng = np.random.default_rng(0)
    per = limit // 2
    rows_real = df[df.label == 0].sample(min(per, (df.label == 0).sum()), random_state=0)
    rows_fake = df[df.label == 1].sample(min(per, (df.label == 1).sum()), random_state=0)

    curves = {}
    for tag, rows in [("real", rows_real), ("fake", rows_fake)]:
        for jq, jlabel in [(None, "c23"), (jpeg, f"jpeg{jpeg}")]:
            acc = None
            n = 0
            for p in rows.path:
                try:
                    radial = _radial_psd(_load_gray(p, jq, size))
                except (FileNotFoundError, OSError):
                    continue
                acc = radial if acc is None else acc + radial
                n += 1
            curves[f"{tag}_{jlabel}"] = acc / max(n, 1)
            print(f"{tag:4s} {jlabel:8s}  n={n}")

    freq = np.arange(len(next(iter(curves.values())))) / (size / 2)  # normalised 0..1
    out = pd.DataFrame({"freq": freq, **curves})
    out.to_csv(out_dir / "radial_psd.csv", index=False)

    # gap = log10(fake) - log10(real), per condition
    for cond in ("c23", f"jpeg{jpeg}"):
        gap = np.log10(out[f"fake_{cond}"]) - np.log10(out[f"real_{cond}"])
        out[f"gap_{cond}"] = gap
        print(f"mean |log10 fake/real| gap  {cond}: {np.nanmean(np.abs(gap)):.4f}"
              f"   high-freq (>0.5): {np.nanmean(np.abs(gap[freq > 0.5])):.4f}")
    out.to_csv(out_dir / "radial_psd.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for i, cond in enumerate(("c23", f"jpeg{jpeg}")):
            ax[i].semilogy(freq, out[f"real_{cond}"], label="real", lw=2)
            ax[i].semilogy(freq, out[f"fake_{cond}"], label="fake", lw=2)
            ax[i].set_title(cond)
            ax[i].set_xlabel("normalised radial frequency")
            ax[i].grid(alpha=0.3)
        ax[0].set_ylabel("mean power")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "radial_psd.png", dpi=150)
        plt.close(fig)
        print(f"wrote {out_dir/'radial_psd.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--jpeg", type=int, default=30)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--out-dir", default="results/spectra")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.manifest), a.limit, a.jpeg, Path(a.out_dir), size=a.image_size)


if __name__ == "__main__":
    main()
