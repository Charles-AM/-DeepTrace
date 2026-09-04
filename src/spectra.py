"""Real-vs-fake frequency analysis of FaceForensics++ face crops.

Motivation experiment for the analysis paper. Frequency-based detection assumes a
measurable real-vs-fake gap in the image spectrum (Durall et al. CVPR'20, Frank et
al. ICML'20). This script measures that gap directly and asks what heavy JPEG does
to it.

Fixes over the first version:
  * NO interpolated resize before the transform (bilinear resize is a low-pass that
    erases the high-frequency fingerprint). Crops are centre-cropped at native
    resolution instead.
  * 2-D Hann window before the FFT to suppress edge-leakage structure.
  * DCT-domain 2-D maps (the domain F3-Net / Frank et al. actually use), reported as
    the fake-minus-real difference and as a per-coefficient t-statistic map, so
    "is there a gap and does it exceed image-to-image noise" is answerable.
  * High-pass residual (image minus Gaussian blur) variant — where upsampling /
    GAN grid artifacts concentrate.
  * ±1 std bands on the radial curves.

    python -m src.spectra --manifest results/manifests/ffpp_seed0_sz128.csv \
        --limit 3000 --jpeg 30 --crop 160 --splits all --out-dir results/spectra
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from scipy.fft import dctn
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    dctn = None
    gaussian_filter = None

_EPS = 1e-8
_R601 = np.array([0.299, 0.587, 0.114])


def _luma(path: str, jpeg: int | None, crop: int) -> np.ndarray | None:
    """Native-resolution centre-cropped luma in [0, 1]; optional in-memory JPEG."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if jpeg is not None:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=jpeg)
                buf.seek(0)
                im = Image.open(buf).convert("RGB")
            a = np.asarray(im, dtype=np.float64) / 255.0
    except (FileNotFoundError, OSError):
        return None
    g = a @ _R601
    h, w = g.shape
    if h < crop or w < crop:
        return None
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return g[y0 : y0 + crop, x0 : x0 + crop]


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n)
    return np.outer(w, w)


def _radial_average(power: np.ndarray) -> np.ndarray:
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy).astype(int)
    nbins = min(cy, cx)
    tbin = np.bincount(r.ravel(), power.ravel())[:nbins]
    nr = np.bincount(r.ravel())[:nbins]
    return tbin / np.maximum(nr, 1)


class _Accum:
    """Streaming mean/var for a fixed-shape array."""

    def __init__(self) -> None:
        self.n = 0
        self.s = None
        self.ss = None

    def add(self, x: np.ndarray) -> None:
        self.n += 1
        self.s = x.copy() if self.s is None else self.s + x
        self.ss = x * x if self.ss is None else self.ss + x * x

    @property
    def mean(self) -> np.ndarray:
        return self.s / max(self.n, 1)

    @property
    def var(self) -> np.ndarray:
        return np.maximum(self.ss / max(self.n, 1) - self.mean**2, 0.0)


def _tmap(a: _Accum, b: _Accum) -> np.ndarray:
    se = np.sqrt(a.var / max(a.n, 1) + b.var / max(b.n, 1)) + _EPS
    return (b.mean - a.mean) / se


def run(manifest: Path, limit: int, jpeg: int, crop: int, splits: str, out_dir: Path) -> dict:
    if dctn is None:
        raise ImportError("scipy is required (scipy.fft.dctn, scipy.ndimage.gaussian_filter)")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    if splits != "all" and "split" in df.columns:
        df = df[df.split.isin(splits.split(","))]
    per = limit // 2
    rows_real = df[df.label == 0].sample(min(per, int((df.label == 0).sum())), random_state=0)
    rows_fake = df[df.label == 1].sample(min(per, int((df.label == 1).sum())), random_state=0)
    win = _hann2d(crop)
    conds = [(None, "c23"), (jpeg, f"jpeg{jpeg}")]

    # accumulators[(cond, class, domain)] -> _Accum ; radial[(cond, class)] -> _Accum
    acc: dict[tuple, _Accum] = {}
    rad: dict[tuple, _Accum] = {}
    for jq, clabel in conds:
        for tag, rows in (("real", rows_real), ("fake", rows_fake)):
            n = 0
            for p in rows.path:
                g = _luma(p, jq, crop)
                if g is None:
                    continue
                x = g - g.mean()
                power = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
                rad.setdefault((clabel, tag), _Accum()).add(_radial_average(power))
                acc.setdefault((clabel, tag, "raw"), _Accum()).add(
                    np.log10(np.abs(dctn(g, type=2, norm="ortho")) + _EPS)
                )
                resid = g - gaussian_filter(g, 2.0)
                acc.setdefault((clabel, tag, "residual"), _Accum()).add(
                    np.log10(np.abs(dctn(resid, type=2, norm="ortho")) + _EPS)
                )
                n += 1
            print(f"{clabel:8s} {tag:4s}  n={n}")

    # ---- radial curves + gap ------------------------------------------------
    freq = np.arange(rad[(conds[0][1], "real")].mean.shape[0]) / (crop / 2)
    rad_df = {"freq": freq}
    summary: dict[str, float] = {}
    for _, clabel in conds:
        r_m, f_m = rad[(clabel, "real")].mean, rad[(clabel, "fake")].mean
        r_s, f_s = np.sqrt(rad[(clabel, "real")].var), np.sqrt(rad[(clabel, "fake")].var)
        rad_df.update({
            f"real_{clabel}": r_m, f"real_{clabel}_std": r_s,
            f"fake_{clabel}": f_m, f"fake_{clabel}_std": f_s,
        })
        gap_db = 10 * (np.log10(f_m + _EPS) - np.log10(r_m + _EPS))
        rad_df[f"gap_db_{clabel}"] = gap_db
        hi = freq > 0.5
        summary[f"radial_absgap_db_{clabel}"] = float(np.nanmean(np.abs(gap_db)))
        summary[f"radial_absgap_db_hi_{clabel}"] = float(np.nanmean(np.abs(gap_db[hi])))
    pd.DataFrame(rad_df).to_csv(out_dir / "radial.csv", index=False)

    # ---- DCT-domain difference + t maps -----------------------------------
    npz: dict[str, np.ndarray] = {}
    for _, clabel in conds:
        for domain in ("raw", "residual"):
            a, b = acc[(clabel, "real", domain)], acc[(clabel, "fake", domain)]
            diff = b.mean - a.mean
            t = _tmap(a, b)
            npz[f"dct_diff_{domain}_{clabel}"] = diff
            npz[f"dct_t_{domain}_{clabel}"] = t
            summary[f"dct_{domain}_{clabel}_maxabs_t"] = float(np.nanmax(np.abs(t)))
            summary[f"dct_{domain}_{clabel}_frac_t_gt3"] = float(np.mean(np.abs(t) > 3))
            summary[f"dct_{domain}_{clabel}_frac_t_gt5"] = float(np.mean(np.abs(t) > 5))
    np.savez_compressed(out_dir / "dct_maps.npz", **npz)
    pd.Series(summary).to_csv(out_dir / "summary.csv")
    for k, v in summary.items():
        print(f"  {k:38s} {v:.4f}")

    _plot(rad_df, freq, npz, jpeg, out_dir)
    return summary


def _plot(rad_df, freq, npz, jpeg, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
        return

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for j, cl in enumerate(("c23", f"jpeg{jpeg}")):
        a = ax[0, j]
        for tag, c in (("real", "tab:blue"), ("fake", "tab:red")):
            m, s = rad_df[f"{tag}_{cl}"], rad_df[f"{tag}_{cl}_std"]
            a.loglog(freq[1:], m[1:], c=c, lw=2, label=tag)
            a.fill_between(freq[1:], np.maximum(m[1:] - s[1:], _EPS), m[1:] + s[1:], color=c, alpha=0.2)
        a.set_title(f"radial power — {cl}"); a.set_xlabel("norm. radial freq"); a.grid(alpha=0.3, which="both")
        a.legend()
    ax[0, 2].axhline(0, color="k", lw=0.8)
    for cl, c in (("c23", "tab:green"), (f"jpeg{jpeg}", "tab:orange")):
        ax[0, 2].plot(freq, rad_df[f"gap_db_{cl}"], label=cl, c=c, lw=2)
    ax[0, 2].set_title("fake − real gap (dB)"); ax[0, 2].set_xlabel("norm. radial freq")
    ax[0, 2].grid(alpha=0.3); ax[0, 2].legend()

    panels = [("dct_diff_raw_c23", "DCT log|coef| fake−real (raw, c23)", "RdBu_r", None),
              ("dct_diff_residual_c23", "DCT fake−real (residual, c23)", "RdBu_r", None),
              ("dct_t_raw_c23", "per-coef t (raw, c23)", "RdBu_r", 5.0)]
    for k, (key, title, cmap, vlim) in enumerate(panels):
        arr = npz[key]
        v = vlim or np.nanpercentile(np.abs(arr), 99)
        im = ax[1, k].imshow(arr, cmap=cmap, vmin=-v, vmax=v, origin="upper")
        ax[1, k].set_title(title); ax[1, k].set_xlabel("freq u"); ax[1, k].set_ylabel("freq v")
        fig.colorbar(im, ax=ax[1, k], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "spectra.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_dir/'spectra.png'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--jpeg", type=int, default=30)
    p.add_argument("--crop", type=int, default=160, help="native-resolution centre crop; NO resize")
    p.add_argument("--splits", default="test", help="'all', or comma list e.g. train,val,test")
    p.add_argument("--out-dir", default="results/spectra")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.manifest), a.limit, a.jpeg, a.crop, a.splits, Path(a.out_dir))


if __name__ == "__main__":
    main()
