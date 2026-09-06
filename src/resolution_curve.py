"""V7 — how much resolution does an FF++ ablation actually have?

Contribution 2's core figure. Every other analysis in this project reports an
interval for *our* study; this one asks the general question behind it:

    given N independent test videos, how wide is the interval on an
    architectural difference — and how large must N be before a claimed
    effect of +0.014 AUC is resolvable at all?

Method. Take one paired prediction set (model A vs model B on identical test
items). For each candidate size ``n``, draw ``n`` units *without replacement*,
run the paired cluster bootstrap inside that subsample, and record the 95%
interval's half-width. Repeat over many draws and report the median. That traces
the empirical width-vs-N curve over the range our data actually covers.

Beyond that range we fit ``log(halfwidth) = a + b*log(n)`` and solve for the N at
which the half-width falls below a target. **That number is a projection, not a
measurement** — it assumes additional videos would resemble ours in difficulty and
correlation structure. It is reported with the fitted exponent so a reader can see
how close to the -0.5 of independent sampling the data actually sits.

Class balance. A video is entirely real or entirely fake, so a small random draw
of videos can be single-class and leave AUC undefined. Video draws are therefore
stratified to preserve the real:fake unit ratio. Components need no such fix: a
source-target component carries its own real and fake videos, which is one reason
its intervals behave more stably than the video unit's.

    python -m src.resolution_curve --a preds_xception.csv --b preds_fad.csv \\
        --unit video --out-dir results/analysis/resolution
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# The externally-anchored thresholds from docs/ESSENCE.md section 7. 0.014 is the
# verified FAD-specific gain (arXiv 2007.09355v2 Fig 7a); the others are the
# stakeholder-judgement practical margins carried for sensitivity.
DEFAULT_TARGETS = (0.020, 0.014, 0.010, 0.005)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-corrected.

    Equivalent to sklearn's roc_auc_score but ~an order of magnitude faster on the
    small arrays this module evaluates hundreds of thousands of times, and it
    keeps the module importable without sklearn. Pinned against a brute-force
    pairwise definition in tests/test_resolution_curve.py.
    """
    # np.unique sorts and gives tie-group sizes in one pass, so average ranks come
    # out fully vectorised. An earlier version walked tie blocks in a Python loop,
    # which made this O(m) *interpreted* steps per call — the dominant cost when
    # the curve evaluates it ~10^6 times.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    ranks = ((starts + ends + 1) / 2.0)[inv]     # mean of ranks start+1..ends
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _units(groups: list[str]) -> dict[str, list[int]]:
    index_of: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        index_of.setdefault(g, []).append(i)
    return index_of


def _unit_class(labels: np.ndarray, idx: list[int]) -> int:
    """1 if every item in the unit is fake, 0 if all real, -1 if mixed."""
    vals = set(labels[idx].tolist())
    return vals.pop() if len(vals) == 1 else -1


def _draw(keys, key_class, n, rng) -> list[str] | None:
    """Draw `n` units, stratified when units are single-class so the draw cannot
    be degenerate. Returns None when `n` cannot be filled."""
    mixed = [k for k in keys if key_class[k] == -1]
    if len(mixed) == len(keys):                       # every unit self-balanced
        return list(rng.choice(keys, size=n, replace=False))
    pos = [k for k in keys if key_class[k] == 1]
    neg = [k for k in keys if key_class[k] == 0]
    share = len(pos) / len(keys)
    n_pos = max(1, min(len(pos), int(round(n * share))))
    n_neg = n - n_pos
    if n_neg < 1 or n_neg > len(neg):
        return None
    return (list(rng.choice(pos, size=n_pos, replace=False))
            + list(rng.choice(neg, size=n_neg, replace=False)))


def curve(labels, sa, sb, groups, ns, n_draws=200, n_boot=400, seed=0) -> list[dict]:
    """Median 95% half-width as a function of the number of independent units."""
    rng = np.random.default_rng(seed)
    index_of = _units(groups)
    keys = np.array(list(index_of))
    key_class = {k: _unit_class(labels, index_of[k]) for k in index_of}

    rows = []
    for n in ns:
        if n > len(keys):
            continue
        widths, points = [], []
        for _ in range(n_draws):
            picked = _draw(keys, key_class, n, rng)
            if picked is None:
                continue
            sub = np.concatenate([index_of[k] for k in picked])
            y = labels[sub]
            if y.min() == y.max():
                continue
            # paired bootstrap *within* the subsample: resample its n units
            sub_index = [index_of[k] for k in picked]
            diffs = []
            for _ in range(n_boot):
                take = rng.integers(0, n, size=n)
                bi = np.concatenate([sub_index[t] for t in take])
                yb = labels[bi]
                if yb.min() == yb.max():
                    continue
                diffs.append(auc(yb, sb[bi]) - auc(yb, sa[bi]))
            if len(diffs) < n_boot // 4:      # too many degenerate replicates
                continue
            d = np.asarray(diffs)
            lo, hi = np.percentile(d, [2.5, 97.5])
            widths.append((hi - lo) / 2)
            points.append(auc(y, sb[sub]) - auc(y, sa[sub]))
        if not widths:
            continue
        w = np.asarray(widths)
        # A zero-width interval is degenerate, not precise: at small n with a
        # skewed unit ratio (1 real : 4 fake here) a draw can hold so few real
        # units that every bootstrap replicate returns the same AUC. Reporting
        # that as high resolution would invert the figure's meaning, so the size
        # is recorded, flagged, and excluded from the fit.
        zero_frac = float((w <= 0).mean())
        if np.median(w) <= 0:
            print(f"  n={n:4d}  DEGENERATE — {zero_frac:.0%} of draws gave a "
                  f"zero-width interval; excluded from the fit")
            rows.append({"n_units": n, "n_draws_used": len(w),
                         "halfwidth_median": 0.0, "halfwidth_q25": 0.0,
                         "halfwidth_q75": round(float(np.percentile(w, 75)), 5),
                         "diff_median": round(float(np.median(points)), 5),
                         "zero_width_frac": round(zero_frac, 4), "degenerate": True})
            continue
        rows.append({"zero_width_frac": round(zero_frac, 4), "degenerate": False,
                     "n_units": n, "n_draws_used": len(w),
                     "halfwidth_median": round(float(np.median(w)), 5),
                     "halfwidth_q25": round(float(np.percentile(w, 25)), 5),
                     "halfwidth_q75": round(float(np.percentile(w, 75)), 5),
                     "diff_median": round(float(np.median(points)), 5)})
        print(f"  n={n:4d}  half-width median={rows[-1]['halfwidth_median']:.4f} "
              f"[q25 {rows[-1]['halfwidth_q25']:.4f}, q75 {rows[-1]['halfwidth_q75']:.4f}]")
    return rows


def fit_power_law(rows: list[dict], targets=DEFAULT_TARGETS) -> dict:
    """Fit log(halfwidth) = a + b*log(n) and solve for the N reaching each target.

    b is the informative number: -0.5 is the rate independent sampling would give,
    and a shallower slope means added videos buy less than sqrt(n) would suggest.
    """
    usable = [r for r in rows if r["halfwidth_median"] > 0
              and not r.get("degenerate", False)]
    dropped = [r["n_units"] for r in rows if r not in usable]
    n = np.array([r["n_units"] for r in usable], dtype=float)
    w = np.array([r["halfwidth_median"] for r in usable], dtype=float)
    if len(n) < 3:
        raise ValueError(
            f"need >=3 non-degenerate curve points to fit, got {len(n)} "
            f"(sizes {dropped} were degenerate). Widen --ns or raise --n-boot.")
    x, y = np.log(n), np.log(w)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1.0 - float(resid.var() / y.var()) if y.var() > 0 else float("nan")

    out = {"slope": round(float(b), 4), "intercept": round(float(a), 4),
           "r2": round(r2, 4), "n_max_observed": int(n.max()),
           "n_points_fitted": len(n),
           "sizes_excluded_degenerate": ";".join(str(d) for d in dropped) or "none",
           "slope_note": "-0.5 == independent-sampling rate"}
    for t in targets:
        need = float(np.exp((np.log(t) - a) / b))
        out[f"n_for_halfwidth_{t}"] = int(np.ceil(need))
        out[f"extrapolated_{t}"] = bool(need > n.max())
    return out


def run(path_a: Path, path_b: Path, unit: str = "video", aggregate: bool = True,
        ns=None, n_draws: int = 200, n_boot: int = 400, seed: int = 0,
        targets=DEFAULT_TARGETS, out_dir: Path | None = None) -> tuple[list[dict], dict]:
    # imported lazily: cluster_boot pulls in sklearn, and the curve machinery
    # above is deliberately dependency-light so it stays unit-testable anywhere.
    from .cluster_boot import _load, _prepare

    labels, sa, sb, vids, comps = _prepare(_load(path_a), _load(path_b), aggregate)
    groups = {"video": vids, "component": comps}[unit]
    n_avail = len(set(groups))
    if ns is None:
        ns = [n for n in (5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150)
              if n <= n_avail]
        if n_avail not in ns:
            ns.append(n_avail)
    print(f"unit={unit}  available={n_avail}  items={len(labels)}  sizes={ns}")

    rows = curve(labels, sa, sb, groups, ns, n_draws=n_draws, n_boot=n_boot, seed=seed)
    fit = fit_power_law(rows, targets=targets)
    print(f"\n  fit: halfwidth ~ n^{fit['slope']}  (R2={fit['r2']})")
    for t in targets:
        flag = " [EXTRAPOLATED beyond observed range]" if fit[f"extrapolated_{t}"] else ""
        print(f"  to resolve {t:+.3f}: ~{fit[f'n_for_halfwidth_{t}']} {unit}s{flag}")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{unit}_{Path(path_a).stem}_vs_{Path(path_b).stem}"
        with (out_dir / f"curve_{stem}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with (out_dir / f"fit_{stem}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fit)); w.writeheader(); w.writerow(fit)
        print(f"wrote {out_dir}/curve_{stem}.csv and fit_{stem}.csv")
    return rows, fit


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--unit", default="video", choices=["video", "component"])
    p.add_argument("--frame-level", action="store_true",
                   help="score individual crops instead of video-averaged scores")
    p.add_argument("--ns", type=int, nargs="*", default=None)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--n-boot", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", type=float, nargs="*", default=list(DEFAULT_TARGETS))
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.a), Path(a.b), unit=a.unit, aggregate=not a.frame_level, ns=a.ns,
        n_draws=a.n_draws, n_boot=a.n_boot, seed=a.seed, targets=tuple(a.targets),
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
