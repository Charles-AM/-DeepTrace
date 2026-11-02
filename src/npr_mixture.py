"""Expose the mixture structure that the crossed bootstrap's one-row summary hides.

`crossed_boot.py` is protected — its empty diff is part of the isolation proof —
and its `--out-dir` writes a single summary row. This module is **reporting only**.
It changes no estimate and recomputes nothing differently.

Two things it produces that the summary cannot:

1. **The exact seed-only distribution.** `crossed_boot`'s `seed_only` statistic is
   the mean of the per-seed full-test differences over a resample of seeds
   (`crossed_boot.py:153-159`), so with n runs there are only n**n ordered
   resamples and the distribution can be *enumerated* rather than sampled.

   ⚠️ Monte Carlo error is **not** the limitation. The tail mass can be computed
   exactly. The limitation is inferential: the whole distribution rests on n
   observed run values, so a probability from it is descriptive of those runs, not
   evidence about the population of possible training executions.

2. **The crossed distribution stratified by copies of the collapsed run.** A third
   of five-run resamples contain none of it, so the pooled interval is a mixture
   whose lower tail is one failed optimisation resampled with replacement.

   This cannot be built from per-seed full-test differences. A crossed draw changes
   the component set, so the per-seed differences must be recomputed on each
   component draw *before* seeds are resampled.

    python -m src.npr_mixture \
        --a-glob 'results/predictions_npr/*_xception_seed*_test.csv' \
        --b-glob 'results/predictions_npr/*_xception_npr_seed*_test.csv' \
        --out-dir results/analysis/npr/mixture
"""
from __future__ import annotations

import argparse
import csv
import itertools
from math import comb
from pathlib import Path

import numpy as np

from .crossed_boot import load_matched
from .resolution_curve import auc

STRATA = ((0, 0), (1, 1), (2, 2), (3, None))     # (low, high) inclusive; None = open


def per_seed_diffs(labels, A, B, idx=None) -> list[float | None]:
    """B minus A per seed on `idx` (default: the full test set). None where the
    draw left a single class, matching crossed_boot's `diff_on`."""
    idx = np.arange(len(labels)) if idx is None else idx
    y = labels[idx]
    if y.min() == y.max():
        return [None] * A.shape[0]
    return [float(auc(y, B[s][idx]) - auc(y, A[s][idx])) for s in range(A.shape[0])]


def exact_seed_only(diffs, collapsed: int = 0):
    """Every ordered resample of n runs from n, enumerated.

    Returns (values, weights, copies). n**n rows — 3125 at n=5, 256 at n=4 — so
    this is exact, not sampled.
    """
    d = np.asarray([x for x in diffs], dtype=float)
    n = len(d)
    vals, cps = [], []
    for pick in itertools.product(range(n), repeat=n):
        vals.append(d[list(pick)].mean())
        cps.append(pick.count(collapsed))
    w = np.full(len(vals), 1.0 / n ** n)
    return np.asarray(vals), w, np.asarray(cps)


def compositions(n: int):
    """Distinct multisets of size n from n, with their exact multinomial weights.

    C(2n-1, n-1) of them: 35 at n=4, 126 at n=5.
    """
    out = []
    for combo in itertools.combinations_with_replacement(range(n), n):
        counts = [combo.count(i) for i in range(n)]
        ways = 1
        rem = n
        for c in counts:
            ways *= comb(rem, c)
            rem -= c
        out.append((combo, ways / n ** n, counts[0]))
    return out


def copy_count_probs(n: int, collapsed: int = 0) -> dict[int, float]:
    """Exact P(k copies of one particular run) in an ordered n-from-n resample."""
    return {k: comb(n, k) * (n - 1) ** (n - k) / n ** n for k in range(n + 1)}


def crossed_replicates(labels, A, B, groups, n_boot: int = 4000, seed: int = 0,
                       collapsed: int = 0):
    """Crossed replicates, each tagged with how many copies of `collapsed` it drew.

    Components are resampled first and the per-seed differences recomputed on that
    draw, then seeds are resampled — the same order as crossed_boot.
    """
    rng = np.random.default_rng(seed)
    n_seeds = A.shape[0]
    index_of: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        index_of.setdefault(g, []).append(i)
    keys = list(index_of)
    reps, cps, skipped = [], [], 0
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([index_of[keys[k]] for k in picked])
        d = per_seed_diffs(labels, A, B, idx)
        seeds_used = rng.integers(0, n_seeds, size=n_seeds)
        vals = [d[s] for s in seeds_used if d[s] is not None]
        if not vals:
            skipped += 1
            continue
        reps.append(float(np.mean(vals)))
        cps.append(int((seeds_used == collapsed).sum()))
    return np.asarray(reps), np.asarray(cps), skipped


def _wq(v, w, q):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, v))


def summarise(v, w=None) -> dict:
    if len(v) == 0:
        return {"n": 0}
    w = np.ones(len(v)) if w is None else np.asarray(w)
    return {"n": len(v), "weight": float(w.sum()),
            "mean": float(np.average(v, weights=w)), "median": _wq(v, w, 0.5),
            "p2.5": _wq(v, w, 0.025), "p97.5": _wq(v, w, 0.975),
            "p_above_0": float(w[v > 0].sum() / w.sum())}


def _row(label, s):
    if not s.get("n"):
        return f"  {label:28s} (no replicates)"
    return (f"  {label:28s} n={s['n']:>5d} wt={s['weight']:.5f} "
            f"mean={s['mean']:+.4f} med={s['median']:+.4f} "
            f"[{s['p2.5']:+.4f}, {s['p97.5']:+.4f}] P(>0)={s['p_above_0']:.4f}")


def _figure(ev, ew, ec, cv, cc, out: Path, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].hist(ev, bins=60, weights=ew, color="#555")
    ax[0].axvline(0, color="crimson", lw=1)
    ax[0].set_title(f"Exact seed-only distribution ({n}**{n} = {n ** n} ordered resamples)")
    ax[0].set_xlabel("mean AUC difference (B − A)")
    for lo, hi in STRATA:
        m = (cc >= lo) if hi is None else (cc == lo)
        if m.sum():
            lab = f"{lo}+ copies" if hi is None else f"{lo} cop{'y' if lo == 1 else 'ies'}"
            ax[1].hist(cv[m], bins=40, alpha=0.6, label=f"{lab} (n={m.sum()})")
    ax[1].axvline(0, color="crimson", lw=1)
    ax[1].set_title("Crossed replicates, stratified by copies of the collapsed run")
    ax[1].set_xlabel("mean AUC difference (B − A)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-glob", required=True)
    p.add_argument("--b-glob", required=True)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--collapsed-seed", type=int, default=0,
                   help="the SEED NUMBER whose run collapsed (default 0)")
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)

    import glob as _g
    labels, A, B, groups, seeds = load_matched(sorted(_g.glob(a.a_glob)),
                                               sorted(_g.glob(a.b_glob)), True)
    n = A.shape[0]
    if a.collapsed_seed not in seeds:
        raise SystemExit(f"--collapsed-seed {a.collapsed_seed} not among {seeds}")
    ci = seeds.index(a.collapsed_seed)
    print(f"seeds={seeds}  items={len(labels)}  components={len(set(groups))}  "
          f"collapsed seed {a.collapsed_seed} is index {ci}")

    d = per_seed_diffs(labels, A, B)
    print(f"\npoint estimate (mean-logit) = {np.mean([x for x in d if x is not None]):+.4f}")
    print("per-seed full-test differences: "
          + "  ".join(f"s{s}={v:+.4f}" for s, v in zip(seeds, d)))

    print(f"\n== exact copy-count weights (ordered {n}-from-{n} resamples) ==")
    for k, pk in copy_count_probs(n).items():
        print(f"  {k} copies of seed {a.collapsed_seed}: {pk:.5f}")
    comps = compositions(n)
    print(f"  distinct compositions: {len(comps)}  (C(2n-1, n-1) = {comb(2 * n - 1, n - 1)})")

    ev, ew, ec = exact_seed_only(d, collapsed=ci)
    print(f"\n== exact seed-only distribution ==")
    print(_row("all replicates", summarise(ev, ew)))
    for lo, hi in STRATA:
        m = (ec >= lo) if hi is None else (ec == lo)
        print(_row(f"{lo}{'+' if hi is None else ''} copies", summarise(ev[m], ew[m])))

    cv, cc, skipped = crossed_replicates(labels, A, B, groups, a.n_boot,
                                          a.boot_seed, collapsed=ci)
    print(f"\n== crossed replicates, stratified (n_boot={a.n_boot}, skipped={skipped}) ==")
    share = np.full(len(cv), 1.0 / len(cv))     # so the wt column reads as a share
    print(_row("all replicates", summarise(cv, share)))
    for lo, hi in STRATA:
        m = (cc >= lo) if hi is None else (cc == lo)
        print(_row(f"{lo}{'+' if hi is None else ''} copies", summarise(cv[m], share[m])))

    if a.out_dir:
        o = Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
        with open(o / "crossed_replicates.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["replicate", "diff", "collapsed_copies"])
            for i, (v, c) in enumerate(zip(cv, cc)):
                w.writerow([i, f"{v:.6f}", c])
        with open(o / "seed_compositions.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["composition", "exact_weight", "collapsed_copies"])
            for combo, wt, c0 in comps:
                w.writerow(["|".join(map(str, combo)), f"{wt:.8f}", c0])
        _figure(ev, ew, ec, cv, cc, o / "mixture.png", n)
        print(f"\nwrote {o}/crossed_replicates.csv, seed_compositions.csv, mixture.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
