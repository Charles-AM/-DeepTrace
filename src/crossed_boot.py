"""Crossed bootstrap: test content and training run resampled together.

Every interval in this project so far is **conditional on a trained model**. The
component bootstrap resamples test content with the weights held fixed; the
seed-level sd varies the weights while ignoring test-content sampling entirely.
Neither answers the question the paper actually poses about FAD:

    over the population of test content AND the population of training runs,
    what is the expected FAD - Xception difference, and how uncertain is it?

Those two sources are **crossed**, not nested: the same test videos are scored by
every trained model. Combining the per-seed intervals — averaging their bounds,
or pooling their sds — treats them as independent and understates or overstates
nothing in a predictable direction. The correct treatment resamples both.

This was impossible before V8. Under the varying-split protocol each seed drew a
different split, so the same component sample could not be applied across seeds.
V8's models share one verified split, which is what makes this computable.

Each replicate: draw components with replacement, draw seeds with replacement,
score every drawn seed on that one component sample, average. The component draw
is shared across seeds within a replicate — that is what "crossed" means, and
re-drawing per seed would silently make them nested again.

    python -m src.crossed_boot --a-glob 'preds/*_xception_seed*_test.csv' \\
        --b-glob 'preds/*_xception_fad_seed*_test.csv' --out-dir results/analysis/crossed
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import numpy as np

from .cluster_boot import _load, _video_key
from .clusters import build_component_clusters
from .resolution_curve import auc


def _seed_of(path: str) -> int:
    m = re.search(r"_seed(\d+)", Path(path).stem)
    if not m:
        raise ValueError(f"cannot read a seed from {path}")
    return int(m.group(1))


def load_matched(a_paths, b_paths, aggregate: bool = True):
    """Align every run on one identical set of test items.

    A crossed bootstrap is only defined if all models were scored on the SAME
    test content — that is the property V8 was run to obtain, so it is verified
    here rather than assumed.
    """
    a_by, b_by = {_seed_of(p): p for p in a_paths}, {_seed_of(p): p for p in b_paths}
    seeds = sorted(set(a_by) & set(b_by))
    if not seeds:
        raise ValueError("no seeds present for both configurations")

    rows = {s: (_load(a_by[s]), _load(b_by[s])) for s in seeds}
    item_sets = {s: frozenset(r["path"] for r in ra) for s, (ra, _) in rows.items()}
    if len({*item_sets.values()}) != 1:
        sizes = {s: len(v) for s, v in item_sets.items()}
        raise ValueError(
            "runs were not scored on the same test items, so a crossed bootstrap "
            f"is undefined (sizes {sizes}). This is the failure V8's fixed split "
            "exists to prevent — check that every run used --split-seed.")

    shared = sorted(next(iter(item_sets.values())))
    meta = {r["path"]: r for r in rows[seeds[0]][0]}
    pairs = sorted({(meta[p]["target_seq"], meta[p]["source_seq"]) for p in shared})
    comp_of = build_component_clusters(pairs)

    if aggregate:
        keys = sorted({_video_key(meta[p]) for p in shared})
        idx_of = {k: i for i, k in enumerate(keys)}
        labels = np.zeros(len(keys), dtype=int)
        groups = [""] * len(keys)
        for p in shared:
            r = meta[p]; i = idx_of[_video_key(r)]
            labels[i] = int(r["label"])
            groups[i] = comp_of.get(r["target_seq"], r["target_seq"])
        def scores(rs):
            acc = {}
            for r in rs:
                acc.setdefault(_video_key(r), []).append(float(r["logit_margin"]))
            return np.array([float(np.mean(acc[k])) for k in keys])
    else:
        labels = np.array([int(meta[p]["label"]) for p in shared])
        groups = [comp_of.get(meta[p]["target_seq"], meta[p]["target_seq"]) for p in shared]
        def scores(rs):
            by = {r["path"]: float(r["logit_margin"]) for r in rs}
            return np.array([by[p] for p in shared])

    A = np.stack([scores(rows[s][0]) for s in seeds])
    B = np.stack([scores(rows[s][1]) for s in seeds])
    return labels, A, B, groups, seeds


def crossed_bootstrap(labels, A, B, groups, n_boot: int = 4000, seed: int = 0) -> dict:
    """Resample components and seeds together.

    Returns the crossed interval plus the two conditional ones — components only
    (seeds fixed) and seeds only (components fixed) — because the comparison
    between them is what shows the crossed interval is doing something.
    """
    rng = np.random.default_rng(seed)
    n_seeds = A.shape[0]
    index_of: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        index_of.setdefault(g, []).append(i)
    keys = list(index_of)
    if len(keys) < 2:
        raise ValueError(f"{len(keys)} component(s): a bootstrap over this is degenerate")
    if n_seeds < 3:
        print(f"  WARNING: {n_seeds} seeds — the seed dimension is very coarse")

    def diff_on(idx, s):
        y = labels[idx]
        if y.min() == y.max():
            return None
        return auc(y, B[s][idx]) - auc(y, A[s][idx])

    all_idx = np.arange(len(labels))
    point = float(np.mean([diff_on(all_idx, s) for s in range(n_seeds)]))

    out = {"n_seeds": n_seeds, "n_components": len(keys), "n_items": len(labels),
           "diff": round(point, 4)}

    # Var_crossed / (Var_component_only + Var_seed_only). Under an additive
    # independent-variance model this is 1; the departure is reported as an
    # exploratory diagnostic, not as proof that independence fails -- AUC is
    # nonlinear, the seed sample is five, and the bootstrap itself adds noise.
    for name, draw_comp, draw_seed in (("crossed", True, True),
                                       ("component_only", True, False),
                                       ("seed_only", False, True)):
        reps, skipped = [], 0
        for _ in range(n_boot):
            if draw_comp:
                picked = rng.integers(0, len(keys), size=len(keys))
                idx = np.concatenate([index_of[keys[k]] for k in picked])
            else:
                idx = all_idx
            seeds_used = (rng.integers(0, n_seeds, size=n_seeds) if draw_seed
                          else np.arange(n_seeds))
            vals = [d for s in seeds_used if (d := diff_on(idx, s)) is not None]
            if not vals:
                skipped += 1
                continue
            reps.append(float(np.mean(vals)))
        d = np.asarray(reps)
        out[name] = {"ci_lo": round(float(np.percentile(d, 2.5)), 4),
                     "ci_hi": round(float(np.percentile(d, 97.5)), 4),
                     # 6 dp: this SE gets SQUARED for the non-additivity ratio in
                     # the README, and 4 dp loses meaningful precision there
                     "se": round(float(d.std(ddof=1)), 6),
                     "halfwidth": round(float(np.percentile(d, 97.5)
                                              - np.percentile(d, 2.5)) / 2, 5),
                     "n_used": len(d), "n_skipped": skipped}
        print(f"  {name:<15} 95% CI [{out[name]['ci_lo']:+.4f}, {out[name]['ci_hi']:+.4f}]"
              f"  half-width {out[name]['halfwidth']:.4f}  se {out[name]['se']:.6f}")

    vc, vs, vx = (out[k]["se"] ** 2 for k in ("component_only", "seed_only", "crossed"))
    out["nonadditivity_ratio"] = round(vx / (vc + vs), 4)
    out["nonadditivity_formula"] = "var(crossed) / (var(component_only) + var(seed_only))"
    print(f"  non-additivity ratio {out['nonadditivity_ratio']:.3f}  "
          f"[{out['nonadditivity_formula']}] — exploratory")
    return out


def run(a_glob: str, b_glob: str, aggregate: bool = True, n_boot: int = 4000,
        margins=(0.014,), out_dir: Path | None = None) -> dict:
    a_paths, b_paths = sorted(glob.glob(a_glob)), sorted(glob.glob(b_glob))
    if not a_paths or not b_paths:
        raise SystemExit(f"no files matched:\n  a: {a_glob}\n  b: {b_glob}")
    labels, A, B, groups, seeds = load_matched(a_paths, b_paths, aggregate)
    print(f"seeds={seeds}  items={len(labels)}  components={len(set(groups))}  "
          f"aggregate={'video' if aggregate else 'frame'}")
    res = crossed_bootstrap(labels, A, B, groups, n_boot=n_boot)
    res["seeds"] = seeds
    res["aggregate"] = "video" if aggregate else "frame"
    for m in margins:
        c = res["crossed"]
        res[f"excludes_{m:g}"] = bool(c["ci_hi"] < m)
        res[f"equivalent_{m:g}"] = bool(c["ci_lo"] > -m and c["ci_hi"] < m)
        print(f"  vs margin {m:g}: excludes={res[f'excludes_{m:g}']}  "
              f"equivalent={res[f'equivalent_{m:g}']}")
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        flat = {k: v for k, v in res.items() if not isinstance(v, dict)}
        for name in ("crossed", "component_only", "seed_only"):
            flat.update({f"{name}_{k}": v for k, v in res[name].items()})
        flat["seeds"] = ";".join(map(str, seeds))
        p = out_dir / f"crossed_{res['aggregate']}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat)); w.writeheader(); w.writerow(flat)
        print(f"wrote {p}")
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-glob", required=True, help="reference config prediction CSVs")
    p.add_argument("--b-glob", required=True, help="comparison config prediction CSVs")
    p.add_argument("--frame-level", action="store_true")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--margins", type=float, nargs="*", default=[0.014])
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.a_glob, a.b_glob, aggregate=not a.frame_level, n_boot=a.n_boot,
        margins=tuple(a.margins), out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
