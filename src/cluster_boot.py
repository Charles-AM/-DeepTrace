"""Cluster-aware paired inference for detector comparisons.

The problem this solves: a test AUC computed over 3,000 crops drawn from 30
videos is not backed by 3,000 independent observations. Frames within a video are
heavily correlated, and several manipulations share a target sequence, so even
resampling videos is optimistic. Treating *seeds* as the unit does not fix this
either — it estimates optimisation variability while ignoring test-content
sampling entirely (and in our setup each seed also draws a different split, so
the two are confounded).

This module separates three uncertainty sources and lets you compare them
directly:

  * **frame**    — resample individual crops i.i.d. (the naive baseline; expect it
                   to be over-optimistic and to keep shrinking as frames/video
                   grows, without ever reflecting real test-content uncertainty)
  * **video**    — resample whole videos
  * **identity** — resample connected components of the FF++ pair graph
                   (`src/clusters.py`), the safest unit

Reporting all three is the point: the gap between them *is* the methodological
finding.

    python -m src.cluster_boot --a results/predictions/..._xception_seed0_test.csv \\
        --b results/predictions/..._f3net_seed0_test.csv --out-dir results/analysis/boot
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .clusters import build_identity_clusters

UNITS = ("frame", "video", "identity")


def _load(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def _video_key(r: dict) -> str:
    """Unique per actual video file. NOTE: `video_id` alone is the *target
    sequence*, which groups a real video together with every manipulation that
    targets it — mixing labels. Aggregation must key on the video itself."""
    return f"{r['manipulation']}|{r['target_seq']}|{r['source_seq']}"


def _prepare(rows_a: list[dict], rows_b: list[dict], aggregate: bool):
    """Align two runs on identical test items, optionally aggregating frames to
    video-level scores. Returns (labels, score_a, score_b, cluster_of_item)."""
    by_a = {r["path"]: r for r in rows_a}
    by_b = {r["path"]: r for r in rows_b}
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        raise ValueError("no overlapping test items — are these the same split/seed?")
    if len(shared) != len(rows_a) or len(shared) != len(rows_b):
        print(f"warning: aligning on {len(shared)} shared items "
              f"(a={len(rows_a)}, b={len(rows_b)})")

    seq_pairs = sorted({(by_a[p]["target_seq"], by_a[p]["source_seq"]) for p in shared})
    ident = build_identity_clusters(seq_pairs)

    if not aggregate:
        labels = np.array([int(by_a[p]["label"]) for p in shared])
        sa = np.array([float(by_a[p]["logit_margin"]) for p in shared])
        sb = np.array([float(by_b[p]["logit_margin"]) for p in shared])
        vids = [_video_key(by_a[p]) for p in shared]
        idents = [ident.get(by_a[p]["target_seq"], by_a[p]["target_seq"]) for p in shared]
        return labels, sa, sb, vids, idents

    acc: dict[str, dict] = {}
    for p in shared:
        r = by_a[p]
        k = _video_key(r)
        d = acc.setdefault(k, {"label": int(r["label"]), "a": [], "b": [],
                               "target": r["target_seq"]})
        d["a"].append(float(r["logit_margin"]))
        d["b"].append(float(by_b[p]["logit_margin"]))
    keys = sorted(acc)
    labels = np.array([acc[k]["label"] for k in keys])
    sa = np.array([float(np.mean(acc[k]["a"])) for k in keys])
    sb = np.array([float(np.mean(acc[k]["b"])) for k in keys])
    idents = [ident.get(acc[k]["target"], acc[k]["target"]) for k in keys]
    return labels, sa, sb, keys, idents


def paired_bootstrap(labels, sa, sb, vids, idents, unit: str = "identity",
                     n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if unit == "frame":
        groups = [str(i) for i in range(len(labels))]
    elif unit == "video":
        groups = vids
    elif unit == "identity":
        groups = idents
    else:
        raise ValueError(f"unit must be one of {UNITS}")

    index_of: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        index_of.setdefault(g, []).append(i)
    keys = list(index_of)

    point = roc_auc_score(labels, sb) - roc_auc_score(labels, sa)
    diffs, skipped = [], 0
    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([index_of[keys[k]] for k in picked])
        y = labels[idx]
        if y.min() == y.max():          # single-class replicate -> AUC undefined
            skipped += 1
            continue
        diffs.append(roc_auc_score(y, sb[idx]) - roc_auc_score(y, sa[idx]))
    d = np.asarray(diffs)
    return {"unit": unit, "n_units": len(keys), "n_items": len(labels),
            "auc_a": round(float(roc_auc_score(labels, sa)), 4),
            "auc_b": round(float(roc_auc_score(labels, sb)), 4),
            "diff": round(float(point), 4),
            "ci_lo": round(float(np.percentile(d, 2.5)), 4),
            "ci_hi": round(float(np.percentile(d, 97.5)), 4),
            "se": round(float(d.std(ddof=1)), 4),
            "n_boot_used": len(d), "n_boot_skipped": skipped}


def run(path_a: Path, path_b: Path, aggregate: bool = True, n_boot: int = 2000,
        margin: float | None = None, out_dir: Path | None = None) -> list[dict]:
    rows_a, rows_b = _load(path_a), _load(path_b)
    labels, sa, sb, vids, idents = _prepare(rows_a, rows_b, aggregate)
    print(f"aggregate={'video-level' if aggregate else 'frame-level'}  "
          f"items={len(labels)}  videos={len(set(vids))}  identities={len(set(idents))}")

    results = []
    for unit in UNITS:
        if aggregate and unit == "frame":
            continue          # already aggregated; frame unit is meaningless here
        r = paired_bootstrap(labels, sa, sb, vids, idents, unit=unit, n_boot=n_boot)
        r["aggregate"] = "video" if aggregate else "frame"
        if margin is not None:
            # equivalence: is the whole CI inside (-margin, +margin)?
            r["margin"] = margin
            r["equivalent"] = bool(r["ci_lo"] > -margin and r["ci_hi"] < margin)
            r["excludes_margin"] = bool(r["ci_hi"] < margin)
        results.append(r)
        print(f"  unit={unit:9s} n={r['n_units']:4d}  diff={r['diff']:+.4f}  "
              f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  se={r['se']:.4f}")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"boot_{Path(path_a).stem}_vs_{Path(path_b).stem}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"wrote {out}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="prediction CSV for model A (reference)")
    p.add_argument("--b", required=True, help="prediction CSV for model B")
    p.add_argument("--frame-level", action="store_true",
                   help="skip video aggregation and score individual crops")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--margin", type=float, default=None,
                   help="smallest worthwhile AUC difference, for equivalence testing")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.a), Path(a.b), aggregate=not a.frame_level, n_boot=a.n_boot,
        margin=a.margin, out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
