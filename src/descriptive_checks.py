"""Descriptive checks answering an external critique (2026-10-02).

⚠️ **NOT PRESPECIFIED. DESCRIPTIVE ONLY.** Neither check below was named in any
prespecification, so neither may be reported as a confirmatory finding. They exist
because a reviewer asked two factual questions about the data, and both are
answerable from the committed prediction dumps without retraining anything.

1. Frames per video sequence — does unequal frame count skew video-level scores?
2. Per-manipulation AUC — is the pooled result driven by one easy manipulation?

Aggregation matches the canonical estimand exactly: mean of `logit_margin` per
video, keyed on `src.cluster_boot._video_key`, AUC from `src.resolution_curve.auc`.
The pooled difference is printed as a cross-check against results/canonical.json.

    python -m src.descriptive_checks
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .cluster_boot import _video_key
from .resolution_curve import auc

SEEDS = (0, 1, 2, 3, 4)


def _rows(p: str) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def frames_per_sequence(root: str) -> None:
    print("\n== 1. Frames per video sequence ==")
    # Each dump is one run over one test split, so counts are only meaningful
    # WITHIN a dump. Unioning keys across dumps double-counts shared sequences.
    ranges, seqcounts = set(), set()
    for p in sorted(glob.glob(f"{root}/*.csv")):
        c = Counter(_video_key(r) for r in _rows(p))
        v = np.array(list(c.values()))
        print(f"  {Path(p).name:<46} seqs={len(c):4d}  min={v.min()}  max={v.max()}  mean={v.mean():.2f}")
        ranges.add((int(v.min()), int(v.max())))
        seqcounts.add(len(c))
    if len(ranges) == 1 and next(iter(ranges))[0] == next(iter(ranges))[1]:
        n = next(iter(ranges))[0]
        print(f"  -> every dump: EXACTLY {n} frames per sequence, "
              f"{sorted(seqcounts)} sequences. Unequal-frame weighting does not "
              f"arise in this data, so no disclosure is needed -- it is a fact to state.")
    else:
        print(f"  -> frame counts VARY across or within dumps {sorted(ranges)}; "
              f"video scores carry unequal weight. Disclose.")


def _agg(rows: list[dict]) -> dict[str, tuple[int, float]]:
    """video key -> (label, mean logit_margin). Canonical aggregation."""
    acc: dict[str, list[float]] = defaultdict(list)
    lab: dict[str, int] = {}
    for r in rows:
        k = _video_key(r)
        acc[k].append(float(r["logit_margin"]))
        lab[k] = int(r["label"])
    return {k: (lab[k], float(np.mean(v))) for k, v in acc.items()}


def per_manipulation(root: str, tag: str, arm_b: str = "xception_fad") -> None:
    print(f"\n== 2. Per-manipulation video-aggregated AUC ({tag}) ==")
    manips = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    diffs: dict[str, list[float]] = defaultdict(list)
    pooled: list[float] = []

    for s in SEEDS:
        # filenames are irregular: seed0 has no _split0 suffix, seeds 1-4 do
        ga_ = glob.glob(f"{root}/{tag}_xception_seed{s}*_test.csv")
        gb_ = glob.glob(f"{root}/{tag}_{arm_b}_seed{s}*_test.csv")
        if len(ga_) != 1 or len(gb_) != 1:
            print(f"  seed {s}: SKIP (expected 1 dump per arm, got {len(ga_)}/{len(gb_)})")
            continue
        a, b = ga_[0], gb_[0]
        ra, rb = _rows(a), _rows(b)
        mk = {r["path"]: r["manipulation"] for r in ra}
        ga, gb = _agg(ra), _agg(rb)
        keys = sorted(set(ga) & set(gb))
        kman = {}
        for r in ra:
            kman[_video_key(r)] = r["manipulation"]

        def sub(sel):  # labels, score_a, score_b for a key subset
            ks = [k for k in keys if sel(kman[k])]
            return (np.array([ga[k][0] for k in ks]),
                    np.array([ga[k][1] for k in ks]),
                    np.array([gb[k][1] for k in ks]))

        L, SA, SB = sub(lambda m: True)
        pooled.append(auc(L, SB) - auc(L, SA))
        for m in manips:
            L, SA, SB = sub(lambda x, m=m: x in ("real", m))
            diffs[m].append(auc(L, SB) - auc(L, SA))

    print(f"  {'manipulation':<16}{'mean diff':>11}{'min':>10}{'max':>10}{'sign':>7}")
    for m in manips:
        d = np.array(diffs[m])
        if not len(d):
            continue
        sign = "+" if d.mean() > 0 else "-"
        print(f"  {m:<16}{d.mean():>+11.4f}{d.min():>+10.4f}{d.max():>+10.4f}{sign:>7}")
    if pooled:
        p = np.array(pooled)
        print(f"  {'POOLED':<16}{p.mean():>+11.4f}{p.min():>+10.4f}{p.max():>+10.4f}")
        print(f"\n  cross-check: canonical.json mean difference is -0.0092 "
              f"(this run: {p.mean():+.4f})")
        spread = max(np.mean(diffs[m]) for m in manips) - min(np.mean(diffs[m]) for m in manips)
        print(f"  spread across manipulations: {spread:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/predictions_v8")
    ap.add_argument("--tag", default="ffpp_c40_vid")
    ap.add_argument("--arm-b", default="xception_fad")
    a = ap.parse_args()
    print("DESCRIPTIVE CHECKS — not prespecified, not confirmatory")
    frames_per_sequence(a.root)
    per_manipulation(a.root, a.tag, a.arm_b)


if __name__ == "__main__":
    main()
