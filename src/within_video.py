"""How much of the variation in crop scores sits between videos rather than within.

⚠️ **DESCRIPTIVE. NOT PRESPECIFIED.** Reproduce with `python -m src.within_video`
(CPU, seconds, committed dumps only).

Supplies the direct empirical basis for clustering. Elsewhere the dependence
argument rests on construction ("twenty crops, same face, same room") plus the
indirect fact that grouping by video tripled the standard error. This measures it.

The intraclass correlation is

    ICC = var(video means) / (var(video means) + mean(within-video variance))

read as the share of total score variation attributable to which video a crop came
from. ICC near 0 would mean crops vary independently and treating them as separate
observations is defensible. ICC near 1 would mean a video's twenty crops carry
almost no more information than one of them.

Keyed on `src.cluster_boot._video_key`, the same unit the analyses aggregate over.
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

from .cluster_boot import _video_key


def icc_for(path: str) -> tuple[int, int, float, float, float]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    byvid: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        byvid[_video_key(r)].append(float(r["prob_fake"]))
    groups = [v for v in byvid.values() if len(v) > 1]
    within = st.mean([st.pvariance(v) for v in groups])
    means = [st.mean(v) for v in byvid.values()]
    between = st.pvariance(means)
    return len(rows), len(byvid), within, between, between / (between + within)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="results/predictions_v8/*_test.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    if not files:
        print(f"SKIP: no files match {a.glob}"); return

    print("DESCRIPTIVE — not prespecified\n")
    print(f"  {'dump':<48}{'crops':>7}{'videos':>8}{'ICC':>8}")
    iccs = []
    for f in files:
        n, v, w, b, icc = icc_for(f)
        iccs.append(icc)
        print(f"  {Path(f).name:<48}{n:>7}{v:>8}{icc:>8.3f}")

    print(f"\n  ICC across {len(iccs)} dumps: min {min(iccs):.3f}  "
          f"median {st.median(iccs):.3f}  max {max(iccs):.3f}")
    print(f"  -> roughly {100*st.median(iccs):.0f}% of the variation between crop scores is")
    print("     between videos rather than within them.")
    print("\n  This is why 3,000 crops do not provide 3,000 independent observations,")
    print("  and it is measured rather than assumed.")
    print("\n  x  Do NOT read ICC as an effective sample size. The design effect for a")
    print("     cluster bootstrap depends on cluster sizes and the statistic being")
    print("     resampled, not on ICC alone.")


if __name__ == "__main__":
    main()
