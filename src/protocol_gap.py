"""Protocol gap (L1 crop-randomised vs L2 video-disjoint) from committed summaries.

⚠️ **DESCRIPTIVE ONLY. NOT PRESPECIFIED. NOT AN UNCERTAINTY ESTIMATE.**

The L1 campaign predates the artifact-retention policy: per-run metrics were saved,
prediction-level outputs were not (`docs/ARTIFACT-POLICY.md`, ledger §25). So the
protocol gap cannot be given a cluster-aware interval and cannot be recomputed at
video level. What *can* be done, from committed CSVs and with no new compute, is to
show the gap across every architecture and seed rather than as a lone point
estimate.

What the reported sd is, and is not:
  IS     run-to-run spread of the quantity across training seeds
  IS NOT test-content uncertainty, which needs prediction dumps L1 does not have
  IS NOT comparable between campaigns: L1 has 3 seeds, L2 has 5

"Corresponding seeds", never "paired". A shared seed fixes initialisation and data
order. It does **not** make the two test sets the same — L1 and L2 partition the
data differently, which is the whole point of the comparison.

Naming: the FAD arm is `xception_fad`. `f3net` is an alias retained only because it
is embedded in artifact filenames produced before 2026-09-06 (`src/config.py`).
**Never print either as "F3-Net"** — this builds the FAD branch alone, without LFS
or MixBlock.

    python -m src.protocol_gap
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

DISPLAY = {"baseline_spatial": "baseline_spatial",
           "xception": "xception",
           "f3net": "xception_fad",          # see module docstring
           "xception_fad": "xception_fad"}
ORDER = ("baseline_spatial", "xception", "xception_fad")


def _load(path: str) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            cfg = DISPLAY.get(r["config"], r["config"])
            out[(cfg, int(r["seed"]))] = float(r["roc_auc"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--l1", default="results/in_domain_c40/summary.csv")
    ap.add_argument("--l2", default="results/in_domain_c40_vid/summary.csv")
    a = ap.parse_args()
    for p in (a.l1, a.l2):
        if not Path(p).is_file():
            print(f"SKIP: {p} not found"); return

    L1, L2 = _load(a.l1), _load(a.l2)
    common = sorted({s for _, s in L1} & {s for _, s in L2})

    print("DESCRIPTIVE — not prespecified, not an uncertainty estimate")
    print(f"frame-pooled ROC-AUC, c40.  corresponding seeds: {common}\n")

    print("== 1. Protocol gap  (L1 crop-randomised  minus  L2 video-disjoint) ==")
    print(f"  {'architecture':<18}{'per-seed gap':<30}{'mean':>9}{'sd':>9}")
    lo = hi = None
    for cfg in ORDER:
        g = [L1[(cfg, s)] - L2[(cfg, s)] for s in common if (cfg, s) in L1 and (cfg, s) in L2]
        if not g:
            continue
        lo = min(g) if lo is None else min(lo, min(g))
        hi = max(g) if hi is None else max(hi, max(g))
        sd = st.stdev(g) if len(g) > 1 else float("nan")
        print(f"  {cfg:<18}{', '.join(f'{x:.4f}' for x in g):<30}{st.mean(g):>+9.4f}{sd:>9.4f}")
    print(f"\n  -> every observed gap lies in [{lo:.4f}, {hi:.4f}];"
          f" the smallest is {lo/0.014:.0f}x the +0.014 reference effect.")
    print("  -> the gap is not carried by one architecture or one seed.")

    print("\n== 2. FAD minus Xception, within each split ==")
    for tag, D in (("L1 crop-randomised", L1), ("L2 video-disjoint", L2)):
        seeds = sorted(s for c, s in D if c == "xception_fad")
        d = [D[("xception_fad", s)] - D[("xception", s)] for s in seeds]
        sd = st.stdev(d) if len(d) > 1 else float("nan")
        print(f"  {tag:<20} n={len(d)}  {', '.join(f'{x:+.4f}' for x in d)}")
        print(f"  {'':<20} mean {st.mean(d):+.4f}   run-to-run sd {sd:.4f}")
    print("\n  -> small under both protocols. The conclusion was stable; its")
    print("     warrant was not.")

    print("\n== What must NOT be concluded ==")
    print("  x  'L1 was more precise.' L1 sits near the AUC ceiling (~0.993-0.995),")
    print("     which compresses differences mechanically, and rests on 3 seeds")
    print("     against L2's 5. ESSENCE forbids claiming L1 could not have detected")
    print("     a difference: near-ceiling AUC does not prove low power.")
    print("  x  'the gap measures leakage.' L1 vs L2 changes prior exposure AND the")
    print("     test partition together. V2 is the experiment that separates them.")
    print("  ok 'agreement across training seeds can create an appearance of")
    print("     stability while leaving test-content uncertainty unmeasured.'")


if __name__ == "__main__":
    main()
