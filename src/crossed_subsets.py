"""Seed-subset sensitivity for the crossed bootstrap.

The crossed interval rests on five training runs, which is coarse for a bootstrap
dimension. This recomputes it on **every** 3-of-5 and 4-of-5 subset — all ten and
all five, not one arbitrary sequence — plus the full set.

⚠️ These subsets overlap heavily: each 3-of-5 shares two runs with most others.
They are a **sensitivity analysis**, not independent replication, and no exclusion
frequency should be computed from them.

Read positively as well as defensively: if the point estimate scatters widely
across subsets, that IS the paper's third contribution — the five-run mean is
itself unstable — rather than merely a caveat about this analysis.

    python -m src.crossed_subsets --a-glob '...' --b-glob '...' --out-dir results/analysis/crossed
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import statistics as st
from pathlib import Path

from .crossed_boot import crossed_bootstrap, load_matched


def run(a_glob, b_glob, aggregate=True, n_boot=4000, boot_seed=0, margin=0.014,
        out_dir=None):
    labels, A, B, groups, seeds = load_matched(sorted(glob.glob(a_glob)),
                                               sorted(glob.glob(b_glob)), aggregate)
    n = len(seeds)
    subsets = [tuple(range(n))]
    for k in (4, 3):
        subsets += list(itertools.combinations(range(n), k))

    rows = []
    for sub in subsets:
        r = crossed_bootstrap(labels, A, B, groups, n_boot=n_boot, seed=boot_seed,
                              margins=(margin,), seed_subset=sub)
        c = r["crossed"]
        rows.append({"k": len(sub), "seeds": ";".join(str(seeds[i]) for i in sub),
                     "diff": r["diff"], "ci_lo": c["ci_lo"], "ci_hi": c["ci_hi"],
                     "halfwidth": c["halfwidth"],
                     f"p_above_{margin:g}": c[f"p_above_{margin:g}"],
                     "contains_margin": bool(c["ci_lo"] < margin < c["ci_hi"])})

    print(f"\n{'k':<4}{'seeds':<12}{'diff':>9}{'95% CI':>22}{'P(>marg)':>10}{'contains':>10}")
    for r in rows:
        print(f"{r['k']:<4}{r['seeds']:<12}{r['diff']:>+9.4f}"
              f"{f'[{r[chr(99)+chr(105)+chr(95)+chr(108)+chr(111)]:+.4f}, {r[chr(99)+chr(105)+chr(95)+chr(104)+chr(105)]:+.4f}]':>22}"
              f"{r[f'p_above_{margin:g}']:>10.4f}{str(r['contains_margin']):>10}")
    for k in (3, 4):
        sub = [r for r in rows if r["k"] == k]
        d = [r["diff"] for r in sub]
        print(f"\n{k}-of-{n} subsets (n={len(sub)}): diff {min(d):+.4f} to {max(d):+.4f} "
              f"(spread {max(d)-min(d):.4f}), all contain margin: "
              f"{all(r['contains_margin'] for r in sub)}")

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"seed_subsets_{'video' if aggregate else 'frame'}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {p}")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-glob", required=True)
    p.add_argument("--b-glob", required=True)
    p.add_argument("--frame-level", action="store_true")
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.014)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(a.a_glob, a.b_glob, aggregate=not a.frame_level, n_boot=a.n_boot,
        boot_seed=a.boot_seed, margin=a.margin,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
