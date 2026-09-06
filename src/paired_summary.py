"""Paired seed-level summary with confidence intervals — and a seed-subset
instability check.

Two jobs.

1. Produce the paired comparison table from a `summary.csv` so headline numbers
   are script-derived and re-derivable, never hand-computed.

2. Report **what conclusion each seed subset would have supported**. This is the
   most legible evidence that a design lacks resolution: if seeds {0,1,2} say one
   thing and seeds {3,4} say the opposite, the point estimate was never stable,
   and no interval is needed to see it.

⚠️ These are **seed-level** intervals. In this project each seed also draws a
different split, so they conflate optimisation noise with split composition, and
they ignore test-content clustering entirely. Report them as a diagnostic of
estimate stability — the inferential interval comes from `src/cluster_boot.py`.

Deliberately stdlib-only so it runs anywhere, including without numpy/pandas.

    python -m src.paired_summary --summary results/in_domain_c40_vid/summary.csv \\
        --reference xception --prefix ffpp_c40_vid
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from itertools import combinations
from pathlib import Path

# two-sided 95% t critical values by degrees of freedom (n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086}


def _ci(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = st.mean(diffs)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "se": None, "lo": None, "hi": None}
    sd = st.stdev(diffs)
    se = sd / (n ** 0.5)
    half = _T95.get(n - 1, 1.96) * se
    return {"n": n, "mean": mean, "sd": sd, "se": se, "lo": mean - half, "hi": mean + half}


def load(summary: Path, prefix: str | None, metric: str) -> dict[str, dict[int, float]]:
    by: dict[str, dict[int, float]] = {}
    with Path(summary).open() as fh:
        for r in csv.DictReader(fh):
            if prefix and not r["run"].startswith(prefix):
                continue
            by.setdefault(r["config"], {})[int(r["seed"])] = float(r[metric])
    return by


def run(summary: Path, reference: str, prefix: str | None = None,
        metric: str = "roc_auc", thresholds: tuple[float, ...] = (0.010, 0.014),
        out_dir: Path | None = None) -> list[dict]:
    by = load(summary, prefix, metric)
    if reference not in by:
        raise SystemExit(f"reference {reference!r} not found; have {sorted(by)}")

    print(f"=== per-config {metric} ===")
    for cfg in sorted(by):
        vals = [by[cfg][s] for s in sorted(by[cfg])]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {cfg:20s} n={len(vals)}  mean={st.mean(vals):.4f}  sd={sd:.4f}  "
              f"{[round(v, 4) for v in vals]}")

    rows = []
    for cfg in sorted(by):
        if cfg == reference:
            continue
        seeds = sorted(set(by[cfg]) & set(by[reference]))
        diffs = [by[cfg][s] - by[reference][s] for s in seeds]
        c = _ci(diffs)
        row = {"config": cfg, "reference": reference, "metric": metric,
               "n_seeds": c["n"], "mean_diff": round(c["mean"], 5),
               "sd": round(c["sd"], 5) if c["sd"] is not None else "",
               "se": round(c["se"], 5) if c["se"] is not None else "",
               "ci_lo": round(c["lo"], 5) if c["lo"] is not None else "",
               "ci_hi": round(c["hi"], 5) if c["hi"] is not None else "",
               "per_seed": ";".join(f"{d:+.4f}" for d in diffs)}
        for t in thresholds:
            row[f"excludes_{t}"] = (c["hi"] is not None and c["hi"] < t)
        row["excludes_zero"] = (c["lo"] is not None and (c["lo"] > 0 or c["hi"] < 0))
        rows.append(row)

        print(f"\n=== {cfg} - {reference} (paired, n={c['n']}) ===")
        print(f"  per-seed {[f'{d:+.4f}' for d in diffs]}")
        print(f"  mean={c['mean']:+.5f}  sd={c['sd']:.5f}  se={c['se']:.5f}")
        print(f"  95% CI [{c['lo']:+.5f}, {c['hi']:+.5f}]")
        print(f"  excludes zero: {row['excludes_zero']}")
        for t in thresholds:
            print(f"  excludes +{t:.3f}: {row[f'excludes_{t}']}")

        # --- seed-subset instability -------------------------------------
        if len(diffs) >= 4:
            print(f"  --- what each 3-seed subset would have concluded ---")
            signs = []
            for combo in combinations(range(len(diffs)), 3):
                sub = [diffs[i] for i in combo]
                m = st.mean(sub)
                signs.append(m)
                print(f"    seeds {[seeds[i] for i in combo]}: mean={m:+.4f}")
            pos = sum(1 for m in signs if m > 0)
            print(f"    -> {pos}/{len(signs)} subsets positive, "
                  f"{len(signs) - pos}/{len(signs)} negative "
                  f"({'SIGN UNSTABLE' if 0 < pos < len(signs) else 'sign stable'})")
            row["subsets_positive"] = pos
            row["subsets_total"] = len(signs)
            row["sign_unstable"] = bool(0 < pos < len(signs))

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "paired_summary.csv"
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {out}")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--reference", default="xception")
    p.add_argument("--prefix", default=None, help="only rows whose run starts with this")
    p.add_argument("--metric", default="roc_auc")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.summary), a.reference, prefix=a.prefix, metric=a.metric,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
