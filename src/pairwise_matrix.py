"""All pairwise cluster-aware comparisons among a set of configurations.

Answers the obvious objection to every interval in this project: they all come
from one comparison (Xception vs Xception + FAD), so are the wide intervals a
property of the **evaluation design** or of that pair?

Running every pair at the same scale separates those. Large effects that clear
their intervals show the design is not uniformly uninformative; small effects that
do not show where the limit falls.

Needs no GPU or checkpoints — it reads `results/predictions/*.csv`.

    python -m src.pairwise_matrix --compression c23 --out-dir results/analysis/generality

The `--configs` order is explicit rather than discovered, because it fixes the
reference/model orientation of each pair and therefore the sign of every
difference. Sorting would silently flip signs relative to the committed results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from .cluster_boot import _load, _prepare, paired_bootstrap

DEFAULT_CONFIGS = ["baseline_spatial", "frequency_only", "xception", "f3net"]
DISPLAY = {"f3net": "Xception+FAD", "xception_fad": "Xception+FAD",
           "baseline_spatial": "spatial", "frequency_only": "freq_only",
           "xception": "Xception"}


def run(preds: Path, compression: str = "c23", configs=None, seeds=(0, 1, 2),
        unit: str = "component", n_boot: int = 2000, aggregate: bool = True,
        out_dir: Path | None = None) -> list[dict]:
    configs = list(configs or DEFAULT_CONFIGS)
    rows = []
    for a, b in itertools.combinations(configs, 2):
        for s in seeds:
            pa = Path(preds) / f"ffpp_{compression}_vid_{a}_seed{s}_test.csv"
            pb = Path(preds) / f"ffpp_{compression}_vid_{b}_seed{s}_test.csv"
            if not (pa.exists() and pb.exists()):
                print(f"skip (missing): {a} vs {b} seed {s}")
                continue
            labels, sa, sb, vids, comps = _prepare(_load(pa), _load(pb), aggregate)
            r = paired_bootstrap(labels, sa, sb, vids, comps, unit=unit, n_boot=n_boot)
            rows.append({"compression": compression,
                         "reference": DISPLAY.get(a, a), "model": DISPLAY.get(b, b),
                         "seed": s, "n_components": r["n_units"], "diff": r["diff"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "halfwidth": round((r["ci_hi"] - r["ci_lo"]) / 2, 5),
                         "excludes_zero": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)})

    if rows:
        big = [r for r in rows if abs(r["diff"]) > 0.10]
        small = [r for r in rows if abs(r["diff"]) < 0.03]
        print(f"\n{len(rows)} comparisons")
        print(f"  |diff| > 0.10: {sum(r['excludes_zero'] for r in big)}/{len(big)} exclude zero")
        print(f"  |diff| < 0.03: {sum(r['excludes_zero'] for r in small)}/{len(small)} exclude zero")
    if out_dir and rows:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"pairwise_{compression}_{unit}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {out}")
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", default="results/predictions")
    p.add_argument("--compression", default="c23")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--unit", default="component", choices=["frame", "video", "component"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--frame-level", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.preds), compression=a.compression, configs=a.configs, seeds=a.seeds,
        unit=a.unit, n_boot=a.n_boot, aggregate=not a.frame_level,
        out_dir=Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
