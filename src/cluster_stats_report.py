"""Report the FF++ source-target component structure of a crop set.

Closes an evidence gap: the finding that our 300 sequences form 150 disjoint
size-2 components was originally computed in an ad-hoc shell pipeline. That number
matters — it determines whether component-level bootstrapping is even possible (a
cyclic pair graph collapses everything into one component and makes the bootstrap
degenerate) and how many independent units survive at the strictest level.

Accepts either a crop directory or a text file listing crop paths (e.g.
`tar tzf crops.tar.gz > listing.txt`), so it can run without unpacking archives.

    python -m src.cluster_stats_report --listing listing.txt --out-dir results/analysis/clusters
    python -m src.cluster_stats_report --crops /path/to/ffpp_crops --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from .clusters import build_component_clusters, cluster_stats

_PAIR_RE = re.compile(r"videos-(\d+)-(\d+)_")
_REAL_RE = re.compile(r"original-sequences-\w+-(?:c\d+|raw)-videos-(\d+)_")


def collect_pairs(lines) -> tuple[set[tuple[str, str]], set[str]]:
    """(target, source) for manipulated crops; target ids seen in originals."""
    pairs: set[tuple[str, str]] = set()
    reals: set[str] = set()
    for line in lines:
        m = _PAIR_RE.search(line)
        if m:
            pairs.add((m.group(1), m.group(2)))
            continue
        m = _REAL_RE.search(line)
        if m:
            reals.add(m.group(1))
    return pairs, reals


def run(lines, out_dir: Path | None = None) -> dict:
    pairs, reals = collect_pairs(lines)
    clusters = build_component_clusters(sorted(pairs) + [(r, "") for r in sorted(reals)])
    stats = cluster_stats(clusters)

    sizes: dict[str, int] = defaultdict(int)
    for cid in clusters.values():
        sizes[cid] += 1
    hist: dict[int, int] = defaultdict(int)
    for n in sizes.values():
        hist[n] += 1

    n_targets = len({t for t, _ in pairs}) | 0
    report = {
        "manipulated_pairs": len(pairs),
        "unique_target_sequences": len({t for t, _ in pairs}),
        "unique_source_sequences": len({s for _, s in pairs if s}),
        "original_sequences": len(reals),
        "total_sequences": stats["n_sequences"],
        "source_target_components": stats["n_components"],
        "largest_component": stats["largest"],
        "singleton_components": stats["singletons"],
        "component_size_histogram": ";".join(f"{k}:{v}" for k, v in sorted(hist.items())),
        "degenerate_for_bootstrap": stats["n_components"] < 2,
    }
    for k, v in report.items():
        print(f"  {k:28s} {v}")

    # the number that actually matters downstream
    ratio = (report["unique_target_sequences"] / report["source_target_components"]
             if report["source_target_components"] else float("nan"))
    print(f"\n  video-groups per source-target component: {ratio:.2f}")
    print(f"  -> component-level bootstrapping has ~{1/ratio:.0%} of the units that "
          f"video-level grouping does")
    report["video_groups_per_component"] = round(ratio, 3)

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "component_stats.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report))
            w.writeheader()
            w.writerow(report)
        print(f"\nwrote {out}")
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--listing", help="text file of crop paths (one per line)")
    g.add_argument("--crops", help="crop directory to walk")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.listing:
        with open(a.listing) as fh:
            report = run(fh, Path(a.out_dir) if a.out_dir else None)
    else:
        paths = (str(p) for p in Path(a.crops).rglob("*.jpg"))
        report = run(paths, Path(a.out_dir) if a.out_dir else None)
    return report


if __name__ == "__main__":
    main()
