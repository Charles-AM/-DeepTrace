"""Pre-analysis gate for the NPR supplementary campaign.

🚨 **Run this BEFORE any bootstrap on the NPR campaign.** The analysis entry
points (`src.crossed_boot`, `src.agg_space`) discover dumps by shell glob and will
happily accept whatever a pattern matches. A glob that reaches
`results/predictions_v8/` as well as the campaign directory would mix two training
sessions **silently** and still print a plausible interval — the same failure mode
as the night-1 output collision.

This module asserts the conditions `docs/npr-supplementary-prespecification.md`
requires, then exits non-zero if any fails. It is a separate file on purpose: the
training and analysis modules are protected (their `git diff` is the isolation
proof), so the gate must not live inside them.

What it refuses to let through:

1. Wrong dump count, or a seed missing from either arm.
2. Any dump resolving outside the campaign directory (the contamination case).
3. A `config` column that disagrees with the arm it was globbed as.
4. Mixed `split_seed` — the campaign must sit on one fixed split.
5. Test-item membership differing across the ten dumps.
6. Component membership differing across the ten dumps.
7. Frames-per-video differing across the ten dumps.
8. A seed whose two arms ran in different sessions, per the run manifest.

    python -m src.campaign_gate \\
        --a-glob 'results/predictions_npr/*_xception_seed*_test.csv' \\
        --b-glob 'results/predictions_npr/*_xception_npr_seed*_test.csv' \\
        --dir results/predictions_npr --manifest results/predictions_npr/runs.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

from .cluster_boot import _video_key
from .clusters import clusters_from_prediction_rows

SEEDS = (0, 1, 2, 3, 4)
ITEM_FIELDS = ("path", "label", "manipulation", "target_seq", "source_seq", "video_id")


def _rows(p: str) -> list[dict]:
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _seed_of(path: str) -> int | None:
    # Seed 0 dumps carry no `_split0` suffix while seeds 1-4 do, so match on the
    # seed token alone rather than on a whole-filename pattern.
    m = re.search(r"_seed(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def _partition(clusters: dict[str, str]) -> set[frozenset[str]]:
    """Compare component *structure*, not the generated component ids."""
    by_cid: dict[str, set[str]] = {}
    for vid, cid in clusters.items():
        by_cid.setdefault(cid, set()).add(vid)
    return {frozenset(v) for v in by_cid.values()}


class Gate:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}" if detail else name)
        return ok


def run(a_glob: str, b_glob: str, campaign_dir: str, manifest: str | None = None,
        n_seeds: int = 5, a_config: str = "xception", b_config: str = "xception_npr") -> int:
    g = Gate()
    a_paths, b_paths = sorted(glob.glob(a_glob)), sorted(glob.glob(b_glob))
    print(f"\n== NPR campaign pre-analysis gate ==\n  arm A ({a_config}): {len(a_paths)} dumps"
          f"\n  arm B ({b_config}): {len(b_paths)} dumps")

    # 1. Count and seed coverage, per arm.
    g.check(len(a_paths) == n_seeds and len(b_paths) == n_seeds,
            "dump count", f"expected {n_seeds} per arm, got {len(a_paths)} and {len(b_paths)}")
    a_seeds, b_seeds = [_seed_of(p) for p in a_paths], [_seed_of(p) for p in b_paths]
    want = list(SEEDS[:n_seeds])
    got_a = sorted(s for s in a_seeds if s is not None)
    got_b = sorted(s for s in b_seeds if s is not None)
    g.check(got_a == want and got_b == want, "one dump per seed per arm",
            f"A={a_seeds} B={b_seeds}")

    # 2. Containment. This is the check that stops a stray v8 glob.
    root = Path(campaign_dir).resolve()
    strays = [p for p in a_paths + b_paths if root not in Path(p).resolve().parents]
    g.check(not strays, "every dump inside the campaign directory",
            f"outside {root}: {strays}" if strays else str(root))

    if not (a_paths and b_paths):
        print("\n🚨 GATE FAILED — no dumps to inspect.")
        return 1

    # 3-7. Content checks. Load once; these files are ~3k rows each.
    loaded = {p: _rows(p) for p in a_paths + b_paths}
    for paths, cfg in ((a_paths, a_config), (b_paths, b_config)):
        bad = []
        for p in paths:
            got = {r["config"] for r in loaded[p]}
            if got != {cfg}:
                bad.append(f"{Path(p).name}={sorted(got)}")
        g.check(not bad, f"config column is '{cfg}' throughout arm", "; ".join(bad))

    split_seeds = {s for rows in loaded.values() for s in {r["split_seed"] for r in rows}}
    g.check(len(split_seeds) == 1, "single split_seed across campaign", f"found {sorted(split_seeds)}")

    items = {p: frozenset(tuple(r[f] for f in ITEM_FIELDS) for r in rows) for p, rows in loaded.items()}
    sizes = {len(v) for v in items.values()}
    g.check(len(set(items.values())) == 1, "identical test-item membership",
            f"{len(set(items.values()))} distinct sets, sizes {sorted(sizes)}")

    parts = {p: _partition(clusters_from_prediction_rows(rows)) for p, rows in loaded.items()}
    g.check(len(set(map(frozenset, parts.values()))) == 1, "identical component membership",
            f"{len({frozenset(v) for v in parts.values()})} distinct partitions, "
            f"n_components {sorted({len(v) for v in parts.values()})}")

    frames = {p: frozenset(Counter(_video_key(r) for r in rows).items()) for p, rows in loaded.items()}
    g.check(len(set(frames.values())) == 1, "identical frames-per-video",
            f"{len(set(frames.values()))} distinct frame profiles")

    # 8. Session pairing, per the prespecified session rule.
    if manifest and Path(manifest).exists():
        recs = json.loads(Path(manifest).read_text())
        by_seed: dict[int, set[str]] = {}
        for r in recs:
            by_seed.setdefault(int(r["seed"]), set()).add(str(r["session_id"]))
        split = {s: v for s, v in by_seed.items() if len(v) != 1}
        g.check(not split, "each seed pair completed in one session",
                f"seeds spanning sessions: {split}" if split else f"{len(by_seed)} seeds")
        g.check(set(by_seed) == set(SEEDS[:n_seeds]), "manifest covers every seed",
                f"manifest seeds {sorted(by_seed)}")
    else:
        g.check(False, "run manifest present",
                f"missing {manifest} — the session rule cannot be verified"
                if manifest else "no --manifest given")

    if g.failures:
        print(f"\n🚨 GATE FAILED — {len(g.failures)} check(s). Do not run the bootstrap.")
        for f in g.failures:
            print(f"  - {f}")
        return 1
    print("\n✅ GATE PASSED — dumps are one campaign, one split, matched membership.")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a-glob", required=True, help="baseline arm prediction CSVs")
    p.add_argument("--b-glob", required=True, help="comparison arm prediction CSVs")
    p.add_argument("--dir", required=True, dest="campaign_dir",
                   help="campaign directory every dump must live inside")
    p.add_argument("--manifest", default=None, help="per-run JSON log (session ids, order)")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--a-config", default="xception")
    p.add_argument("--b-config", default="xception_npr")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    return run(a.a_glob, a.b_glob, a.campaign_dir, a.manifest,
               a.n_seeds, a.a_config, a.b_config)


if __name__ == "__main__":
    raise SystemExit(main())
