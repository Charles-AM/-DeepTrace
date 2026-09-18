"""Validate V2 seen/unseen dumps before any analysis touches them.

Distinct file hashes prove the files differ. They do NOT prove the right examples
were scored. This checks membership, composition and the recorded AUC, so a
silently wrong evaluation cannot reach the analysis.

    python -m src.verify_v2 --seen <csv> --unseen <csv> \
        --manifest-seen <csv> --manifest-unseen <csv> --expect-seen-auc 0.9884
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from .agg_space import fast_auc

checks: list[tuple[str, bool, str]] = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {detail}")
    return ok


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _groups(rows, key="video_id"):
    return {r[key] for r in rows}


def _manifest_split(path, split):
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                out.append(r["path"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seen", required=True)
    p.add_argument("--unseen", required=True)
    p.add_argument("--manifest-seen", required=True)
    p.add_argument("--manifest-unseen", required=True)
    p.add_argument("--expect-seen-auc", type=float, default=None)
    p.add_argument("--tol", type=float, default=0.002)
    a = p.parse_args(argv)

    print("=" * 74)
    print("V2 DUMP VALIDATION — membership, composition, recorded AUC")
    print("=" * 74)

    # 1. distinct files
    ck("seen and unseen paths differ", Path(a.seen).resolve() != Path(a.unseen).resolve())
    h = [hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in (a.seen, a.unseen)]
    ck("dumps are not byte-identical", h[0] != h[1], f"{h[0][:12]} vs {h[1][:12]}")

    S, U = _rows(a.seen), _rows(a.unseen)

    # 2. size
    ck("seen has 750 predictions", len(S) == 750, f"got {len(S)}")
    ck("unseen has 750 predictions", len(U) == 750, f"got {len(U)}")

    # 3. no exact evaluation crop appears in training
    train = set(_manifest_split(a.manifest_seen, "train"))
    sp, up = {r["path"] for r in S}, {r["path"] for r in U}
    ck("zero seen-eval crops in training", len(sp & train) == 0, f"overlap {len(sp & train)}")
    ck("zero unseen-eval crops in training", len(up & train) == 0, f"overlap {len(up & train)}")

    # 4/5. group-level representation — the actual seen/unseen distinction
    import re
    def vid_of(path):
        m = re.search(r"videos-(\d+)", path)
        return m.group(1) if m else None
    train_groups = {vid_of(x) for x in train} - {None}
    sg, ug = _groups(S), _groups(U)
    ck("seen groups ARE represented in training",
       sg <= train_groups, f"{len(sg & train_groups)}/{len(sg)} represented")
    ck("unseen groups are NOT represented in training",
       len(ug & train_groups) == 0, f"{len(ug & train_groups)} leaked")
    ck("seen and unseen groups are disjoint", len(sg & ug) == 0,
       "independent resampling required, NOT paired")

    # 6. composition matches
    def comp(rows):
        d = {}
        for r in rows:
            d[r["manipulation"]] = d.get(r["manipulation"], 0) + 1
        return d
    ck("manipulation composition matches", comp(S) == comp(U), f"{comp(S)}")
    ck("class balance matches",
       sum(int(r["label"]) for r in S) == sum(int(r["label"]) for r in U))
    ck("group counts match", len(sg) == len(ug), f"{len(sg)} vs {len(ug)}")

    # 7. recorded AUC reproduces
    auc_s = fast_auc([float(r["prob_fake"]) for r in S], [int(r["label"]) for r in S])
    auc_u = fast_auc([float(r["prob_fake"]) for r in U], [int(r["label"]) for r in U])
    print(f"\n  seen AUC   {auc_s:.4f}")
    print(f"  unseen AUC {auc_u:.4f}")
    print(f"  difference {auc_s - auc_u:+.4f}   <- the seen-video advantage\n")
    if a.expect_seen_auc is not None:
        ck("seen AUC reproduces the training log",
           abs(auc_s - a.expect_seen_auc) <= a.tol,
           f"{auc_s:.4f} vs {a.expect_seen_auc}")

    failed = [n for n, ok, _ in checks if not ok]
    print("=" * 74)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("\nDO NOT ANALYSE THESE DUMPS. Failed:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("\nValidated. Resample components INDEPENDENTLY within each set --")
    print("the two sets contain disjoint groups, so this is not a paired design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
