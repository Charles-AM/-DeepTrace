"""V2 — isolate seen-video leakage with the model and test composition held fixed.

The L1-vs-L2 gap this project reports (~18 AUC points at c40) compares two
*different partitions*, so it confounds seen-video leakage with the two test sets
differing in difficulty. Until that is separated the honest name is "protocol
gap", not leakage — a hedge that currently appears in five places in ESSENCE.

This builds the experiment that separates them. From one video-disjoint manifest
it writes two more:

  SEEN manifest    train = L2 train minus a held-out slice of crops
                   val   = L2 val, unchanged (checkpoint selection is identical)
                   test  = those held-out crops — videos the model trained on,
                           crops it never saw. The L1-like condition.

  UNSEEN manifest  test  = an equally-sized, equally-shaped slice of the L2 test
                           videos. The L2 condition.

Train once on the SEEN manifest, then score that single fixed model against both.
The difference is the seen-video advantage, with architecture, weights, training
data, checkpoint-selection rule and test composition all held constant.

Matching, which is what makes the comparison mean anything:

  * same number of videos, and the same number of crops from each
  * same target-group structure (each group is 1 real + 4 manipulations)
  * same manipulation mix and compression
  * same sampling positions — crops are taken at the same evenly spaced ranks
    within each video, so the two sets do not differ in where in the clip they
    were sampled

    python -m src.seen_unseen --manifest results/manifests/ffpp_c40_vid_seed0_sz128.csv \\
        --out-root results --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .cropname import parse_crop_name


def _read(manifest: Path) -> list[dict]:
    with Path(manifest).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _by_video(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group rows by the actual video file, keeping frame order."""
    vids: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        m = parse_crop_name(r["path"])
        if not m["video_id"]:
            raise ValueError(f"unparsed crop name in manifest: {r['path']}")
        r["_meta"] = m
        vids[(m["manipulation"], m["target_seq"], m["source_seq"])].append(r)
    for k in vids:
        vids[k].sort(key=lambda r: r["_meta"]["frame_idx"])
    return dict(vids)


def _ranks(n_total: int, k: int) -> list[int]:
    """k evenly spaced positions within a video, identical for every video so the
    two evaluation sets cannot differ in sampling position."""
    if k > n_total:
        raise ValueError(f"asked for {k} crops but videos have {n_total}")
    step = n_total / k
    return [min(n_total - 1, int(i * step)) for i in range(k)]


def build(manifest: Path, out_root: Path, seed: int = 0, image_size: int = 128,
          n_groups: int = 30, crops_per_video: int = 5,
          name_seen: str = "ffpp_c40_v2seen",
          name_unseen: str = "ffpp_c40_v2unseen") -> dict:
    import random

    rows = _read(manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    tr_vids, te_vids = _by_video(train), _by_video(test)

    # Pick target GROUPS, not videos: a group is one real plus its manipulations,
    # so selecting groups preserves the 1:4 class balance and the manipulation mix
    # automatically. Picking videos independently would not.
    tr_groups = sorted({k[1] for k in tr_vids})
    te_groups = sorted({k[1] for k in te_vids})
    if len(tr_groups) < n_groups:
        raise ValueError(f"train has {len(tr_groups)} groups, need {n_groups}")
    picked = sorted(random.Random(seed).sample(tr_groups, n_groups))

    def slice_of(vids, groups):
        held, kept = [], []
        for key, crops in vids.items():
            if key[1] not in groups:
                kept.extend(crops)
                continue
            idx = set(_ranks(len(crops), crops_per_video))
            for i, r in enumerate(crops):
                (held if i in idx else kept).append(r)
        return held, kept

    seen_eval, train_kept = slice_of(tr_vids, set(picked))
    unseen_eval, _ = slice_of(te_vids, set(te_groups))

    if len(seen_eval) != len(unseen_eval):
        raise ValueError(f"unmatched sizes: seen={len(seen_eval)} unseen={len(unseen_eval)}")

    def write(path: Path, splits: dict[str, list[dict]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["path", "label", "split"])
            for name, rs in splits.items():
                for r in rs:
                    w.writerow([r["path"], r["label"], name])

    mdir = Path(out_root) / "manifests"
    p_seen = mdir / f"{name_seen}_seed{seed}_sz{image_size}.csv"
    p_unseen = mdir / f"{name_unseen}_seed{seed}_sz{image_size}.csv"
    write(p_seen, {"train": train_kept, "val": val, "test": seen_eval})
    write(p_unseen, {"train": train_kept, "val": val, "test": unseen_eval})

    def shape(rs):
        vids = {(r["_meta"]["manipulation"], r["_meta"]["target_seq"],
                 r["_meta"]["source_seq"]) for r in rs}
        manip = defaultdict(int)
        for r in rs:
            manip[r["_meta"]["manipulation"]] += 1
        return {"crops": len(rs), "videos": len(vids),
                "groups": len({r["_meta"]["target_seq"] for r in rs}),
                "reals": sum(1 for r in rs if int(r["label"]) == 0),
                "by_manipulation": dict(sorted(manip.items()))}

    report = {
        "source_manifest": str(manifest), "seed": seed,
        "n_groups": n_groups, "crops_per_video": crops_per_video,
        "seen_groups": picked,
        "seen_eval": shape(seen_eval), "unseen_eval": shape(unseen_eval),
        "train_before": len(train), "train_after": len(train_kept),
        "manifest_seen": str(p_seen), "manifest_unseen": str(p_unseen),
    }
    (mdir / f"v2_report_seed{seed}.json").write_text(json.dumps(report, indent=2))

    print(f"seen_eval   {report['seen_eval']}")
    print(f"unseen_eval {report['unseen_eval']}")
    print(f"train {report['train_before']} -> {report['train_after']} crops")
    print(f"wrote {p_seen}\n      {p_unseen}")
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-root", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--n-groups", type=int, default=30)
    p.add_argument("--crops-per-video", type=int, default=5)
    p.add_argument("--name-seen", default="ffpp_c40_v2seen")
    p.add_argument("--name-unseen", default="ffpp_c40_v2unseen")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    build(Path(a.manifest), Path(a.out_root), seed=a.seed, image_size=a.image_size,
          n_groups=a.n_groups, crops_per_video=a.crops_per_video,
          name_seen=a.name_seen, name_unseen=a.name_unseen)


if __name__ == "__main__":
    main()
