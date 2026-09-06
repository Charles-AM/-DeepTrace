"""Per-manipulation breakdown computed from prediction dumps, not checkpoints.

`src/permanip.py` reloads models and re-runs inference. That made it expensive
and, worse, it was last run against the **crop-randomised (L1) checkpoints** —
the leaky protocol this project exists to criticise. Any figure derived from it
describes a model evaluated on frames from videos it trained on.

This module recomputes the same breakdown from `results/predictions/*.csv`, which
are video-disjoint (L2) and committed to the repo, so it needs no GPU and no
checkpoints, and it cannot silently drift back onto the wrong protocol.

Each manipulation is scored against the SAME real videos: AUC for method M uses
the real crops plus M's fakes only. Scores are aggregated to one value per video
first, matching the paper's video-level metric rather than the frame-pooled one.

    python -m src.permanip_preds --preds results/predictions --out-dir results/analysis/permanip_l2
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .resolution_curve import auc


def _video_key(r: dict) -> str:
    return f"{r['manipulation']}|{r['target_seq']}|{r['source_seq']}"


def per_manipulation(rows: list[dict]) -> list[dict]:
    """One row per manipulation: video-level AUC against the shared real set."""
    acc: dict[str, dict] = {}
    for r in rows:
        k = _video_key(r)
        d = acc.setdefault(k, {"label": int(r["label"]), "manip": r["manipulation"],
                               "scores": []})
        d["scores"].append(float(r["logit_margin"]))
    vids = [{"label": d["label"], "manip": d["manip"],
             "score": float(np.mean(d["scores"]))} for d in acc.values()]

    reals = [v for v in vids if v["label"] == 0]
    out = []
    for manip in sorted({v["manip"] for v in vids if v["label"] == 1}):
        fakes = [v for v in vids if v["label"] == 1 and v["manip"] == manip]
        sub = reals + fakes
        y = np.array([v["label"] for v in sub])
        s = np.array([v["score"] for v in sub])
        out.append({"manipulation": manip, "n_real_videos": len(reals),
                    "n_fake_videos": len(fakes),
                    "auc": round(auc(y, s), 4)})
    return out


def run(preds_dir: Path, out_dir: Path | None = None) -> list[dict]:
    rows_out = []
    for f in sorted(Path(preds_dir).glob("*_test.csv")):
        rows = list(csv.DictReader(open(f)))
        m = re.match(r"(?P<ds>ffpp_c\d+)_vid_(?P<cfg>.+?)_seed(?P<seed>\d+)", f.stem)
        for r in per_manipulation(rows):
            rows_out.append({"run": f.stem,
                             "compression": m.group("ds").split("_")[1] if m else "?",
                             "config": m.group("cfg") if m else "?",
                             "seed": int(m.group("seed")) if m else -1, **r})
        print(f"{f.stem}: " + "  ".join(
            f"{r['manipulation']}={r['auc']:.4f}" for r in per_manipulation(rows)))

    if out_dir and rows_out:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "per_manipulation_l2.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {out} ({len(rows_out)} rows)")
    return rows_out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", default="results/predictions")
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run(Path(a.preds), Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    main()
