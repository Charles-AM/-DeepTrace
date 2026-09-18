"""Aggregation-space sensitivity: mean-logit vs mean-probability video scores.

"Video-aggregated AUC" does not fully specify an estimand. Frame scores can be
averaged in LOGIT space -- what src/cluster_boot.py does, and what produced the
canonical result -- or in PROBABILITY space. Averaging and the sigmoid do not
commute, so the two can rank videos differently and yield different AUCs. Neither
is incorrect; they encode different aggregation rules, and no paper in
docs/split-policy-audit.md specifies which it uses.

The comparison is PAIRED: one set of component x training-run resamples is drawn
once and applied to both arms, so the contrast carries no Monte Carlo noise
between them. That is what licenses the per-replicate difference statistics.

⚠️ POST-HOC. The ambiguity was found while validating other numbers (ledger §19).

    python -m src.agg_space \
        --a-glob 'results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv' \
        --b-glob 'results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv' \
        --n-boot 50000 --boot-seed 0 --margin 0.014 \
        --out-dir results/analysis/agg_space
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

FIELDS = ("logit_margin", "prob_fake")
NAMES = {"logit_margin": "mean_logit", "prob_fake": "mean_probability"}


def _load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fast_auc(scores, labels):
    """Rank AUC with mid-ranks for ties. Same implementation as resolution_curve."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    ranks = ((starts + ends + 1) / 2.0)[inv]
    n_pos = int((labels == 1).sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: one class absent")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _components(rows):
    """Union-find over the source-target pair graph."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in rows:
        if r["source_seq"]:
            ra, rb = find(r["target_seq"]), find(r["source_seq"])
            if ra != rb:
                parent[ra] = rb
    return [find(r["target_seq"]) for r in rows]


def run(a_glob, b_glob, n_boot=50000, boot_seed=0, margin=0.014, out_dir=None):
    a_paths, b_paths = sorted(glob.glob(a_glob)), sorted(glob.glob(b_glob))
    if not a_paths or len(a_paths) != len(b_paths):
        raise SystemExit(f"need matching run sets, got {len(a_paths)} and {len(b_paths)}")
    A = [_load(p) for p in a_paths]
    B = [_load(p) for p in b_paths]

    ref = [r["path"] for r in A[0]]
    for x in A + B:
        if [r["path"] for r in x] != ref:
            raise SystemExit(
                "runs do not share identical test items. The paired contrast is only "
                "meaningful on one fixed split -- refusing to proceed.")

    key = [r["video_id"] + "|" + r["manipulation"] for r in A[0]]
    vids = sorted(set(key))
    idx_of = {v: [i for i, k in enumerate(key) if k == v] for v in vids}
    comp_of_row = _components(A[0])
    vlab = np.array([int(A[0][idx_of[v][0]]["label"]) for v in vids])
    vcomp = [comp_of_row[idx_of[v][0]] for v in vids]
    comps = sorted(set(vcomp))
    cidx = {c: np.array([i for i, cc in enumerate(vcomp) if cc == c]) for c in comps}

    def agg(rows, field):
        return np.array([np.mean([float(rows[i][field]) for i in idx_of[v]]) for v in vids])

    scores = {f: (np.array([agg(a, f) for a in A]), np.array([agg(b, f) for b in B]))
              for f in FIELDS}

    # ---- one set of draws, applied to BOTH arms -------------------------
    rng = np.random.default_rng(boot_seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(comps), len(comps))
        idx = np.concatenate([cidx[comps[i]] for i in pick])
        if vlab[idx].sum() in (0, len(idx)):
            continue
        draws.append((idx, rng.integers(0, len(A), len(A))))

    out, dists = [], {}
    for f in FIELDS:
        sa, sb = scores[f]
        per = np.array([fast_auc(sb[s], vlab) - fast_auc(sa[s], vlab) for s in range(len(A))])
        o = np.array([np.mean([fast_auc(sb[s][i], vlab[i]) - fast_auc(sa[s][i], vlab[i])
                               for s in ss]) for i, ss in draws])
        dists[f] = o
        lo, hi = np.percentile(o, 2.5), np.percentile(o, 97.5)
        out.append({
            "aggregation_space": NAMES[f],
            "point_estimate": round(float(per.mean()), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "half_width": round(float((hi - lo) / 2), 4),
            f"exceedance_above_{margin:g}".replace(".", "p"): round(float((o > margin).mean()), 4),
            "contains_zero": bool(lo < 0 < hi),
            f"contains_{margin:g}".replace(".", "p"): bool(lo < margin < hi),
            "n_boot": len(draws), "n_components": len(comps),
            "n_videos": len(vids), "n_runs": len(A),
            "per_seed": "; ".join(f"{d:+.4f}" for d in per),
        })

    d = dists["prob_fake"] - dists["logit_margin"]
    paired = {
        "paired_diff_mean": round(float(d.mean()), 5),
        "paired_diff_p2.5": round(float(np.percentile(d, 2.5)), 4),
        "paired_diff_p97.5": round(float(np.percentile(d, 97.5)), 4),
        "replicates_disagreeing_on_margin": round(
            float(np.mean((dists["logit_margin"] > margin) != (dists["prob_fake"] > margin))), 4),
    }

    for r in out:
        print("  %-16s %+.4f  [%+.4f, %+.4f]  hw %.4f" %
              (r["aggregation_space"], r["point_estimate"], r["ci_lo"], r["ci_hi"], r["half_width"]))
    print("  paired diff mean %+.5f   replicates disagreeing on margin %.4f" %
          (paired["paired_diff_mean"], paired["replicates_disagreeing_on_margin"]))

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "agg_space_c40_v8.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        (out_dir / "paired_contrast.json").write_text(json.dumps(paired, indent=2))
        print(f"wrote {out_dir}/agg_space_c40_v8.csv and paired_contrast.json")
    return out, paired


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--a-glob", required=True)
    p.add_argument("--b-glob", required=True)
    p.add_argument("--n-boot", type=int, default=50000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=0.014)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    run(a.a_glob, a.b_glob, a.n_boot, a.boot_seed, a.margin, a.out_dir)


if __name__ == "__main__":
    main()
