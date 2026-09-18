"""V2 seen-vs-unseen advantage with INDEPENDENT component resampling.

The two evaluation sets contain DISJOINT video groups -- seen_eval is drawn from
training videos, unseen_eval from test videos (src/seen_unseen.py). They are matched
in composition (counts, manipulation mix, sampling positions) but share no groups,
so this is NOT a paired design. Each set is resampled independently and the
difference of the two resampled AUCs is taken.

⚠️ This is a MECHANISM experiment conditional on one trained model. It propagates
test-content variation only -- no training-run variation -- so it is not a
population-level protocol estimate.

    python -m src.seen_unseen_boot --seen <csv> --unseen <csv> \
        --n-boot 50000 --boot-seed 0 --out-dir results/analysis/v2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .agg_space import fast_auc, _components


def _rows(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


ESTIMANDS = (
    ("frame_pooled", None),
    ("video_mean_logit", "logit_margin"),
    ("video_mean_probability", "prob_fake"),
)


def _prep(rows, field=None):
    """field=None -> frame-pooled. Otherwise aggregate to one score per video."""
    comp = _components(rows)
    if field is None:
        lab = np.array([int(r["label"]) for r in rows])
        sc = np.array([float(r["prob_fake"]) for r in rows])
        units = comp
    else:
        key = [r["video_id"] + "|" + r["manipulation"] for r in rows]
        vids = sorted(set(key))
        by = {v: [i for i, k in enumerate(key) if k == v] for v in vids}
        lab = np.array([int(rows[by[v][0]]["label"]) for v in vids])
        sc = np.array([np.mean([float(rows[i][field]) for i in by[v]]) for v in vids])
        units = [comp[by[v][0]] for v in vids]
    keys = sorted(set(units))
    idx = {c: np.array([i for i, cc in enumerate(units) if cc == c]) for c in keys}
    return lab, sc, keys, idx


def run(seen, unseen, n_boot=50000, boot_seed=0, out_dir=None):
    """Report the advantage under all three estimands.

    The paper's first contribution is that estimand choice changes reported
    quantities, so reporting this result under one estimand would undercut it.
    """
    S, U = _rows(seen), _rows(unseen)
    results = {}
    for name, field in ESTIMANDS:
        sl, ss, sk, si = _prep(S, field)
        ul, us, uk, ui = _prep(U, field)
        if set(sk) & set(uk):
            raise SystemExit("seen and unseen share components -- not the V2 design")
        auc_s, auc_u = fast_auc(ss, sl), fast_auc(us, ul)
        rng = np.random.default_rng(boot_seed)
        diffs, skipped = [], 0
        for _ in range(n_boot):
            a = np.concatenate([si[sk[i]] for i in rng.integers(0, len(sk), len(sk))])
            b = np.concatenate([ui[uk[i]] for i in rng.integers(0, len(uk), len(uk))])
            if sl[a].sum() in (0, len(a)) or ul[b].sum() in (0, len(b)):
                skipped += 1
                continue
            diffs.append(fast_auc(ss[a], sl[a]) - fast_auc(us[b], ul[b]))
        d = np.array(diffs)
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        results[name] = {
            "auc_seen": round(auc_s, 4), "auc_unseen": round(auc_u, 4),
            "advantage": round(float(auc_s - auc_u), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "half_width": round(float((hi - lo) / 2), 4),
            "n_units_seen": len(sl), "n_units_unseen": len(ul),
            "n_components_seen": len(sk), "n_components_unseen": len(uk),
            "n_boot_used": len(d), "n_boot_skipped": skipped,
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
        print(f"  {name:24s} seen {auc_s:.4f}  unseen {auc_u:.4f}  "
              f"adv {auc_s-auc_u:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    out = {
        "primary_estimand": "frame_pooled",
        "resampling": "independent component resampling within each set (NOT paired)",
        "boot_seed": boot_seed,
        "by_estimand": results,
        "_scope": ("Conditional on ONE trained model. Propagates test-content "
                   "variation only; NO training-run variation. A mechanism "
                   "experiment, not a population-level protocol estimate."),
        "_residual_confound": ("The two sets contain DIFFERENT video groups. "
                               "Composition is matched (counts, manipulation mix, "
                               "sampling positions) and assignment was random under "
                               "the seeded L2 split, but equal composition does not "
                               "guarantee equal intrinsic difficulty. With 30 groups "
                               "per side the realised difficulty may differ."),
    }
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "v2_advantage.json").write_text(json.dumps(out, indent=2))
        print(f"wrote {out_dir}/v2_advantage.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seen", required=True)
    p.add_argument("--unseen", required=True)
    p.add_argument("--n-boot", type=int, default=50000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    a = p.parse_args(argv)
    run(a.seen, a.unseen, a.n_boot, a.boot_seed, a.out_dir)


if __name__ == "__main__":
    main()
