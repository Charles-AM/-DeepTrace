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


def _prep(rows):
    comp = _components(rows)
    lab = np.array([int(r["label"]) for r in rows])
    sc = np.array([float(r["prob_fake"]) for r in rows])
    keys = sorted(set(comp))
    idx = {c: np.array([i for i, cc in enumerate(comp) if cc == c]) for c in keys}
    return lab, sc, keys, idx


def run(seen, unseen, n_boot=50000, boot_seed=0, out_dir=None):
    S, U = _rows(seen), _rows(unseen)
    sl, ss, sk, si = _prep(S)
    ul, us, uk, ui = _prep(U)
    if set(sk) & set(uk):
        raise SystemExit("seen and unseen share components -- this is not the V2 design")

    auc_s, auc_u = fast_auc(ss, sl), fast_auc(us, ul)
    print(f"  seen   AUC {auc_s:.4f}  ({len(sk)} components, {len(S)} crops)")
    print(f"  unseen AUC {auc_u:.4f}  ({len(uk)} components, {len(U)} crops)")
    print(f"  advantage  {auc_s - auc_u:+.4f}")

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

    out = {
        "estimand": "frame-pooled AUC, one fixed model, two matched disjoint-group sets",
        "resampling": "independent component resampling within each set (NOT paired)",
        "auc_seen": round(auc_s, 4), "auc_unseen": round(auc_u, 4),
        "advantage": round(float(auc_s - auc_u), 4),
        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "half_width": round(float((hi - lo) / 2), 4),
        "n_components_seen": len(sk), "n_components_unseen": len(uk),
        "n_boot_used": len(d), "n_boot_skipped": skipped, "boot_seed": boot_seed,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "_scope": ("Conditional on one trained model. Propagates test-content "
                   "variation only; no training-run variation. A mechanism "
                   "experiment, not a population-level protocol estimate."),
    }
    print(f"  95% CI     [{lo:+.4f}, {hi:+.4f}]  half-width {(hi-lo)/2:.4f}")
    print(f"  excludes zero: {out['excludes_zero']}   ({len(d)} replicates, {skipped} skipped)")
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
