"""Regenerate every headline number from committed data and check it.

One command that answers: does this repository still reproduce its own claims?

    python -m src.reproduce

Uses only committed prediction dumps. No GPU, no training, no network. Each check
prints PASS or FAIL against a value recorded in the repository, so a silent drift
between the prose and the data becomes a failing line rather than a discovery
during review.

Claims that CANNOT be reproduced are reported explicitly rather than skipped --
see UNREPRODUCIBLE at the end. Hiding them would defeat the purpose.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOL = 0.0015          # Monte Carlo tolerance for bootstrap endpoints
results: list[tuple[str, bool, str]] = []


def check(name, got, want, tol=0.0, note=""):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    results.append((name, ok, f"got {got}, want {want}" + (f"  ({note})" if note else "")))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {got} vs {want}")
    return ok


def main():
    print("=" * 74)
    print("REPRODUCING HEADLINE NUMBERS FROM COMMITTED DATA")
    print("=" * 74)

    # ---- 1. unit structure ------------------------------------------------
    print("\n[1] Unit structure — the fact all three contributions rest on")
    sys.path.insert(0, str(REPO))
    import csv
    from src.agg_space import _components
    rows = list(csv.DictReader(
        open(REPO / "results/predictions_v8/ffpp_c40_vid_xception_seed0_test.csv")))
    check("test crops", len(rows), 3000)
    check("video files (video x manipulation)",
          len({r["video_id"] + "|" + r["manipulation"] for r in rows}), 150)
    check("target groups", len({r["video_id"] for r in rows}), 30)
    check("source-target components", len(set(_components(rows))), 29)
    check("video_id == target_seq (all rows)",
          sum(r["video_id"] == r["target_seq"] for r in rows), 3000)

    # ---- 2. canonical crossed result --------------------------------------
    print("\n[2] Canonical crossed result — src/crossed_boot.py")
    canon = json.loads((REPO / "results/canonical.json").read_text())
    out = subprocess.run(
        [sys.executable, "-m", "src.crossed_boot",
         "--a-glob", "results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv",
         "--b-glob", "results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv",
         "--margins", "0.014", "--n-boot", "50000", "--boot-seed", "0",
         "--out-dir", "/tmp/_repro"],
        cwd=REPO, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if "crossed " in l), "")
    if not line:
        results.append(("crossed reproduction", False, out.stderr[-300:]))
        print("  FAIL  crossed_boot produced no result line")
    else:
        lo = float(line.split("[")[1].split(",")[0])
        hi = float(line.split(",")[1].split("]")[0])
        check("crossed ci_lo", lo, canon["ci_lo"], TOL)
        check("crossed ci_hi", hi, canon["ci_hi"], TOL)
        check("exceedance rate", float(line.split("P(>0.014)=")[1].split("±")[0]),
              canon["exceedance_rate_above_reference"], 0.002)

    # ---- 3. aggregation space ---------------------------------------------
    print("\n[3] Aggregation space — src/agg_space.py")
    from src.agg_space import run as agg_run
    rows_out, paired = agg_run(
        "results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv",
        "results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv",
        n_boot=50000, boot_seed=0, out_dir=None)
    by = {r["aggregation_space"]: r for r in rows_out}
    check("mean-logit point estimate", by["mean_logit"]["point_estimate"], -0.0092, 0.0002)
    check("mean-probability point estimate", by["mean_probability"]["point_estimate"], -0.0060, 0.0002)
    check("paired difference mean", paired["paired_diff_mean"], 0.00286, 0.0002)

    # ---- 4. V2 seen-vs-unseen ---------------------------------------------
    print("\n[4] V2 seen-vs-unseen — src/seen_unseen_boot.py")
    seen = REPO / "results/predictions_v2/v2seen_xception_seed0_test.csv"
    unseen = REPO / "results/predictions_v2/v2unseen_xception_seed0_test.csv"
    if not (seen.exists() and unseen.exists()):
        print("  SKIP  dumps not yet committed to results/predictions_v2/")
        print("        (the result exists but is aggregate-only until they are)")
    else:
        from src.seen_unseen_boot import run as v2_run
        # Point estimates are deterministic; a small bootstrap is enough to verify
        # them. The full 50,000-replicate intervals live in
        # results/analysis/v2/v2_advantage.json.
        r = v2_run(str(seen), str(unseen), n_boot=2000, boot_seed=0, out_dir=None)
        fp = r["by_estimand"]["frame_pooled"]
        vl = r["by_estimand"]["video_mean_logit"]
        check("V2 seen AUC (frame-pooled)", fp["auc_seen"], 0.9884, 0.0002)
        check("V2 unseen AUC (frame-pooled)", fp["auc_unseen"], 0.7886, 0.0002)
        check("V2 advantage (frame-pooled)", fp["advantage"], 0.1998, 0.0004)
        check("V2 advantage (video mean-logit)", vl["advantage"], 0.1731, 0.0004)
        check("V2 advantage excludes zero", fp["excludes_zero"], True)

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 74)
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFAILED:")
        for n, ok, d in results:
            if not ok:
                print(f"  {n}: {d}")

    print("\n" + "=" * 74)
    print("UNREPRODUCIBLE FROM COMMITTED DATA — stated, not skipped")
    print("=" * 74)
    print("""
  Protocol gap (~18 AUC points, L1 vs L2)
      The L1 campaign has NO prediction dumps -- 9 per-run provenance JSONs and
      aggregate metrics only (results/in_domain_c40/). No checkpoints are committed
      either, so regenerating dumps requires retraining. The figure is traceable to
      per-run roc_auc written at training time; it cannot be recomputed from rows.
      Recorded in docs/contribution-1-evidence.md section 0.

  V2 seen/unseen (until the dumps are committed)
      The result is computed and recorded in results/analysis/v2/, but until the
      two prediction CSVs are committed it cannot be recomputed here. Check [4]
      above says SKIP in that state.

  C1 learning-rate sweep and C2 repeat audit (2026-09-17 and -18)
      Trained without a prediction dump; train.py writes a per-run JSON and a
      checkpoint but not per-frame scores. Recoverable by inference while the
      Kaggle version outputs survive -- see docs/ARTIFACT-POLICY.md.
""")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
