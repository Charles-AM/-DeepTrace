# Tier 0 — paste-ready Kaggle cells

Copy each block into its own cell, in order. No placeholders to fill in: Cell 2
finds the prediction dumps itself.

**Accelerator: None** (CPU) unless Cell 2 reports the dumps are missing.

---

### Cell 1 — clone

```python
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace
!cd /kaggle/working/-DeepTrace && git log --oneline -1
```

### Cell 2 — run the test suite

Fast, and it is the only place the full suite actually runs — pytest and sklearn
are not installed on the dev machine.

```python
!cd /kaggle/working/-DeepTrace && python -m pytest tests/ -q 2>&1 | tail -15
```

### Cell 3 — find the prediction dumps

`os.environ` is used deliberately: a shell variable set in a `!` line dies with
that line's shell, but an environ entry set from Python is visible to every later
`!` cell.

```python
import os, glob, csv, collections, pathlib

cands = collections.Counter()
for p in glob.glob("/kaggle/input/**/*_test.csv", recursive=True):
    cands[str(pathlib.Path(p).parent)] += 1

if not cands:
    print("NO PREDICTION DUMPS FOUND — run Cell 3b (needs GPU)")
else:
    preds, n = cands.most_common(1)[0]
    os.environ["PREDS"] = preds
    print(f"PREDS = {preds}   ({n} files)\n")
    for f in sorted(glob.glob(preds + "/*_test.csv")):
        print("   ", pathlib.Path(f).name)

    # Validate one file before anything downstream trusts it.
    sample = sorted(glob.glob(preds + "/*_test.csv"))[0]
    rows = list(csv.DictReader(open(sample)))
    vids = {(r["manipulation"], r["target_seq"], r["source_seq"]) for r in rows}
    print(f"\ncheck {pathlib.Path(sample).name}: {len(rows)} rows, "
          f"{len({r['target_seq'] for r in rows})} target groups, {len(vids)} videos")
    print("  empty video_id rows:", sum(1 for r in rows if not r["video_id"]))
    assert len(rows) == 3000, "expected 3000 crops"
    assert len({r["target_seq"] for r in rows}) == 30, \
        "expected 30 target groups — 300 means a crop-randomised manifest, STOP"
    assert not any(not r["video_id"] for r in rows), "unparsed crop names present"
    print("  OK")
```

### Cell 3b — ONLY if Cell 3 found nothing (switch accelerator to GPU T4 x2, ~15 min)

Replace `<C40VID_RESULTS>` with the attached c40 video-level output path.

```python
!cd /kaggle/working/-DeepTrace && for cfg in xception f3net; do for s in 0 1 2 3 4; do python -m src.predict --run ffpp_c40_vid_${cfg}_seed${s} --results-root <C40VID_RESULTS> --dataset-name ffpp_c40_vid --seed $s --image-size 128 --out-dir /kaggle/working/preds; done; done
```

```python
import os; os.environ["PREDS"] = "/kaggle/working/preds"
```

### Cell 4 — V7, the resolution curve (~10 min)

```python
!cd /kaggle/working/-DeepTrace && python -m src.resolution_curve --a $PREDS/ffpp_c40_vid_xception_seed0_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed0_test.csv --unit video --out-dir results/analysis/resolution
```

```python
!cd /kaggle/working/-DeepTrace && python -m src.resolution_curve --a $PREDS/ffpp_c40_vid_xception_seed0_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed0_test.csv --unit component --out-dir results/analysis/resolution
```

Small sizes being flagged `DEGENERATE` and dropped is the guard working, not a
failure. `slope` must be negative; `n_for_halfwidth_0.014` is the headline.

### Cell 5 — V6, threshold sensitivity (~5 min)

```python
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --margins 0.005 0.010 0.014 0.020 --out-dir results/analysis/thresholds; done
```

```python
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.cluster_boot --a $PREDS/ffpp_c23_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c23_vid_f3net_seed${s}_test.csv --margins 0.005 0.010 0.014 0.020 --out-dir results/analysis/thresholds; done
```

### Cell 6 — V4, the aggregation effect (~5 min)

Also regenerates the c40 seed-0 video-level file that an earlier `--frame-level`
run overwrote.

```python
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --margins 0.014 --out-dir results/analysis/aggregation; python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --frame-level --margins 0.014 --out-dir results/analysis/aggregation; done
```

### Cell 7 — collect and package

```python
import csv, glob, pathlib, os
os.chdir("/kaggle/working/-DeepTrace")
for name in ("resolution", "thresholds", "aggregation"):
    files = sorted(glob.glob(f"results/analysis/{name}/*.csv"))
    files = [f for f in files if not f.endswith("ALL.csv")]
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            rows.append({"source_file": pathlib.Path(f).name, **r})
    if not rows:
        print(f"{name}: NO ROWS"); continue
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "source_file", k))
    with open(f"results/analysis/{name}/ALL.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows from {len(files)} files -> ALL.csv")
```

```python
!cd /kaggle/working && tar czf tier0-results.tar.gz -C ./-DeepTrace results/analysis && ls -lh tier0-results.tar.gz
```

Download `tier0-results.tar.gz` from the notebook output.

### Cell 8 — read the headline before you leave

```python
import csv, glob
for f in sorted(glob.glob("/kaggle/working/-DeepTrace/results/analysis/resolution/fit_*.csv")):
    r = next(csv.DictReader(open(f)))
    print(f.split("/")[-1])
    print(f"   slope={r['slope']}  R2={r['r2']}  fitted={r['n_points_fitted']}  "
          f"excluded={r['sizes_excluded_degenerate']}")
    for t in ("0.02", "0.014", "0.01", "0.005"):
        k = f"n_for_halfwidth_{t}"
        if k in r:
            print(f"   +-{t}: {r[k]} units   extrapolated={r[f'extrapolated_{t}']}")
```
