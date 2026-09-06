# Tier 0 runbook — resolution, thresholds, aggregation

**Zero GPU if the V1 prediction dumps are still attached.** Everything here reads
`results/predictions/*.csv`; nothing trains and nothing needs a checkpoint unless
those CSVs have to be regenerated (see Cell 2b).

Produces three things ESSENCE.md puts in the main paper and that do not yet exist:

| # | output | why it matters |
|---|---|---|
| **V7** | interval half-width vs independent-video count, + the N needed for +0.014 | §9 calls it "contribution 2's core figure"; nothing produced it |
| **V6** | equivalence/exclusion at 0.005 / 0.010 / 0.014 / 0.020 | §12 lists thresholds as main-paper content |
| **V4** | frame-pooled vs video-level AUC on the same videos | separates the aggregation effect from the split effect (C0b) |

~25 min wall clock, CPU only.

---

## Inputs to attach

- the **V1 notebook output** (contains `preds/*.csv`) — this is the only hard requirement
- if that output is gone: the **c40 video-level** and **c23 video-level** outputs
  (checkpoints + `results/manifests/`) plus the matching crop datasets, and run Cell 2b

## Cell 1 — clone

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```

Expect the tip to be at or after `3fab34c` (V6). The leading `-` in the directory
name needs the `./` prefix on every `cd`, `rm` and `cp`.

## Cell 2 — point at the prediction dumps

```bash
!ls /kaggle/input/<V1_OUTPUT>/preds/*.csv | head -20 && ls /kaggle/input/<V1_OUTPUT>/preds/*.csv | wc -l
```

Expect 17 files.

⚠️ Set `PREDS` from **Python**, not a shell line: a variable assigned in a `!`
cell dies with that cell's shell and `$PREDS` is empty in the next one.

```python
import os; os.environ["PREDS"] = "/kaggle/input/<V1_OUTPUT>/preds"
```

Paste-ready cells that discover this path automatically: `docs/tier0-cells.md`.

## Cell 2b — ONLY if the dumps are missing (needs GPU, ~15 min)

```bash
!cd /kaggle/working/-DeepTrace && for cfg in xception xception_fad; do for s in 0 1 2 3 4; do python -m src.predict --run ffpp_c40_vid_${cfg}_seed${s} --results-root <C40VID_RESULTS> --dataset-name ffpp_c40_vid --seed $s --image-size 128 --out-dir /kaggle/working/preds; done; done
```

Check the line says **30 video groups**, not 300. If it says 300 the manifest is
crop-randomised and every number downstream is invalid.

## Cell 3 — V7, the resolution curve (~10 min)

Video unit first: it is the axis the figure is about and it has 150 units, so the
power-law fit has a real range to sit on.

```bash
!cd /kaggle/working/-DeepTrace && python -m src.resolution_curve --a $PREDS/ffpp_c40_vid_xception_seed0_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed0_test.csv --unit video --out-dir results/analysis/resolution
```

Then the component unit (29 units — expect the small sizes to be flagged
degenerate and dropped from the fit; that is the guard working, not a failure):

```bash
!cd /kaggle/working/-DeepTrace && python -m src.resolution_curve --a $PREDS/ffpp_c40_vid_xception_seed0_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed0_test.csv --unit component --out-dir results/analysis/resolution
```

**What to read.** `slope` near -0.5 means added videos buy what independent
sampling would give; shallower means they buy less. `n_for_halfwidth_0.014` is
the headline number — how many test videos an FF++ ablation needs before it can
resolve the effect F3-Net reports. If `extrapolated_0.014` is `True`, it is a
projection beyond our data and must be written up as one.

## Cell 4 — V6, threshold sensitivity (~5 min)

```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --margins 0.005 0.010 0.014 0.020 --out-dir results/analysis/thresholds; done
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.cluster_boot --a $PREDS/ffpp_c23_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c23_vid_f3net_seed${s}_test.csv --margins 0.005 0.010 0.014 0.020 --out-dir results/analysis/thresholds; done
```

Columns are `equivalent_<m>` and `excludes_<m>`. They are different claims —
report both, and never write `excludes` up as "no effect".

## Cell 5 — V4, the aggregation effect (~5 min)

Same predictions, same videos, two aggregations. This regenerates the c40 seed-0
video-level file that an earlier `--frame-level` run overwrote (the filename now
encodes the mode, so it cannot recur).

```bash
!cd /kaggle/working/-DeepTrace && for s in 0 1 2 3 4; do python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --margins 0.014 --out-dir results/analysis/aggregation; python -m src.cluster_boot --a $PREDS/ffpp_c40_vid_xception_seed${s}_test.csv --b $PREDS/ffpp_c40_vid_f3net_seed${s}_test.csv --frame-level --margins 0.014 --out-dir results/analysis/aggregation; done
```

Pair `boot_frame_*` against `boot_video_*` per seed. Seed 0 already showed the
estimate itself moving (-0.0196 frame-pooled vs -0.0303 video-level), not just
the interval — this establishes whether that holds across seeds.

## Cell 6 — collect

```bash
!cd /kaggle/working/-DeepTrace && python - <<'PY'
import csv, glob, pathlib
for name in ("resolution", "thresholds", "aggregation"):
    files = sorted(glob.glob(f"results/analysis/{name}/*.csv"))
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            rows.append({"source_file": pathlib.Path(f).name, **r})
    if not rows:
        print(f"{name}: NO ROWS"); continue
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "source_file", k))
    with open(f"results/analysis/{name}/ALL.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows from {len(files)} files")
PY
!cd /kaggle/working && tar czf tier0-results.tar.gz -C ./-DeepTrace results/analysis && ls -lh tier0-results.tar.gz
```

Download `tier0-results.tar.gz` and unpack into the repo.

---

## Sanity checks before trusting any of it

- Cell 2b (if run) prints **30 video groups**, never 300.
- V7 at the component unit drops small sizes as degenerate. If it drops
  *everything*, `--n-boot` is too low or the split is broken — do not fit.
- V7's `slope` should be negative. A positive slope means the curve is noise.
- V6 `equivalent_*` should never be True where `excludes_*` is False.
- The bootstrap uses a fixed seed, so re-running a cell must reproduce the CSV
  byte for byte. If it does not, something is reading the RNG out of order.
