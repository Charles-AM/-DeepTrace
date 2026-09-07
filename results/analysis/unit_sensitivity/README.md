# Target-group vs component resampling (2026-09-06)

Run against `docs/target-group-prespecification.md`, committed at `0639ba0`
**before this code existed** — `cluster_boot` and `crossed_boot` offered only
frame / video / component at that commit.

## Why

`../resolution/README.md` notes that component construction merged only one
additional pair of target groups (29 vs 30). An earlier draft inferred from that
count that components "added almost nothing". **Similar unit counts do not
guarantee similar uncertainty**, so the inference was unsupported. This measures
it.

## A. The crossed primary result

V8 fixed split, 5 training runs, video-level, 50,000 replicates, RNG seed 0:

| content unit | n | 95% CI | half-width | exceedance > +0.014 | verdict |
|---|---|---|---|---|---|
| **component** | 29 | [−0.0358, +0.0180] | 0.0269 | 0.0407 ± 0.0018 | included |
| target group | 30 | [−0.0361, +0.0182] | 0.0271 | 0.0442 ± 0.0018 | included |
| **difference** | | | **+0.0002** (ratio 1.007) | +0.0035 | **unchanged** |

The half-width difference is smaller than the endpoint's own Monte Carlo band
(±0.0004). **The paper's primary result does not depend on this choice.**

## B. The eight conditional per-seed comparisons

Video-level, 2,000 replicates, the same comparisons reported in `../thresholds/`:

| run | n target | n comp | hw target | hw comp | diff | ratio | verdict |
|---|---|---|---|---|---|---|---|
| c23 s0 | 30 | 29 | 0.0154 | 0.0152 | +0.0002 | 1.010 | same |
| c23 s1 | 30 | 28 | 0.0301 | 0.0292 | +0.0010 | 1.033 | same |
| c23 s2 | 30 | 29 | 0.0261 | 0.0255 | +0.0007 | 1.026 | same |
| c40 s0 | 30 | 29 | 0.0370 | 0.0370 | −0.0001 | 0.997 | same |
| c40 s1 | 30 | 28 | 0.0411 | 0.0442 | −0.0032 | 0.929 | same |
| c40 s2 | 30 | 29 | 0.0605 | 0.0624 | −0.0020 | 0.968 | same |
| c40 s3 | 30 | 28 | 0.0490 | 0.0489 | +0.0001 | 1.002 | same |
| c40 s4 | 30 | 28 | 0.0445 | 0.0452 | −0.0007 | 0.983 | same |

Half-width difference −0.0032 to +0.0010 (median +0.0000); ratio 0.929–1.033.
**No exclusion verdict at +0.014 changed, in any of the eight.**

Note the difference is not signed consistently: target grouping gives a slightly
wider interval in four comparisons and a slightly narrower one in four, despite
always having one or two more units. That is what a difference within noise looks
like, and it is worth more than the median being zero.

## Conclusion — as prespecified

> **Target-group resampling approximated component-level uncertainty in this
> split.**

⚠️ It does **not** establish that target grouping is generally sufficient for
FF++. One fixed split, one subset, one contrast.

⚠️ **Component resampling remains the preferred inferential unit**, as fixed in
advance and independent of this outcome: it captures dependence through the
source-target graph that target grouping cannot see. In a split where more
partners co-occurred, the two would diverge. This measures what that choice cost
here — it was not a search for a cheaper unit.

The recommendation stands unchanged: *identify the relevant dependence structure
empirically rather than assuming either crops or the strictest available grouping
is automatically appropriate.*

## Files

`boot_video_*.csv` — eight comparisons × four units (frame, video, target,
component). The crossed rows are in `../crossed/crossed_video_target_b50000_s0.csv`
and `../crossed/crossed_video_b50000_s0.csv`.

```
python -m src.cluster_boot --a <preds_a> --b <preds_b> --margins 0.014 --out-dir results/analysis/unit_sensitivity
python -m src.crossed_boot --a-glob '...' --b-glob '...' --unit target --n-boot 50000 --out-dir results/analysis/crossed
```
