# V1 — cluster-aware paired inference (2026-09-06)

The project's **first valid inferential intervals**. Everything reported before
this was seed-level: those measure optimisation variability confounded with split
composition, and say nothing about uncertainty over test content.

Produced by `src/predict.py` → `src/cluster_boot.py` on video-disjoint
checkpoints. `ALL.csv` concatenates every row; per-comparison files are alongside.

---

## 1. The headline: one estimate, three uncertainty models, opposite
naive-versus-clustered conclusions

FAD − Xception, c40 seed 0, **no aggregation** (individual crops scored):

| resampling unit | n units | 95% CI | SE | vs frame |
|---|---|---|---|---|
| **frame** (naive) | 3000 | **[−0.0325, −0.0079]** | 0.0063 | 1.00× |
| **video** | 150 | [−0.0606, +0.0167] | 0.0199 | **3.16×** |
| **component** | 29 | [−0.0534, +0.0096] | 0.0158 | **2.51×** |

Point estimate is **−0.0196 in all three**. Only the uncertainty model differs.

Three intervals, but **two** inferential outcomes: the frame bootstrap excludes
zero, and both clustered bootstraps include it. Do not describe this as "three
different conclusions" — the video and component analyses agree.

**The naive frame bootstrap excludes zero.** It would be reported as *"Xception +
FAD performs significantly worse than Xception (p<0.05)"*. Resampling videos or
source-target components instead — the units that are actually independent — the
interval spans zero and no such claim survives.

A researcher using the standard frame-pooled bootstrap on this exact data would
have published a significant architectural finding the evidence does not support.
That is contribution 2, measured rather than argued.

Note the component interval is *narrower* than the video interval despite having
fewer units (29 vs 150). Bootstrap width depends on within-cluster homogeneity as
well as cluster count, so "fewer clusters ⇒ wider interval" does not hold
mechanically. Report both; do not assume monotonicity.

## 2. Cluster-aware per-seed results (video-level aggregation)

Frames averaged to one score per video, then bootstrapped over clusters:

| | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **c40** diff | −0.0303 | −0.0208 | +0.0203 | +0.0306 | +0.0164 |
| **c23** diff | −0.0061 | −0.0044 | +0.0111 | — | — |

**No cluster-aware interval, at any seed or either compression, excludes zero.**
Two of eight (c40 seed 0, c23 seed 0) exclude the verified +0.014 FAD gain; the
other six do not.

⚠️ These are per-seed intervals **conditional on a trained model**. Combining them
properly across seeds needs a hierarchical treatment (test-content and
optimisation are crossed, not nested). Do not average the bounds.

## 3. Aggregation changes the estimate, not just the interval

c40 seed 0: frame-pooled AUC difference is **−0.0196**; video-level aggregated is
**−0.0303**. Same model, same test set. This is C0b appearing in the data — our
frame-pooled metric is not the video-level metric published FF++ work reports, and
the two do not merely differ in precision.

## 4. Cluster counts

Each comparison: 3000 crops → **150 videos** (30 target groups × 5 videos: one
real plus four manipulations) → **28–29 identity components**.

⚠️ Correcting an earlier estimate: identity clustering does **not** halve the test
units. The 150-components figure applies to all 300 sequences; within a 30-target
test split most partners fall outside the split, so components stay near-singleton
and ~29 survive.

## Files

`ALL.csv` (17 rows, every comparison), `boot_*.csv` per comparison,
`../efficiency/params_flops_latency.csv` (regenerated — closes G1).

⚠️ **Known gap:** the frame-level run overwrote the c40 seed-0 *video-level* file
(the output name did not encode the aggregation mode; fixed in `cluster_boot.py`
this commit). The seed-0 video-level numbers above are from console output and
should be regenerated to file on the next run.
