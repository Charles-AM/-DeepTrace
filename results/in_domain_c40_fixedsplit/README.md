# V8 — fixed split, varying training seed (2026-09-06)

10 runs: c40, `--split-seed 0`, training seeds 0–4, `xception` and `xception_fad`,
15 epochs, fp32. 6 h, 10/10 trained, 0 failed. Split verified against
`results/reference/ffpp_c40_vid_seed0_test_split.json` before training began.
Training command built from `results/reference/train_args_c40_vid.json`, so every
hyperparameter matches the varying-split runs this is compared against.

Supersedes the AMP attempt described in the git history (commit `c131486`), which
confounded precision mode with split policy.

---

## 1. The decomposition — point estimate

FAD − Xception, frame-pooled test AUC:

| seed | varying split | **fixed split** |
|---|---|---|
| 0 | −0.0196 | +0.0140 |
| 1 | −0.0101 | −0.0127 |
| 2 | +0.0142 | −0.0201 |
| 3 | +0.0253 | +0.0044 |
| 4 | +0.0145 | −0.0152 |
| **mean** | +0.0049 | −0.0059 |
| **sd** | **0.0188** | **0.0145** |

| quantity | value |
|---|---|
| complete-pipeline sd (varying split) | 0.0188 |
| **fixed-split training-run sd** | **0.0145** |
| implied additional split-related sd | 0.0121 |
| illustrative allocation | 59% / 41% |

**Naming.** 0.0145 is *fixed-split training-run variability*, not "optimisation
variability". It contains initialisation, data order, augmentation and
nondeterministic runtime operations, and possibly session-level effects. It is a
point estimate, not a bound.

⚠️ **Not an established variance partition.** Under an additive
independent-variance model the observed sds imply an illustrative 59/41
allocation, but the two variances are not statistically distinguishable at five
runs per condition — F = 1.70 on (4, 4) df against a two-sided 5% critical value
of 9.60.

⚠️ **The component changes sign between attempts.** Computed from the AMP
attempt (`c131486`) the same subtraction gives a *negative* variance component
(−0.000098); from this fp32 run it gives +0.000146. A quantity whose sign depends
on the precision mode of the run is not a measured partition. This is the
strongest single reason to keep the 59/41 figure clearly secondary.

Both components are **larger than the +0.014 effect being adjudicated**, which is
the substantive point and does not depend on separating them.

## 2. The reproducibility control failed — and that is a finding

Seed 0 is nominally the same configuration in both runs: same verified split, same
seed, same code, same hyperparameters, both fp32.

| | Xception | + FAD | difference |
|---|---|---|---|
| varying-split run | 0.81347 | 0.79391 | −0.0196 |
| V8 run | 0.79705 | 0.81104 | **+0.0140** |
| movement | −0.0164 | **+0.0171** | **+0.0336** |

The paired difference moves by 0.034 and flips sign.

**The two configurations moved in opposite directions**, so this is not a constant
session offset.

⚠️ **No causal attribution.** We have not isolated a cause — cuDNN kernel
selection, hardware, and library versions are all candidates and none was tested.
The empirical finding does not require an explanation, and offering an untested
one would be a claim we cannot support.

⚠️ **n = 1.** Only seed 0 is a repeated configuration: under the varying-split
protocol seed *s* draws split *s*, so seeds 1–4 are not nominally identical across
the two arms. This is a **reproducibility audit**, not an estimate of
session-level variance. A second independent repeat (~36 min GPU) would show
whether the movement is typical.

Supporting texture, not a measurement — seed-0 Xception across three near-identical
executions: 0.81347 (session A, fp32), 0.78158 (session B, AMP), 0.79705
(session C, fp32); spread **0.0319**.

**Manuscript wording.** *Repeating a nominally identical seed-0 configuration
across execution sessions changed the FAD−Xception difference from −0.0196 to
+0.0140, a movement of 0.0336 — 2.4× the +0.014 reference effect. The recorded
seed, split, code and hyperparameters did not guarantee exact repeatability in our
execution environment. Because this comparison comprises one repeated
configuration, we treat it as a reproducibility audit rather than an estimate of
session-level variance.*

## 3. Does the control failure invalidate the decomposition?

No, under one stated assumption. Environment is **constant within each arm** — all
five varying-split runs came from one session, all five fixed-split runs from
another — so it contributes a location shift, not within-arm variance. The
subtraction remains valid provided environmental noise has similar magnitude in
both sessions. That is an assumption we cannot check, and it should be stated.

What the control does rule out is any claim that the two arms are exactly
comparable run-for-run. They are comparable in distribution, not in outcome.

## 4. Scope

Frame-pooled AUC from `summary.csv`, matching the metric the 0.0188 figure was
computed on. The cluster-aware, video-level version needs the prediction dumps
(`preds/` in the notebook output) and is not yet computed.

## Files

`summary.csv` (10 runs). Checkpoints and prediction dumps in the Kaggle version
output; the dumps should be committed here as was done for V1.
