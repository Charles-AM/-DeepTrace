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
| varying-split sd (split composition + optimisation) | 0.0188 |
| fixed-split sd (optimisation alone) | 0.0145 |
| implied split-composition sd | **0.0121** |
| optimisation share of total variance | **59%** |

⚠️ **The two sds are not statistically distinguishable at this sample size.**
F = 1.70 on (4, 4) df against a two-sided 5% critical value of 9.60. The point
estimate says roughly 59% optimisation / 41% split composition; the data cannot
exclude 100% optimisation. Report the decomposition as an estimate with that
caveat, never as a demonstrated split.

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
session offset — it is consistent with independent redraws from the run-to-run
distribution. Setting a seed does not pin the outcome when cuDNN kernel selection
and hardware differ between sessions.

For the paper: *reporting a seed is not sufficient for reproducibility.* Our own
attempt to reproduce a run from its recorded seed, split and hyperparameters moved
the estimate by more than twice the effect under study.

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
