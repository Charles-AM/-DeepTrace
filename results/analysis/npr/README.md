# NPR supplementary — analysis of two campaigns

Prespecification: `docs/npr-supplementary-prespecification.md` (campaign-2 rules in
its addendum). Observations: `docs/npr-two-campaign-observations.md`.

Both campaigns passed `src.campaign_gate` locally. Every one of the twenty dumps
reproduces its training-loop frame-pooled AUC exactly. Campaign 1 is session
`S-29005a04`, campaign 2 is `S-fc95918c`.

🚨 **The campaigns are analysed separately and never pooled.** The crossed bootstrap
resamples training runs, so pooling would charge execution-level variance to the run
level.

Margin fixed at **0**. The 0.014 reference effect is FAD-specific and is not used
here as an NPR benchmark.

## Primary — all five runs, video-aggregated

The prespecified analysis. Includes the seed-0 constant-prediction run, which is the
complete outcome of running the frozen recipe five times.

| variation propagated | campaign 1 | campaign 2 |
|---|---|---|
| test components only | **[−0.1028, −0.0278]** | **[−0.1127, −0.0221]** |
| training runs only | [−0.2099, +0.0088] | [−0.1998, +0.0198] |
| both (crossed) | [−0.2203, +0.0379] | [−0.2094, +0.0380] |
| point (mean-logit) | −0.0667 | −0.0676 |
| non-additivity ratio | 1.045 | 1.048 |

`crossed_boot.py` reads **only** `logit_margin` (lines 90 and 98), so its point
estimate and all three intervals are mean-logit throughout. The mean-probability
figure (−0.0675 for campaign 1) belongs to the `agg_space` sensitivity analysis
alone and must not be placed beside these intervals. Their four-decimal agreement
is numerical coincidence between two estimands, not interchangeability.

✅ **The conditioning claim replicates across two independent executions.** Holding
the five trained model pairs fixed and resampling only test content supports a
negative conclusion — the interval excludes zero in *both* campaigns. Propagating
training-run variation removes that conclusion in *both* campaigns.

❌ The negative conditional interval is **not** an NPR architecture effect. It is a
statement about these five trained pairs, and it is driven by one run that predicts a
constant.

## Sensitivity — seeds 1–4, video-aggregated

**Post hoc.** Not prespecified. It answers a different question: the NPR effect among
runs that produced discriminative models.

| variation propagated | campaign 1 | campaign 2 |
|---|---|---|
| test components only | [−0.0333, +0.0479] | [−0.0531, +0.0488] |
| training runs only | [−0.0006, +0.0110] | [−0.0293, +0.0333] |
| both (crossed) | [−0.0396, +0.0515] | [−0.0679, +0.0593] |
| point (mean-logit) | **+0.0047** | **−0.0037** |
| P(>0), runs only | 0.94 | 0.39 |

🚨 **The point estimate flips sign between executions** (+0.0047 → −0.0037 on logit;
+0.0042 → −0.0021 on probability). Every interval contains zero in both campaigns.

Campaign 1's `P(>0) = 0.94` **did not replicate** (0.39 in campaign 2). It was never
a significance statement: run-level resampling at n=4 has only
**35 distinct resample compositions** (`C(2n−1, n−1)`), so its tail mass cannot be
read at two decimal places. Campaign 2 demonstrates that directly.

## Estimand effects

| | campaign 1 | campaign 2 |
|---|---|---|
| seeds 1–4, frame-pooled | −0.00085 | −0.01685 |
| seeds 1–4, video-aggregated | +0.0042 | −0.0021 |

Aggregation **changed the sign** in campaign 1 and changed the magnitude roughly
eightfold in campaign 2. Neither estimate establishes a directional effect.

Mean-logit versus mean-probability moves the point estimate by under 0.001 in all
four analyses, and even *which* of the two is more negative is not stable across
campaigns (c1: logit −0.0667, prob −0.0675; c2: logit −0.0676, prob −0.0666). Treat
that ordering as noise, not a finding. **Not every evaluation choice matters
equally** — aggregation unit does, aggregation space does not.

## 🚨 The five-run bootstrap distribution is bimodal — do not report the interval alone

Run-level resampling draws 5 runs from 5 with replacement:

| copies of seed 0 | exact weight |
|---|---|
| 0 | **0.32768** |
| 1 | 0.40960 |
| 2 | 0.20480 |
| 3 | 0.05120 |
| 4 | 0.00640 |
| 5 | 0.00032 |
| 3 or more | 0.05792 |

`src.npr_mixture` enumerates this exactly — `5**5 = 3125` ordered resamples, 126
distinct compositions — and the result is not a tail but a **four-spike ladder**,
each stratum narrow and separated by roughly one fifth of the collapsed run's
deficit:

| copies | exact seed-only mean | 95% range | campaign 2 mean |
|---|---|---|---|
| 0 | **+0.0047** | [−0.0001, +0.0102] | **−0.0037** |
| 1 | −0.0667 | [−0.0709, −0.0616] | −0.0676 |
| 2 | −0.1381 | [−0.1416, −0.1342] | −0.1314 |
| 3+ | −0.2181 | [−0.2823, −0.2057] | −0.2031 |

Two things this makes visible that the pooled interval cannot:

- **The 0-copies stratum *is* the seeds 1–4 sensitivity analysis.** Its exact range
  [−0.0001, +0.0102] reproduces the seeds 1–4 `seed_only` interval
  [−0.0006, +0.0110] to within Monte Carlo error. The "post hoc sensitivity" is
  simply the 33% of the prespecified distribution that omits the collapsed run.
- **The 1-copy stratum carries the modal weight (0.41)** and its crossed range
  [−0.1080, −0.0273] closely tracks the five-run `component_only` interval
  [−0.1028, −0.0278]. The "significant negative" conditional result is largely a
  statement about resamples containing exactly one copy of a constant predictor.

❌ Reporting the five-run interval alone is therefore not conservative — it hides
that the estimate is a weighted average over how many times one failed
optimisation happened to be drawn.

⚠️ **Monte Carlo error is not the limitation.** The tail mass is exactly
computable. The limitation is inferential: the whole distribution rests on five
observed run values, so `P(>0)` from it is descriptive of those five runs and is
not evidence about the population of possible training executions. Run-level
bootstrap probabilities should be reported descriptively, and the 35 (n=4) or 126
(n=5) distinct compositions stated alongside them.

Artifacts: `mixture_c1/` and `mixture_c2/` — `crossed_replicates.csv` (each
replicate with its collapsed-run copy count), `seed_compositions.csv` (all 126
compositions with exact weights), `mixture.png`.

## Constant-prediction audit

| | collapsed seed-0 NPR | healthy runs |
|---|---|---|
| `sd(prob_fake)` | **0.00000** (both campaigns) | 0.206 – 0.290 |
| `mean(prob_fake)` | 0.5001 (c1), 0.5010 (c2) | 0.617 – 0.864 |

The collapsed mean matches the **0.500** constant optimum that the loss-weighting
arithmetic predicts (`src/losses.FocalLoss`, α=0.2 exactly balancing the 1:4 ratio at
3840 vs 3840). This confirms the *location* of the solution empirically.

❌ It does **not** establish why this configuration entered that solution, nor that
NPR caused it — `xception_seed3` entered the same region in both campaigns and
escaped both times.

### The loss-geometry diagnostic ran, and refuted its own prediction

Three runs at seed 0 on the NPR arm (`S-827a87f7`, 2026-11-03; full record in
`docs/npr-loss-geometry-prespecification.md`):

| run | α | γ | p\* | test AUC | sd(prob_fake) | verdict |
|---|---|---|---|---|---|---|
| `ctrl` | 0.2 (derived) | 2.0 | 0.500 | 0.5000 | 0.00000 | collapsed |
| `alpha` | 0.5 | 2.0 | 0.627 | 0.7832 | 0.17086 | escaped |
| `gamma` | 0.2 (derived) | 0.0 | 0.500 | 0.7948 | 0.30005 | escaped |

🚨 **`gamma` was predicted to collapse and did not.** It leaves the best constant
prediction at p=0.500, exactly on the decision threshold, and still trained to
`test_auc=0.7948`. So the coincidence of the optimum with the threshold is **not
sufficient** for collapse. The arithmetic describes the collapsed *state* — `ctrl`
held its computed floor of 0.0555 for all 15 epochs — but does not explain entry
into it. **Outcome 4** of the prespecified table: neither factor isolated.

❌ **No mechanism is identified**, and nothing in the diagnostic bears on NPR:
initialisation and input representation were held constant across all three runs.
The three AUCs use different loss configurations and may never enter an arm
comparison. n=1 per condition.

## FAD and NPR reach fragile inference through different uncertainty structures

Both cases support Contribution 3; neither is a replication of the other. FAD
figures are from `results/analysis/crossed/` (canonical: `results/canonical.json`).

| | FAD | NPR (c1 / c2) |
|---|---|---|
| representation | explicit DCT frequency decomposition | pixel-domain neighbouring-pixel operator |
| reference point | published **+0.014** | zero |
| component only | [−0.0226, +0.0040] — excludes +0.014, contains 0 | [−0.1028, −0.0278] — **excludes 0** |
| runs only | [−0.0257, +0.0041] — **also excludes +0.014** | [−0.2099, +0.0088] — contains 0 |
| crossed | [−0.0358, +0.0180] — excludes neither | [−0.2203, +0.0379] — contains 0 |
| variance non-additivity | **1.58 – 1.74** | **1.045 / 1.048** |

Three structural differences, all of which should be reported rather than smoothed
over:

1. **In FAD both conditionals excluded the reference effect; in NPR only one did.**
   FAD's run-only interval also excludes +0.014, so propagating training-run
   variation *alone* would not have corrected it — only crossing the two sources
   widened the interval enough. In NPR the run-only interval already contains zero,
   because the pathology lives in the run dimension. **The general recommendation is
   therefore to cross both sources, not to add the run dimension.**
2. **The over-generalisation points in opposite directions.** FAD's conditionals
   supported "the published gain is not here"; NPR's supported "this is worse than
   the baseline". Narrow conditioning produced confidence on either side of the null,
   so this is not a directional bias.
3. **Variance non-additivity does not replicate.** FAD's combined variance is
   substantially greater than the sum of the two conditional variances; NPR's is
   approximately additive, twice.

✅ **Each conditional analysis is valid for its own estimand.** The error is
substituting a conditional result for an architecture-level conclusion, which
requires both sources. See `results/analysis/crossed/README.md` §2.

⚠️ **The non-additivity ratio is exploratory** — `src/crossed_boot.py` prints it with
that label. It is `var(crossed) / (var(component_only) + var(seed_only))`, a
description of how the bootstrap variance behaved in these five-run studies. It is
**not** evidence of a causal interaction in FAD or of independence in NPR, and AUC is
nonlinear. Report the observed **range**, not a point value: Monte Carlo variation
alone moves FAD's ratio across 1.5771 – 1.7439 (video, b=4000, bootstrap seeds 0–4),
with 1.6973 at b=50,000 and 1.7311 at the target unit. The FAD–NPR *contrast* is
robust because ≈1.6–1.74 and ≈1.05 are far outside that band; the third digit of
either value is not.

## Outstanding

Resampling-unit comparison (frame/video/target/component), test-content-only interval
at other units, per-manipulation descriptives, and the stratified bootstrap histogram
— which needs a new reporting module, since `crossed_boot.py` is protected and its
`--out-dir` writes only a summary row.
