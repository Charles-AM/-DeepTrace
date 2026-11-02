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

| copies of seed 0 | share of replicates |
|---|---|
| 0 | **0.3277** |
| 1 | 0.4096 |
| 2 | 0.2048 |
| 3+ | 0.0512 |

A third of replicates never contain the collapsed run. The distribution is therefore
a mixture, and a single percentile interval hides that its lower tail is one failed
optimisation resampled. Any report of the five-run result must carry the
stratification or a histogram alongside it.

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
escaped both times. The `alpha`/`gamma` diagnostic
(`docs/npr-loss-geometry-prespecification.md`) remains unrun.

## Outstanding

Resampling-unit comparison (frame/video/target/component), test-content-only interval
at other units, per-manipulation descriptives, and the stratified bootstrap histogram
— which needs a new reporting module, since `crossed_boot.py` is protected and its
`--out-dir` writes only a summary row.
