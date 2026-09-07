# Stability check on the crossed interval (2026-09-06)

Executed against `docs/crossed-prespecification.md`, committed at `86e6c98`
**before** these runs. The decision rule below was fixed in advance.

---

## 1. Verdict

**Case 1 — the reference remains included.**

50,000 replicates, video-level, bootstrap seed 0:

| | |
|---|---|
| crossed 95% CI | **[−0.0358, +0.0180]** |
| upper endpoint MC band | **[+0.0176, +0.0184]** |
| rule: U − MC > +0.014? | **0.0176 > 0.014 ✓**, margin 0.0036 |
| **P(replicate > +0.014)** | **0.0407 ± 0.0018** (2σ) |

The upper endpoint is above the reference, and that inclusion is **robust to
Monte Carlo error** — not "comfortable". The exceedance rate of 4.07% sits fairly
close to the 2.5% percentile boundary; what has been demonstrated is that
numerical bootstrap error cannot explain the inclusion, which was not a foregone
conclusion: the pre-check margin was 0.0043, comparable to plausible MC error at
4,000 replicates.

## 2. Monte Carlo error — two methods, agreeing

| method | 2σ band at B=50,000 |
|---|---|
| batch spread (5 × 4,000, scaled by √(4000/50000)) | ±0.00033 |
| order statistic (rank ~ Binomial(B, 0.975)) | ±0.00040 |
| **difference** | **0.00007 — agree** |

Batch endpoints at B=4,000: 0.0183, 0.0172, 0.0171, 0.0173, 0.0182 (sd 0.00058).

## 3. Bootstrap exceedance rate — the better statistic

Reporting the **bootstrap exceedance rate** above +0.014 instead of a binary
endpoint test turns the conditional-vs-crossed contrast from a verdict into a
magnitude. B = 50,000 in every row.

| variation propagated | replicates above +0.014 | exceedance rate |
|---|---|---|
| test components only | 20 of 50,000 | 0.0004 |
| training runs only | **0 of 50,000** | < 0.00002 |
| **both (crossed)** | **2,037 of 50,000** | **0.0407** |

> The crossed-bootstrap exceedance rate above +0.014 was **4.07%**, against
> **0.04%** when only test-component variation was propagated.

⚠️ **Naming.** This is an exceedance rate of the bootstrap distribution, not a
probability that FAD is compatible with the reference. The ~102× figure is a ratio
of exceedance rates and nothing more. For the training-run-only row, report "0 of
50,000 replicates" — the underlying probability is not exactly zero, it is merely
below what 50,000 draws can resolve.

## 4. Frame-pooled estimand — BORDERLINE

Same resampling units (components × training runs); AUC computed over crops
instead of video-averaged scores.

| bootstrap seed | 95% CI | upper MC band | P(>0.014) |
|---|---|---|---|
| 0 | [−0.0256, +0.0147] | [+0.0137, +0.0164] | 0.0290 ± 0.0053 |
| 1 | [−0.0259, +0.0148] | [+0.0136, +0.0162] | 0.0278 ± 0.0052 |
| 2 | [−0.0261, +0.0148] | [+0.0138, +0.0159] | 0.0288 ± 0.0053 |
| 3 | [−0.0262, +0.0145] | [+0.0136, +0.0158] | 0.0285 ± 0.0053 |
| 4 | [−0.0268, +0.0151] | [+0.0140, +0.0163] | 0.0293 ± 0.0053 |

**Every MC band spans +0.014**, and the exceedance-rate 2σ interval spans the
0.025 threshold. By the prespecified rule this is **case 3: borderline /
bootstrap-sensitive**, and no binary verdict is forced.

> Under frame-pooled AUC, both the endpoint uncertainty band and the
> exceedance-rate interval crossed their prespecified decision thresholds. The
> result was therefore classified as **borderline** rather than forced into an
> inclusion or exclusion verdict.

⚠️ **This is sensitivity to the estimand, not a return to pseudoreplication.**
Resampling remained component-aware throughout — components × training runs, as in
every other row. Only the AUC estimand changed, from video-averaged scores to
frame-pooled. That is what makes it evidence for contribution 2 rather than a
methodological regression.

## 5. Seed-subset sensitivity

All ten 3-of-5 and all five 4-of-5 subsets, plus the full five
(`seed_subsets_video.csv`).

| | point estimate range | spread |
|---|---|---|
| 3-of-5 (n=10) | −0.0194 to +0.0015 | **0.0209** |
| 4-of-5 (n=5) | −0.0144 to −0.0017 | 0.0127 |
| full 5 | −0.0092 | — |

> Across all prespecified three- and four-run subsets, omitting two runs shifted
> the estimated effect by as much as **0.021 AUC — approximately 1.5× the +0.014
> reference magnitude**.

Read positively, that is contribution 3 appearing inside the primary analysis
rather than beside it. Used to demonstrate **sensitivity to run selection**, never
to estimate an exclusion probability.

Two subsets do not contain the reference: the 4-run {1,2,3,4} and the 3-run
{1,2,4}. Both omit seed 0, the run with the most positive difference.

⚠️ Per the prespecification, **no exclusion frequency is computed from these**.
The subsets overlap heavily — each 3-of-5 shares two runs with most others — so
they are sensitivity, not replication, and a "k of 16" tally would be
meaningless.

## 6. What this check does not establish

More replicates reduce **numerical** error only. They do not enlarge five training
runs or 28–29 components, and they say nothing about whether the percentile
bootstrap has correct coverage with clusters this few. Those limitations stand
whatever the verdict.

## Files

`crossed_video_b50000_s0.csv` (primary), `crossed_video_b4000_s{0..4}.csv`,
`crossed_frame_b4000_s{0..4}.csv`, `seed_subsets_video.csv`.

```
python -m src.crossed_boot --a-glob 'results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv' \
  --b-glob 'results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv' \
  --margins 0.014 --n-boot 50000 --boot-seed 0 --out-dir results/analysis/crossed
python -m src.crossed_subsets --a-glob '<same>' --b-glob '<same>' --out-dir results/analysis/crossed
```
