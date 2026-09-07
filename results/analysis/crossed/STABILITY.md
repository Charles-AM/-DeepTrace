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

The upper endpoint is stably above the reference. This was not a foregone
conclusion — the pre-check margin was 0.0043, comparable to plausible Monte Carlo
error at 4,000 replicates.

## 2. Monte Carlo error — two methods, agreeing

| method | 2σ band at B=50,000 |
|---|---|
| batch spread (5 × 4,000, scaled by √(4000/50000)) | ±0.00033 |
| order statistic (rank ~ Binomial(B, 0.975)) | ±0.00040 |
| **difference** | **0.00007 — agree** |

Batch endpoints at B=4,000: 0.0183, 0.0172, 0.0171, 0.0173, 0.0182 (sd 0.00058).

## 3. The proportion is the better statistic

Reporting P(replicate > +0.014) instead of a binary endpoint test turns the
conditional-vs-crossed contrast from a verdict into a magnitude:

| variation propagated | P(replicate > +0.014) |
|---|---|
| test components only | **0.0004** |
| training runs only | **0.0000** |
| **both (crossed)** | **0.0407** |

The conditional analyses put essentially **no** bootstrap mass above the reference;
the crossed analysis puts 4%. That is a ~100× difference in compatibility, and it
states the paper's argument far more informatively than "excludes / does not
exclude".

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

**Every MC band spans +0.014**, and the proportion's 2σ interval spans the 0.025
threshold. By the prespecified rule this is **case 3: borderline /
bootstrap-sensitive**, and no binary verdict is forced.

This is a finding, not a nuisance. Contribution 2 argues that aggregation defines
the estimand; here the primary conclusion is comfortably included under
video-level aggregation and borderline under frame-pooling. The estimand choice
matters at exactly the margin under adjudication.

## 5. Seed-subset sensitivity

All ten 3-of-5 and all five 4-of-5 subsets, plus the full five
(`seed_subsets_video.csv`).

| | point estimate range | spread |
|---|---|---|
| 3-of-5 (n=10) | −0.0194 to +0.0015 | **0.0209** |
| 4-of-5 (n=5) | −0.0144 to −0.0017 | 0.0127 |
| full 5 | −0.0092 | — |

**Dropping two of five training runs moves the point estimate by up to 0.021 —
about 1.5× the effect being adjudicated.** Read positively, that is contribution 3
appearing inside the primary analysis rather than beside it.

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
