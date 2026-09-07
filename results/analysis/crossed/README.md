# The crossed interval — the paper's primary FAD result (2026-09-06)

Test content and training run resampled **together**. Every other interval in this
project is conditional on something: the component bootstrap holds the trained
weights fixed, the seed-level sd ignores test-content sampling. This is the one
that answers the question the FAD claim actually poses.

Produced by `src/crossed_boot.py` from the V8 fixed-split prediction dumps
(`results/predictions_v8/`, 10 runs). CPU, seconds.

---

## 1. Result

**Xception + FAD − Xception, c40, video-level aggregation, mean over 5 training runs:**

| | 95% CI | half-width | SE |
|---|---|---|---|
| **crossed** (components + seeds) | **[−0.0351, +0.0183]** | **0.0267** | 0.0130 |
| component only (seeds fixed) | [−0.0224, +0.0042] | 0.0133 | 0.0068 |
| seed only (components fixed) | [−0.0257, +0.0041] | 0.0149 | 0.0078 |

Point estimate **−0.0092**. 29 components, 150 videos, seeds 0–4, one verified
split shared by all ten runs.

**The crossed interval contains both zero and +0.014.** So after accounting for
test-content sampling and training-run variation together, this evaluation can
neither detect FAD's published gain nor rule it out.

`excludes_0.014 = False`, `equivalent_0.014 = False`.

## 2. Why the crossed interval is the right one

It is **1.8× wider** than either conditional interval, and either conditional
alone would have supported a stronger claim than the data permit: the
component-only interval excludes +0.014, the seed-only interval excludes +0.014,
and the crossed interval — the honest one — does not.

That is the whole argument of the paper appearing in a single table. Two
defensible-looking analyses agree on a conclusion that vanishes once both sources
of variation are admitted at once.

## 3. The two sources are not independent

| | variance |
|---|---|
| component-only + seed-only | 0.000107 |
| crossed | 0.000169 |
| **ratio** | **1.58** |

Under independence the crossed variance would equal the sum. It is 58% larger, so
the additive independent-variance model used for the V8 decomposition
(`../../in_domain_c40_fixedsplit/` §1) **understates** the combined uncertainty.
This is direct evidence for that section's caveat, and another reason its 59/41
allocation should not be read as a variance partition.

## 4. Note on width

The crossed half-width (0.0267) is **narrower** than a single run's component
interval (0.037–0.062, `../resolution/`). That is correct, not contradictory: this
estimates the *expected* difference over training runs, and averaging five runs
reduces that component. A single-run interval answers a different question — what
this model achieves — and is wider because it carries one run's noise in full.

## 5. What this licenses

> Resampling source-target components and training runs together, the FAD −
> Xception difference at c40 was −0.0092 AUC (95% CI [−0.0351, +0.0183]) over five
> runs on one fixed video-disjoint split. The interval contains both zero and the
> +0.014 reference effect, so this evaluation neither detects nor excludes a gain
> of the published magnitude.

❌ Not "FAD does not work."
❌ Not "FAD is equivalent to Xception" — equivalence fails too.

## Files

`crossed_video.csv`. Inputs: `results/predictions_v8/*.csv` (10 dumps).

```
python -m src.crossed_boot \
  --a-glob 'results/predictions_v8/ffpp_c40_vid_xception_seed*_test.csv' \
  --b-glob 'results/predictions_v8/ffpp_c40_vid_xception_fad_seed*_test.csv' \
  --margins 0.014 --out-dir results/analysis/crossed
```

Refuses to run unless every run was scored on identical test items — verified
here, and the reason V8's fixed split was necessary.
