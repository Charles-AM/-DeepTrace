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

| variation propagated | 95% CI | contains +0.014? | half-width | SE |
|---|---|---|---|---|
| test components only; five runs fixed | [−0.0224, +0.0042] | **no** | 0.0133 | 0.006824 |
| training runs only; test content fixed | [−0.0257, +0.0041] | **no** | 0.0149 | 0.007781 |
| **training runs and test components** | **[−0.0351, +0.0183]** | **yes** | 0.0267 | 0.012997 |

Point estimate **−0.0092**. 29 components, 150 videos, seeds 0–4, one verified
split shared by all ten runs.

⚠️ **This is the fixed-split (V8) result.** The earlier c40 five-run mean of
**+0.0049** came from the varying-split pipeline and answers a different
question — there each seed drew its own split, so that mean confounds training
run with split composition. The two are not alternative estimates of the same
quantity.

**The crossed interval contains both zero and +0.014.** So after accounting for
test-content sampling and training-run variation together, this evaluation can
neither detect FAD's published gain nor rule it out.

`excludes_0.014 = False`, `equivalent_0.014 = False`.

## 2. Why the crossed interval is the right one

It is **1.8× wider** than either conditional interval, and either conditional
alone would have supported a stronger claim than the data permit: the
component-only interval excludes +0.014, the seed-only interval excludes +0.014,
and the crossed interval — the honest one — does not.

The conditional analyses are **not mathematically wrong** — they answer restricted
questions, and answer them correctly. Neither alone supports an unconditional
architectural conclusion. Once both observed sources of variation are propagated,
**the apparent exclusion no longer holds**: compatibility with +0.014 returns.

Note what does *not* change. The architectural estimate stays negative (−0.0092)
under every treatment. What changes is the evidential interpretation, not the
direction of the effect.

## 3. The two sources are not independent

$$\text{ratio} = \frac{\operatorname{var}(\text{crossed})}
{\operatorname{var}(\text{component only}) + \operatorname{var}(\text{seed only})}
= \frac{0.012997^2}{0.006824^2 + 0.007781^2} = \mathbf{1.577}$$

Computed from the **bootstrap variances** — the sd of the replicate statistics —
not from CI half-widths, and emitted by `crossed_boot.py` as
`nonadditivity_ratio` so it is not recomputed by hand.

⚠️ **Exploratory.** Under an additive independent-variance model this would be 1.
The excess is *consistent with* seed-by-content interaction or non-additivity, but
it is not proof that independence fails: AUC is nonlinear in the resampled data,
the seed sample is only five, and the bootstrap adds estimation noise of its own.

Read as a caution rather than a result, it is one more reason the 59/41 allocation
in `../../in_domain_c40_fixedsplit/` §1 should not be treated as a variance
partition.

## 4. Note on width

The crossed half-width (0.0267) is **narrower** than a single run's component
interval (0.037–0.062, `../resolution/`). That is correct, not contradictory: this
estimates the *expected* difference over training runs, and averaging five runs
reduces that component. A single-run interval answers a different question — what
this model achieves — and is wider because it carries one run's noise in full.

## 5. What this licenses

> On the fixed c40 split, the mean FAD−Xception difference across five training
> runs was **−0.0092 AUC**. A crossed bootstrap propagating both training-run and
> source-target-component variation produced a 95% interval of
> **[−0.0358, +0.0180]**. Its upper endpoint remained above the +0.014 reference
> effect after accounting for Monte Carlo uncertainty **[0.0176, 0.0184]**. The
> evaluation therefore neither demonstrated a FAD advantage nor excluded a gain of
> the published magnitude.

**FROZEN** — 50,000 replicates, prespecified at `86e6c98`, stability check in
`STABILITY.md`. Quote the interval as [−0.0358, +0.0180] (the 50k run), not the
earlier 4,000-replicate [−0.0351, +0.0183].

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
