# Thesis and contribution list — restructured 2026-09-18

Supersedes the ordering in `docs/CONTRIBUTION-MAP.md`. Adopted after the
split-policy audit established that **no audited paper crop-randomises**, which
changes what the protocol-gap experiment can carry.

## Thesis

> Deepfake-detector comparisons depend not only on the models being compared, but
> also on how predictions are aggregated, which observations are treated as
> independent, and which sources of variation are propagated. Using F3-Net's FAD
> component as a controlled case study, we show that these choices can change
> reported quantities and the conclusions supported by them.

Every clause is demonstrated by our own experiments, and the two central problems —
estimand ambiguity and incomplete uncertainty propagation — are **visibly present
in the literature we audited**.

## Contributions

1. **Incompletely specified evaluation choices materially change reported
   quantities.** Frame- versus video-level aggregation, and the score space in
   which video scores are formed.
2. **The resampling unit changes uncertainty and can reverse a statistical verdict
   without changing the point estimate.**
3. **Architecture-level conclusions require variation over both test content and
   training runs.** Shown by the crossed analysis of FAD.

**Not a contribution — motivation and scale-setting:** crop-randomised splitting,
a controlled stress test motivated by our own original pipeline, worth
approximately 18 AUC points of inflation at c40.

### Why the crop experiment is demoted

It documents a real failure in our original pipeline and quantifies the
consequence, so it belongs in the paper. But **no audited paper does it**, so it
cannot carry a novelty claim about current practice. Its role is to explain how the
project began and to set the scale against which small effects are judged.

> *"It explains how the project began; the estimand and uncertainty results explain
> why the finished paper matters."*

For a 4–5 page paper, place it early as motivation, then give the inferential
weight to estimand, clustering and the crossed bootstrap.

## ⚠️ The 13× comparison — wording is fixed

✅ **"The protocol gap was numerically about thirteen times the published FAD
effect."**

❌ **"Protocol uncertainty was thirteen times the FAD effect."**

**The 18 points is not an uncertainty interval.** It is a change in absolute
performance caused by changing the evaluation task; the +0.014 is a paired
architectural difference. They are different statistical objects. The comparison is
rhetorically useful and numerically true — the word *uncertainty* must not attach
to it.

## Mapping from the previous structure

| old | new |
|---|---|
| C1 protocol **and** estimand | splitting → **motivation**; estimand → **contribution 1** |
| C2 dependence-aware inference | split into **contribution 2** (resampling unit) and **contribution 3** (crossed / training-run) |
| C3 FAD case study | the **vehicle throughout**, not a separate contribution |

No experiment changes. No number changes. Only what each is asked to support.
