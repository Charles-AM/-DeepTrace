# Prespecification — crossed-bootstrap stability check

**Written and committed before the confirmatory runs.** Git history is the
evidence. In a paper about evidential warrant, fixing the decision rule after
seeing the numbers it governs would be the exact failure being criticised.

⚠️ **Timing, stated precisely** (amendment, 2026-09-06). This is *not*
preregistration before any result. A preliminary 4,000-replicate crossed interval
had already been inspected — that is why this check was commissioned, and §2 below
quotes its endpoint as the motivation. The accurate description is:

> After observing that the preliminary endpoint lay near the reference effect, we
> prospectively specified a three-case Monte Carlo stability rule before
> conducting the confirmatory high-replicate analysis.

One arm is stronger than that: **no frame-pooled crossed result existed when this
was written**, and the frame-pooled analysis is where the rule actually bound,
returning the borderline verdict rather than a forced one.

Only this timing description was amended. The runs, quantities and interpretation
rule below are unchanged from the original commit; see `git log -p` on this file.

## Why this check exists

The crossed interval's upper endpoint is **+0.0183**; the reference effect is
**+0.014**. The margin is **0.0043**. If Monte Carlo error moves that endpoint
below +0.014, the paper's conclusion changes from *"neither demonstrates nor
excludes"* to *"excludes the published effect"*. One endpoint from one
4,000-replicate run is carrying contribution 3.

## What will be run

| run | replicates | bootstrap seeds |
|---|---|---|
| batch | 4,000 | 0, 1, 2, 3, 4 |
| high-replicate | 50,000 | 0 |
| frame-pooled estimand | 4,000 | 0, 1, 2, 3, 4 |
| seed-subset sensitivity | 4,000 | 0 |

All on `results/predictions_v8/`, c40, video-level unless stated. **Resampling
units are always source-target components and training runs** — "frame-pooled"
refers to the estimand (AUC computed over crops) and never to the resampling unit.

Seed-subset sensitivity covers **all ten 3-of-5 and all five 4-of-5 subsets** plus
the full five, not one arbitrary sequence. These subsets overlap heavily, so they
are a sensitivity analysis, not independent replication, and no exclusion
frequency will be computed from them.

## Primary quantity — decided in advance

Report **P(replicate > +0.014)**, not only whether the 97.5th percentile clears
it. The two are equivalent — the endpoint test is `P > 0.025` — but the proportion
has closed-form binomial Monte Carlo error (±0.002 at 2σ for B=50,000 near
p=0.04), does not depend on the density near a quantile, and degrades gracefully
instead of forcing a binary verdict on a knife-edge.

Also report P(replicate > 0) and P(replicate < −0.014) to characterise the
distribution against the reference scale in both directions.

## Monte Carlo error — two independent estimates

1. **Batch spread**: sd of the endpoint across the five 4,000-replicate runs.
   Scales as 1/√B, so the 50,000-replicate MC sd ≈ batch sd × 0.283.
2. **Order statistic**: the 97.5th percentile of B draws has rank
   ~Binomial(B, 0.975); at B=50,000 the rank sd is ≈35, and ±2σ of rank converts
   to a value by lookup in the sorted replicate array.

Agreement between the two is the check. Disagreement means the tail is
misbehaving and the endpoint should not be trusted.

## Interpretation rule — fixed before seeing results

Let *U* be the 50,000-replicate 97.5th percentile and *MC* its Monte Carlo band.

| condition | conclusion |
|---|---|
| U − MC > +0.014 | the reference remains **included**; report as such |
| U + MC < +0.014 | the reference is **excluded** under this analysis |
| MC band spans +0.014 | **borderline / bootstrap-sensitive** — report the proportion and both endpoints, force no binary verdict |

## What this check cannot do

More replicates reduce **numerical** Monte Carlo error only. They do not enlarge
the five training runs, do not enlarge the 28–29 components, and do not establish
that the percentile bootstrap has correct coverage with clusters this few. Those
remain limitations regardless of the outcome.

## Freeze

After these runs the experimental results are frozen and tagged. Remaining GPU
work (V2, and anything else) becomes optional robustness, not a prerequisite.
