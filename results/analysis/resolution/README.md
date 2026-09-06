# V7 — evaluation resolution at the component scale (2026-09-06)

**Contribution 2's core evidence.** Produced by `src/resolution_curve.py` and
`src/cluster_boot.py` from `results/predictions/` (committed). CPU only.

**Definition.** Interval half-width is $h=(U-L)/2$, where $L,U$ are the
percentile-bootstrap endpoints. Percentile intervals are **not** centred on the
point estimate — measured offsets reach 0.0028, which is 20% of the +0.014
reference effect — so tables report endpoints, never `estimate ± h`.

---

## 1. Primary result

> Across five c40 pipeline runs, source-target-component bootstrap intervals had
> half-widths of **0.037–0.062 AUC**, approximately **2.6–4.5×** the +0.014 FAD
> reference effect. An estimate near zero would therefore not separate the null
> from a reference-sized positive effect at this evaluation scale. Four of five
> intervals included +0.014. The remaining interval, from seed 0, excluded the
> published positive effect through its upper endpoint because its point estimate
> was −0.030; however, its lower endpoint extended to −0.070, preventing any
> conclusion of equivalence or bounded harm. Compatibility with the published
> magnitude varied across pipeline runs despite every run containing the same
> nominal 3,000 test crops.

| seed | n comp. | diff | 95% CI | half-width | ×0.014 | excludes +0.014 |
|---|---|---|---|---|---|---|
| 0 | 29 | −0.0303 | [−0.0702, +0.0039] | 0.0370 | 2.6× | **yes** |
| 1 | 28 | −0.0208 | [−0.0633, +0.0252] | 0.0442 | 3.2× | no |
| 2 | 29 | +0.0203 | [−0.0398, +0.0851] | 0.0624 | 4.5× | no |
| 3 | 28 | +0.0306 | [−0.0206, +0.0773] | 0.0489 | 3.5× | no |
| 4 | 28 | +0.0164 | [−0.0311, +0.0593] | 0.0452 | 3.2× | no |

Video-file unit, secondary: **0.042–0.067** (3.0–4.8×). Do not splice the two
ranges — they are different resampling units.

⚠️ Five intervals over five different splits are descriptive. "One of five" is
not an exclusion *rate*, and these are not a combined test.

⚠️ **Video files are not independent units.** The 150 files are nested within
28–29 source-target components. The component unit is our closest defensible
independence unit and is why it is primary here.

## 2. Compression, on matched test content

> For the three seeds evaluated at both compression levels, c23 and c40 used
> identical test content, verified by hashing complete manipulation–target–source
> membership. Component-bootstrap half-widths were **1.5–2.5× greater at c40** in
> all three matched comparisons. This difference cannot be explained by
> test-partition composition and instead reflects the compression condition
> together with its associated trained models.

| seed | c23 half-width | c40 half-width | ratio | membership SHA-256 |
|---|---|---|---|---|
| 0 | 0.0152 | 0.0370 | 2.43× | `71fa4298df62b46b` |
| 1 | 0.0292 | 0.0442 | 1.52× | `73689d81b0cad66f` |
| 2 | 0.0255 | 0.0624 | 2.45× | `bd1c4e1dc0464bc2` |

The same seed produces the same partition at both compressions because
`make_splits` groups on `videos-([0-9]+)`, which is compression-independent.

**Heavy compression is associated with wider cluster-aware intervals under matched
test content.** Not "c40 has lower achievable resolution" — three matched runs
support an association, not a universal property.

## 3. Stratified-bootstrap sensitivity

`../sensitivity/stratified_bootstrap.csv`. Every reported interval uses the
standard unstratified cluster bootstrap; the class-stratified variant holds each
replicate's class composition fixed.

**Conclusions were unchanged under a class-stratified bootstrap sensitivity
analysis.** Exclusion verdicts flipped in 0 of 10 comparisons; video-unit
half-widths moved by at most 0.0011 (2.4% of the width).

At the **component unit the two procedures are bit-identical** in all five seeds.
That is structural, not luck: a component carries both real and fake videos, so
every cluster is mixed, there is one stratum, and stratification is a no-op. It
can only bite for the video unit, where a file is entirely one class.

## 4. ⚠️ Exploratory — sample-size projection (not a headline result)

Subsampling confirms width falls as evaluated content grows. Converting that into
a required N is **not reliable from 28–29 observed components** and is reported
only as sensitivity.

| c40 seed | slope | half-width @150 files | projected N for ±0.014 |
|---|---|---|---|
| 0 | −0.510 | 0.0421 | 1300 |
| 1 | −0.491 | 0.0445 | 1576 |
| 2 | −0.417 | 0.0668 | 6371 |
| 3 | −0.479 | 0.0610 | 3234 |
| 4 | −0.491 | 0.0483 | 1868 |

**The projection spans 1300–6371 — a 4.9× range across pipeline runs.** Seed 0,
which an earlier draft of this file used alone, is the most favourable of the five
on every axis.

Three further reasons it stays exploratory:

1. **The sampling model is wrong even where the statistical fit is stable.** Our
   150 files come from 28–29 components. Extrapolating to 700 files implicitly
   assumes ~135 components, a structure the fit never observed. A projection
   requires specifying how *both* $N_\text{files}$ and $N_\text{components}$ grow.
2. **Model dependence at the component unit.** Unconstrained fit gives ~520;
   imposing the theoretical $N^{-1/2}$ gives ~223 — a 2.3× disagreement. (The
   video projections happen to agree with theory to within 4%, but see 1.)
3. **A slope near −0.5 is not evidence of independence.** Bootstrap SE scales
   near $n^{-1/2}$ largely by construction of the resampling. Earlier drafts read
   the −0.510 slope as showing video-level resampling behaves i.i.d., and the
   shallower component slope as within-component correlation. Neither inference
   is supported; the component slope is more plausibly small-$n$ instability from
   29 units.

**"Resolvable" here means an exclusion criterion** — 95% half-width ≤ margin,
i.e. the ability to exclude a margin-sized effect when the estimate is near zero.
It is *not* 80% power to detect a true effect of that size.

Fits exclude subsamples below `min_fit_n=10` and any flagged degenerate.

## Files

`curve_{unit}_*.csv`, `fit_{unit}_*.csv`, `ALL.csv`;
`../thresholds/` (V6), `../aggregation/` (V4), `../sensitivity/` (§3).

```
python -m src.resolution_curve --a results/predictions/ffpp_c40_vid_xception_seed0_test.csv \
    --b results/predictions/ffpp_c40_vid_f3net_seed0_test.csv --unit component \
    --out-dir results/analysis/resolution
```

Fixed bootstrap seed: re-running reproduces these CSVs byte for byte.
