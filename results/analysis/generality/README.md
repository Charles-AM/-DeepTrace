# Does the resolution limit generalise beyond the FAD pair? (2026-09-06)

Every cluster-aware result so far came from one comparison: Xception vs
Xception + FAD. A reviewer can reasonably ask whether the wide intervals are a
property of the **evaluation design** or of that particular pair.

All six pairwise comparisons among the four c23 configurations, three seeds each,
component unit, video-level aggregation. **No GPU** — computed from the committed
prediction dumps.

---

## 1. The design resolves large effects and not small ones

| comparison | mean \|diff\| | mean half-width | ratio | excludes 0 |
|---|---|---|---|---|
| freq_only − spatial | 0.2016 | 0.0687 | 2.93 | **3/3** |
| Xception − freq_only | 0.2123 | 0.0620 | 3.43 | **3/3** |
| Xception+FAD − freq_only | 0.2125 | 0.0640 | 3.32 | **3/3** |
| Xception − spatial | 0.0208 | 0.0377 | 0.55 | 1/3 |
| Xception+FAD − spatial | 0.0136 | 0.0338 | 0.40 | 0/3 |
| Xception+FAD − Xception | 0.0072 | 0.0233 | 0.31 | 0/3 |

Across all 18 comparisons:

- **effects above 0.10: 9 of 9 exclude zero**
- **effects below 0.03: 0 of 8 exclude zero**

This is the answer to "is your study just underpowered?" — **no, it is
selectively powered**, and the selection threshold sits above the effect the
literature reports for FAD.

## 2. The sharpest single demonstration

**Xception + FAD − baseline_spatial has a mean difference of 0.0136** — within
0.0004 of F3-Net's published +0.014 FAD gain — and **none of its three intervals
excludes zero**.

An effect of exactly the published magnitude, present in our own data as a point
estimate, cannot be distinguished from zero at this evaluation scale. That is the
paper's claim demonstrated on a live example rather than argued from an interval
width.

## 3. Half-widths are a design property, with one modifier

Across all six pairs: **0.015–0.075, median 0.054**. Comparisons involving
`frequency_only` sit at the wide end (0.062–0.069) and comparisons among the three
spatial models at the narrow end (0.023–0.038).

So the limit is broadly a property of the evaluation design — 28–29 components,
whatever is being compared — with a secondary contribution from how noisy the
weaker model is. It is not an artifact of the FAD comparison.

## 4. Scope

⚠️ c23 only; c40 has just two configurations. ⚠️ Three seeds per pair, and each
seed is a different split, so these are three related estimates, not three
independent ones. ⚠️ The 0.03/0.10 boundary is descriptive of these 18
comparisons, not an estimated detection threshold.

## Files

`pairwise_c23_component.csv` — 18 rows (compression, reference, model, seed,
n_components, diff, ci_lo, ci_hi, halfwidth, excludes_zero).

```
python -m src.cluster_boot --a results/predictions/ffpp_c23_vid_baseline_spatial_seed0_test.csv \
    --b results/predictions/ffpp_c23_vid_f3net_seed0_test.csv --margins 0.014
```
