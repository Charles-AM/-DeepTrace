# Positive and negative controls: what this evaluation can and cannot resolve (2026-09-06)

Every cluster-aware result elsewhere in this project comes from one comparison,
Xception vs Xception + FAD. This runs **all six pairwise comparisons** among the
four c23 configurations, three seeds each, component unit, video-level
aggregation. No GPU — computed from the committed prediction dumps by
`src/pairwise_matrix.py`.

Its job is to act as **empirical controls**, not as a power analysis.

---

## 1. What the 18 comparisons establish

| | |
|---|---|
| observed \|diff\| > 0.10 | **9 of 9** excluded zero |
| 0.03 ≤ \|diff\| ≤ 0.10 | 1 of 1 excluded zero *(Xception − spatial, seed 0, +0.0300)* |
| observed \|diff\| < 0.03 | **0 of 8** excluded zero |

⚠️ **This is not a power analysis.** Power is defined against a prespecified true
effect; these are sorted *after the fact* by observed difference, and an observed
difference far larger than its half-width excludes zero close to by construction.
Do not write "selectively powered".

⚠️ These are **related** comparisons — six pairs drawn from four models over three
shared splits — not 18 independent experiments.

What they do establish is a **positive control**: the machinery does produce
zero-excluding intervals when differences are large. Without that, "nothing
excluded zero" could not be distinguished from a broken pipeline. Together with
the negative control at small differences, they locate the practical resolution
boundary for these data somewhere near **0.03 AUC**.

**Defensible statement:** *the evaluation was informative for large architectural
differences but had insufficient resolution for FAD-sized differences.*

## 2. The informative content is the half-width structure

The exclusion counts follow from these, so this table — not the counts — is the
actual finding.

| comparison | half-width range | mean |
|---|---|---|
| Xception+FAD − Xception | 0.0152–0.0292 | 0.0233 |
| Xception+FAD − spatial | 0.0258–0.0408 | 0.0338 |
| Xception − spatial | 0.0273–0.0510 | 0.0377 |
| Xception − freq_only | 0.0575–0.0680 | 0.0620 |
| Xception+FAD − freq_only | 0.0599–0.0688 | 0.0640 |
| freq_only − spatial | 0.0599–0.0747 | 0.0687 |

⚠️ **Correcting an earlier draft of this file**, which called half-widths
"broadly a design property". They are **bimodal**: 0.023–0.038 among the three
spatial-family models, and 0.062–0.069 whenever `frequency_only` is involved — a
~2× jump driven by the weaker model's noise, not by the design.

So the design sets a **floor** of roughly 0.023–0.038 for a matched comparison
between comparable architectures at this scale, which is **1.7–2.7× the +0.014**
under adjudication. Pairing with a much noisier model roughly doubles it.

## 3. One comparison that is easy to misread

⚠️ **Xception + FAD − baseline_spatial = 0.0136 is NOT a matched FAD ablation.**
It changes the backbone (11.34M → 20.81M parameters, different architecture)
*and* adds FAD. F3-Net's published ablation compares Xception + FAD against
matched Xception. The numerical similarity to +0.014 is a coincidence and must
not be presented as the published-sized FAD effect appearing in our data.

Correct use:

> One cross-architecture comparison produced a mean difference of 0.0136 AUC,
> numerically similar to F3-Net's reported FAD ablation gain of 0.014; none of its
> three component-bootstrap intervals excluded zero. This illustrates the limited
> resolution for effects of that scale, but it is **not** a matched estimate of
> FAD's incremental contribution.

Keep it out of the abstract — the mismatched contrast invites an immediate
challenge. The matched FAD-vs-Xception row (mean 0.0072, 0/3 excluding zero) is
the one that speaks to FAD, and the crossed V8 interval is what will make it
properly.

## 4. Scope

c23 only — c40 carries just two configurations. Three seeds per pair, each a
different split, so three related estimates. The 0.03 / 0.10 boundaries are
descriptive of these 18 comparisons, not estimated detection thresholds.

## Files

`pairwise_c23_component.csv` — 18 rows.

```
python -m src.pairwise_matrix --compression c23 --out-dir results/analysis/generality
```

Verified to reproduce the committed CSV byte for byte. `--configs` order is
explicit rather than sorted: it fixes each pair's reference/model orientation and
therefore the sign of every difference.
