# V7 — how much resolution does an FF++ ablation actually have? (2026-09-06)

**Contribution 2's core figure.** Every other interval in this project describes
*our* study. This one asks the general question behind it: given N independent
test videos, how wide is the interval on an architectural difference, and how
large must N be before F3-Net's reported +0.014 FAD gain is resolvable at all?

Produced by `src/resolution_curve.py` from `results/predictions/` (c40, seed 0,
Xception vs Xception + FAD). CPU only, no checkpoints needed.

---

## 1. The headline

| unit | n available | half-width at full n | fitted slope | R² | N for ±0.014 |
|---|---|---|---|---|---|
| **video** | 150 | **0.0403** | **−0.510** | 0.987 | **~1300** ⚠️ |
| **component** | 29 | **0.0361** | −0.353 | 0.902 | ~520 ⚠️ |

⚠️ Both N figures are **projections beyond the observed range**, not measurements.
They assume additional videos would resemble ours in difficulty and correlation
structure.

**At the scale this study operates, the 95% interval is ~0.040 wide either side —
roughly three times the effect being adjudicated.** That is the quantitative form
of the sobering implication ESSENCE.md §7 anticipated, and it is measured now
rather than assumed.

The video-unit slope of **−0.510 (R² = 0.987)** sits almost exactly on the −0.5 of
independent sampling. That matters for the extrapolation: the projection is not
merely a curve fit, it coincides with the rate theory predicts, so extending it is
better founded than an arbitrary trend line. It also says video-level resampling
behaves close to i.i.d. — the correlation that pseudoreplication exploits lives
*within* videos, and grouping by video removes most of it.

## 2. The stricter unit is not the flattering one

The component curve is **shallower (−0.353)**: added components buy less than √n.
Since our components hold ~5.2 videos each, ~520 components is ≈2,700 videos —
about twice what the video curve projects. We report both and do not pick the
smaller number. The disagreement is itself informative: correlation between the
videos inside a component limits what extra components are worth.

Note also that the component curve required **no degeneracy guard at any size**,
while the video curve lost n=5 (69% of draws gave a zero-width interval) and had
a 25th percentile of 0 at n=8 and n=10. A component carries its own real and fake
videos; a video is entirely one class. This is direct evidence for the mechanism
behind the non-monotonic SE noted in `../cluster_boot/README.md` §1.

## 3. Independent validation of the curve

At full n the subsampling curve should reproduce the ordinary cluster bootstrap,
which is computed by a different code path:

| unit | V7 curve at full n | direct bootstrap | agreement |
|---|---|---|---|
| video (n=150) | 0.04025 | 0.0408 | 1.4% |
| component (n=29) | 0.03612 | 0.0370 | 2.4% |

## 4. What this licenses us to say

✅ *"A video-disjoint FF++ ablation of this size resolves differences of about
±0.04 AUC. The +0.014 gain reported for FAD is roughly a third of that, so our
design cannot adjudicate it; on the fitted rate, doing so would take on the order
of a thousand independent test videos."*

❌ Not *"FAD provides no benefit"* — the study is underpowered for the question,
which is a statement about the design, not the architecture.

❌ Not *"published FF++ ablations are wrong."* We have measured **our** subset. A
claim about the literature needs a documented audit of their test-set sizes and
inference procedures, which we have not done.

## 5. Supporting runs

**V6 — threshold sensitivity** (`../thresholds/`, component unit):

| | c23 s0 | c23 s1 | c23 s2 | c40 s0 | c40 s1 | c40 s2 | c40 s3 | c40 s4 |
|---|---|---|---|---|---|---|---|---|
| diff | −0.0061 | −0.0044 | +0.0111 | −0.0303 | −0.0208 | +0.0203 | +0.0306 | +0.0164 |
| excludes +0.014 | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ | ✗ | ✗ |
| excludes +0.020 | ✅ | ✗ | ✗ | ✅ | ✗ | ✗ | ✗ | ✗ |

**Six of eight runs exclude nothing, even at the loosest margin we carry.**
Exactly what §1 predicts. `equivalent_*` is False everywhere: no interval
demonstrates equivalence in both directions.

⚠️ These eight intervals are over eight different test sets. Tallying them is
descriptive; it is not a combined test.

**V4 — aggregation moves the estimate, not just the interval** (`../aggregation/`,
c40, video unit):

| seed | frame-pooled | video-level | shift |
|---|---|---|---|
| 0 | −0.0196 | −0.0303 | −0.0107 |
| 1 | −0.0101 | −0.0208 | −0.0107 |
| 2 | +0.0142 | +0.0203 | +0.0061 |
| 3 | +0.0253 | +0.0306 | +0.0053 |
| 4 | +0.0145 | +0.0164 | +0.0019 |

Same model, same test videos, two aggregations. The shift is non-zero in all five
seeds and **carries the sign of the difference** — video-level aggregation
magnifies whichever direction the estimate already points. This confirms C0b
beyond seed 0: our frame-pooled metric is not the video-level metric published
FF++ work reports, and the two differ in more than precision.

## Files

`curve_{unit}_*.csv` (half-width by n, with `zero_width_frac` and `degenerate`),
`fit_{unit}_*.csv` (slope, R², projected N per margin, extrapolation flags),
`ALL.csv`. Inputs: `results/predictions/*.csv`, committed.

Reproduce:

```
python -m src.resolution_curve --a results/predictions/ffpp_c40_vid_xception_seed0_test.csv \
    --b results/predictions/ffpp_c40_vid_f3net_seed0_test.csv --unit video \
    --out-dir results/analysis/resolution
```

Fixed bootstrap seed: re-running reproduces these CSVs byte for byte.
