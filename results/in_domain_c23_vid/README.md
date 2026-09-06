# c23 VIDEO-LEVEL results — C0.2 (2026-09-06)

FF++ **c23**, video-disjoint splits (`--group-by 'videos-([0-9]+)'`), 5 configs ×
3 seeds. Completes the protocol-matched c23↔c40 comparison and adds `full` and
`frequency_only`, which the c40 video-level run does not carry.

## 1. The protocol gap scales with compression

⚠️ **Terminology.** This is the **protocol gap** — crop-randomised (L1) minus
video-disjoint (L2) — *not* a leakage measurement. The two protocols differ in
more than leakage: they induce entirely different train/test partitions, so the
gap confounds seen-video leakage with partition difficulty. Isolating leakage
requires **V2** (one fixed trained model, matched seen/unseen test sets). Until
then, "protocol gap" is what this number is.

| config | c23 gap | c40 gap | **c40 − c23** |
|---|---|---|---|
| baseline_spatial | +0.0955 | +0.1921 | **+0.0966** |
| xception | +0.0814 | +0.1764 | **+0.0950** |
| Xception + FAD | +0.0780 | +0.1817 | **+0.1037** |
| full | +0.0944 | — | — |
| frequency_only | +0.0075 | — | — |

**Heavy compression enlarged the protocol gap by 9.5–10.4 AUC points across all
three matched configurations.**

Lead with this difference-in-differences rather than the ratio (2.01×, 2.17×,
2.33×): the absolute enlargement spans 0.0087 (~9% relative) against the ratio's
0.32 (~15%), so it is the tighter and more defensible statistic. Underlying
absolutes: c23 L1 0.9949/0.9977/0.9971 → L2 0.8993/0.9163/0.9191; c40 L1
0.9878/0.9933/0.9935 → L2 0.7957/0.8169/0.8117.

Plausible mechanism: heavier compression leaves stronger, more video-specific
encoder artifacts, so "recognise this video's compression signature" is an easier
shortcut at c40 than at c23, where more of the available signal is genuine content.

**This is the sharpest link between the paper's two halves.** Frequency methods
claim their largest gains under heavy compression — which is precisely the regime
where crop-randomised evaluation is *most* misleading.

## 2. `frequency_only` shows almost no protocol gap — contrary to prediction

`frequency_only`'s protocol gap is only **+0.0075**, an order of magnitude smaller
than the spatial models (+0.078 to +0.096).

We predicted the opposite: that a frequency model would be *more* protocol-sensitive,
since compression signatures are video-specific and live in the spectrum. That
prediction was wrong. The likely reason is capacity — **exploiting a
seen-video shortcut requires the ability to memorise, and `frequency_only` (176k
trainable parameters, 0.69 AUC) has too little of either.** It cannot benefit
from a shortcut it lacks the capacity to learn. (Stated as a mechanism
hypothesis; V2 would test it directly.)

Consistent with the earlier robustness finding: this model is stable across
conditions because it is weak, not because it is well-founded.

## 3. FAD − Xception, video-level, at both compressions

| | n | per-seed | mean | 95% CI |
|---|---|---|---|---|
| **c23** | 3 | −0.0004, −0.0041, +0.0129 | **+0.0028** | [−0.0194, +0.0250] |
| **c40** | 3 | −0.0196, −0.0101, +0.0142 | −0.0052 | [−0.0484, +0.0381] |
| c40 | 5 | (see `in_domain_c40_vid/`) | +0.0049 | [−0.0185, +0.0283] |

**Small point estimates at both compressions** (|Δ| ≤ 0.005). The intervals shown
are **seed-level diagnostics** — they measure optimisation variability confounded
with split composition, not test-content uncertainty — and none excludes zero,
+0.010, or the verified +0.014. Whether the comparison is *resolvable* with
respect to independent test content is pending V1.

## 4. Compression interaction — no evidence, and the sign opposes F3-Net

The quantity F3-Net's claim implies (FAD should help *more* at low quality):

$$(\text{FAD}-\text{Xception})_{c40} - (\text{FAD}-\text{Xception})_{c23}$$

Per-seed: −0.0192, −0.0060, +0.0013 → **mean −0.0080, 95% CI [−0.0338, +0.0179]**.

The interval **includes zero**, so we find no evidence of compression-dependent
FAD benefit. The point estimate is *negative* — FAD faring relatively worse at c40,
opposite to the reported direction — but it is not significant and must not be
reported as a finding.

⚠️ Computed as an interaction interval, not by comparing two significance labels.

## 5. External validity improved

Video-level c23 absolute AUCs (0.90–0.92) sit far closer to published FF++ ranges
than the frame-level figures (0.99+) ever did, and c40 (0.79–0.82) sits below
F3-Net's LQ numbers as expected for our much smaller subset. Being *below*
published SOTA on a scoped subset is a credible position; being above it was the
red flag that started this whole investigation.

## Files

`summary.csv` (15 rows), `ablation_c23_vid.md`, `ablation_table.csv`,
`paired_summary.csv` (script-derived CIs + subset sensitivity),
`per_run/*.json` (15 raw `test_metrics.json`). Checkpoints in the Kaggle version
output.
