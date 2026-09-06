# c23 VIDEO-LEVEL results — C0.2 (2026-09-06)

FF++ **c23**, video-disjoint splits (`--group-by 'videos-([0-9]+)'`), 5 configs ×
3 seeds. Completes the protocol-matched c23↔c40 comparison and adds `full` and
`frequency_only`, which the c40 video-level run does not carry.

## 1. Leakage inflation scales with compression

Frame-level (L1) minus video-level (L2), matched configs:

| config | c23 L1 | c23 L2 | inflation | c40 L1 | c40 L2 | inflation | ratio |
|---|---|---|---|---|---|---|---|
| baseline_spatial | 0.9949 | 0.8993 | **+0.0955** | 0.9878 | 0.7957 | **+0.1921** | 2.01× |
| xception | 0.9977 | 0.9163 | **+0.0814** | 0.9933 | 0.8169 | **+0.1764** | 2.17× |
| f3net (Xception+FAD) | 0.9971 | 0.9191 | **+0.0780** | 0.9935 | 0.8117 | **+0.1817** | 2.33× |
| full | 0.9947 | 0.9003 | +0.0944 | — | — | — | — |
| frequency_only | 0.7007 | 0.6932 | **+0.0075** | — | — | — | — |

**Crop-randomised splits inflate roughly twice as much at c40 as at c23** — ~18
points versus ~8.5 — and the ratio is consistent across all three matched
architectures (2.0–2.3×).

Plausible mechanism: heavier compression leaves stronger, more video-specific
encoder artifacts, so "recognise this video's compression signature" is an easier
shortcut at c40 than at c23, where more of the available signal is genuine content.

**This is the sharpest link between the paper's two halves.** Frequency methods
claim their largest gains under heavy compression — which is precisely the regime
where crop-randomised evaluation is *most* misleading.

## 2. `frequency_only` is nearly leakage-immune — contrary to prediction

`frequency_only` inflates by only **+0.0075**, an order of magnitude less than the
spatial models (+0.078 to +0.096).

We predicted the opposite: that a frequency model would be *more* leakage-prone,
since compression signatures are video-specific and live in the spectrum. That
prediction was wrong, and the likely reason is capacity — **exploiting leakage
requires the ability to memorise, and `frequency_only` (176k trainable
parameters, 0.69 AUC) has too little of either.** It cannot benefit from a
shortcut it lacks the capacity to learn.

Consistent with the earlier robustness finding: this model is stable across
conditions because it is weak, not because it is well-founded.

## 3. FAD − Xception, video-level, at both compressions

| | n | per-seed | mean | 95% CI |
|---|---|---|---|---|
| **c23** | 3 | −0.0004, −0.0041, +0.0129 | **+0.0028** | [−0.0194, +0.0250] |
| **c40** | 3 | −0.0196, −0.0101, +0.0142 | −0.0052 | [−0.0484, +0.0381] |
| c40 | 5 | (see `in_domain_c40_vid/`) | +0.0049 | [−0.0185, +0.0283] |

**Unresolvable at both compressions.** No interval excludes zero, +0.010, or the
verified published FAD gain of +0.014.

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
