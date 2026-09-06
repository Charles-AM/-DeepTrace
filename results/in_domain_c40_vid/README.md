# c40 VIDEO-LEVEL results — C0.1, the leakage fix (2026-09-05)

FF++ c40, **video-level splits** (`--group-by 'videos-([0-9]+)'`), 3 configs × 3
seeds. Same crops as `results/in_domain_c40/` (frame-level), so this is a direct
L1↔L2 protocol comparison on identical data.

Split verified in the run's own log before training: **300 video groups, 0 spanning
more than one split**, train/val/test = 24000/3000/3000, real:fake exactly 4800:19200.

## 1. The leakage finding

| config | L1 (frame) | L2 (video) | inflation |
|---|---|---|---|
| baseline_spatial | 0.9878 | 0.7957 | **+19.2 AUC points** |
| xception | 0.9933 | 0.8169 | **+17.6 AUC points** |
| f3net | 0.9935 | 0.8117 | **+18.2 AUC points** |

**Frame-level splits inflate FaceForensics++ AUC by ~18 points**, remarkably
consistently across three different architectures. Every number this project
produced before 2026-09-05 sits on that inflation.

L2 absolute values (~0.79–0.82) now sit *below* F3-Net's published c40 range
(~0.90–0.96), which is consistent with our scoped subset (150 pairs → 240 training
video groups vs full FF++ ~1000 pairs), 128 px inputs, 15 epochs, and frame-level
rather than video-level metric aggregation (see C0b).

## 2. The frequency question, now measured with headroom

Per-seed ROC-AUC:

| config | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| baseline_spatial | 0.7971 | 0.7977 | 0.7925 | 0.7957 | 0.0029 |
| xception | 0.8135 | 0.8527 | 0.7845 | 0.8169 | 0.0342 |
| f3net | 0.7939 | 0.8426 | 0.7987 | 0.8117 | 0.0268 |

**`f3net − xception`, paired: −0.0196, −0.0101, +0.0142 → mean −0.0052**
(sd 0.0174, 95% CI **[−0.048, +0.038]**, p = 0.659).

**The central claim survives the ceiling objection.** With ~18 points of headroom
restored, F3-Net's FAD still does not beat a matched Xception at c40 — it is
marginally *worse* on average, and not significantly different either way. The
previous null was not purely a ceiling artifact.

## 3. ⚠️ The new limiting factor: variance, not ceiling

Seed variance exploded going L1→L2: `xception` sd went from 0.0012 to **0.0342**
(~28×). With 240 training and 30 test video groups, run-to-run variation now
dominates.

**Seed 1 is high for both `xception` (0.8527) and `f3net` (0.8426)** — a *shared*
seed effect, meaning the split itself (which videos land in test) drives more
variance than the architecture does. With only 30 test groups, split luck swamps
architectural differences.

Consequence: the 95% CI half-width is ±0.043. F3-Net's published c40 advantage over
Xception is ≈ +0.035, so **at n=3 we are marginally underpowered to rule out their
claimed effect.** At n=5, t(4)=2.776 gives a half-width of ≈ ±0.022 — comfortably
below their claimed effect, which would let us state that we would have detected it.

**→ 5 seeds on the headline pair is now the top priority, ahead of adding configs.**

Note also `baseline_spatial` has tiny variance (sd 0.0029) while the Xception-family
models are ~10× noisier — consistent with 20.8M parameters overfitting harder on 240
groups (best val epoch was 3 in every f3net run, with train loss still falling).

## Files

`summary.csv` (9 rows), `ablation_c40_vid.md` (means ± sd + paired-t vs xception),
`ablation_table.csv`, `per_run/*.json` (9 raw `test_metrics.json`).
Checkpoints not in git — Kaggle version output.
