# Analysis experiments (2026-09-03)

Mechanistic follow-ups on the c23 checkpoints. All use existing checkpoints /
crops — no retraining.

## `fusion_alpha.csv` — learned gated-fusion weight

`alpha_spatial = sigmoid(alpha_logit)`, the weight on the spatial branch
(`fused = alpha*spatial + (1-alpha)*freq`), init 0.5.

**Result:** alpha = **0.496 ± 0.001** for every gated config (`full`,
`full_banddrop`, `no_mask`) and every seed. The gate never moved from init
(|Δlogit| < 0.023). With the spatial branch already at ~0.997 AUC on c23 the loss
is essentially flat in alpha, so there is no gradient pressure to re-weight. The
"learnable" fusion behaves as a fixed 50/50 average here — corroborates
`baseline_spatial` ≈ `full` and `no_fusion_concat` ≈ `full`.

## `permanip/` — per-forgery-method breakdown, 3 seeds

`seed{0,1,2}.csv` long form (run × method: roc_auc/eer/accuracy, real vs that
method only); `auc_seed{0,1,2}.csv` pivots.

**3-seed mean ROC-AUC:**

| model | Deepfakes | Face2Face | FaceSwap | NeuralTextures |
|---|---|---|---|---|
| xception | 0.9991 | **0.9979** | **0.9992** | **0.9944** |
| f3net | 0.9990 | 0.9975 | 0.9989 | 0.9930 |
| baseline_spatial | 0.9985 | 0.9955 | 0.9978 | 0.9877 |
| full | 0.9976 | 0.9953 | 0.9975 | 0.9884 |
| full_banddrop+sas | 0.9978 | 0.9958 | 0.9978 | 0.9882 |
| frequency_only | 0.852 | 0.668 | 0.665 | 0.617 |

- On **NeuralTextures** (the subtlest manipulation — the case frequency methods
  claim to win) the plain spatial Xception is best; the frequency hybrids match the
  spatial baseline, not beat it.
- `frequency_only` does **best on Deepfakes** (crudest, most artifact-heavy) and
  **worst on NeuralTextures** — the opposite of the frequency-detection premise
  that spectral analysis surfaces subtle traces. The spatial CNN already scores
  0.999 on Deepfakes.
- Ordering is stable every seed, every method:
  `xception ≥ f3net > full ≈ full_banddrop+sas ≈ baseline_spatial`.

## `spectra/` — real-vs-fake frequency gap (`src/spectra.py`, 3000 crops/class, native 160 px)

`radial.csv` (azimuthal DFT power + std bands, c23 & jpeg30), `summary.csv`,
`spectra.png` (radial curves + 2-D DCT fake−real difference + per-coefficient
t-map), `dct_maps.npz` (the 2-D arrays).

| metric | c23 | jpeg30 |
|---|---|---|
| radial power gap fake−real, mean | 0.94 dB | 0.79 dB |
| radial gap, high-freq band (norm.f > 0.5) | 1.24 dB | 0.94 dB |
| DCT coefficients with \|t\| > 3 | 5.6 % | 4.8 % |
| DCT coefficients with \|t\| > 5 | 0.04 % | 0.02 % |
| max \|t\| | 5.76 | 5.55 |

- A stationary real-vs-fake spectral gap **does exist** in FF++ c23 (5.6 % of DCT
  coefficients differ at |t|>3 vs ~0.3 % by chance), concentrated at high
  frequency.
- **Effect size is small** — max t ≈ 5.8 at n = 3000/class → Cohen's d ≈ 0.15 at
  the single most discriminative coefficient.
- **JPEG-q30 attenuates it only ~15–25 %** — it does not "destroy" the signal.
- Conclusion: frequency modelling fails here not because the signal is absent or
  compression-fragile, but because it is **marginal and redundant** — a spatial
  CNN already captures whatever discriminative content it holds
  (`baseline_spatial` ≈ `full`; `frequency_only` = 0.70).
- The high-pass **residual** t-map is numerically identical to the raw t-map (the
  Gaussian high-pass only touches the lowest ~8 % of frequencies, where there is no
  signal) — the residual variant adds nothing; can be dropped.

`spectra/` also once held a flawed v1 (`radial_psd.*`): it bilinear-resized crops
to 128 px before the FFT, low-passing away the very signal being measured. Not
kept. `src/spectra.py` was rewritten (commit fda0d73).
