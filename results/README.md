# Results index

All numbers for the controlled-study paper. Checkpoints (`*.pt`) and the split
manifests are **not** in git — checkpoints live in the Kaggle `deeptrace3` notebook
output; the 80/10/10 splits are seeded and regenerate from
`src/data.py` given the same crop set, seed, and image size (128).

## `in_domain/` — FaceForensics++ c23, 3 seeds, 15 epochs (12 effective), image 128

| file | what |
|---|---|
| `summary.csv` | one row per (config, seed): accuracy/precision/recall/f1/roc_auc/eer on the held-out test split |
| `ablation_table.md` / `.csv` | seed-aggregated mean±std + paired-t p-values vs the reference config |
| `ablation_auc.png` | bar chart of test ROC-AUC per config |
| `per_run/*.json` | raw `test_metrics.json` for each of the 30 runs |

**Headline (3-seed mean ROC-AUC):** xception 0.9977 · f3net 0.9971 · no_mask 0.9954 ·
full_banddrop 0.9954 · full_banddrop+sas 0.9949 · baseline_spatial 0.9949 ·
full 0.9947 · no_fusion_concat 0.9945 · efficientnet_b0 0.9933 ·
**frequency_only 0.7007** (near chance — v1 frequency branch GAPs over a raw DCT map
and loses frequency position).

No frequency component (learnable mask, gated vs concat fusion, band-dropout, SAS)
produces a significant AUC gain. F3-Net > proposed (full_banddrop+sas), p=0.049.

## `robustness/` — test-time perturbations on the c23 checkpoints (`src/robustness.py`, --limit 3000)

| file | what |
|---|---|
| `seed0_all_perturbations.csv` | seed 0, all 5 perturbations (jpeg/blur/noise/resize/contrast) × 4 severities, all 7 key runs |
| `seed0_scores.csv` | seed 0 retention = AUC(severity 4) / AUC(clean) per (run, perturbation) |
| `seed1_jpeg.csv`, `seed2_jpeg.csv` | JPEG-only sweep, seeds 1 & 2, 4 runs (baseline_spatial/xception/f3net/full) |
| `seed1_scores.csv`, `seed2_scores.csv` | matching retention scores |
| `robustness_{jpeg,blur,noise,resize,contrast}.png` | AUC vs severity, all models (seed 0) |

### JPEG severity 4 (≈quality 30) — absolute ROC-AUC

| model | seed0 | seed1 | seed2 | mean ± sd |
|---|---|---|---|---|
| baseline_spatial | 0.855 | 0.827 | 0.858 | 0.846 ± 0.017 |
| xception | 0.839 | 0.842 | 0.862 | 0.848 ± 0.012 |
| f3net | 0.868 | 0.845 | 0.860 | 0.857 ± 0.011 |
| full (proposed) | 0.828 | 0.831 | 0.827 | 0.829 ± 0.002 |

Per-seed (f3net − xception) = **+2.9 / +0.3 / −0.3** — straddles zero. F3-Net's
apparent JPEG-robustness advantage is a seed-0 artifact; over 3 seeds it ties the
spatial baselines. The proposed hybrid (`full`) is reliably ~2 AUC below both
spatial baselines under JPEG (and blur, noise).

`frequency_only` retains a high *fraction* of its score under every perturbation
(JPEG 0.94, blur 0.74) but its absolute AUC stays 0.5–0.65 throughout — insensitive
to the fine detail that carries both compression and generation artifacts.
Robustness by ignorance, not a competitive result.

## `analysis/` — mechanistic follow-ups (2026-09-03, no retraining)

See `analysis/README.md`. Summary:
- **fusion α** never moved from 0.5 init (0.496 ± 0.001, all configs/seeds) — flat
  loss at ceiling, gate is effectively a fixed average.
- **per-manipulation** (3 seeds): spatial Xception best on every method incl.
  NeuralTextures; `frequency_only` best on Deepfakes / worst on NeuralTextures —
  opposite of the frequency premise.
- **spectra** (`src/spectra.py`, native-res, Hann-windowed, DCT t-maps): a real but
  small (Cohen's d ≈ 0.15) real-vs-fake spectral gap exists and is largely
  JPEG-robust (5.6 % → 4.8 % of coeffs at |t|>3). Frequency modelling fails because
  the signal is marginal and redundant, not absent or compression-fragile.

### Not yet run (see `docs/phase3-plan.md`)
- c40 **training** run (Xception vs F3-Net) — F3-Net's literal claim; test-time JPEG
  here is a proxy. The spectra result tempers expectations: the global spectrum
  barely changes c23→jpeg30, so any real F3-Net c40 gain is likely its *local*
  statistics (LFS), not the global spectrum.
- cross-dataset seeds 1 & 2 (seed 0 only so far: Celeb-DF spatial ~0.82 vs
  freq-hybrid ~0.74).
