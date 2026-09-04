# DeepTrace — Progress Log

## 2026-09-03 — Mechanistic analysis (fusion gate, per-manipulation, spectra)

No retraining — c23 checkpoints + crops only. Outputs: `results/analysis/`.

- **Fusion gate α = 0.496 ± 0.001** for every gated config and seed — never moved
  from the 0.5 init. Spatial branch is already at ceiling on c23, loss is flat in
  α, no pressure to re-weight. The learnable fusion behaves as a fixed average.
- **Per-manipulation (3 seeds):** spatial Xception is the best model on all four FF++
  methods including NeuralTextures (the subtlest). `frequency_only` scores best on
  Deepfakes (0.85, crudest) and worst on NeuralTextures (0.62) — the inverse of the
  "frequency reveals subtle traces" argument. Order stable everywhere:
  xception ≥ f3net > full ≈ full_banddrop+sas ≈ baseline_spatial.
- **Spectra (`src/spectra.py`, rewritten):** native-resolution crops, 2-D Hann
  window, DCT-domain fake−real difference + per-coefficient t-map. A real
  real-vs-fake spectral gap exists in FF++ c23 (5.6% of DCT coeffs at |t|>3 vs
  ~0.3% by chance; ~1 dB radial power, larger at high freq) but the **effect size
  is small** (max t≈5.8 at n=3000/class → Cohen's d≈0.15) and **JPEG-q30 attenuates
  it only ~15–25%**, not destroys it. So frequency modelling fails here because the
  signal is marginal and redundant (a spatial CNN captures it), not because it is
  absent or compression-fragile. (First spectra version was flawed — bilinear
  resize to 128 before FFT low-passed away the signal; fixed in fda0d73.)

## 2026-09-03 — Robustness sweep (test-time perturbations, seed-0 all pert + 3-seed JPEG)

Re-evaluated existing FF++ c23 checkpoints under JPEG/blur/noise/resize/contrast at
4 severities (`src/robustness.py`, `--limit 3000`). Outputs: `results/robustness/`.

**JPEG sev4 (~q30), absolute ROC-AUC, 3-seed mean ± sd:**
baseline_spatial 0.846 ± 0.017 · xception 0.848 ± 0.012 · f3net 0.857 ± 0.011 ·
full 0.829 ± 0.002. Per-seed (f3net − xception) = +2.9 / +0.3 / −0.3 → straddles
zero. **F3-Net's apparent JPEG-robustness edge is a seed-0 artifact; it ties the
spatial baselines over 3 seeds.** The proposed hybrid (`full`) is reliably ~2 AUC
below both spatial baselines under JPEG (and under blur/noise), every seed.

**Other perturbations (seed 0):** blur/noise/resize collapse every trained model to
AUC ~0.5 by severity 2. `frequency_only` retains a higher *fraction* of its score
(JPEG 0.94, blur 0.74 vs ~0.5 for others) but its absolute AUC is 0.5–0.65
throughout — robustness by ignorance (it never used the fine detail that both
compression and generation artifacts occupy), not a competitive result.

**Effect on the claim:** the strong version holds under *test-time* compression — no
frequency method beats a spatial CNN under JPEG once seeds are averaged. Caveat: a
c23-trained model tested with JPEG is a proxy for F3-Net's literal train+test-on-c40
setting. The c40 training run (Xception vs F3-Net) remains the one experiment that
fully closes this; the proxy now favours the spatial baseline.

## 2026-09-03 — Phase 2 results in; project reframed as a controlled study

**The proposed method does not beat well-tuned spatial baselines.** Firm result
across 3 seeds + 2 cross-datasets (Celeb-DF v2, DFDC).

In-domain FF++ c23 (3-seed mean ROC-AUC): Xception 0.9977, F3-Net 0.9971,
full_banddrop+sas 0.9949, baseline_spatial 0.9949. F3-Net significantly beats the
proposed model (p=0.049). No component (learnable mask, gated fusion, band-dropout,
SAS) shows a significant AUC gain. `frequency_only` (raw DCT -> CNN) = 0.70 AUC
(near chance) -- the v1 frequency branch is a poor representation (global-average-
pooling over a DCT map discards frequency position).

Cross-dataset (seed 0): on Celeb-DF the spatial baselines (Xception 0.827,
F3-Net 0.820) beat the frequency-hybrid (0.744) by ~8 points; SAS *hurts* Celeb-DF
transfer (0.756 -> 0.744). On DFDC everything clusters at 0.73-0.75.

**Reframe:** controlled analysis / negative-results paper. Main claim: on a matched
backbone, adding frequency-domain modelling does not improve deepfake detection over
a well-tuned spatial CNN. Target paper: F3-Net (ECCV 2020). Next: compression-
robustness sweep (F3-Net's headline claim was on c40), cross-dataset error bars,
mask-interpretability figures.

Fixed `src/cross_dataset.py` and `src/robustness.py` to accept `--out-dir` (they
crashed writing CSVs into a read-only mounted `--results-root`).

## 2026-09-03 — Stage 4: training matrix launched
- FaceForensics++ scoped subset: 1,500 videos (~300 pristine + ~1,200 manipulated,
  150 pairs x 2 directions x 4 methods, c23), via src/ffpp_fastdl.py.
- 30,000 face crops via src/extract_faces_v2.py (6,000 real / 24,000 fake; 160 px,
  MTCNN, every 12th frame, <=20/video). Path-slug filenames, no method collisions.
- Smoke test: full config, 1,500 imgs, 2 ep -> ran end to end, test ROC-AUC 0.857.
- Training matrix launched on Kaggle (T4): 10 configs x 3 seeds x 15 epochs
  (baseline_spatial, xception, efficientnet_b0, f3net, frequency_only, no_mask,
  no_fusion_concat, full, full_banddrop, full_banddrop+sas), seeded frozen
  80/10/10 split. Output: results/ablation_table.md.

## 2026-09-02 — Tasks 9-13
- Task 9  freq_dropout.py  — FrequencyBandDropout (config full_banddrop)
- Task 10 sas.py           — Spectral Artifact Simulation (--sas)
- Task 11 f3net.py         — F3-Net FAD baseline (--config f3net)
- Task 12 robustness.py    — JPEG/blur/noise/downscale/contrast sweep
- Task 13 cross_dataset.py — multi-target zero-shot eval, mean delta-AUC
- ffpp_fastdl.py, extract_faces_v2.py; ~140 unit tests.

## 2026-08-31 — Tasks 1-8 (core system)
- 1 dct.py: differentiable 2-D DCT (verified vs scipy to 1e-12, gradcheck)
- 2 frequency_mask.py: learnable per-coefficient mask
- 3 detector.py + frequency_branch.py + losses.py: hybrid model, gated fusion, focal loss
- 4 data/engine/metrics/train.py: pipeline, seeded frozen splits, 6 metrics
- 5 run_ablation.py: matrix runner + comparison table + paired t-tests
- 6-8 robustness / cross-dataset / visualize.py (mask heatmap, radial, Grad-CAM, t-SNE)
- baselines.py: Xception / EfficientNet via timm

Code developed locally, run on Kaggle Notebooks (T4). See docs/kaggle-setup.md,
docs/phase2-runbook.md.
