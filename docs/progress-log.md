# DeepTrace — Progress Log

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
