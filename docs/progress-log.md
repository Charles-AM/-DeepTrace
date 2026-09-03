# DeepTrace — Progress Log

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
