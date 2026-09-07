# Experiment inventory — counts and scale for the paper

Everything run to date, tallied. Use this for the Experimental Setup section
("we trained N models…") and for the reproducibility statement. Counts are derived
from committed artifacts, not recollection — each row's evidence lives in the named
file, and every training run has a `per_run/*.json` with its full metrics.

Last updated 2026-09-05.

## Training runs

| campaign | protocol | configs | seeds | runs | evidence |
|---|---|---|---|---|---|
| c23 in-domain matrix | frame-level (L1) | 10 | 3 | **30** | `results/in_domain/summary.csv` |
| c40 in-domain | frame-level (L1) | 3 | 3 | **9** | `results/in_domain_c40/summary.csv` |
| c40 in-domain | **video-level (L2)** | 3 | 3 | **9** | `results/in_domain_c40_vid/summary.csv` |
| gate weight-decay control | frame-level (L1) | 2 | 1 | **2** | `results/in_domain/per_run/ffppfix_*.json` |
| **subtotal 2026-09-05** | | | | **50** | 50 `per_run/*.json` files |
| c23 in-domain | video-level (L2) | 5 | 3 | **15** | `results/in_domain_c23_vid/summary.csv` |
| c40 in-domain seeds 3–4 | video-level (L2) | 3 | 2 | **6** | `results/in_domain_c40_vid/summary.csv` |
| **V8** fixed split, varying train seed | video-level (L2), `--split-seed 0` | 2 | 5 | **10** | `results/in_domain_c40_fixedsplit/summary.csv` |
| ~~V8 attempt 1~~ (AMP, invalid) | video-level (L2) | 2 | 5 | *10* | discarded — see `c131486`; DO NOT CITE |
| **total to 2026-09-06** | | | | **81** | (plus 10 discarded) |
| *queued, quota permitting* | | | | | |
| V2 seen/unseen | one fixed model, matched sets | 1 | 1–3 | *1–3* | `docs/post-v8-queue.md` |
| Repeatability audit | 2 in-session repeats of seed 0 | 2 | 1 | *4* | `docs/repeat_audit.py` |

**Configurations exercised** (`src/config.py`): `baseline_spatial`, `xception`,
`efficientnet_b0`, `f3net`, `full`, `full_banddrop`, `full_banddrop+sas`,
`no_mask`, `no_fusion_concat`, `frequency_only`.

## Evaluation-only experiments (no training)

| experiment | scale | evidence |
|---|---|---|
| Robustness sweep | 7 models × 5 perturbations × 4 severities + clean (seed 0); 4 models × JPEG × 4 severities (seeds 1–2) ≈ **170+ evaluations** | `results/robustness/` |
| ~~Per-manipulation (L1)~~ | 6 configs × 3 seeds × 4 methods | `results/analysis/permanip/` — **DO NOT CITE**, leaky checkpoints |
| **Per-manipulation (L2)** | 22 runs × 4 methods = **88 evaluations**, video-level | `results/analysis/permanip_l2/` |
| CKA (representation similarity) | 3 seeds, spatial vs frequency branch + random null | `results/analysis/cka/` |
| Late-fusion complementarity | 3 seeds × 2 combiners (L2-regularised + unregularised) | `results/analysis/late_fusion/` |
| Fusion-gate readout | 14 checkpoints | `results/analysis/fusion_alpha.csv` |
| Spectral analysis | 3,000 crops/class × 2 compression conditions, DCT t-maps + radial PSD | `results/analysis/spectra/` |
| Efficiency profiling | 4 architectures (params / FLOPs / latency) | `results/analysis/efficiency/` |
| Cross-dataset (preliminary) | 2 targets × 1 seed — **superseded**, non-official Celeb-DF sample | see `docs/REPRODUCIBILITY.md` |
| **V1** cluster-aware paired bootstrap | 8 comparisons × 3 units × 2000 replicates | `results/analysis/cluster_boot/` |
| **V7** resolution curve | 3 curves (c40 video/component, c23 video) × 13 sizes × 200 draws × 400 replicates, + 4 c40 seeds | `results/analysis/resolution/` |
| **V6** threshold sensitivity | 8 runs × 2 units × 4 margins | `results/analysis/thresholds/` |
| **V4** aggregation effect | 5 seeds × 2 aggregations | `results/analysis/aggregation/` |
| Stratified-bootstrap sensitivity | 5 seeds × 2 units × 2 procedures = 20 | `results/analysis/sensitivity/` |
| Component-structure report | 300 sequences → 150 components | `results/analysis/clusters/` |
| **Crossed bootstrap** (primary result) | 50,000 replicates × 3 resampling modes; stability batch 5 × 4,000; frame-pooled 5 × 4,000 | `results/analysis/crossed/` |
| **Seed-subset sensitivity** | all ten 3-of-5 and five 4-of-5 subsets + full = **16 crossed analyses** | `results/analysis/crossed/seed_subsets_video.csv` |
| **Target vs component unit sensitivity** | 8 conditional comparisons × 4 units + crossed at 2 units | `results/analysis/unit_sensitivity/` |
| **Pairwise generality matrix** | 6 config pairs × 3 seeds = **18 cluster-aware comparisons** | `results/analysis/generality/` |
| Prediction dumps (substrate for all of the above) | 22 runs × 3,000 crops = **66,000 predictions**, committed | `results/predictions/` |

## Approximate compute

| campaign | wall time |
|---|---|
| c23 matrix (30 runs) | ~10 h |
| c40 frame-level (9 runs + download + extraction) | ~4.6 h |
| c40 video-level (9 runs) | ~3.9 h |
| gate-fix control (2 runs) | ~0.7 h |
| all evaluation-only analyses | ~4 h |
| overnight 2026-09-05 (21 runs, 2 parallel) | ~5.5 h wall / ~8.5 h GPU |
| **≈ total** | **~32 GPU-hours** |

All on Kaggle Notebooks, NVIDIA **T4 ×2**. Environment pinned in
`docs/pip_freeze_2026-09-05.txt` (torch 2.10.0+cu128, verified in the same
container that produced the c40 results).

## Dataset scale

- **FaceForensics++ c23:** 150 pairs → ~1,500 videos → **30,000 face crops**
  (6,000 real / 24,000 fake), 160 px, every 12th frame, ≤20 per video.
- **FaceForensics++ c40:** identical video set re-downloaded at c40 → **30,000
  crops**, enabling a paired c23↔c40 comparison on the same source material.
- Splits: 24,000 / 3,000 / 3,000 train/val/test. At video level this is
  **240 / 30 / 30 video groups** — the small group count is the source of the
  seed variance documented in `results/in_domain_c40_vid/README.md`.

## Protocol variants compared

A deliberate axis of the study, not an accident:

| level | grouping | status |
|---|---|---|
| L1 frame | none | complete (c23, c40) |
| L2 video | `--group-by 'videos-([0-9]+)'` | c40 complete, c23 queued |
| L3 identity | connected components over the FF++ pair graph | not started (C0.4) |

Headline protocol finding: **frame-level splits inflate FF++ AUC by 17.6–19.2
points**, consistent across three architectures.

## Statistical practice

- ≥3 seeds for every headline claim; moving to 5 on the c40 headline pair.
- Paired t-tests on the metric *difference*, not independent per-arm intervals.
- Confidence intervals reported alongside means wherever variance is material.
- Negative and falsified results retained and documented (CKA falsifying the
  redundancy hypothesis; the weight-decay confound; the split leak) — see
  `docs/progress-log.md`.
