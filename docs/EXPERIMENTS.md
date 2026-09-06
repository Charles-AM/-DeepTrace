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
| **total to 2026-09-05** | | | | **50** | 50 `per_run/*.json` files |
| *queued 2026-09-05 overnight* | | | | | |
| c23 in-domain | video-level (L2) | 5 | 3 | *15* | pending |
| c40 in-domain seeds 3–4 | video-level (L2) | 3 | 2 | *6* | pending |
| **projected total** | | | | **71** | |

**Configurations exercised** (`src/config.py`): `baseline_spatial`, `xception`,
`efficientnet_b0`, `f3net`, `full`, `full_banddrop`, `full_banddrop+sas`,
`no_mask`, `no_fusion_concat`, `frequency_only`.

## Evaluation-only experiments (no training)

| experiment | scale | evidence |
|---|---|---|
| Robustness sweep | 7 models × 5 perturbations × 4 severities + clean (seed 0); 4 models × JPEG × 4 severities (seeds 1–2) ≈ **170+ evaluations** | `results/robustness/` |
| Per-manipulation breakdown | 6 configs × 3 seeds × 4 forgery methods = **72 evaluations** | `results/analysis/permanip/` |
| CKA (representation similarity) | 3 seeds, spatial vs frequency branch + random null | `results/analysis/cka/` |
| Late-fusion complementarity | 3 seeds × 2 combiners (L2-regularised + unregularised) | `results/analysis/late_fusion/` |
| Fusion-gate readout | 14 checkpoints | `results/analysis/fusion_alpha.csv` |
| Spectral analysis | 3,000 crops/class × 2 compression conditions, DCT t-maps + radial PSD | `results/analysis/spectra/` |
| Efficiency profiling | 4 architectures (params / FLOPs / latency) | `results/analysis/efficiency/` |
| Cross-dataset (preliminary) | 2 targets × 1 seed — **superseded**, non-official Celeb-DF sample | see `docs/REPRODUCIBILITY.md` |

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
