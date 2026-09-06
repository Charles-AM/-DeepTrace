# Per-manipulation breakdown, recomputed on video-disjoint predictions (2026-09-06)

Supersedes `../permanip/`, which was computed on the **crop-randomised (L1)
checkpoints** — the leaky protocol this project exists to criticise. Any number
from that directory describes models evaluated on frames from videos they trained
on. **Do not cite `../permanip/`.**

Produced by `src/permanip_preds.py` from the committed `results/predictions/*.csv`.
No GPU, no checkpoints. Each manipulation is scored against the same real videos,
with frames aggregated to one score per video first, matching the paper's
video-level metric.

## Mean video-level AUC across seeds

**c23** (n=3 seeds)

| config | Deepfakes | Face2Face | FaceSwap | NeuralTextures | DF − NT |
|---|---|---|---|---|---|
| baseline_spatial | 0.9355 | 0.9485 | 0.9415 | 0.8918 | +0.0437 |
| Xception | 0.9515 | 0.9544 | 0.9263 | 0.9278 | +0.0237 |
| Xception + FAD | 0.9503 | 0.9519 | 0.9378 | 0.9207 | +0.0296 |
| **frequency_only** | **0.8863** | 0.7023 | 0.6819 | **0.6404** | **+0.2459** |

**c40** (n=5 seeds)

| config | Deepfakes | Face2Face | FaceSwap | NeuralTextures | DF − NT |
|---|---|---|---|---|---|
| Xception | 0.8833 | 0.8576 | 0.8309 | 0.7647 | +0.1187 |
| Xception + FAD | 0.9005 | 0.8478 | 0.8565 | 0.7447 | +0.1558 |

## The one finding that survives the protocol fix

`frequency_only` is **strongest on Deepfakes and weakest on NeuralTextures**, with
a 0.246 spread — five to ten times the spread of any spatial model at the same
compression. Deepfakes is a full generated face swap; NeuralTextures modifies
expression in a small region. The frequency-only model tracks how much of the
frame was synthesised.

This was previously reported from L1 checkpoints. It replicates at L2, so the
protocol correction does not overturn it — unlike the absolute AUCs, which fall
throughout.

⚠️ Descriptive. Three (c23) or five (c40) seeds, no cluster-aware intervals per
manipulation: each cell rests on 30 real and 30 fake videos, so per-cell
uncertainty is larger than the differences between adjacent cells. The
frequency_only spread is large enough to read; nothing else here is.

⚠️ NeuralTextures is hardest for every model at both compressions. That ordering
is consistent across all 22 runs but is not tested here.

## Files

`per_manipulation_l2.csv` (88 rows: run, compression, config, seed, manipulation,
n_real_videos, n_fake_videos, auc).

```
python -m src.permanip_preds --preds results/predictions --out-dir results/analysis/permanip_l2
```


## Scope note

Of the four analyses in `results/analysis/` that predate the protocol fix, only
those that **reload a checkpoint** are invalidated by it: `late_fusion/` and
`cka/` (and this one, now redone). `spectra/` takes a manifest and no model — it
describes the data, so leaky checkpoints do not touch it. Its separate limitation
is frame-level pseudoreplication, recorded in ESSENCE §8.
