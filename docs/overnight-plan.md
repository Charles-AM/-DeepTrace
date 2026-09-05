# Overnight validation plan (~15 h budget, 2 parallel GPU slots)

**Context:** C0 revealed frame-level split leakage inflating FF++ AUC by ~20 points
(f3net c40: 0.995 frame-level → 0.794 video-level). Every result in the project was
measured at that ceiling and needs re-verification on video-level (L2) splits.

Kaggle allows 2 concurrent GPU sessions, so run **A and B in parallel** — ~6 h wall
clock, ~11 h quota, leaving margin.

---

## Notebook A — c23 video-level matrix (~5–6 h)

**Why:** the c23↔c40 comparison is only meaningful if both sides use the same
protocol. Also produces the video-level checkpoints needed to re-verify every
mechanistic analysis (CKA, late fusion, gate, per-manipulation).

Five configs, not three — `full` and `frequency_only` are required for the
mechanistic re-runs, and `frequency_only` at L2 is independently interesting: it
was 0.70 AUC *with* leakage, so it may be at or near chance without it.

Settings: GPU T4 x2, Internet ON, attach `ffpp-crops-160`. Delete the starter cell.

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```
```bash
!cd /kaggle/working/-DeepTrace && python -m src.run_ablation --data-root /kaggle/input/datasets/charlesappiahmanu/ffpp-crops-160/ffpp_crops --dataset-name ffpp_c23_vid --configs baseline_spatial xception f3net full frequency_only --seeds 0 1 2 --epochs 15 --image-size 128 --batch-size 64 --reference xception --extra --group-by 'videos-([0-9]+)'
```
```bash
!cd /kaggle/working && cp -r ./-DeepTrace/results ./c23vid_results && mv c23vid_results/ablation_table.md c23vid_results/ablation_c23_vid.md && tar czf c23vid_metrics.tar.gz --exclude="*.pt" --exclude="*/tb/*" c23vid_results && du -h c23vid_metrics.tar.gz
```

## Notebook B — c40 video-level, remaining configs (~4–5 h)

**Why:** the in-flight c40 run covers only `baseline_spatial`/`xception`/`f3net`.
Adding `full` and `frequency_only` completes the config set so c23 and c40 are
symmetric and the mechanistic analyses can run at both compression levels.

**Start this only after the current c40-vid run finishes** (it holds a GPU slot).

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```
```bash
!cd /kaggle/working/-DeepTrace && python -m src.run_ablation --data-root /kaggle/input/notebooks/charlesappiahmanu/c40-run/ffpp_c40_crops --dataset-name ffpp_c40_vid2 --configs full frequency_only --seeds 0 1 2 --epochs 15 --image-size 128 --batch-size 64 --reference full --extra --group-by 'videos-([0-9]+)'
```
```bash
!cd /kaggle/working && cp -r ./-DeepTrace/results ./c40vid2_results && tar czf c40vid2_metrics.tar.gz --exclude="*.pt" --exclude="*/tb/*" c40vid2_results && du -h c40vid2_metrics.tar.gz
```

⚠️ `--dataset-name ffpp_c40_vid2` is deliberately distinct so it builds a fresh
manifest rather than reusing a cached one. `--extra` must stay last.

---

## Tomorrow (cheap, inference-only, on the new checkpoints)

All of these were previously run on leaky checkpoints and must be repeated:

| analysis | script | note |
|---|---|---|
| CKA | `src.cka` | did "distinct not redundant" survive de-leaking? |
| Late fusion | `src.late_fusion` | needs `baseline_spatial` + `frequency_only` at L2 |
| Fusion gate α | `src.gate_readout` | free, reads checkpoints |
| Per-manipulation | `src.permanip` | |
| Band ablation | `src.band_ablation` | C4/C4b, now meaningful with headroom |
| Robustness | `src.robustness` | the JPEG story may change entirely |

## Deferred (worth doing, not tonight)

- **Disentangle leakage from sample size.** The 20-point drop conflates removing
  leakage with collapsing effective sample size (240 video groups vs 24,000
  quasi-independent frames — note the overfitting: best val at epoch 3). Test by
  training L2 on progressively fewer groups (120, 180, 240) to get a scaling curve,
  or by downloading more pairs. A reviewer will ask which effect dominates.
- **L3 identity-level splits** (C0.4) — connected components over the FF++ pair
  graph, or the official FF++ split JSONs. L2 groups by target id only, so
  `004_982` and `982_004` still separate.
- **5 seeds** on the headline pair once the L2 picture is clear.
- **Video-level metric aggregation** (C0b) — cheap, no retraining, needed for
  comparability with published numbers.
