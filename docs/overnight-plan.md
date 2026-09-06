# Overnight validation plan (~15 h budget, 2 parallel GPU slots)

**Context:** C0 revealed frame-level split leakage inflating FF++ AUC by ~18 points.
The video-level c40 run (`results/in_domain_c40_vid/`) confirmed the central claim
survives with headroom restored (`f3net − xception` = −0.005, p = 0.659), **but
shifted the bottleneck from ceiling to variance**: xception seed sd went 0.0012 →
0.0342, CI half-width ±0.043 vs F3-Net's claimed +0.035 advantage. At n=3 we are
marginally underpowered to rule their effect out; n=5 gives ±0.022.

**→ Revised priority (2026-09-05): more seeds beats more configs.**

Kaggle allows 2 concurrent GPU sessions, so run **A and B in parallel** — ~5.5 h
wall clock, well inside a 15 h budget.

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

## Notebook B — c40 video-level, seeds 3 & 4 (~3 h) — **now the higher priority**

**Why:** takes the headline comparison to **n = 5**, dropping the CI half-width from
±0.043 to ≈ ±0.022 — below F3-Net's claimed +0.035 advantage, which converts "not
significant" into "we would have detected an effect of the size they report." That
is the difference between a weak null and a defensible one.

Attach the c40 notebook output. Note `--dataset-name ffpp_c40_vid` is **reused
deliberately here** so the manifests match seeds 0–2 exactly (each seed builds its
own manifest, so seeds 3–4 get fresh ones under the same naming scheme).

```bash
!cd /kaggle/working && rm -rf ./-DeepTrace && git clone -q https://github.com/Charles-AM/-DeepTrace.git ./-DeepTrace && cd ./-DeepTrace && git log --oneline -1
```
```bash
!cd /kaggle/working/-DeepTrace && python -m src.run_ablation --data-root /kaggle/input/notebooks/charlesappiahmanu/c40-run/ffpp_c40_crops --dataset-name ffpp_c40_vid --configs baseline_spatial xception f3net --seeds 3 4 --epochs 15 --image-size 128 --batch-size 64 --reference xception --extra --group-by 'videos-([0-9]+)'
```
```bash
!cd /kaggle/working && cp -r ./-DeepTrace/results ./c40vid_s34_results && tar czf c40vid_s34_metrics.tar.gz --exclude="*.pt" --exclude="*/tb/*" c40vid_s34_results && du -h c40vid_s34_metrics.tar.gz
```

Deferred to a later session: `full` and `frequency_only` at c40 video-level (needed
eventually for symmetry with c23 and for the mechanistic re-runs at both compression
levels, but less urgent than tightening the headline CI).

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
