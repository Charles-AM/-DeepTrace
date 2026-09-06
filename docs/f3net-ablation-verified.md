# F3-Net ablation numbers — verified from source (closes G2)

**Source:** Qian, Yin, Sheng, Chen, Shao, *"Thinking in Frequency: Face Forgery
Detection by Mining Frequency-Aware Clues"*, ECCV 2020. Verified against
**arXiv:2007.09355v2** (local copy: `~/Downloads/2007.09355v2.pdf`), Fig. 7(a)
p.12 and Table 3 p.14. Verified 2026-09-06.

Every threshold comparison in this project depends on these numbers, and the
specific risk being checked was whether the reported gain is **accuracy or AUC**.
It is reported as both; the value we use is AUC.

## Fig. 7(a), p.12 — main ablation, FaceForensics++ **LQ (c40)**

| ID | FAD | LFS | MixBlock | Acc | AUC |
|---|---|---|---|---|---|
| 1 | – | – | – | 86.86% | **0.893** |
| 2 | ✓ | – | – | 87.95% | **0.907** |
| 3 | – | ✓ | – | 88.73% | 0.920 |
| 4 | ✓ | ✓ | – | 89.89% | 0.928 |
| 5 | ✓ | ✓ | ✓ | 90.43% | **0.933** |

## Table 3, p.14 — FAD component analysis (same LQ task)

| model | Acc | AUC | |
|---|---|---|---|
| Xception | 86.86% | 0.893 | baseline |
| Xception + PAFilters | 87.16% | 0.902 | hand-crafted filters |
| Xception + FAD (`f_base`) | 87.12% | 0.901 | **fixed** filters only |
| Xception + FAD (`f_base + f_w`) | 87.95% | **0.907** | **learnable — the row we match** |

Right half of the same table (band ablation): FAD-Low 0.901, FAD-Mid 0.904,
FAD-High 0.906, **FAD-All 0.907**. `FAD-All` uses all three bands.

## Derived thresholds

| quantity | AUC | Acc |
|---|---|---|
| **ΔFAD (our reference)** | **+0.014** (0.907 − 0.893) | +1.09 pts |
| Δ full F3-Net (**do not use**) | +0.040 (0.933 − 0.893) | +3.57 pts |

**Metric confirmed: AUC.** The paper reports Acc and AUC side by side, so our
AUC-based comparison is on the correct scale. This was the risk that would have
invalidated E2 rather than merely widened it.

## Does their FAD match ours?

Yes, on the two axes that matter:

| | F3-Net | ours |
|---|---|---|
| Filter form | `f_base + f_w` — fixed band indicator plus a learnable additive term | `1[r ∈ B_i] + tanh(w_i)` — same structure |
| Bands | three (low / mid / high), `FAD-All` | three radial bands |
| Reconstruction | inverse-DCT per band, stacked as backbone input | inverse-DCT per band → 9-channel Xception input |

Fixed-filter-only FAD reaches just 0.901, so **+0.014 is specifically the learnable
variant's gain** — the one we implement.

## ⚠️ Protocol differences to disclose (not resolved by this verification)

The +0.014 is a valid AUC anchor, but it was measured under a materially different
setup. These belong in the comparability audit:

| dimension | F3-Net (p.10) | ours |
|---|---|---|
| Frames per video | **270** | ≤20 (every 12th) |
| Dataset scope | full FF++ | 150 pairs |
| Optimiser | SGD, lr 0.002, cosine, momentum 0.9 | AdamW, lr 3e-4 |
| Batch size | 128 | 64 |
| Budget | ~150k iterations | 15 epochs |
| Backbone init | Xception, ImageNet-pretrained | same |

**Metric aggregation is also not identical.** p.10 states: *"for single-frame
methods, we average the accuracy scores of each frame in a video"* and *"we also
average the AUC scores of each frame in a video."* That phrasing is ambiguous and
is **not** the frame-pooled AUC we currently compute. It reinforces C0b (report
video-level aggregation alongside frame-level), and any claim of direct
comparability should be hedged until we match the aggregation.
