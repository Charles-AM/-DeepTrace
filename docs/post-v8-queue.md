# Post-V8 queue — the tick-off list

Everything left that needs a GPU, in priority order. **~1 h 50 min total**, fits one
session. Requires: `c40-run` (crops), the **c23 video-level** output (for G7), and
the **V8 output** (for nothing here, but keep it).

| # | item | what it buys | cost |
|---|---|---|---|
| 1 | **V2** — seen vs unseen, matched | turns "protocol gap" into a **leakage measurement**; removes a hedge from 5 places in ESSENCE | ~80 min |
| 2 | **G7a** — late fusion on c23 L2 checkpoints | a **main-paper** figure currently computed on leaky checkpoints | ~15 min |
| 3 | **G7b** — CKA on L2 checkpoints | supplementary, same defect | ~10 min |
| 4 | **G3** — `band_ablation.py` smoke test | closes the last register gap | ~5 min |

---

## 1. V2 — the one that matters

**The problem.** Our ~18-point L1→L2 gap compares two *different partitions*, so it
confounds seen-video leakage with the two test sets differing in difficulty. That
is why every claim says "protocol gap".

**The design.** One fixed model, two matched evaluation sets. Verified by
`tests/test_seen_unseen.py` to be identical in every respect but seen-vs-unseen:

| | seen_eval | unseen_eval |
|---|---|---|
| crops | 750 | 750 |
| videos | 150 | 150 |
| target groups | 30 | 30 |
| reals | 150 | 150 |
| manipulation mix | 150 each × 4 | 150 each × 4 |
| sampling positions | ranks 0,4,8,12,16 | ranks 0,4,8,12,16 |
| **model saw the video** | **yes** | **no** |

Held-out crops are removed from training, so `seen_eval` measures memorisation of
the *video*, not of the exact crops. Val is untouched, so checkpoint selection is
identical to every other run.

```python
# build the two manifests (CPU, seconds) — repeat for seeds 0,1,2
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.seen_unseen --manifest /kaggle/working/results/manifests/ffpp_c40_vid_seed${s}_sz128.csv --out-root /kaggle/working/results --seed $s; done
```

⚠️ Needs `ffpp_c40_vid_seed{1,2}_sz128.csv`. V8 only regenerated seed 0. Either
regenerate the others first (deterministic, CPU) or run V2 at seed 0 only.

```python
# train on the SEEN manifest: its test split is the seen-video evaluation
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.train --data-root <CROPS> --config xception --dataset-name ffpp_c40_v2seen --seed $s --split-seed $s --epochs 15 --batch-size 64 --lr 0.0003 --weight-decay 0.05 --image-size 128 --num-workers 2 --focal-gamma 2.0 --grad-clip 1.0 --group-by 'videos-([0-9]+)' --out-root /kaggle/working/results; done
```

⚠️ Hyperparameters must match `results/reference/train_args_c40_vid.json` exactly,
and **no `--amp`**. That flag cost a 155-minute run once already.

```python
# score the SAME model against both sets
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do \
  python -m src.predict --run ffpp_c40_v2seen_xception_seed${s} --results-root /kaggle/working/results --dataset-name ffpp_c40_v2seen   --seed $s --image-size 128 --out-dir /kaggle/working/preds_v2; \
  python -m src.predict --run ffpp_c40_v2seen_xception_seed${s} --results-root /kaggle/working/results --dataset-name ffpp_c40_v2unseen --seed $s --image-size 128 --out-dir /kaggle/working/preds_v2_unseen; done
```

**Analysis (to build when the data lands).** `cluster_boot` compares two *models*
on one test set; V2 is one model on two *different* sets, so the samples are
independent rather than paired. The difference needs an independent-samples
cluster bootstrap — resample components within each set separately, take the
difference per replicate. Small script, written once V2 data exists.

**What it licenses.** *"With the model, architecture, training data,
checkpoint-selection rule and test composition all held fixed, evaluating on seen
videos inflated AUC by X points."* That is a leakage measurement, and it also
tests the capacity hypothesis in `results/in_domain_c23_vid/README.md` §2.

## 2–3. G7 — analyses still on leaky checkpoints

Late fusion needs `baseline_spatial` + `frequency_only`. **Only c23 has both at
L2** — c40 never trained `frequency_only`. So it is redone at c23, 3 seeds.

```python
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.late_fusion --spatial-run ffpp_c23_vid_baseline_spatial_seed${s} --freq-run ffpp_c23_vid_frequency_only_seed${s} --results-root <C23VID_RESULTS> --dataset-name ffpp_c23_vid --seed $s --image-size 128; done
```

```python
!cd /kaggle/working/-DeepTrace && for s in 0 1 2; do python -m src.cka --run ffpp_c23_vid_f3net_seed${s} --results-root <C23VID_RESULTS> --dataset-name ffpp_c23_vid --seed $s --image-size 128 --out-dir /kaggle/working/results/analysis/cka_l2; done
```

`spectra/` is **not** affected — `src/spectra.py` takes a manifest and loads no
checkpoint.

## 4. G3 — the last register gap

```python
!cd /kaggle/working/-DeepTrace && python -m src.band_ablation --help && echo SMOKE-OK
```

## Packaging — do not skip

```python
!cd /kaggle/working && tar czf postv8.tar.gz results preds_v2 preds_v2_unseen && du -h postv8.tar.gz
```

The V1 notebook's outputs were not retained and its prediction dumps were lost.
Download this, and commit the prediction CSVs to the repo as we did for V1.
