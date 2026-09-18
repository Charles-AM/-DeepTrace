# V2 prediction dumps — seen vs unseen

Two scorings of **one fixed checkpoint** against two matched, disjoint-group test
sets. These are the substrate for `results/analysis/v2/`; without them that result
would be aggregate-only, like the L1 campaign.

| file | what |
|---|---|
| `v2seen_xception_seed0_test.csv` | held-out crops from videos the model **trained on** |
| `v2unseen_xception_seed0_test.csv` | crops from videos the model **never saw** |

Both: 750 predictions, 30 video groups, identical manipulation composition
(150 each of Deepfakes, Face2Face, FaceSwap, NeuralTextures, real).

⚠️ `predict.py` names its output from `--run`, not `--dataset-name`, so both files
are produced under the **same** name. They are renamed on commit. Writing both to
one directory silently overwrites the first — that happened twice on 2026-09-18.

## Provenance

| | |
|---|---|
| checkpoint | `ffpp_c40_v2seen_xception_seed0/best.pt`, trained 2026-09-18 18:23 |
| training | seen manifest, 23,250 train / 3,000 val / 750 test, 15 epochs, fp32, lr 3e-4 |
| seen test AUC at training time | **0.9884** (best val epoch 6) |
| manifests | built by `src/seen_unseen.py` from the verified L2 seed-0 manifest |
| checkpoint location | Kaggle notebook output `contribution-3` — **not committed**, see `docs/ARTIFACT-POLICY.md` |

## ✅ Committed and independently reproduced, 2026-09-18

Both dumps are committed. Their SHA-256 prefixes match the Kaggle run exactly
(`cdfbb5439525` seen, `3d086527cc66` unseen), and all three estimands recompute from
these copies on a different machine:

| estimand | advantage | 95% CI |
|---|---|---|
| frame-pooled | **+0.1998** | [+0.1428, +0.2543] |
| video mean-logit | +0.1731 | [+0.1044, +0.2352] |
| video mean-probability | +0.1778 | [+0.1109, +0.2384] |

Interval endpoints differ from the Kaggle run in the fourth decimal — independent
RNG draws at 50,000 replicates. Point estimates are exact.

⚠️ **The four membership checks cannot run locally**, because the manifests are not
committed. `verify_v2.py` reports them as **SKIPPED, not FAILED** — an unrunnable
check is not a failing one. They passed on Kaggle with the real manifests (13/13);
to re-verify membership elsewhere, the two V2 manifests must be attached.

## Validation performed before analysis

`python -m src.verify_v2 ...` — **13/13 passed**:

distinct paths and hashes · 750 predictions each · zero exact evaluation crops in
training · all 30 seen groups represented in training · zero unseen groups
represented · manipulation and class composition matching · groups disjoint ·
seen AUC reproducing the training log exactly.

## Regenerate

From the checkpoint (inference only, ~2 min, no GPU needed):

```
python -m src.predict --run ffpp_c40_v2seen_xception_seed0 \
    --results-root <dir containing manifests/ and the run dir> \
    --dataset-name ffpp_c40_v2seen --seed 0 --split-seed 0 --split test \
    --out-dir <out>/seen
```

and again with `--dataset-name ffpp_c40_v2unseen --out-dir <out>/unseen`.
**Separate `--out-dir` is mandatory.**

⚠️ Manifests store **absolute** crop paths, so they resolve only when the crops are
mounted at the same location. Both the `c40-run` crops and the night-1 output must
be attached.
