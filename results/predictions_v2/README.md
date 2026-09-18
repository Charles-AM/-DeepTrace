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
